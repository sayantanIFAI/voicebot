"""The call log as ONE record per call, written when the call ends, loaded into the database once a night.

Owner, 2026-10-29: "call log must be single event to purge and archive by batch at 1 am" and "no parallel transactions and thread issues". The old way sent every event of a call to the clinic API as the
call went (one database commit each, ~150 ms on the network volume). Now:

  * During the call each event is queued to ONE writer thread, which appends it to a small per-call file (inflight/<call_id>.jsonl). That is a local file append: no network, no database, nothing the
    caller's turn waits for, and it survives a killed process (the events are in the file).
  * When the call ends the writer thread writes ONE line for the whole call to the day's journal (YYYY-MM-DD.jsonl, one fsync) and removes the in-flight file.
  * Once a night (01:00 clinic time, tools/call_log_archiver.py, run by the watchdog) a single process applies the previous days' lines to the database ONE CALL AT A TIME (the batch endpoint, one
    commit per call), writes a CSV summary, compresses the journal into the archive folder and deletes it. Nothing runs in parallel, so there is no transaction or thread contention.

Memory: a call holds a few KB of events; the writer thread holds nothing between calls. Latency: the turn never waits (a queue put). Redis is not needed: the file IS the durable queue and it lives on the
persistent volume.

What the database no longer has until the nightly load: today's call records (history, summaries, audit lines for calls). The bookings, holds and payments themselves are written at once by their own
endpoints and are not part of this.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import queue
import re
import threading

logger = logging.getLogger("call_journal")

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def safe_name(call_id: str) -> str:
    return _SAFE.sub("_", call_id or "unknown")[:80] or "unknown"


class CallJournal:
    """Writes call events to files from a single background thread. `append` and `close_call` return at once."""

    def __init__(self, root: str, today=None):
        self.root = root
        self.inflight = os.path.join(root, "inflight")
        os.makedirs(self.inflight, exist_ok=True)
        self._today = today or self._clinic_today  # a function returning the clinic's date as YYYY-MM-DD (a test can pin it)
        self._q: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="call-journal", daemon=True)
        self._thread.start()

    @staticmethod
    def _clinic_today() -> str:
        import datetime

        return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5, minutes=30)).date().isoformat()

    # -- called from the event loop (never blocks) ---------------------------
    def append(self, call_id: str, event: dict) -> None:
        self._q.put(("append", call_id, event, None))

    def close_call(self, call_id: str, caller_phone: str | None, events: list[dict]) -> concurrent.futures.Future:
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._q.put(("close", call_id, {"caller_phone": caller_phone, "events": events}, fut))
        return fut

    def stop(self) -> None:
        self._q.put(("stop", "", {}, None))
        self._thread.join(timeout=5)

    # -- the one writer thread -----------------------------------------------
    def _run(self) -> None:
        while True:
            op, call_id, data, fut = self._q.get()
            if op == "stop":
                return
            try:
                if op == "append":
                    with open(os.path.join(self.inflight, safe_name(call_id) + ".jsonl"), "a", encoding="utf-8") as f:
                        f.write(json.dumps(data, ensure_ascii=False) + "\n")
                elif op == "close":
                    line = {"call_id": call_id, "caller_phone": data["caller_phone"], "events": data["events"]}
                    path = os.path.join(self.root, self._today() + ".jsonl")
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(line, ensure_ascii=False) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    try:
                        os.remove(os.path.join(self.inflight, safe_name(call_id) + ".jsonl"))
                    except FileNotFoundError:
                        pass
                    if fut is not None:
                        fut.set_result(True)
            except Exception as e:  # noqa: BLE001 - the journal must never take a call down
                logger.warning("call journal %s failed for %s: %s", op, call_id, e)
                if fut is not None and not fut.done():
                    fut.set_exception(e)
