"""KCD-501: every call leaves a complete record.

"Intent, actions taken, values confirmed, escalation reason and outcome are written
within thirty seconds of the call ending. The write is idempotent and a dropped call
still records what was completed."

How each part is met:

  * WRITTEN AS THE CALL GOES. Every event (a language, an intent, a completed action,
    confirmed values, an escalation) is queued the moment it happens and flushed
    right away, so a call that drops -- socket closed, process killed, network cut --
    already has everything that was completed before it dropped. The end-of-call write
    only adds the outcome.

  * IDEMPOTENT. Each event carries a sequence number that this recorder assigns ONCE
    and keeps until the server acknowledges it (clinic-api applies an event only if
    its `seq` is higher than the last applied). A retry after a lost response is
    acknowledged as a duplicate and changes nothing: no doubled action, no doubled
    intent.

  * WITHIN THIRTY SECONDS. finish() flushes with a deadline (FINISH_DEADLINE_S, 25 s),
    retrying failed events with a short backoff until it either drains or the deadline
    passes; the un-flushed count is returned and logged so a miss is visible, never
    silent.

  * MINIMAL. Values confirmed are redacted server-side (clinic-api redact_confirmed):
    an allow-list of fields, a phone as its last four digits, no name and no address.

Pure asyncio around an injected writer: tests use a fake that fails, duplicates and
drops on demand.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Awaitable, Callable

FINISH_DEADLINE_S = 25.0
RETRY_BACKOFF_S = (0.2, 0.5, 1.0, 2.0)

Writer = Callable[[str, int, str, dict, "str | None"], Awaitable[dict]]
BatchWriter = Callable[[str, "list[dict]", "str | None"], Awaitable[dict]]

# Owner, 2026-10-29: one commit per event was the clinic API's largest steady write load (a commit costs ~150 ms on the network volume). With a batch writer, events are held up to this long and sent
# together in one request. A COMPLETED ACTION, an escalation and the end of the call are never held: they go at once, so a call that drops keeps everything that mattered.
BATCH_DELAY_S = 2.0
URGENT_KINDS = frozenset({"action", "escalation", "end"})


@dataclasses.dataclass
class _Event:
    seq: int
    kind: str
    payload: dict


class CallRecorder:
    def __init__(
        self,
        call_id: str,
        writer: Writer,
        caller_phone: str | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep=asyncio.sleep,
        batch_writer: BatchWriter | None = None,
        journal=None,
    ):
        self.call_id, self.caller_phone = call_id, caller_phone
        self._writer, self._clock, self._sleep = writer, clock, sleep
        self._batch_writer = batch_writer
        self._flush_timer: asyncio.Task | None = None
        # Owner, 2026-10-29: with a journal (agent/call_journal.py) NOTHING is sent to the clinic API during the call: each event is appended to a local file by one writer thread, the whole call is
        # written as one line when it ends, and a nightly batch loads it into the database (tools/call_log_archiver.py).
        self._journal = journal
        self._all_events: list[dict] = []
        self._seq = 0
        self._pending: list[_Event] = []
        self._lock = asyncio.Lock()
        self.acknowledged = 0
        self.finished = False
        self.unflushed_at_finish = 0

    # -- events ----------------------------------------------------------
    def add(self, kind: str, payload: dict | None = None) -> int:
        """Queue an event; returns its sequence number. Cheap and synchronous, so it can be
        called from anywhere in a turn without awaiting a network round trip."""
        self._seq += 1
        self._pending.append(_Event(self._seq, kind, dict(payload or {})))
        if self._journal is not None:
            ev = {"seq": self._seq, "kind": kind, "payload": dict(payload or {})}
            self._all_events.append(ev)
            self._journal.append(self.call_id, ev)
        return self._seq

    def start(self, channel: str = "voice", disclosure_version: str = "") -> None:
        self.add("start", {"channel": channel, "disclosure_version": disclosure_version})

    def language(self, lang: str) -> None:
        self.add("language", {"language": lang})

    def intent(self, name: str) -> None:
        self.add("intent", {"intent": name})

    def action(self, name: str, ref: str | None = None) -> None:
        self.add("action", {"name": name, "ref": ref})

    def confirmed(self, slots: dict) -> None:
        self.add("confirmed", {"slots": dict(slots)})

    def history(self, statement_ids: list[str]) -> None:
        if statement_ids:
            self.add("history", {"statements": list(statement_ids)})

    def patient(self, patient_ref) -> None:
        self.add("patient", {"patient_ref": patient_ref})

    def satisfaction(self, payload: dict) -> None:
        """The call's implicit happiness score (agent/call_score.py): counts, flags and reasons, never any text."""
        self.add("satisfaction", payload)

    def escalation(self, reason: str) -> None:
        self.add("escalation", {"reason": reason})

    def reask(self, payload: dict) -> None:
        """A re-ask or a tool/LLM failure ("sorry, didn't catch that") -- WHICH reason, per call, not just the
        running count already in satisfaction_json (agent/outcome_metrics.py's process-wide counters reset on
        restart and are not tied to a call_id)."""
        self.add("reask", dict(payload))

    # -- delivery --------------------------------------------------------
    @property
    def pending(self) -> int:
        return len(self._pending)

    def flush_soon(self) -> None:
        """Deliver what is queued: at once when it holds an urgent event (or there is no batch writer), else after BATCH_DELAY_S so that the events of a turn travel together."""
        if self._journal is not None:
            return  # journal mode: nothing is sent during the call
        if self._batch_writer is None or any(e.kind in URGENT_KINDS for e in self._pending):
            asyncio.ensure_future(self.flush())
            return
        if self._flush_timer is None or self._flush_timer.done():
            self._flush_timer = asyncio.ensure_future(self._flush_after_delay())

    async def _flush_after_delay(self) -> None:
        await self._sleep(BATCH_DELAY_S)
        await self.flush()

    async def flush(self) -> int:
        """Send what is queued, in order. An event stays queued (with its seq) if the
        write fails, and everything after it waits behind it so order is preserved.
        Returns how many are still pending."""
        if self._journal is not None:
            return 0 if self.finished else len(self._pending)  # the journal holds the events; the call's single line is written by finish()
        async with self._lock:
            if self._batch_writer is not None and self._pending:
                batch = list(self._pending)
                try:
                    await self._batch_writer(self.call_id, [{"seq": e.seq, "kind": e.kind, "payload": e.payload} for e in batch], self.caller_phone)
                except Exception:  # noqa: BLE001 - the whole batch stays queued for the retry (the server skips what it already has)
                    return len(self._pending)
                del self._pending[: len(batch)]
                self.acknowledged += len(batch)
                return len(self._pending)
            while self._pending:
                ev = self._pending[0]
                try:
                    await self._writer(self.call_id, ev.seq, ev.kind, ev.payload, self.caller_phone)
                except Exception:  # noqa: BLE001 - any failure leaves it queued for the retry
                    break
                self._pending.pop(0)
                self.acknowledged += 1
        return len(self._pending)

    async def finish(self, outcome: str) -> int:
        """Close the record. Retries until drained or FINISH_DEADLINE_S; returns the number
        of events that could NOT be written (0 is the goal)."""
        if not self.finished:
            self.add("end", {"outcome": outcome})
            self.finished = True
        if self._journal is not None:
            try:
                await asyncio.wait_for(asyncio.wrap_future(self._journal.close_call(self.call_id, self.caller_phone, list(self._all_events))), FINISH_DEADLINE_S)
                self._pending.clear()
                self.unflushed_at_finish = 0
            except Exception:  # noqa: BLE001 - reported, never raised: the in-flight file still holds every event
                self.unflushed_at_finish = len(self._pending)
            return self.unflushed_at_finish
        deadline = self._clock() + FINISH_DEADLINE_S
        attempt = 0
        left = await self.flush()
        while left and self._clock() < deadline:
            await self._sleep(RETRY_BACKOFF_S[min(attempt, len(RETRY_BACKOFF_S) - 1)])
            attempt += 1
            left = await self.flush()
        self.unflushed_at_finish = left
        return left
