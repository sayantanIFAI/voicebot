"""Nightly: load the previous days' call journals into the database, write a CSV summary, compress and purge.

    python tools/call_log_archiver.py            # every journal file dated before today (clinic time)
    python tools/call_log_archiver.py --all      # including today's (for a test)
    python tools/call_log_archiver.py --dry-run  # say what would be done

Run once a night at 01:00 by deploy/gateway/watchdog.sh. ONE process, ONE call at a time (the clinic API's batch endpoint, one commit per call), so nothing runs in parallel and no transactions
contend. Idempotent: the endpoint skips events it already has, so a night that is interrupted and run again changes nothing twice.

Per journal file (agent/call_journal.py writes one line per call):
  1. every line is POSTed to /api/v1/calls/{id}/events/batch; a line that fails after 3 tries is kept for the next night (the file is rewritten with only those lines);
  2. when ALL lines were accepted: <archive>/<yyyy>/<mm>/<date>.csv (one row per call) and <date>.jsonl.gz are written, the gzip is read back and its line count checked, and only then the journal is deleted.
In-flight files left by a call that died with its process (no end line, older than 6 h) are closed with the outcome "dropped" and folded into the journal of the day they were last written.

Environment: CLINIC_API_BASE (http://127.0.0.1:8080), CLINIC_API_TOKEN, CALL_JOURNAL_DIR (/workspace/call_journal), CALL_ARCHIVE_DIR (/workspace/call_archive).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import gzip
import json
import os
import sys
import time

JOURNAL_DIR = os.environ.get("CALL_JOURNAL_DIR", "/workspace/call_journal")
ARCHIVE_DIR = os.environ.get("CALL_ARCHIVE_DIR", "/workspace/call_archive")
API_BASE = os.environ.get("CLINIC_API_BASE", "http://127.0.0.1:8080").rstrip("/")
ORPHAN_AFTER_S = 6 * 3600


def clinic_today() -> str:
    return (datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=5, minutes=30)).date().isoformat()


def summarise(line: dict) -> dict:
    """One CSV row for a call, from its events (counts and labels only; the phone as its last four digits)."""
    langs, intents, actions, outcome, channel, escalation = [], [], [], "", "", ""
    reasks = 0
    for ev in sorted(line.get("events", []), key=lambda e: e.get("seq", 0)):
        k, p = ev.get("kind"), ev.get("payload") or {}
        if k == "start":
            channel = p.get("channel", "")
        elif k == "language" and p.get("language") and p["language"] not in langs:
            langs.append(p["language"])
        elif k == "intent" and p.get("intent") and (not intents or intents[-1] != p["intent"]):
            intents.append(p["intent"])
        elif k == "action":
            actions.append(str(p.get("name")))
        elif k == "escalation":
            escalation = str(p.get("reason") or "")
        elif k == "reask":
            reasks += 1
        elif k == "end":
            outcome = str(p.get("outcome", ""))
    phone = str(line.get("caller_phone") or "")
    return {
        "call_id": line.get("call_id"),
        "channel": channel,
        "caller_last4": phone[-4:] if phone else "",
        "languages": "|".join(langs),
        "intents": "|".join(intents),
        "actions": "|".join(actions),
        "escalation": escalation,
        "reasks": reasks,
        "outcome": outcome,
        "events": len(line.get("events", [])),
    }


def close_orphans(root: str, now: float | None = None) -> int:
    """Fold in-flight files of calls that died with their process into the journal of the day they were last written."""
    now = now or time.time()
    n = 0
    for path in glob.glob(os.path.join(root, "inflight", "*.jsonl")):
        if now - os.path.getmtime(path) < ORPHAN_AFTER_S:
            continue
        events = []
        with open(path, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if raw:
                    try:
                        events.append(json.loads(raw))
                    except ValueError:
                        continue
        if events and not any(e.get("kind") == "end" for e in events):
            events.append({"seq": max(e.get("seq", 0) for e in events) + 1, "kind": "end", "payload": {"outcome": "dropped"}})
        day = datetime.date.fromtimestamp(os.path.getmtime(path)).isoformat()
        call_id = os.path.basename(path)[: -len(".jsonl")]
        with open(os.path.join(root, day + ".jsonl"), "a", encoding="utf-8") as out:
            out.write(json.dumps({"call_id": call_id, "caller_phone": None, "events": events}, ensure_ascii=False) + "\n")
        os.remove(path)
        n += 1
    return n


def load_line(client, line: dict) -> bool:
    for attempt in range(3):
        try:
            r = client.post(
                f"{API_BASE}/api/v1/calls/{line['call_id']}/events/batch",
                json={"events": line.get("events", []), "caller_phone": line.get("caller_phone")},
            )
            if r.status_code == 200:
                return True
        except Exception:  # noqa: BLE001 - retried, then kept for tomorrow night
            pass
        time.sleep(1.0 * (attempt + 1))
    return False


def archive_day(client, path: str, dry_run: bool = False) -> dict:
    day = os.path.basename(path)[: -len(".jsonl")]
    lines, bad = [], 0
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except ValueError:
                bad += 1
    if dry_run:
        return {"day": day, "calls": len(lines), "loaded": 0, "kept": 0, "bad": bad, "archived": False}
    failed = [ln for ln in lines if not load_line(client, ln)]
    result = {"day": day, "calls": len(lines), "loaded": len(lines) - len(failed), "kept": len(failed), "bad": bad, "archived": False}
    if failed:
        with open(path + ".tmp", "w", encoding="utf-8") as f:
            for ln in failed:
                f.write(json.dumps(ln, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)  # only what still has to be loaded stays
        return result
    out_dir = os.path.join(ARCHIVE_DIR, day[:4], day[5:7])
    os.makedirs(out_dir, exist_ok=True)
    rows = [summarise(ln) for ln in lines]
    csv_path, gz_path = os.path.join(out_dir, day + ".csv"), os.path.join(out_dir, day + ".jsonl.gz")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["call_id"])
        w.writeheader()
        w.writerows(rows)
    with open(path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        dst.write(src.read())
    with gzip.open(gz_path, "rt", encoding="utf-8") as chk:
        back = sum(1 for raw in chk if raw.strip())
    if back != len(lines) + 0:
        raise RuntimeError(f"{gz_path}: wrote {len(lines)} calls, read back {back}; the journal is NOT deleted")
    os.remove(path)
    result["archived"] = True
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include today's journal")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    import httpx

    token = os.environ.get("CLINIC_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    orphans = 0 if args.dry_run else close_orphans(JOURNAL_DIR)
    today = clinic_today()
    files = sorted(p for p in glob.glob(os.path.join(JOURNAL_DIR, "????-??-??.jsonl")) if args.all or os.path.basename(p)[:10] < today)
    print(f"call log archiver: {len(files)} journal file(s), {orphans} dropped call(s) closed")
    failures = 0
    with httpx.Client(timeout=30.0, headers=headers) as client:
        for path in files:
            res = archive_day(client, path, args.dry_run)
            print(json.dumps(res))
            failures += res["kept"] + res["bad"]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
