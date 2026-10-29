"""Owner, 2026-10-29: the call log is one line per call, loaded into the database once a night, then archived and purged."""

import asyncio
import gzip
import json
import os
import sys

import httpx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

import call_log_archiver as arch  # noqa: E402

from agent import call_record  # noqa: E402
from agent.call_journal import CallJournal  # noqa: E402


def _record_a_call(root, call_id="c-1"):
    async def run():
        j = CallJournal(root, today=lambda: "2026-10-28")
        rec = call_record.CallRecorder(call_id, None, journal=j, caller_phone="9000012345")
        rec.start("voice")
        rec.language("bn")
        rec.flush_soon()
        rec.intent("clinic_faq")
        rec.action("book_appointment", "KC-9")
        left = await rec.finish("completed")
        j.stop()
        return left

    return asyncio.run(run())


def test_a_call_becomes_one_line_in_the_days_journal_and_its_inflight_file_goes(tmp_path):
    root = str(tmp_path)
    assert _record_a_call(root) == 0
    lines = open(os.path.join(root, "2026-10-28.jsonl"), encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 1
    call = json.loads(lines[0])
    assert call["call_id"] == "c-1" and [e["kind"] for e in call["events"]] == ["start", "language", "intent", "action", "end"]
    assert not os.listdir(os.path.join(root, "inflight"))


def test_nothing_is_sent_to_the_clinic_api_during_a_journalled_call(tmp_path):
    async def run():
        sent = []

        async def writer(*a):
            sent.append(a)
            return {}

        j = CallJournal(str(tmp_path), today=lambda: "2026-10-28")
        rec = call_record.CallRecorder("c-2", writer, batch_writer=writer, journal=j)
        rec.intent("clinic_faq")
        rec.flush_soon()
        await rec.flush()
        await rec.finish("completed")
        j.stop()
        return sent

    assert asyncio.run(run()) == []


def _client(status=200, seen=None):
    def handler(request):
        if seen is not None:
            seen.append(json.loads(request.content))
        return httpx.Response(status, json={"applied": 1})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_the_nightly_run_loads_archives_and_purges(tmp_path, monkeypatch):
    root, archive = str(tmp_path / "j"), str(tmp_path / "a")
    os.makedirs(root)
    _record_a_call(root)
    monkeypatch.setattr(arch, "ARCHIVE_DIR", archive)
    seen = []
    res = arch.archive_day(_client(200, seen), os.path.join(root, "2026-10-28.jsonl"))
    assert res["archived"] and res["loaded"] == 1 and res["kept"] == 0
    assert len(seen) == 1 and len(seen[0]["events"]) == 5
    assert not os.path.exists(os.path.join(root, "2026-10-28.jsonl"))
    gz = os.path.join(archive, "2026", "10", "2026-10-28.jsonl.gz")
    assert len(gzip.open(gz, "rt", encoding="utf-8").read().strip().splitlines()) == 1
    csv_text = open(os.path.join(archive, "2026", "10", "2026-10-28.csv"), encoding="utf-8").read()
    assert "c-1" in csv_text and "2345" in csv_text and "9000012345" not in csv_text  # the phone only as its last four digits


def test_a_call_the_database_refused_is_kept_for_the_next_night(tmp_path, monkeypatch):
    root, archive = str(tmp_path / "j"), str(tmp_path / "a")
    os.makedirs(root)
    _record_a_call(root)
    monkeypatch.setattr(arch, "ARCHIVE_DIR", archive)
    monkeypatch.setattr(arch.time, "sleep", lambda s: None)
    res = arch.archive_day(_client(500), os.path.join(root, "2026-10-28.jsonl"))
    assert res["archived"] is False and res["kept"] == 1
    assert len(open(os.path.join(root, "2026-10-28.jsonl"), encoding="utf-8").read().strip().splitlines()) == 1
    assert not os.path.exists(archive)
