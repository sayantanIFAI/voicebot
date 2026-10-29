"""Owner, 2026-10-29: a call's events travel together (one request, one commit) instead of one commit each."""

import asyncio
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app  # noqa: E402

from agent import call_record  # noqa: E402


def test_the_events_of_a_turn_go_in_one_request_after_a_short_delay():
    async def run():
        sent = []

        async def batch(call_id, events, phone):
            sent.append([e["kind"] for e in events])
            return {"applied": len(events)}

        async def single(*a):  # must not be used
            raise AssertionError("single writer used")

        async def fast_sleep(_s):
            await asyncio.sleep(0)

        rec = call_record.CallRecorder("c1", single, batch_writer=batch, sleep=fast_sleep)
        rec.language("bn")
        rec.flush_soon()
        rec.intent("doctor_availability")
        rec.flush_soon()
        rec.intent("book_appointment")
        rec.flush_soon()
        await asyncio.sleep(0.05)
        return sent, rec.pending

    sent, pending = asyncio.run(run())
    assert sent == [["language", "intent", "intent"]] and pending == 0


def test_a_completed_action_is_sent_at_once_with_what_is_queued_before_it():
    async def run():
        sent = []

        async def batch(call_id, events, phone):
            sent.append([e["kind"] for e in events])
            return {}

        async def never_sleep(_s):
            await asyncio.sleep(3600)

        rec = call_record.CallRecorder("c2", None, batch_writer=batch, sleep=never_sleep)
        rec.intent("book_appointment")
        rec.flush_soon()
        rec.action("book_appointment", "KC-1")
        rec.flush_soon()
        await asyncio.sleep(0.05)
        return sent

    assert asyncio.run(run()) == [["intent", "action"]]


def test_a_failed_batch_stays_queued_and_the_next_one_carries_it():
    async def run():
        calls = []

        async def batch(call_id, events, phone):
            calls.append(len(events))
            if len(calls) == 1:
                raise RuntimeError("clinic API busy")
            return {}

        rec = call_record.CallRecorder("c3", None, batch_writer=batch)
        rec.language("bn")
        left1 = await rec.flush()
        rec.intent("clinic_faq")
        left2 = await rec.flush()
        return calls, left1, left2

    assert asyncio.run(run()) == ([1, 2], 1, 0)


def test_the_server_applies_a_batch_in_order_once_and_a_retry_changes_nothing():
    with clinic_app(sample_patients=False):
        import patient_context as pc
        from db import SessionLocal

        db = SessionLocal()
        try:
            events = [
                {"seq": 1, "kind": "start", "payload": {"channel": "voice"}},
                {"seq": 2, "kind": "language", "payload": {"language": "bn"}},
                {"seq": 3, "kind": "intent", "payload": {"intent": "clinic_faq"}},
                {"seq": 4, "kind": "mystery", "payload": {}},
                {"seq": 5, "kind": "intent", "payload": {"intent": "book_appointment"}},
            ]
            first = pc.record_call_events(db, "call-batch-1", events, "9000000000")
            assert first["applied"] == 4 and first["last_seq"] == 5  # the unknown kind is skipped but does not block seq 5
            again = pc.record_call_events(db, "call-batch-1", events, "9000000000")
            assert again["applied"] == 0 and again["duplicates"] == 5 and again["last_seq"] == 5
            rec = pc.get_call_record(db, "call-batch-1")
            assert rec["languages"] in (["bn"], "bn") or "bn" in str(rec["languages"])
            assert "book_appointment" in str(rec["intents"])
        finally:
            db.close()
