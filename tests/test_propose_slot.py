"""The booking asks for no time: _propose_slot chooses and says one (agent/slot_proposal.py), with the clinic's free slots faked."""
import asyncio
import datetime
import types

import pytest

main = pytest.importorskip("main")  # needs the full voice-agent environment (torch, nemo ...): runs on the pod, skipped elsewhere


class _Tools:
    def __init__(self, info):
        self.info = info

    async def free_slots(self, doctor, date):
        return self.info


def _run(info, slots, lang="bn", now=None, monkeypatch=None):
    spoken = []

    async def fake_speak(session, text, lang_, **kw):
        spoken.append(text)

    monkeypatch.setattr(main, "_tools", _Tools(info))
    monkeypatch.setattr(main, "_speak", fake_speak)
    if now is not None:
        monkeypatch.setattr(main.slot_proposal, "clinic_now", lambda: now)
    st = types.SimpleNamespace(slots=dict(slots), hold_token=None)
    session = types.SimpleNamespace(call_id="t")
    done = asyncio.run(main._propose_slot(session, st, lang))
    return done, st, spoken


INFO = {"found": True, "sits": True, "doctor_name": "Dr. A. Sen", "doctor_name_bn": "সেন", "valid_slots": ["10:00", "10:15", "10:30", "10:45"],
        "free_slots": ["10:00", "10:15", "10:30", "10:45"], "next": None}


def test_the_caller_is_first_told_whether_the_doctor_sits_that_day_and_asked_to_book_it(monkeypatch):
    # owner 2026-10-18: "কালকে ডাক্তার ... কালকে কি বসছেন উনি" -> first "he sits tomorrow at 10:00. Shall I book that day?", the name only after the yes
    now = datetime.datetime(2026, 10, 17, 10, 12)
    done, st, spoken = _run(INFO, {"doctor_name": "Sen", "date": "2026-10-17"}, now=now, monkeypatch=monkeypatch)
    assert done and st.slots["time_slot"] == "10:45" and st.slots["_time_ok"] == "pending" and st.slots["_sits_told"] == "1"
    assert "আজই" in spoken[0] and "বসবেন" in spoken[0] and "অ্যাপয়েন্টমেন্ট করব?" in spoken[0] and "এক থেকে দুই ঘণ্টা" not in spoken[0]


def test_a_caller_already_told_the_sitting_is_not_told_twice(monkeypatch):
    spoken = []

    async def fake_speak(session, text, lang_, **kw):
        spoken.append(text)

    monkeypatch.setattr(main, "_tools", _Tools(INFO))
    monkeypatch.setattr(main, "_speak", fake_speak)
    monkeypatch.setattr(main.slot_proposal, "clinic_now", lambda: datetime.datetime(2026, 10, 17, 9, 0))
    st = types.SimpleNamespace(slots={"doctor_name": "Sen", "date": "2026-10-17"}, hold_token=None)
    session = types.SimpleNamespace(call_id="t", turn_count=3, last_doctor_enquiry={"doctor_name": "Sen", "date": "2026-10-17", "turn": 2})
    assert not asyncio.run(main._propose_slot(session, st, "bn")) and spoken == [] and st.slots["_time_ok"] == "yes"


def test_a_closing_time_taken_for_between_ten_and_twelve_is_replaced(monkeypatch):
    now = datetime.datetime(2026, 10, 17, 9, 0)
    done, st, spoken = _run(INFO, {"doctor_name": "Sen", "date": "2026-10-17", "time_slot": "12:00"}, now=now, monkeypatch=monkeypatch)
    assert done and st.slots["time_slot"] == "10:00"


def test_a_free_time_the_caller_named_is_kept(monkeypatch):
    done, st, spoken = _run(INFO, {"doctor_name": "Sen", "date": "2026-10-18", "time_slot": "10:30"}, now=datetime.datetime(2026, 10, 17, 9, 0), monkeypatch=monkeypatch)
    assert not done and st.slots["time_slot"] == "10:30" and spoken == []


def test_nothing_left_today_offers_the_next_day(monkeypatch):
    info = dict(INFO, free_slots=[], next={"date": "2026-10-18", "time_slot": "10:00", "alternatives": []})
    done, st, spoken = _run(info, {"doctor_name": "Sen", "date": "2026-10-17"}, now=datetime.datetime(2026, 10, 17, 14, 30), monkeypatch=monkeypatch)
    assert done and st.slots["date"] == "2026-10-18" and st.slots["time_slot"] == "10:00"
    assert "আর কোনো স্লট নেই" in spoken[0] and "আগামীকাল" in spoken[0]


def test_a_day_the_doctor_does_not_sit_goes_to_his_next_sitting_day_never_to_a_time_question(monkeypatch):
    # SPEC CHANGE 2026-10-17: the caller is never asked for a time -- a non-sitting day is moved to the next sitting day, said aloud
    info = dict(INFO, sits=False, valid_slots=[], free_slots=[], next={"date": "2026-10-19", "time_slot": "10:00", "alternatives": []})
    done, st, spoken = _run(info, {"doctor_name": "Sen", "date": "2026-10-17"}, now=datetime.datetime(2026, 10, 17, 9, 0), monkeypatch=monkeypatch)
    assert done and st.slots["date"] == "2026-10-19" and st.slots["time_slot"] == "10:00"
    assert "বসেন না" in spoken[0] and "পরশু" in spoken[0] and "10:00" in spoken[0]


def test_a_doctor_with_no_next_sitting_day_is_left_to_the_old_path(monkeypatch):
    done, st, spoken = _run(dict(INFO, sits=False, valid_slots=[], free_slots=[], next=None), {"doctor_name": "Sen", "date": "2026-10-19"}, monkeypatch=monkeypatch)
    assert not done and "time_slot" not in st.slots and spoken == []
