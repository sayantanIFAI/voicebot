"""The owner's booking order, driven through the REAL turn logic with typed sentences (tools/text_call_sim.py). Pod only: it needs the intent model (Ollama), the clinic
API and the fast-path catalogue.

    cd /workspace/kolkata-care-voice-agent && source deploy/env.sh && VOICEBOT_POD_SIM=1 /workspace/venv/bin/python3 -m pytest tests/test_text_call_sim_pod.py -q

Each scenario was a real failure on a real call (2026-10-17 / 18). The last scenario writes a booking for a test phone number; the test cancels it afterwards.
"""

import asyncio
import os
import sys

import pytest

if os.environ.get("VOICEBOT_POD_SIM") != "1":
    pytest.skip("pod-only: set VOICEBOT_POD_SIM=1 on the pod", allow_module_level=True)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))

import text_call_sim  # noqa: E402


def _run(*utterances):
    return asyncio.run(text_call_sim.run(list(utterances)))


def _said(turns, i):
    return " ".join(turns[i][1])


def test_asked_in_one_breath_whether_he_sits_the_doctor_sitting_is_told_first_then_the_name():
    t = _run("হ্যাঁ ওই কালকে ডাক্তার বিশ্বজিৎ সেনের অযাপ্ফোন্টমেন্ট লাগবে কালকে কি বসছেন উনি", "হ্যাঁ")
    assert "বসবেন" in _said(t, 0) and "অ্যাপয়েন্টমেন্ট করব?" in _said(t, 0) and "নামটা" not in _said(t, 0)
    assert "আপনার নামটা বলবেন" in _said(t, 1)


def test_another_day_is_the_day_and_the_sitting_of_that_day_is_told():
    t = _run("ডাক্তার বিশ্বজিৎ সেনের অ্যাপয়েন্টমেন্ট লাগবে কালকে কি বসছেন উনি", "না, পরশু করুন")
    said = _said(t, 1)
    assert "2026-10-19" not in said  # not the day already told
    assert "বসবেন" in said or "বসেন না" in said


def test_a_day_the_doctor_does_not_sit_gives_his_next_sitting_day():
    t = _run("আজকে ডাক্তার সৌরভ মুখার্জীর অ্যাপয়েন্টমেন্ট চাই", "হ্যাঁ")
    assert "বসেন না" in _said(t, 0) and "অ্যাপয়েন্টমেন্ট করব?" in _said(t, 0)
    assert "আপনার নামটা বলবেন" in _said(t, 1)


def test_kalkei_after_the_doctors_day_was_told_keeps_that_day_and_asks_the_name():
    t = _run("ডাক্তার বিশ্বজিৎ সেনের অ্যাপয়েন্টমেন্ট নেব", "কালকেই দিল আমাকে অযাপ্ফোন্টমেন্ট টা")
    said = _said(t, 1)
    assert "2026-10-17" not in said and "কনফার্মেশন নম্বর" not in said and "আপনার নামটা বলবেন" in said

def test_a_number_that_came_out_one_digit_short_is_read_back_so_far_and_never_given_up_on():
    t = _run(
        "ডাক্তার বিশ্বজিৎ সেনের অ্যাপয়েন্টমেন্ট নেব", "কালকেই করে দিন", "সায়ন দাস", "ঠিক",
        "নাইন সিক্স থ্রি ডবল ফাইভল নাইট থ্রি জিরো ওন", "ঠিক আছে",
    )
    assert "9 6 3 5 5 8 3 0 1 পেয়েছি" in _said(t, 4)  # "ফাইভল" is a five; nine digits so far
    assert "বয়স" not in _said(t, 5) and "9 6 3 5 5 8 3 0 1 পেয়েছি। বাকিটা বলুন" in _said(t, 5)  # still on the number (what was heard, and the rest asked for), not on to the age


def test_the_number_given_in_pieces_and_then_again_whole_is_the_whole_one():
    t = _run(
        "ডাক্তার বিশ্বজিৎ সেনের অ্যাপয়েন্টমেন্ট নেব", "কালকেই করে দিন", "সায়ন দাস", "ঠিক", "নয় আট", "সাত ছয়",
        "নাইন সিক্স থ্রি ডবল ফাইভ এইট থ্রি জিরো ওয়ান ফোর",
    )
    assert "9 6 3 5 5 8 3 0 1 4। এটা কি ঠিক?" in _said(t, 6)


def test_the_whole_order_name_number_age_address_each_read_back_then_payment_notice_then_the_arrival_question():
    t = _run(
        "ডাক্তার বিশ্বজিৎ সেনের অ্যাপয়েন্টমেন্ট নেব", "কালকেই দিল আমাকে অযাপ্ফোন্টমেন্ট টা", "সায়ন দাস", "ठीक",
        "নয় আট সাত ছয় পাঁচ চার তিন দুই এক শূন্য", "হ্যাঁ", "পঁয়তাল্লিশ", "ঠিক আছে", "বেনিয়াপুকুর কলকাতা", "হ্যাঁ ঠিক আছে", "হ্যাঁ ঠিক আছে",
    )
    assert "আপনার নামটা বলবেন" in _said(t, 1)
    assert "এটা কি ঠিক?" in _said(t, 2)  # the name, read back
    assert "আপনার নম্বরটা বলবেন" in _said(t, 3)  # a Hindi "ठीक" is a yes
    assert "এটা কি ঠিক?" in _said(t, 4) and "9 8 7 6 5 4 3 2 1 0" in _said(t, 4)  # the number, read back
    assert "রোগীর বয়স কত?" in _said(t, 5)
    assert "বয়স 45 বছর, ঠিক আছে?" in _said(t, 6)
    assert "ঠিকানাটা বলবেন?" in _said(t, 7)
    assert "ঠিকানা বেনিয়াপুকুর কলকাতা। ঠিক আছে?" in _said(t, 8)
    last_question = _said(t, 9)
    assert last_question.index("চল্লিশ টাকা") < last_question.index("চলে আসুন") and last_question.rstrip().endswith("ঠিক আছে?")
    assert "এক থেকে দুই ঘণ্টা সময় নিয়ে আসবেন" in last_question
    closing = _said(t, 10)
    assert "স্লট রাখা হয়েছে" in closing and "বুকিং আইডি" in closing and "KCD-" not in closing

    async def cancel():
        tools = text_call_sim.main_module().ClinicToolsClient(text_call_sim.main_module().CLINIC_API_BASE)
        for b in (await tools.lookup_bookings(phone="9876543210")).get("bookings") or []:
            await tools.cancel_appointment(b["confirmation_id"], True)

    asyncio.run(cancel())
