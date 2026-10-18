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
    assert "আমি নম্বরটা শুনেছি 9 6 3 5 5 8 3 0 1 4। রোগীর বয়স কত?" in _said(t, 6)  # the whole number, read back, with the age asked next

def test_in_bengali_the_answer_starts_with_hanyan_not_ji():
    t = _run("আমি কি সোনা স্কান এর সাথে কথা বলছি")
    assert _said(t, 0).startswith("হ্যাঁ, আপনি সোনোস্ক্যান কলকাতা")


def test_anything_the_agent_said_can_be_asked_back_across_scripts():
    # the call is in Hindi (the caller asked for it); the address is read in Hindi, the caller asks back in Bengali letters
    t = asyncio.run(text_call_sim.run(["आप लोगों का एड्रेस क्या है", "বেনিয়া পুকুর কেয়া"], lang="hi"))
    assert "बेनियापुकुर" in _said(t, 0)
    assert "सही" in _said(t, 1) and "बेनियापुकुर" in _said(t, 1) and "समझ नहीं पाई" not in _said(t, 1)


def test_a_test_that_needs_a_prescription_asks_for_it_and_understands_a_mangled_no():
    t = _run("এইচ আই ভি টেস্ট করাতে চাই", "আমার কাছে প্রেস্কৃপ্তিন এ লেখা নাই")
    assert "প্রেসক্রিপশন আছে?" in _said(t, 0)
    assert "হোয়াটসঅ্যাপ নম্বরে" in _said(t, 1) and "৯৬৩৫৫৮৮৩০৬" in _said(t, 1) and "বুঝতে পারিনি" not in _said(t, 1)


def test_asked_whether_the_tests_are_done_here_the_answer_starts_with_yes():
    t = _run("আমাকে বলুন তো ওখানে আপনাদের ওই সিবিসি আর লিপিড প্রোফাইল এই দুটো টেস্ট করবো হয় কি")
    assert _said(t, 0).startswith("হ্যাঁ, ") and "আমাদের এখানে হয়।" in _said(t, 0).split("।")[0] + "।" and "টাকা" in _said(t, 0)  # the yes names the tests, then the rates


def test_each_read_back_asks_the_next_question_and_the_next_answer_is_the_yes():
    t = _run(
        "ডাক্তার বিশ্বজিৎ সেনের অ্যাপয়েন্টমেন্ট নেব", "কালকেই করে দিন", "সায়ন দাস", "9876543210", "৪৫", "বেনিয়াপুকুর কলকাতা", "বেনিয়াপুকুর থানা",
    )
    assert "আপনার নামটা বলবেন" in _said(t, 1)
    assert "আমি নামটা শুনেছি সায়ন দাস। আপনার নম্বরটা বলবেন" in _said(t, 2) and "এটা কি ঠিক?" not in _said(t, 2)
    assert "আমি নম্বরটা শুনেছি 9 8 7 6 5 4 3 2 1 0। রোগীর বয়স কত?" in _said(t, 3)  # the number was the yes for the name
    assert "রোগীর বয়স 45 বছর। ঠিকানাটা বলবেন?" in _said(t, 4)  # the age was the yes for the number
    assert "থানার নামটা বলবেন?" in _said(t, 5)  # no police station in the address
    last = _said(t, 6)
    assert "ঠিকানা বেনিয়াপুকুর কলকাতা, থানা বেনিয়াপুকুর থানা শুনেছি" in last and last.rstrip().endswith("ঠিক আছে?")  # the address goes into the closing question

# ---- the 2026-10-18 issues list
def test_a_picture_of_a_body_part_says_which_tests_there_are_before_the_prescription_line():
    t = _run("আমার ওই পেটের ছবি তুলব কোন টেস্ট এ আমার মনে নাই পেটের ছবি তুলতে হবে ডাক্তার বলেছিল কখন আসবো কত টাকা পড়বে")
    said = _said(t, 0)
    assert "পেটের ছবির জন্য আমাদের এখানে এই টেস্টগুলো হয়" in said and "টাকা" in said
    assert said.index("এই টেস্টগুলো হয়") < said.index("প্রেসক্রিপশনের একটা ছবি")


def test_a_clear_bengali_sentence_is_answered_in_bengali_even_when_the_call_was_locked_to_english():
    t = asyncio.run(text_call_sim.run(["আমার বুকে খুব ব্যাথা করছে কোন ডাক্তার দেখাবো আপনাদের ওখানে কে আছে"], lang="en"))
    said = _said(t, 0)
    assert any("ঀ" <= c <= "৿" for c in said) and "Dr." not in said


def test_the_day_the_doctor_does_not_sit_is_named_with_no_acknowledgement_of_it_first():
    t = _run("ডাক্তার বিশ্বজিৎ সেনের অ্যাপয়েন্টমেন্ট নেব", "কালকেই করে দিন", "আজকে পাওয়া যাবে")
    said = _said(t, 2)
    assert "আজ ডাক্তার বসেন না" in said and "বদলে" not in said and "করে দিলাম" not in said


def test_a_changed_name_is_read_back_and_the_number_is_not_said_twice_or_short():
    t = _run(
        "ডাক্তার বিশ্বজিৎ সেনের অ্যাপয়েন্টমেন্ট নেব", "কালকেই করে দিন", "সায়ন দাস", "ঠিক", "নয় দুই এক তিন দুই এক চার এক জিরো", "শেষে একটা জিরো হবে আরেকটা",
        "হ্যাঁ", "নামটা চাঞ্জ হবে বিপ্লব সাহা",
    )
    assert "পেয়েছি" in _said(t, 4) and "9 2 1 3 2 1 4 1 0 পেয়েছি" in _said(t, 4)  # nine digits so far
    assert "আমি নম্বরটা শুনেছি 9 2 1 3 2 1 4 1 0 0" in _said(t, 5) and _said(t, 5).count("9 2 1 3 2 1 4 1 0 0") == 1  # the zero went on the end; said once
    assert "ঠিক আছে, 9213214100" not in _said(t, 5)
    assert "আমি নামটা শুনেছি বিপ্লব সাহা" in _said(t, 7)


def test_the_whole_order_name_number_age_address_each_read_back_then_payment_notice_then_the_arrival_question():
    t = _run(
        "ডাক্তার বিশ্বজিৎ সেনের অ্যাপয়েন্টমেন্ট নেব", "কালকেই দিল আমাকে অযাপ্ফোন্টমেন্ট টা", "সায়ন দাস", "ठीक",
        "নয় আট সাত ছয় পাঁচ চার তিন দুই এক শূন্য", "হ্যাঁ", "পঁয়তাল্লিশ", "ঠিক আছে", "বেনিয়াপুকুর কলকাতা", "বেনিয়াপুকুর থানা", "হ্যাঁ ঠিক আছে", "হ্যাঁ ঠিক আছে",
    )
    assert "আপনার নামটা বলবেন" in _said(t, 1)
    assert "আমি নামটা শুনেছি সায়ন দাস। আপনার নম্বরটা বলবেন" in _said(t, 2)  # the name, with the number asked in the same breath
    assert "আপনার নম্বরটা বলবেন" in _said(t, 3)  # a Hindi "ठीক" is a yes; the number is asked again
    assert "আমি নম্বরটা শুনেছি 9 8 7 6 5 4 3 2 1 0। রোগীর বয়স কত?" in _said(t, 4)  # the number, with the age asked next
    assert "রোগীর বয়স কত?" in _said(t, 5)  # a bare yes: the age is asked again
    assert "রোগীর বয়স 45 বছর। ঠিকানাটা বলবেন?" in _said(t, 6)
    assert "ঠিকানাটা বলবেন?" in _said(t, 7)
    assert "থানার নামটা বলবেন?" in _said(t, 8)  # no police station in the address
    last_question = _said(t, 9)
    assert "ঠিকানা বেনিয়াপুকুর কলকাতা, থানা বেনিয়াপুকুর থানা শুনেছি" in last_question
    assert last_question.index("চল্লিশ টাকা") < last_question.index("চলে আসুন") and last_question.rstrip().endswith("ঠিক আছে?")
    assert "আপনি সিরিয়াল নম্বর পেয়ে যাবেন" in last_question and "এক থেকে দুই ঘণ্টা সময় নিয়ে আসবেন" in last_question
    closing = _said(t, 10)
    assert "স্লট রাখা হয়েছে" in closing and "বুকিং আইডি" in closing and "KCD-" not in closing

    async def cancel():
        tools = text_call_sim.main_module().ClinicToolsClient(text_call_sim.main_module().CLINIC_API_BASE)
        for b in (await tools.lookup_bookings(phone="9876543210")).get("bookings") or []:
            await tools.cancel_appointment(b["confirmation_id"], True)

    asyncio.run(cancel())