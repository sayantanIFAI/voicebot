"""Owner's 2026-10-18 batch, agent side (pure functions; the clinic-API half is in test_owner_extras_2026_10_18.py)."""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import booking_capture as bc  # noqa: E402
from agent import doctor_search, nato_letters, owner_answers, phrases, slot_proposal, speech_policy  # noqa: E402
from agent import reply_templates as rt  # noqa: E402


# ---- 8. which branch is this?
def test_the_agent_confirms_it_is_sonoscan_kolkata_and_says_no_to_another_branch(monkeypatch):
    monkeypatch.setattr(owner_answers, "CLINIC_BRANCH", "Kolkata")
    assert "Sonoscan Kolkata" in owner_answers.branch_identity("is it sonoscan", "en")
    assert owner_answers.branch_identity("is it kolkata sonoscan?", "en").startswith("Yes")
    no = owner_answers.branch_identity("is this malda sonoscan", "en")
    assert no.startswith("No") and "Malda" in no and "Kolkata" in no
    assert "কলকাতা" in owner_answers.branch_identity("আপনারা কি সোনোস্ক্যান কলকাতা", "bn")
    assert owner_answers.branch_identity("সোনোস্ক্যান মালদা?", "bn").startswith("না")
    assert owner_answers.branch_identity("sonoscan CBC price", "en") is None  # a question about something else
    assert owner_answers.branch_identity("what is the weather", "en") is None


def test_the_branch_follows_the_configuration(monkeypatch):
    monkeypatch.setattr(owner_answers, "CLINIC_BRANCH", "Malda")
    assert owner_answers.branch_identity("is it malda sonoscan", "en").startswith("Yes")
    assert owner_answers.branch_identity("is it kolkata sonoscan", "en").startswith("No")


# ---- 13. schemes
def test_swasthya_sathi_and_ayushman_are_not_covered():
    for q in ("স্বাস্থ্যসাথী কার্ডে হবে?", "does ayushman bharat card work", "swasthya sathi card", "आयुष्मान कार्ड चलेगा"):
        lang = "bn" if "স্বাস্থ্য" in q else "hi" if "आयुष्मान" in q else "en"
        reply = owner_answers.scheme_cover(q, lang)
        assert reply and ("না" in reply or "नहीं" in reply or "No" in reply)
    assert owner_answers.scheme_cover("what is the price of CBC", "en") is None


# ---- 16. distance
def test_distance_from_the_stations_gives_no_figure_but_30_to_60_minutes_and_the_address():
    for q, key in (("শিয়ালদা স্টেশন থেকে কত দূর", "sealdah"), ("howrah station how far", "howrah"), ("bus stand theke koto dur", "bus"), ("চিতপুর থেকে ড্রাইভার নিয়ে আসবে", "chitpur")):
        assert owner_answers.distance_from(q) == key
    reply = owner_answers.distance_reply("sealdah", "en")
    assert "do not know the exact distance" in reply and "Google Maps" in reply and "30 minutes to 1 hour" in reply and "44, C I T Road" in reply
    assert "৩০ মিনিট থেকে ১ ঘণ্টা" in owner_answers.distance_reply("howrah", "bn")
    assert owner_answers.distance_from("what is the price") is None


# ---- 17 / 18. token
def test_token_twenty_with_a_ten_oclock_doctor_comes_between_eleven_and_twelve():
    assert owner_answers.arrival_window("10:00", 20) == (11, 12)
    assert owner_answers.arrival_window("10:00", 3) == (10, 11)
    assert owner_answers.token_number("আমার টোকেন বিশ") == 20 and owner_answers.token_number("token number 20") == 20
    assert owner_answers.asks_token_arrival("doctor sits at 10 and my token is 20 when should i come")
    reply = owner_answers.token_arrival_reply(20, "10:00", "en")
    assert "11 AM and 12 PM" in reply and "two to five minutes" in reply
    assert "এগারোটা" in owner_answers.token_arrival_reply(20, "10:00", "bn")


def test_a_serial_cannot_be_pushed_up_but_paying_sooner_gives_an_earlier_token():
    assert owner_answers.asks_token_push("আমার টোকেন আগে করে দিন")
    assert owner_answers.asks_token_push("can you move my token up, early token please")
    assert not owner_answers.asks_token_push("what is my token")
    assert "payment link" in owner_answers.token_push_reply("en") and "cannot" in owner_answers.token_push_reply("en")


# ---- 14. visit fee cue
def test_a_visit_fee_question_is_recognised():
    assert owner_answers.asks_visit_fee("ডাক্তারের ভিজিট কত")
    assert owner_answers.asks_visit_fee("what is the visit fee of dr sen")
    assert not owner_answers.asks_visit_fee("I want to visit tomorrow")


# ---- 3. a lady doctor
def test_a_lady_or_male_doctor_is_asked_for_in_all_three_languages():
    assert doctor_search.wanted_sex("I want a lady gynecologist") == "F"
    assert doctor_search.wanted_sex("মহিলা গাইনো ডাক্তার চাই") == "F"
    assert doctor_search.wanted_sex("लेडी डॉक्टर चाहिए") == "F"
    assert doctor_search.wanted_sex("male doctor please") == "M"
    assert doctor_search.wanted_sex("I need a doctor") is None
    reply = doctor_search.filtered_reply({"doctors": ["Dr. A"], "doctors_today": []}, "F", "en", "Gynaecology")
    assert "lady doctors are Dr. A" in reply
    none = doctor_search.filtered_reply({"doctors": [], "doctors_today": []}, "F", "en", "Neurology")
    assert "no lady doctor" in none


# ---- 19. neuro / euro
def test_neuro_and_euro_are_confirmed_and_neurologist_is_not_neurosurgeon():
    existing = {"Neurology", "Neurosurgery", "Urology"}
    plain = doctor_search.confusable("neuro ডাক্তার", existing)
    assert [o["dept"] for o in plain["options"]] == ["Neurology", "Neurosurgery"]
    heard = doctor_search.confusable("ইউরো ডাক্তার চাই", existing)
    assert [o["dept"] for o in heard["options"]] == ["Neurology", "Neurosurgery", "Urology"]
    assert doctor_search.confusable("I need a neurologist", existing) is None  # said in full: no question
    assert doctor_search.confusable("cardio doctor", existing) is None
    assert doctor_search.pick_from("সার্জন", plain["options"])["dept"] == "Neurosurgery"
    assert doctor_search.pick_from("the first one", plain["options"])["dept"] == "Neurology"
    assert "নিউরোলজিস্ট" in doctor_search.question(plain["options"], "bn")


# ---- 9. letters said with words
def test_s_for_september_a_for_alpha_n_for_november_is_san():
    assert nato_letters.collapse("S for September A for Alpha N for November") == "S A N"
    assert nato_letters.collapse("my name is S for Sam, A for Alpha, N for November") == "my name is S A N"
    assert nato_letters.collapse("এস ফর সেপ্টেম্বর এ ফর আলফা এন ফর নভেম্বর") == "S A N"
    assert nato_letters.spelled_run("S A N") == "SAN"
    assert nato_letters.nato_run("alpha bravo charlie") == "ABC"
    assert nato_letters.collapse("hello there") == "hello there"


def _booking(**slots):
    from agent.booking_flow import new_state

    st = new_state("book_appointment")
    st.slots.update(slots)
    return st


def test_a_name_spelled_with_words_is_read_back_for_a_yes():
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00")
    plan = bc.before_model(st, "S for September A for Alpha N for November", "en")
    assert st.pending_name == "San" and "San" in plan.speak


# ---- 6. age said with the number is not part of the number
def test_an_age_beside_the_phone_number_is_kept_apart_from_it():
    assert bc.split_age("আমার বয়স ৪৫ নাম সায়ন দাস")[0] == 45
    assert bc.split_age("সায়ন দাস 9876543210 45 years old") == (45, "সায়ন দাস 9876543210")
    assert bc.split_age("9876543210")[0] is None
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00", patient_name="Rahul Dey")
    st.asked_fields.add("phone")
    plan = bc.before_model(st, "number 9876543210 age 45", "en")
    assert plan.phone == "9876543210" and plan.age == 45


# ---- 10. "ok" in the gaps
def test_a_number_said_in_groups_is_answered_with_ok_in_each_gap():
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00", patient_name="Rahul Dey")
    st.asked_fields.add("phone")
    assert bc.before_model(st, "98 76", "en").speak == "Okay, go on."
    # six digits or more and still short: what was heard is said, so a caller who FINISHED the number knows it came out short (real call 2026-10-18)
    assert bc.before_model(st, "54 32", "en").speak == "Okay, I have 9 8 7 6 5 4 3 2 so far. Please say the rest."
    assert bc.prompt("phone_partial", "bn", n=4) == "ঠিক আছে, বলুন।"


def test_a_digit_word_the_recogniser_lengthened_is_still_that_digit():
    assert bc.read_digits("নাইন সিক্স থ্রি ডবল ফাইভল নাইট থ্রি জিরো ওন")[0] == "963558301"  # "ফাইভল" is a five, "ডবল" doubles it
    assert bc.read_digits("এইটা আমার নাম")[0] == "" and bc.read_digits("তিনি এসেছেন")[0] == ""  # "this one" / "he" are no digits


def test_the_payment_notice_ends_with_the_serial_number_line():
    assert "আপনি সিরিয়াল নম্বর পেয়ে যাবেন।" in slot_proposal.arrival_question("2026-10-19", "10:00", "bn")
    assert "You will get your serial number." in slot_proposal.arrival_question("2026-10-19", "10:00", "en")


# ---- 1 / 5. the doctor sits at ... come then with one to two hours in hand; nothing free -> call again
def test_the_caller_is_told_when_the_doctor_sits_not_a_ten_minute_slot():
    text = slot_proposal.proposal("Dr Sen", "2026-10-30", "10:10", "en", sits_at="10:00")
    assert "10:00" in text and "10:10" not in text and "one to two hours" in text
    bn = slot_proposal.proposal("বিশ্বজিৎ সেন", "2026-10-30", "10:10", "bn", sits_at="10:00")
    assert "10:00" in bn and "10:10" not in bn and "এক থেকে দুই ঘণ্টা" in bn


def test_the_readback_names_the_sitting_hour_unless_the_caller_chose_a_time():
    slots = {"doctor_name": "Sen", "date": "2026-10-30", "time_slot": "10:10", "_sits_start": "10:00", "patient_name": "Rahul"}
    assert "10:00" in rt.booking_confirmation_readback(slots, "book_appointment", "bn") and "10:10" not in rt.booking_confirmation_readback(slots, "book_appointment", "bn")
    chosen = {**slots, "time_slot": "11:00", "_time_explicit": "1"}
    assert "11:00" in rt.booking_confirmation_readback(chosen, "book_appointment", "en")


def test_when_nothing_is_free_the_patient_is_asked_to_call_again_for_the_next_date():
    for lang, word in (("bn", "আবার ফোন"), ("hi", "दोबारा फ़ोन"), ("en", "call again")):
        assert word in slot_proposal.call_again(lang)


# ---- 1. the payment line, spoken
def test_the_payment_notice_comes_before_the_last_question_and_the_closing_line_does_not_repeat_it():
    bn = slot_proposal.arrival_question("2026-10-19", "10:00", "bn")
    assert bn.index("চল্লিশ টাকা") < bn.index("ঠিক আছে?") and "হোয়াটসঅ্যাপে" in bn and "পনেরো মিনিটের মধ্যে" in bn and "বাতিল হয়ে যাবে" in bn
    en = slot_proposal.arrival_question("2026-10-19", "10:00", "en")
    assert "forty rupees within fifteen minutes" in en and "expire automatically" in en and en.endswith("?")
    closing = phrases.PHRASES["bn"]["payment_link_sent"]
    assert "বুকিং আইডি" in closing and "চল্লিশ" not in closing  # said once, before the question
    assert "KCD-" not in rt.booking_reply({}, {"success": True, "doctor_name": "Dr. Sen", "date": "2026-10-30", "time_slot": "10:10", "confirmation_id": "KCD-1"}, "en")



# ---- 2. Bengali is faster
def test_bengali_is_spoken_faster_than_hindi_and_english():
    policy = speech_policy.derive_policy()
    hi = speech_policy.effective_rate(policy, None, "hi")
    assert speech_policy.effective_rate(policy, None, "bn") > hi
    assert speech_policy.effective_rate(policy, None, "en") > hi  # owner 2026-10-18: the English voice is the slowest and is sped up most


# ---- 7 / 11. the phrases exist in all three languages
def test_prescription_and_routing_phrases_exist_in_three_languages():
    for key in ("rx_ask", "rx_yes", "rx_no", "handoff_better"):
        for lang in ("bn", "hi", "en"):
            assert phrases.PHRASES[lang][key]
    assert "9635588306" in phrases.PHRASES["en"]["rx_no"] and "doctor" in phrases.PHRASES["en"]["rx_no"]
    assert "better" in phrases.PHRASES["en"]["handoff_better"]


# ---- issues from the 2026-10-18 pod calls
def test_sonoscan_is_recognised_in_whatever_spelling_the_recogniser_writes():
    # the real transcript was "এটা কি সোনস্কান কলকাতা" -- crystal clear, but spelled without the ো and the ্য the cue list knew. In Bengali the answer starts "হ্যাঁ", not "জি"
    for q in ("এটা কি সোনস্কান কলকাতা", "এটা কি সোনোস্ক্যান", "আপনারা কি সোনোস্কেন কলকাতা", "সোনো স্ক্যান কলকাতা তো", "আমি কি সোনা স্কান এর সাথে কথা বলছি"):
        assert owner_answers.branch_identity(q, "bn").startswith("হ্যাঁ, আপনি সোনোস্ক্যান কলকাতা")
    assert owner_answers.branch_identity("सोनोस्कैन कोलकाता है क्या", "hi").startswith("जी")
    assert owner_answers.branch_identity("is this sonoscan", "en").startswith("Yes")
    assert owner_answers.branch_identity("এটা কি সোনস্কান মালদা", "bn").startswith("না")
    assert owner_answers.branch_identity("এটা কি স্ক্যান সেন্টার", "bn") is None



def test_a_hindi_theek_after_the_name_is_read_back_is_a_yes_and_is_not_withheld():
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00")
    st.pending_name = "সায়ন দাস"
    plan = bc.before_model(st, "ठीक", "bn")
    assert plan.skip_model and plan.readback_answer and plan.slots == {"patient_name": "সায়ন দাস"} and st.name_confirmed == "সায়ন দাস"


def test_the_age_is_read_back_with_the_address_question_and_the_address_asks_for_the_police_station():
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00", patient_name="Rahul Dey", phone="9876543210")
    st.asked_fields.add("patient_age")
    plan = bc.before_model(st, "৪৫", "bn")
    assert plan.speak == "রোগীর বয়স 45 বছর। ঠিকানাটা বলবেন?" and st.slots["_age_pending"] == 45 and not plan.skip_model  # the next question is asked in the same breath
    plan = bc.before_model(st, "হ্যাঁ", "bn")
    assert plan.skip_model and plan.readback_answer and plan.slots == {"patient_age": 45}
    st.slots["patient_age"] = 45
    st.asked_fields.add("patient_address")
    plan = bc.before_model(st, "বেনিয়াপুকুর কলকাতা", "bn")
    assert plan.speak == "থানার নামটা বলবেন?" and st.slots["_ps_pending"] == "বেনিয়াপুকুর কলকাতা"  # no police station was given: it is asked
    plan = bc.before_model(st, "বেনিয়াপুকুর থানা", "bn")
    assert plan.skip_model and plan.slots == {"patient_address": "বেনিয়াপুকুর কলকাতা, থানা বেনিয়াপুকুর থানা"}
    other = _booking(doctor_name="Sen", patient_name="A B", phone="9876543210")
    other.asked_fields.add("patient_address")
    assert bc.before_model(other, "বেনিয়াপুকুর থানা কলকাতা", "bn").slots == {"patient_address": "বেনিয়াপুকুর থানা কলকাতা"}  # the station is there: no question
    third = _booking(doctor_name="Sen", patient_name="A B", phone="9876543210")
    third.asked_fields.add("patient_address")
    assert bc.before_model(third, "না", "bn").slots == {}  # a refusal is let go, never asked twice
    assert bc.bare_age("পঁয়তাল্লিশ") == 45 and bc.bare_age("ষাট বছর") == 60 and bc.bare_age("হ্যাঁ") is None




def test_the_last_question_is_the_day_the_hour_and_come_with_one_to_two_hours_in_hand():
    slots = {"date": "2026-10-19", "time_slot": "10:10", "_sits_start": "10:00"}
    bn = rt.booking_confirmation_readback(slots, "book_appointment", "bn")
    assert bn.endswith("ঠিক আছে?") and "10:00-এ আপনি ওই সময়ে চলে আসুন" in bn and "এক থেকে দুই ঘণ্টা সময় নিয়ে আসবেন" in bn and "10:10" not in bn
    en = rt.booking_confirmation_readback(slots, "book_appointment", "en")
    assert "one to two hours in hand" in en and en.endswith("?")
    # after the caller's yes the booking reply only thanks; the payment-link line follows
    assert rt.booking_reply({}, {"success": True, "doctor_name": "Dr. Sen", "date": "2026-10-19", "time_slot": "10:10", "confirmation_id": "KCD-1"}, "bn") == "ধন্যবাদ।"



# ---- 2026-10-18 calls: "who sits TOMORROW", "কে কে" is not a doctor, and "did you say forty-four?"
def test_who_sits_tomorrow_in_a_department_is_a_list_question_with_or_without_the_word_doctor():
    from agent import dept_list

    assert dept_list.asks_doctor_list("কার্ডিওলজি বিভাগে কালকে কোন কোন ডাক্তার বসছে")
    assert dept_list.asks_doctor_list("কালকে কে কে বসছে কার্ডিওলজি")  # no doctor word: "কে কে" is "who, who"
    assert dept_list.asks_doctor_list("आज कौन कौन बैठ रहे हैं कार्डियोलॉजी")
    assert not dept_list.asks_doctor_list("কার্ডিওলজির ডাক্তার কেমন")


def test_the_list_for_a_day_names_the_doctors_and_their_hours_never_todays_list():
    sitting = [{"name": "কৌশিক ভট্টাচার্য", "start": "10:00", "end": "12:00"}, {"name": "নির্মল রায়", "start": "10:00", "end": "12:00"}]
    text = doctor_search.sitting_list_reply(sitting, "আগামীকাল", "কার্ডিওলজি", "bn")
    assert "আগামীকাল" in text and "কৌশিক ভট্টাচার্য (10:00-12:00)" in text and "আজ" not in text
    assert "Nobody" in doctor_search.sitting_list_reply([], "tomorrow", "Cardiology", "en")


def test_a_spoken_number_is_read_from_digits_and_from_words_in_three_languages():
    assert owner_answers.spoken_number("চুয়াল্লিশ বললেন") == 44
    assert owner_answers.spoken_number("অযাড্ড্রেস টাই চুয়াল্লিশ বললেন") == 44
    assert owner_answers.spoken_number("did you say forty four") == 44
    assert owner_answers.spoken_number("आपने पैंतालीस कहा") == 45
    assert owner_answers.spoken_number("৪৪ বললেন") == 44


def test_the_caller_can_check_a_number_in_the_last_answer(monkeypatch):
    import asyncio
    import types

    main = pytest.importorskip("main")
    if not hasattr(main, "_answer_word_check"):
        pytest.skip("`main` is the clinic API module here (another test imported it first): runs on the pod and alone")
    spoken = []

    async def fake_speak(session, text, lang=None, **kw):
        spoken.append(text)

    monkeypatch.setattr(main, "_speak", fake_speak)
    last = "আমাদের ক্লিনিকের ঠিকানা ৪৪, সি আই টি রোড, লেডিজ পার্কের কাছে, বেনিয়াপুকুর, কলকাতা।"
    session = types.SimpleNamespace(last_spoken=last)
    assert asyncio.run(main._answer_word_check(session, "চুয়াল্লিশ বললেন", "bn"))
    assert "ঠিক ধরেছেন" in spoken[0] and "চার চার" in spoken[0] and "সি আই টি রোড" in spoken[0]
    spoken.clear()
    assert asyncio.run(main._answer_word_check(session, "তেতাল্লিশ বললেন", "bn"))
    assert spoken[0].startswith("না, আমি 43 বলিনি")
    assert not asyncio.run(main._answer_word_check(session, "কালকে কে কে বসছে", "bn"))
    assert not asyncio.run(main._answer_word_check(types.SimpleNamespace(last_spoken=""), "চুয়াল্লিশ বললেন", "bn"))


# ---- the call that lost the 19th: "কালকেই" is tomorrow, not yesterday
def test_kalkei_is_tomorrow_and_the_callers_day_word_beats_the_models_reading():
    import datetime

    today = datetime.date(2026, 10, 18)
    st = _booking(doctor_name="Biswajit Sen", date="2026-10-19")
    plan = bc.before_model(st, "কালকেই দিল আমাকে অযাপ্ফোন্টমেন্ট টা", "bn", today=today)
    assert plan.date is None  # the held day (the 19th) is already tomorrow: nothing to change
    st2 = _booking(doctor_name="Biswajit Sen", date="2026-10-22")
    plan2 = bc.before_model(st2, "কালকেই করে দিন", "bn", today=today)
    assert plan2.date == "2026-10-19"
    assert bc.read_when("পরশুই করুন", False, today=today).date == "2026-10-20" and bc.read_when("আজই", False, today=today).date == "2026-10-18"


def test_a_number_started_again_replaces_the_pieces_instead_of_adding_to_them():
    from agent.booking_capture import absorb

    step = absorb("9876", "নাইন সিক্স থ্রি ডবল ফাইভ এইট থ্রি জিরো ওয়ান ফোর", True)  # 4 old digits + a whole new number: the caller started over
    assert step.status == "complete" and step.number == "9635583014"
    step2 = absorb("9876", "five four three", True)  # a real continuation is still added
    assert step2.status == "partial" and step2.buffer == "9876543"


# ---- confirmations are chained with the next question; a clipped "ह" is a yes (real calls 2026-10-18)
def test_a_read_back_asks_the_next_question_in_the_same_breath_and_the_next_answer_is_the_yes():
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00")
    st.pending_name = "সায়ন দাস"
    assert bc.readback(st, "bn", "name", "সায়ন দাস") == "আমি নামটা শুনেছি সায়ন দাস। আপনার নম্বরটা বলবেন প্লিজ? ওই নম্বরে কনফার্মেশন পাঠাব।"
    plan = bc.before_model(st, "9876543210", "bn")  # the number, not a "yes": it IS the yes for the name
    assert st.name_confirmed == "সায়ন দাস" and st.pending_name is None and plan.slots == {"patient_name": "সায়ন দাস", "phone": "9876543210"}
    st2 = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00", patient_name="Rahul Dey", phone="9876543210")
    st2.slots["_phone_pending"] = "9876543210"
    assert bc.readback(st2, "bn", "phone", "9 8 7 6 5 4 3 2 1 0") == "আমি নম্বরটা শুনেছি 9 8 7 6 5 4 3 2 1 0। রোগীর বয়স কত?"
    plan = bc.before_model(st2, "৪৫", "bn")  # the age is the yes for the number
    assert st2.slots["_phone_ok"] == "9876543210" and "_phone_pending" not in st2.slots and st2.slots["_age_pending"] == 45
    assert plan.speak.startswith("রোগীর বয়স 45 বছর")


def test_a_single_letter_ha_is_a_yes_and_never_a_name():
    assert bc.candidate_name("ह") is None and bc.candidate_name("হ") is None
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00")
    st.pending_name = "সায়ন দাস"
    plan = bc.before_model(st, "ह", "bn")
    assert plan.skip_model and plan.slots == {"patient_name": "সায়ন দাস"} and st.pending_name is None
    st.pending_name = "সায়ন দাস"
    assert bc.confirm_answer("हम्म हम्म", "bn") == "yes"


def test_a_corrected_name_after_the_chained_question_is_read_back_again():
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00")
    st.pending_name = "সায়ন দাস"
    plan = bc.before_model(st, "না, সুমন দাস", "bn")
    assert st.pending_name == "সুমন দাস" and plan.speak.startswith("আমি নামটা শুনেছি সুমন দাস। আপনার নম্বরটা বলবেন")


# ---- anything the AI says can be asked back; clinical words are found by sound, more easily in context; "yes, we do these tests" comes first
def test_a_word_asked_back_is_found_by_sound_across_scripts():
    from agent import echo_check

    last = "हमारे क्लिनिक का पता ४४, सी आई टी रोड, लेडीज़ पार्क के पास, बेनियापुकुर, कोलकाता है।"
    found = echo_check.find("বেনিয়া পুকুর কেয়া", last)  # Bengali letters, two words, against one Devanagari word
    assert found and found[0] == "বেনিয়া পুকুর" and "बेनियापुकुर" in found[1]
    assert echo_check.find("লেডিজ পার্ক কি", last)[0] == "লেডিজ পার্ক"
    assert echo_check.find("শিয়ালদা কেয়া", last) is None and echo_check.find("ঠিক", last) is None and echo_check.find("আপনার নাম কি", last) is None
    assert echo_check.skeleton("বেনিয়াপুকুর") == echo_check.skeleton("बेनियापुकुर") == echo_check.skeleton("Beniapukur")


def test_prescription_is_found_even_when_mangled_but_not_confused_with_ordinary_words():
    from agent import clinical_terms

    said = "টেস্টের বুকিং আর পেমেন্ট আমাদের কাউন্টারে গিয়ে করতে হয়। প্রেসক্রিপশন নিয়ে কাউন্টারে চলে আসুন।"
    for heard in ("আমার কাছে প্রেস্কৃপ্তিন এ লেখা নাই", "আমাকে হয়তো প্রেস্কৃপ্তিন নাই", "আমার প্রেসক্রিপশন আছে"):
        assert clinical_terms.mentions("prescription", heard, context=said)
    for other in ("পরিষ্কার নাই", "প্রেসার নাই", "প্রসঙ্গ নাই", "আমার ব্লাড টেস্ট"):
        assert not clinical_terms.mentions("prescription", other, context=said)


def test_the_caller_asked_whether_the_tests_are_done_here_is_told_yes_first():
    from agent import test_availability

    q = "আমাকে বলুন তো ওখানে আপনাদের ওই ক্ব আর লিপিড প্রোফাইল এই দুটো টেস্ট করবো হয় কি এই দুটো টেস্ট CBC"
    assert test_availability.mentions_done_here(q) and test_availability.mentions_done_here("do you do the lipid profile test")
    assert not test_availability.mentions_done_here("লিপিড প্রোফাইলের দাম কত") and not test_availability.mentions_done_here("সিবিসি টেস্টের রেট কত")


# ---- 2026-10-18 issues list
def test_english_bazar_is_a_place_not_a_request_for_english():
    from agent import language_switch

    for t in ("আমার ঠিকানা ইংলিশ বাজার থানা মালদা", "English Bazar police station", "ইংলিশ বাজার", "इंग्लिश बाज़ार"):
        assert language_switch.detect_language_switch_request(t, "bn") is None, t
    assert language_switch.detect_language_switch_request("ইংলিশে কথা বলুন", "bn") == "en"


def test_one_more_digit_at_the_end_is_added_to_the_number_so_far():
    assert bc.end_append("শেষে একটা জিরো হবে আরেকটা") == "0" and bc.end_append("নয় আট সাত") == ""
    step = bc.absorb("921321410", "শেষে একটা জিরো হবে আরেকটা", True)
    assert step.status == "complete" and step.number == "9213214100"
    again = bc.absorb("921321410", "শেষে আরেকটা জিরো হবে জিরো", True)  # said twice for emphasis: still one zero, never eleven digits
    assert again.status == "complete" and again.number == "9213214100"


def test_a_changed_name_or_age_is_read_back_again():
    assert bc.name_change("নামটা চাঞ্জ হবে বিপ্লব সাহা") == "বিপ্লব সাহা" and bc.name_change("আমার নাম বিপ্লব সাহা") is None
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00", patient_name="সায়ন দাস", phone="9876543210")
    st.name_confirmed = "সায়ন দাস"
    plan = bc.before_model(st, "নামটা চাঞ্জ হবে বিপ্লব সাহা", "bn")
    assert st.pending_name == "বিপ্লব সাহা" and "patient_name" not in st.slots and plan.speak.startswith("আমি নামটা শুনেছি বিপ্লব সাহা")
    st2 = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00", patient_name="সায়ন দাস", phone="9876543210", patient_age=45)
    plan2 = bc.before_model(st2, "বয়স ৫০ হবে", "bn")
    assert st2.slots["_age_pending"] == 50 and "patient_age" not in st2.slots and "50" in plan2.speak


def test_the_day_the_doctor_does_not_sit_is_named_in_the_not_sitting_message():
    from agent import slot_proposal

    said = slot_proposal.sits_statement("বিশ্বজিৎ সেন", "2026-10-19", "10:00", "bn", moved=True, asked_date="2026-10-18")
    assert said.startswith("আজ ডাক্তার বসেন না।") and "2026-10-19" in said and "ঠিক আছে, বদলে" not in said
