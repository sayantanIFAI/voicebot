"""Owner's 2026-10-18 batch, agent side (pure functions; the clinic-API half is in test_owner_extras_2026_10_18.py)."""

import os
import sys

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
    assert bc.before_model(st, "54 32", "en").speak == "Okay, go on."
    assert bc.prompt("phone_partial", "bn", n=4) == "ঠিক আছে, বলুন।"


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
def test_the_spoken_payment_line_says_forty_rupees_within_fifteen_minutes_or_the_link_expires():
    assert "চল্লিশ টাকা পনেরো মিনিট" in phrases.PHRASES["bn"]["payment_link_sent"] and "বাতিল হয়ে যাবে" in phrases.PHRASES["bn"]["payment_link_sent"]
    en = phrases.PHRASES["en"]["payment_link_sent"]
    assert "forty rupees within fifteen minutes" in en and "expire automatically" in en
    assert "KCD-" not in rt.booking_reply({}, {"success": True, "doctor_name": "Dr. Sen", "date": "2026-10-30", "time_slot": "10:10", "confirmation_id": "KCD-1"}, "en")


# ---- 2. Bengali is faster
def test_bengali_is_spoken_faster_than_hindi_and_english():
    policy = speech_policy.derive_policy()
    assert speech_policy.effective_rate(policy, None, "bn") > speech_policy.effective_rate(policy, None, "hi")
    assert speech_policy.effective_rate(policy, None, "hi") == speech_policy.effective_rate(policy, None, "en")


# ---- 7 / 11. the phrases exist in all three languages
def test_prescription_and_routing_phrases_exist_in_three_languages():
    for key in ("rx_ask", "rx_yes", "rx_no", "handoff_better"):
        for lang in ("bn", "hi", "en"):
            assert phrases.PHRASES[lang][key]
    assert "9635588306" in phrases.PHRASES["en"]["rx_no"] and "doctor" in phrases.PHRASES["en"]["rx_no"]
    assert "better" in phrases.PHRASES["en"]["handoff_better"]


# ---- issues from the 2026-10-18 pod calls
def test_sonoscan_is_recognised_in_whatever_spelling_the_recogniser_writes():
    # the real transcript was "এটা কি সোনস্কান কলকাতা" -- crystal clear, but spelled without the ো and the ্য the cue list knew
    for q in ("এটা কি সোনস্কান কলকাতা", "এটা কি সোনোস্ক্যান", "আপনারা কি সোনোস্কেন কলকাতা", "সোনো স্ক্যান কলকাতা তো"):
        assert owner_answers.branch_identity(q, "bn").startswith("জি")
    assert owner_answers.branch_identity("सोनोस्कैन कोलकाता है क्या", "hi").startswith("जी")
    assert owner_answers.branch_identity("is this sonoscan", "en").startswith("Yes")
    assert owner_answers.branch_identity("এটা কি সোনস্কান মালদা", "bn").startswith("না")
    assert owner_answers.branch_identity("এটা কি স্ক্যান সেন্টার", "bn") is None


def test_a_hindi_theek_after_the_name_is_read_back_is_a_yes_and_is_not_withheld():
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00")
    st.pending_name = "সায়ন দাস"
    plan = bc.before_model(st, "ठीक", "bn")
    assert plan.skip_model and plan.readback_answer and plan.slots == {"patient_name": "সায়ন দাস"} and st.name_confirmed == "সায়ন দাস"


def test_the_age_and_the_address_are_taken_one_at_a_time_from_the_bare_answer():
    st = _booking(doctor_name="Sen", date="2026-10-20", time_slot="10:00", patient_name="Rahul Dey", phone="9876543210")
    st.asked_fields.add("patient_age")
    plan = bc.before_model(st, "৪৫", "bn")
    assert plan.skip_model and plan.slots == {"patient_age": 45}
    st.asked_fields.add("patient_address")
    plan = bc.before_model(st, "বেনিয়াপুকুর কলকাতা", "bn")
    assert plan.skip_model and plan.slots == {"patient_address": "বেনিয়াপুকুর কলকাতা"}
    other = _booking(doctor_name="Sen", patient_name="A B", phone="9876543210")
    other.asked_fields.add("patient_address")
    assert bc.before_model(other, "না", "bn").slots == {}  # a refusal is let go, never asked twice
    assert bc.bare_age("পঁয়তাল্লিশ") == 45 and bc.bare_age("ষাট বছর") == 60 and bc.bare_age("হ্যাঁ") is None


def test_the_booking_ends_with_the_sitting_hour_statement():
    reply = rt.booking_reply({"_sits_start": "10:00"}, {"success": True, "doctor_name": "Dr. Biswajit Sen", "date": "2026-10-19", "time_slot": "10:10", "confirmation_id": "KCD-1"}, "en")
    assert "sits" in reply and "one to two hours" in reply and "KCD-" not in reply and "10:10" not in reply
    assert "এক থেকে দুই ঘণ্টা" in slot_proposal.conclusion("বিশ্বজিৎ সেন", "2026-10-19", "10:00", "bn")
