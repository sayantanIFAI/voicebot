"""Owner, 2026-10-23: WhatsApp spellings and the "does this number have WhatsApp?" question (item 15), a rare test that is not in the lab table goes to a person
(item 7), and a doctor asked about on a named weekday (item 25) -- in Bengali, Hindi and English, romanized, garbled and plain.

python -m pytest tests/test_whatsapp_rare_test_weekday.py -q
"""

import datetime
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import booking_capture as bc  # noqa: E402
from agent import early_routing as er  # noqa: E402
from agent import rare_test_cues, speech_norm, weekday_cues, whatsapp_cues  # noqa: E402
from agent.booking_flow import new_state  # noqa: E402

WHATSAPP_SPELLINGS = [
    "whatsapp", "WhatsApp", "whatsap", "watsapp", "watsap", "botsapp", "bhatsapp", "whats app", "what's up", "wasap",
    "ওয়াটসঅ্যাপ", "হোয়াটসঅ্যাপ", "হোয়াটস অ্যাপ", "হোয়াটসএপ", "হোয়াটসাপ", "ওয়াটসআপ",
    "व्हाट्सएप", "व्हाट्सऐप", "व्हाट्स एप", "वाट्सऐप", "वॉट्सऐप", "व्हाट्सअप", "वाट्सअप",
]  # fmt: skip


@pytest.mark.parametrize("word", WHATSAPP_SPELLINGS)
def test_every_spelling_of_whatsapp_is_matched_and_made_canonical(word):
    assert whatsapp_cues.mentions_whatsapp(f"amar {word} e ache")
    assert whatsapp_cues.canonicalise(f"amar {word} e") in {f"amar {c} e" for c in whatsapp_cues.CANONICAL.values()}


def test_ordinary_words_are_not_whatsapp():
    for t in ("what is up doctor", "what time", "whatever", "apple", "hot up", "আমার অ্যাপয়েন্টমেন্ট", "वाट लगा"):
        assert not whatsapp_cues.mentions_whatsapp(t), t


YES = ["ha ache", "haan hai", "yes", "whatsapp ache", "botsapp hai", "হ্যাঁ আছে", "আছে", "হ্যাঁ", "हाँ है", "हां", "ji haan", "okay", "hmm", "ওয়াটসঅ্যাপ আছে"]
NO = ["nei", "nahi hai", "no", "na nei", "whatsapp nei", "botsapp nahi", "না নেই", "নেই", "नहीं है", "नहीं", "keypad phone", "onno number e ache", "dusre number pe hai", "অন্য নম্বরে আছে", "दूसरे नंबर पर है"]  # fmt: skip


@pytest.mark.parametrize("text", YES)
def test_a_yes_to_the_whatsapp_question(text):
    assert whatsapp_cues.answer(text, "en") == "yes"


@pytest.mark.parametrize("text", NO)
def test_a_no_or_another_number_to_the_whatsapp_question(text):
    assert whatsapp_cues.answer(text, "en") == "no"


def test_the_english_voice_says_whatsapp_as_two_words():
    assert "Whats App" in speech_norm.verbalize_en("A link goes to your WhatsApp.")


def _booking_with_details(**extra):
    st = new_state("book_appointment")
    st.slots.update(
        doctor_name="Sen", date="2026-10-30", time_slot="10:00", patient_name="Rahul Dey", phone="9876543210", patient_age=45, patient_address="Benia Pukur, Entally thana", _phone_ok="9876543210"
    )
    st.name_confirmed = "Rahul Dey"
    st.slots.update(extra)
    return st


@pytest.mark.parametrize("lang,word", [("en", "WhatsApp"), ("bn", "হোয়াটসঅ্যাপ"), ("hi", "व्हाट्सऐप")])
def test_the_number_is_asked_about_once_in_every_language(lang, word):
    st = _booking_with_details()
    assert bc.whatsapp_due(st)
    asked = bc.ask_whatsapp(st, lang)
    assert word in asked and "9 8 7 6 5 4 3 2 1 0" in asked
    assert not bc.whatsapp_due(st)  # once


@pytest.mark.parametrize("text,lang", [("ha ache", "en"), ("botsapp ache", "en"), ("হ্যাঁ আছে", "bn"), ("हाँ है", "hi")])
def test_a_yes_lets_the_booking_carry_on(text, lang):
    st = _booking_with_details()
    bc.ask_whatsapp(st, "en")
    plan = bc.before_model(st, text, lang)
    assert plan.skip_model and not plan.speak and st.slots["_wa_done"] == "1"


@pytest.mark.parametrize("text,lang,word", [("nei", "en", "ten digits"), ("না নেই", "bn", "দশ সংখ্যার"), ("नहीं है", "hi", "दस अंकों")])
def test_a_no_asks_for_the_number_that_has_whatsapp(text, lang, word):
    st = _booking_with_details()
    bc.ask_whatsapp(st, "en")
    plan = bc.before_model(st, text, lang)
    assert word in plan.speak and st.slots["_wa_number"] == "1"


def test_the_whatsapp_number_replaces_the_booking_number_and_is_read_back():
    st = _booking_with_details()
    bc.ask_whatsapp(st, "en")
    bc.before_model(st, "no", "en")
    plan = bc.before_model(st, "9 1 2 3 4 5 6 7 8 9", "en")
    assert plan.slots == {"phone": "9123456789"} and plan.skip_model and st.slots["_wa_done"] == "1"
    assert "_phone_ok" not in st.slots  # the existing read-back for a yes runs on the new number


def test_the_number_may_come_in_the_same_breath_as_the_no():
    st = _booking_with_details()
    bc.ask_whatsapp(st, "en")
    plan = bc.before_model(st, "no, the whatsapp is 9123456789", "en")
    assert plan.slots == {"phone": "9123456789"}


def test_an_answer_that_is_neither_is_asked_once_more_then_let_go():
    st = _booking_with_details()
    bc.ask_whatsapp(st, "en")
    assert bc.before_model(st, "blah blah", "en").speak
    plan = bc.before_model(st, "blah blah", "en")
    assert plan.skip_model and not plan.speak and st.slots["_wa_done"] == "1"


def test_no_whatsapp_question_for_a_caller_who_gave_no_number():
    st = new_state("book_appointment")
    st.phone_declined = True
    assert not bc.whatsapp_due(st)


RARE = [
    "usg of thyroid gland", "thyroid gland er usg korano jabe", "tb test in ear", "kaner tb test hoy?", "thyroid ultrasound", "knee x-ray", "ct scan of neck", "doppler of neck",
    "থাইরয়েড গ্ল্যান্ডের ইউএসজি", "গলার আল্ট্রাসাউন্ড", "কানের টিবি টেস্ট", "ঘাড়ের এক্স-রে",
    "कान का टीबी टेस्ट", "थायराइड का यूएसजी", "घुटने का एक्स-रे", "गले का अल्ट्रासाउंड",
]  # fmt: skip
LISTED = [
    "thyroid profile test", "usg whole abdomen", "chest x ray", "cbc test", "usg pregnancy", "tsh test price", "ultrasound abdomen and thyroid profile", "head ct scan",
    "থাইরয়েড টেস্টের রেট", "পেটের আল্ট্রাসাউন্ড", "थायराइड टेस्ट का रेट", "ब्लड शुगर टेस्ट",
]  # fmt: skip


@pytest.mark.parametrize("text", RARE)
def test_a_rare_test_not_in_the_table_goes_to_a_person(text):
    assert rare_test_cues.asks_for_unlisted_test(text)
    assert er.classify(text, "en") == er.RARE_TEST
    assert er.HANDOFF_REASON[er.RARE_TEST] == "unlisted_test"


@pytest.mark.parametrize("text", LISTED)
def test_a_listed_test_is_never_sent_to_a_person(text):
    assert not rare_test_cues.asks_for_unlisted_test(text)
    assert er.classify(text, "en") != er.RARE_TEST


TODAY = datetime.date(2026, 10, 23)  # a Friday
DAYS = [
    ("samner robibar doctor sen ke pabo", "2026-10-25"), ("ei robibar", "2026-10-25"), ("robibar ki doc sen bosbe", "2026-10-25"), ("robi bar doctor sen bosben", "2026-10-25"),
    ("সামনের রবিবার ডাক্তার সেন কি বসবেন", "2026-10-25"), ("এই রবিবার ডাক্তার সেনকে পাব", "2026-10-25"), ("রবিবার ডাক্তার সেন আছেন", "2026-10-25"),
    ("रविवार को डॉक्टर सेन मिलेंगे", "2026-10-25"), ("अगले इतवार डॉक्टर सेन बैठेंगे", "2026-10-25"), ("इस रविवार", "2026-10-25"),
    ("next sunday is dr sen available", "2026-10-25"), ("is doctor sen available on monday", "2026-10-26"), ("somobar doctor sen ke pabo", "2026-10-26"),
    ("mangalvar ko doctor sen milenge", "2026-10-27"), ("বুধবার ডাক্তার সেন বসবেন", "2026-10-28"), ("brihospotibar doc sen bosben", "2026-10-29"),
    ("শুক্রবার ডাক্তার সেন আছেন?", "2026-10-23"), ("samner shukrobar doctor sen ke pabo", "2026-10-30"), ("shonibar doc sen asben", "2026-10-24"),
]  # fmt: skip
NOT_DAYS = ["robibar appointment chai", "robibar report pabo", "sunday clinic khola ache", "sunday serial number", "somobar payment link", "kon kon din bose", "doctor sen ke pabo", "রবিবার রিপোর্ট পাব কি", "रविवार को रिपोर्ट मिलेगी"]  # fmt: skip


@pytest.mark.parametrize("text,date", DAYS)
def test_a_doctor_asked_about_on_a_weekday_resolves_to_the_coming_such_day(text, date):
    assert weekday_cues.asks_day_availability(text, "en"), text
    assert weekday_cues.resolve_date(text, TODAY) == date
    assert er.classify(text, "en") == er.DAY_AVAILABILITY


@pytest.mark.parametrize("text", NOT_DAYS)
def test_a_weekday_beside_another_topic_is_left_alone(text):
    assert not weekday_cues.asks_day_availability(text, "en")
    assert er.classify(text, "en") != er.DAY_AVAILABILITY


def test_next_sunday_asked_on_a_sunday_is_a_week_away_but_this_sunday_is_today():
    sunday = datetime.date(2026, 10, 25)
    assert weekday_cues.resolve_date("samner robibar doctor sen ke pabo", sunday) == "2026-11-01"
    assert weekday_cues.resolve_date("ei robibar doctor sen ke pabo", sunday) == "2026-10-25"
