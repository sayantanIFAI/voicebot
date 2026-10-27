"""Story "Caller has reached the hospital by mistake" -- the pure, offline half of the proof (no orchestrator, no model).

# ADDED BY SOURAV: these tests pin the three rules agent/unrelated_caller.py exists to keep:
#   1. HOSPITAL ALWAYS WINS -- a casual or apologetic sentence that carries any hospital cue is never classified as unrelated;
#   2. a bare NAME is a person being asked for, never a doctor/patient -- but "Dr <name> ..." stays a hospital request;
#   3. nothing on this path asks for personal data (the fixed phrases are scanned for it).
# The orchestrator half (real _dispatch_turn: counter, reset, outcome, websocket close, active flows) is in
# tests/test_orchestrator_unrelated_caller.py and tests/test_orchestrator_unrelated_caller_flows.py.

  python -m pytest tests/test_unrelated_caller.py -v
"""

import os
import re
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import call_score, llm, unrelated_caller
from agent.intent_schema import VALID_INTENTS, parse_extraction
from agent.phrases import PHRASES
from agent.unrelated_caller import (
    PERSON_DIRECTED,
    VERDICT_GREETING,
    VERDICT_HOSPITAL,
    VERDICT_UNRELATED,
    WRONG_NUMBER,
    classify,
    has_hospital_cue,
    llm_verdict,
)

# ========================================================================================= wrong number / wrong place

WRONG_NUMBER_CASES = [
    # English
    ("Sorry, wrong number.", "en"),
    ("I dialled the wrong number", "en"),
    ("I called the wrong place", "en"),
    ("sorry, wrong call", "en"),
    ("oh I dialed this by mistake", "en"),
    ("I thought this was Suresh's number", "en"),
    # Hindi / Hinglish
    ("Galat number lag gaya.", "en"),
    ("wrong number ho gaya", "en"),
    ("Mujhe laga ye Suresh ka number hai", "en"),
    ("Bhai ye Rakesh ka number hai kya?", "en"),
    ("गलत नंबर लग गया", "hi"),
    ("गलत नंबर लगा दिया", "hi"),  # "नंबर लगा" is a FastPath BOOKING cue -- the apology's own words must not veto it
    ("मुझे लगा ये सुरेश का नंबर है", "hi"),
    ("गलती से फोन लग गया", "hi"),
    # Bengali
    ("ভুল নম্বরে ফোন হয়ে গেছে।", "bn"),
    ("ভুল করে ফোন করেছি", "bn"),
    ("ভুল জায়গায় ফোন হয়েছে", "bn"),
    ("ami bhul number e phone korechi", "en"),
    ("আমি ভেবেছিলাম এটা সুরেশের নম্বর", "bn"),
]


@pytest.mark.parametrize("text,lang", WRONG_NUMBER_CASES)
def test_a_clear_wrong_number_is_recognised_in_every_language(text, lang):
    assert classify(text, lang) == WRONG_NUMBER


# ========================================================================================= a named person is being asked for

PERSON_CASES = [
    ("Suresh hai kya?", "en"),
    ("Bhai Amit se baat karwao.", "en"),
    ("Ramesh ko phone diya tha.", "en"),
    ("Dada Suresh ache?", "en"),
    ("Can I speak to Suresh please", "en"),
    ("Is Suresh there", "en"),
    ("सुरेश है क्या?", "hi"),
    ("अमित से बात करवाओ", "hi"),
    ("रमेश को फोन दिया था", "hi"),
    ("সুরেশ আছে?", "bn"),
    ("দাদা সুরেশ আছে কি", "bn"),
    ("সুরেশের সাথে কথা বলিয়ে দিন", "bn"),
]


@pytest.mark.parametrize("text,lang", PERSON_CASES)
def test_a_bare_name_being_asked_for_is_a_person_never_a_doctor(text, lang):
    assert classify(text, lang) == PERSON_DIRECTED


# ========================================================================================= HOSPITAL ALWAYS WINS

HOSPITAL_FIRST_CASES = [
    # the story's own mixed examples
    ("Bhai kaise ho, CBC ka rate kya hai?", "en"),
    ("Arre dada kemon acho, amar test ta bari theke collect hobe?", "en"),
    ("Sorry bhai, actually mera appointment check karna tha.", "en"),
    ("Mujhe laga Suresh ka number hai, but actually CBC report ke baare mein poochna tha.", "en"),
    # an apology plus a real request
    ("Sorry wrong number, actually I need my report", "en"),
    ("sorry wrong number, can I book an appointment", "en"),
    ("ভুল নম্বর, আসলে আমার রিপোর্ট নিয়ে জানতে চাই", "bn"),
    ("गलत नंबर नहीं, मुझे टेस्ट का रेट पूछना है", "hi"),
    # a doctor named WITH a doctor word is a hospital request, not a bare name
    ("Dr Suresh ka appointment chahiye", "en"),
    ("Is Dr Suresh there", "en"),
    ("doctor hai kya", "en"),
    ("डॉक्टर है क्या", "hi"),
    ("ডাক্তার আছেন?", "bn"),
    # a facility / service a hospital caller really asks about ("X hai kya" is not a person)
    ("parking hai kya", "en"),
    ("ambulance ache?", "en"),
    ("report hai kya", "en"),
    ("I thought my test number was different", "en"),
    # home collection
    ("আমার টেস্ট টা বাড়ি থেকে নেওয়া যাবে", "bn"),
]


@pytest.mark.parametrize("text,lang", HOSPITAL_FIRST_CASES)
def test_any_hospital_cue_means_the_turn_is_not_decided_as_unrelated(text, lang):
    assert classify(text, lang) is None


@pytest.mark.parametrize(
    "text",
    [
        "hello",
        "hi bhai",
        "namaste",
        "Hello how are you",  # polite, not unrelated: it is the model's job (and the two-turn rule's) to judge, not a cue's
        "Arre bhai kaise ho",  # soft chit-chat is deliberately NOT a deterministic cue: it is left to the model intent
        "Aur batao kya chal raha hai",
        "koi hai kya",  # "is anyone there": a pronoun, not a name
        "Suresh? Hello?",
    ],
)
def test_soft_or_ambiguous_speech_is_never_decided_deterministically(text):
    assert classify(text, "en") is None


def test_a_catalogue_entity_vetoes_even_when_no_static_cue_does():
    """`entity_match` is how main.py hands in the live catalogue (a test name, an FAQ topic, a full doctor name)."""
    assert classify("Suresh hai kya", "en", entity_match=lambda t, lang: False) == PERSON_DIRECTED
    assert classify("Suresh hai kya", "en", entity_match=lambda t, lang: True) is None
    assert (
        classify("wrong number", "en", entity_match=lambda t, lang: True) == WRONG_NUMBER
    )  # nothing left but the apology


def test_a_failure_inside_the_catalogue_check_is_treated_as_a_hospital_caller():
    def boom(_t, _lang):
        raise RuntimeError("catalogue down")

    assert has_hospital_cue("Suresh hai kya", "en", entity_match=boom) is True
    assert classify("Suresh hai kya", "en", entity_match=boom) is None


def test_a_long_sentence_is_never_decided_by_a_name_pattern():
    assert classify("Suresh hai kya " + "aur batao " * 6, "en") is None


# ========================================================================================= the model's own answer is only a proposal


@pytest.mark.parametrize(
    "text,slots,expected",
    [
        ("Arre bhai kaise ho", {}, VERDICT_UNRELATED),
        ("Aur batao kya chal raha hai", {}, VERDICT_UNRELATED),
        ("How are you brother", {}, VERDICT_UNRELATED),
        ("office mein kya chal raha hai", {}, VERDICT_UNRELATED),
        ("weather kaisa hai", {}, VERDICT_UNRELATED),
        ("দাদা কেমন আছো", {}, VERDICT_UNRELATED),
        ("भाई कैसे हो", {}, VERDICT_UNRELATED),
        # a plain greeting or thanks stays smalltalk
        ("hello", {}, VERDICT_GREETING),
        ("namaste", {}, VERDICT_GREETING),
        ("thank you", {}, VERDICT_GREETING),
        # a hospital sign anywhere overrules the model
        ("Bhai kaise ho CBC ka rate", {}, VERDICT_HOSPITAL),
        ("Arre dada kemon acho, amar test ta bari theke collect hobe", {}, VERDICT_HOSPITAL),
        ("sorry bhai actually mera appointment check karna tha", {}, VERDICT_HOSPITAL),
        # the model filled a slot: it saw a request, whatever it called the intent
        ("Aur batao", {"test_name": "CBC"}, VERDICT_HOSPITAL),
        ("Aur batao", {"doctor_name": "Suresh"}, VERDICT_HOSPITAL),
        ("Aur batao", {"date": None, "test_name": ""}, VERDICT_UNRELATED),  # empty slots are not evidence
    ],
)
def test_the_models_unrelated_answer_is_only_acted_on_when_nothing_contradicts_it(text, slots, expected):
    assert llm_verdict(text, "en", slots) == expected


# ========================================================================================= the schema, the prompt and the phrases






def test_the_two_fixed_phrases_exist_in_every_language_and_ask_for_nothing_personal():
    forbidden = re.compile(
        r"name|phone|number|mobile|id\b|booking|date of birth|birth|address|email|payment|card|"
        r"নাম|ফোন|নম্বর|নাম্বার|বুকিং|ঠিকানা|ইমেইল|জন্ম|পেমেন্ট|"
        r"नाम|फोन|फ़ोन|नंबर|बुकिंग|(?<![ऀ-ॿ])पता(?![ऀ-ॿ])|ईमेल|जन्म|पेमेंट",
        re.I,
    )
    for lang in ("bn", "hi", "en"):
        for key in ("unrelated_first", "unrelated_final"):
            line = PHRASES[lang][key]
            assert line.strip()
            assert "?" not in line and "？" not in line  # a statement: it asks nothing at all
            assert not forbidden.search(line), (lang, key, line)


def test_the_first_reply_leaves_the_door_open_and_the_final_one_says_goodbye():
    assert "hospital" in PHRASES["en"]["unrelated_first"] and "help" in PHRASES["en"]["unrelated_first"]
    assert "goodbye" in PHRASES["en"]["unrelated_final"].lower()


# ========================================================================================= score and metrics








def test_close_after_two_consecutive_turns_is_the_documented_rule():
    assert unrelated_caller.CLOSE_AT == 2


# ========================================================================================= the clinic API stores the new outcome unchanged


@pytest.fixture(scope="module")
def api():
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        yield client


