"""Owner, 2026-10-31: a symptom word the keyword table does not know ("ছুলি", a skin disease) must still reach the right department -- by the symptom-word algorithm in
clinic-api/symptom_lexicon.py -- and the booking's police-station step must drop the words "police station" from the answer.

python -m pytest tests/test_symptom_lexicon.py -q
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "clinic-api"))

import symptom_lexicon as lex  # noqa: E402

from agent import booking_capture as bc  # noqa: E402
from agent.clause_split import split_into_clauses  # noqa: E402


@pytest.mark.parametrize(
    "text,dept",
    [
        ("বলছি আমার ছেলে না ছুলির মতো হয়ে যায় কাকে দেখাবো", "Dermatology"),
        ("আমার ছেলের ছুলি হয়েছে", "Dermatology"),
        ("ছেলের গায়ে ঘামাচি", "Dermatology"),
        ("my son has itching on his skin", "Dermatology"),
        ("बच्चे को दाद हो गया है", "Dermatology"),
        ("amar chele r chulkani", "Dermatology"),
        ("আমার খুব মাথা ব্যথা", "Neurology"),
        ("আমার কানে কম শুনি", "ENT"),
        ("পেচ্ছাপে জ্বালা", "Urology"),
        ("আমার বাবার প্রেসার বেশি", "Cardiology"),
        ("daante byatha", "Dentistry"),
        ("মেয়ের মাসিক হয় না", "Gynaecology & Obstetrics"),
        ("সুগারের সমস্যা", "Diabetology & Endocrinology"),
        ("হাঁটুতে ব্যথা", "Orthopaedics"),
        ("ঘুম আসে না", "Psychiatry"),
        ("পিঠে ফোঁড়া হয়েছে", "General Surgery"),
        ("আমার খুব জ্বর", "General Medicine"),
        ("চোখে ব্যথা", "Ophthalmology"),
    ],
)
def test_symptom_goes_to_its_department(text, dept):
    assert lex.departments_for(text) == [dept]


@pytest.mark.parametrize("text", ["আমার ছেলেকে কাকে দেখাবো", "আমার ছেলে কিছু খায় না", "ডাক্তার সেন কখন বসে", "the canada trip", "আমার ছেলে কেমন আছে", "valo doctor ke ache", "I have a problem"])
def test_a_sentence_without_a_symptom_word_finds_nothing(text):
    assert lex.departments_for(text) == []


def test_a_near_spelling_never_ties_with_the_exact_word():
    # "মাসিক" (period) is one letter from "মানসিক" (mental): the exact word wins alone
    assert lex.departments_for("মেয়ের মাসিক হয় না") == ["Gynaecology & Obstetrics"]
    assert lex.departments_for("আমার মানসিক সমস্যা") == ["Psychiatry"]


def test_a_specific_word_beats_the_vague_fever():
    assert lex.departments_for("ছেলের গায়ে ছুলি আর জ্বর") == ["Dermatology"]


@pytest.mark.parametrize(
    "said,station",
    [("থানার রাজারহাটি", "রাজারহাটি"), ("রাজারহাট থানা", "রাজারহাট"), ("rajarhat thana", "rajarhat"), ("पुलिस स्टेशन राजारहाट", "राजारहाट"), ("থানা", ""), ("Park Street", "Park Street")],
)
def test_police_station_words_are_not_part_of_the_name(said, station):
    assert bc.station_name_only(said) == station


def test_a_reply_with_a_phone_number_is_split_so_only_the_number_is_slow():
    assert split_into_clauses("আমি নম্বরটা শুনেছি 9 6 3 5 5 , 8 8 2 0 1। রোগীর বয়স কত?") == ["আমি নম্বরটা শুনেছি 9 6 3 5 5 , 8 8 2 0 1।", "রোগীর বয়স কত?"]
    assert split_into_clauses("রোগীর বয়স 45 বছর। ঠিকানাটা বলবেন?") == ["রোগীর বয়স 45 বছর। ঠিকানাটা বলবেন?"]
