"""Whole bookings, caller turn by caller turn, through the REAL turn handler of both entry points (main.py and main_pcm.py) -- the cases that only show up across turns:
a phone number said in pieces (the echo check must not eat the digits), the WhatsApp question, the police station checked against the geography tables.

python -m pytest tests/test_booking_sessions.py -q
"""

import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO, "tests", "_booking_session_runner.py")

BOOK = {"en": "I want to book Dr Sen", "bn": "আমি ডাক্তার সেনকে দেখাতে চাই", "hi": "मुझे डॉक्टर सेन को दिखाना है"}
NAME = {"en": "Rahul Dey", "bn": "রাহুল দে", "hi": "राहुल डे"}
YES = {"en": "yes", "bn": "হ্যাঁ", "hi": "हाँ"}
ADDRESS = {"en": "Benia Pukur Road Kolkata", "bn": "বেনিয়াপুকুর রোড কলকাতা", "hi": "बेनिया पुकुर रोड कोलकाता"}
NUMBER = "9 8 7 6 5 4 3 2 1 0"


def _upto_the_address(lang):
    return [BOOK[lang], NAME[lang], NUMBER, YES[lang], "45", YES[lang], ADDRESS[lang]]


def _turns(lang, *rest):
    return [[t, lang] for t in _upto_the_address(lang) + list(rest)]


SCENARIOS = {
    "en_garbled_station_whatsapp_no": _turns("en", "shamppukur thana", "no", "9 1 2 3 4 5 6 7 8 9", "yes", "yes"),
    "bn_station_in_script_whatsapp_no": _turns("bn", "শ্যামপুকুর থানা", "না নেই", "9 1 2 3 4 5 6 7 8 9", "হ্যাঁ", "হ্যাঁ"),
    "hi_station_in_script_whatsapp_no": _turns("hi", "श्यामपुकुर", "नहीं है", "9 1 2 3 4 5 6 7 8 9", "हाँ", "हाँ"),
    "en_whatsapp_yes_said_as_botsapp": _turns("en", "shyampukur", "botsapp ache", "yes"),
    "en_station_refused_three_times_place_in_address": [[t, "en"] for t in _upto_the_address("en")[:-1]]
    + [
        ["Shyambazar Kolkata", "en"],
        ["jani na", "en"],
        ["bolbo na", "en"],
        ["nahi bolunga", "en"],
        ["yes", "en"],
        ["yes", "en"],
    ],
    "en_unknown_station_kept_as_said": _turns("en", "Jadavpur", "yes", "yes"),
    # the echo check once took "বাকিটা" for a word asked back and threw the digits away (real call 2026-10-23)
    "bn_number_in_pieces": [
        [BOOK["bn"], "bn"],
        [NAME["bn"], "bn"],
        ["নয় আট সাত ছয় পাঁচ", "bn"],
        ["বাকিটা হচ্ছে চার তিন দুই এক শূন্য", "bn"],
    ],
    "bn_number_in_pieces_garbled": [
        [BOOK["bn"], "bn"],
        [NAME["bn"], "bn"],
        ["নয় আট সাত ছয় পাঁচ", "bn"],
        ["বাকিটা হচ্ছে ফোর থ্রি টু ওয়ান জিরো", "bn"],
    ],
    "hi_number_in_pieces": [
        [BOOK["hi"], "hi"],
        [NAME["hi"], "hi"],
        ["नौ आठ सात छह पाँच", "hi"],
        ["बाकी है चार तीन दो एक शून्य", "hi"],
    ],
    "en_number_in_pieces": [
        [BOOK["en"], "en"],
        [NAME["en"], "en"],
        ["nine eight seven six five", "en"],
        ["the rest is four three two one zero", "en"],
    ],
}


@pytest.fixture(scope="module")
def sessions():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, RUNNER],
        input=json.dumps(SCENARIOS, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        env=env,
        timeout=900,
        cwd=REPO,
    )
    out = proc.stdout.decode("utf-8", "replace")
    assert "@@RESULT@@" in out, proc.stderr.decode("utf-8", "replace")[-3000:]
    return json.loads(out.split("@@RESULT@@", 1)[1].strip().splitlines()[0])


ENTRIES = ("main", "main_pcm")


def _said(rows, i):
    assert rows[i]["error"] is None, rows[i]["error"]
    return " | ".join(rows[i]["said"])


def test_both_entry_points_answer_every_turn_of_every_booking_the_same(sessions):
    for name in SCENARIOS:
        a, b = sessions["main"][name], sessions["main_pcm"][name]
        assert [r["said"] for r in a] == [r["said"] for r in b], name
        assert [r["closed"] for r in a] == [r["closed"] for r in b], name


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "name", ["bn_number_in_pieces", "bn_number_in_pieces_garbled", "hi_number_in_pieces", "en_number_in_pieces"]
)
def test_a_number_said_in_pieces_is_taken_whole_and_read_back(sessions, entry, name):
    rows = sessions[entry][name]
    assert "9 8 7 6 5 , 4 3 2 1 0" in _said(rows, -1), _said(rows, -1)  # grouped five-and-five, owner 2026-10-26
    assert "ঠিক ধরেছেন" not in _said(rows, -1) and "that is right" not in _said(rows, -1)  # not "you caught the word"


@pytest.mark.parametrize("entry", ENTRIES)
def test_the_whatsapp_question_and_a_garbled_station_in_english(sessions, entry):
    rows = sessions[entry]["en_garbled_station_whatsapp_no"]
    n = len(_upto_the_address("en"))
    assert "Does the number 9 8 7 6 5 , 4 3 2 1 0 have WhatsApp?" in _said(rows, n)  # right after the station is given
    assert "ten digits" in _said(rows, n + 1)  # "no": the number that has WhatsApp is asked for
    assert "9 1 2 3 4 , 5 6 7 8 9" in _said(rows, n + 2)  # ...read back like any phone number
    assert "police station Shyampukur" in _said(rows, n + 3)  # "shamppukur" is stored under the table's own name
    assert rows[-1]["closed"] and "confirm_booking" in rows[-1]["clinic_calls"]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize(
    "lang,question,station", [("bn", "হোয়াটসঅ্যাপ আছে?", "থানা শ্যামপুকুর"), ("hi", "व्हाट्सऐप है?", "थाना श्यामपुकुर")]
)
def test_the_station_is_read_back_in_the_calls_own_script(sessions, entry, lang, question, station):
    from agent import speech_norm

    rows = sessions[entry][f"{lang}_station_in_script_whatsapp_no"]
    n = len(_upto_the_address(lang))
    assert question in _said(rows, n)
    readback = _said(rows, n + 3)
    assert station in readback and "Shyampukur" not in readback  # the Bengali / Hindi voice cannot say English letters
    assert speech_norm.unspeakable_spans(speech_norm.verbalize(readback, lang), lang) == []
    assert rows[-1]["closed"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_whatsapp_said_as_botsapp_is_a_yes(sessions, entry):
    rows = sessions[entry]["en_whatsapp_yes_said_as_botsapp"]
    n = len(_upto_the_address("en"))
    assert "have WhatsApp?" in _said(rows, n)
    assert "police station Shyampukur" in _said(rows, n + 1) and "ten digits" not in _said(
        rows, n + 1
    )  # on to the closing read-back
    assert rows[-1]["closed"]


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_caller_who_will_not_name_the_station_is_asked_twice_then_the_place_in_the_address_fills_it(sessions, entry):
    rows = sessions[entry]["en_station_refused_three_times_place_in_address"]
    n = len(_upto_the_address("en")) - 1  # the address turn is "Shyambazar Kolkata"
    assert "Please say the police station name" in _said(rows, n + 1) and "Please say the police station name" in _said(
        rows, n + 2
    )
    assert "have WhatsApp?" in _said(rows, n + 3)
    assert "police station Shyampukur" in _said(rows, n + 4)


@pytest.mark.parametrize("entry", ENTRIES)
def test_a_station_the_tables_do_not_know_is_kept_as_the_caller_said_it(sessions, entry):
    rows = sessions[entry]["en_unknown_station_kept_as_said"]
    assert "police station Jadavpur" in _said(rows, len(_upto_the_address("en")) + 1)
