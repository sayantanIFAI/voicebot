"""Owner, 2026-10-23: the echo check ("did you mean this word?") must not take over a number being said, or an ordinary question that happens to hold a word the agent just said.

python -m pytest tests/test_older_fixes.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import echo_check  # noqa: E402

LAST_NUMBER = "ঠিক আছে, এখন পর্যন্ত 9 8 7 6 5 পেয়েছি। বাকিটা বলুন।"
LAST_SAMPLE = "লিপিড প্রোফাইল টেস্টের রেট 650 টাকা। এর জন্য রক্তের নমুনা নেওয়া হয়। রিপোর্ট 1 দিনের মধ্যে পাবেন।"


@pytest.mark.parametrize(
    "said",
    [
        "বাকিটা হচ্ছে ফাইভ সিক্স সেবেন নাইট নাইন জিরো",
        "বাকিটা হচ্ছে চার তিন দুই এক শূন্য",
        "বাকিটা বলছি",
        "আমার ঠিকানা বলছি বাকিটা",
    ],
)
def test_digits_and_ordinary_sentences_are_not_a_word_asked_back_in_bengali(said):
    assert echo_check.find(said, LAST_NUMBER) is None, said


def test_a_long_ordinary_question_holding_one_word_of_the_last_answer_is_not_a_word_asked_back():
    assert echo_check.find("আচ্ছা আপনারা কি বাড়ি গিয়ে নমুনা সংগ্রহ করেন", LAST_SAMPLE) is None


def test_the_short_cue_must_be_a_whole_word():
    assert not echo_check.asks_back("বাকিটা হচ্ছে সাত আট")  # "কি" is only letters inside "বাকিটা"
    assert not echo_check.asks_back("আমার ঠিকানা বলছি")  # "ঠিক" is only letters inside "ঠিকানা"
    assert echo_check.asks_back("বেনিয়া পুকুর কি")
    assert echo_check.asks_back("বেনিয়া পুকুর ঠিক")
    assert echo_check.asks_back("बेनिया पुकुर क्या")
    assert echo_check.asks_back("beniapukur ki?")


def test_a_real_word_asked_back_is_still_found():
    last = "ঠিকানা বেনিয়া পুকুর শুনেছি।"
    assert echo_check.find("বেনিয়া পুকুর কি", last) is not None
    assert echo_check.find("बेनिया पुकुर क्या", "ঠিকানা বেনিয়া পুকুর শুনেছি।") is not None
