"""Regression tests for a second real Hindi call on 2026-10-27: a 10-digit phone number came back missing both of
its "eight"s, and a bare "चवालीस" (44) answer to "what is the patient's age?" was not read as an age at all -- the
exact same question was asked again, and the caller gave up and said a different number.

    python -m pytest tests/test_call_2026_10_27_hindi_short_utterances.py -v
"""

from __future__ import annotations

from agent import booking_capture as bc


def test_two_different_devanagari_spellings_of_eight_in_one_number_both_count():
    # real call: "सेवन आइट ज़ीरो ऐट सेवन नाइन सेवन नाइन सिक्स वन" -- the recogniser wrote the English word "eight"
    # two different ways in the SAME sentence ("आइट" the first time, "ऐट" the second), neither of which was the
    # "एट" spelling already in the table. Both digits were silently dropped; the number came back 8 digits long.
    digits, hits, n = bc.read_digits("सेवन आइट ज़ीरो ऐट सेवन नाइन सेवन नाइन सिक्स वन")
    assert digits == "7808797961"
    assert hits == n == 10
    assert bc.phone_from_digits(digits) == "7808797961"


def test_the_colloquial_contraction_of_44_is_read_as_a_bare_age_answer():
    # real call: asked "मरीज़ की उम्र क्या है?", the caller answered with just "चवालीस" -- the everyday colloquial
    # contraction of "चौंतालीस"/"चौवालीस" (44) that drops the औ vowel sign entirely. bare_age is the function the
    # booking flow actually calls for a standalone age answer (no "साल"/"उम्र" word needed, since the question
    # itself already establishes that this turn IS the age).
    assert bc.bare_age("चवालीस") == 44


def test_the_same_colloquial_word_is_also_understood_next_to_an_age_word():
    # the sibling path (agent/number_words.py, used by split_age for a sentence that carries its own "साल"/"उम्र"
    # word) needed the same addition, so "मेरी उम्र चवालीस साल है" is read the same way as the bare answer.
    from agent import number_words

    assert number_words.digits_of("चवालीस") == "44"
