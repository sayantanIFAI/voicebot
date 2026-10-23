"""Owner, 2026-10-23 (item 6): "achha thik ache", "ok", "thik hai" and the like -- said with nothing else open -- end the call, in Bengali, Hindi and English, in
their own scripts, romanized and garbled. The same words after a question, or with anything more in the sentence, do not.

python -m pytest tests/test_closing_acknowledgement.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import call_end  # noqa: E402

CLOSING = [
    # English / romanized Bengali and Hindi, plain and garbled
    "acha thik ache", "achha thik ache", "accha thik hay", "achha theek hai", "atcha thik ache", "aacha thik ase", "achha", "acha", "ok", "okk", "okay", "okey", "ok ok", "okay okay",
    "ok thik ache", "okay thik hai", "thik ache", "thik achhe", "thik ase", "thik hay", "thikai", "thikache", "thik", "theek hai", "theek", "thik hai ji", "haan thik hai", "hmm thik ache",
    "fine", "alright", "all right", "ok fine", "good", "great", "perfect", "noted", "understood", "ok thanks", "okay thank you", "thanks", "thank you", "thank you so much", "thanku", "thankyou",
    "thanx", "ok thnx", "dhanyabad", "dhonnobad", "ok dhonnobad", "thik ache dhonnobad", "shukriya", "ok shukriya", "theek hai shukriya", "bujhlam", "bujhechi", "achha bujhlam",
    "achha bujhechi thik ache", "samajh gaya", "samajh gayi", "achha samajh gaya", "ok samjha", "achha chalo", "thik ache chalo", "achha bhalo", "besh", "besh thik ache", "badhiya",
    "achha thik hai dada", "thik ache didi", "achha thik hai sir", "thik hai madam",
    # Bengali script
    "আচ্ছা", "আচ্ছা ঠিক আছে", "ঠিক আছে", "ঠিক আছে।", "ঠিকাছে", "ঠিক", "ওকে", "ওকে ধন্যবাদ", "ঠিক আছে ধন্যবাদ", "আচ্ছা ঠিক আছে ধন্যবাদ", "ধন্যবাদ", "থ্যাংকস", "বুঝেছি", "বুঝলাম",
    "বেশ ধন্যবাদ আমি এটা জেনে নিলাম", "ধন্যবাদ জেনে নিলাম", "আচ্ছা জানলাম ধন্যবাদ", "সব বুঝে গেলাম ধন্যবাদ", "ঠিক আছে আমি জেনে গেলাম", "ami eta jene nilam dhonnobad", "ok ami sob janlam", "achha janlam", "jene nilam thik ache",
    "मुझे पता चल गया धन्यवाद", "ठीक है मैं सब समझ गया", "जान लिया शुक्रिया", "अच्छा मैंने जान लिया",
    "আচ্ছা বুঝলাম", "বুঝলাম ঠিক আছে", "বেশ", "ভালো", "আচ্ছা ঠিক আছে দাদা", "ওকে ঠিক আছে", "হ্যাঁ ঠিক আছে", "হুম ঠিক আছে", "অচ্ছা ঠিক আছে",
    # Hindi
    "अच्छा", "अच्छा ठीक है", "ठीक है", "ठीक है।", "ठीक", "ओके", "ओके ठीक है", "ठीक है धन्यवाद", "अच्छा ठीक है धन्यवाद", "धन्यवाद", "शुक्रिया", "थैंक्स", "थैंक्यू", "समझ गया", "समझ गई",
    "अच्छा समझ गया", "ठीक है जी", "हाँ ठीक है", "बढ़िया", "सही है", "ठीक है भैया", "अच्छा चलो",
]  # fmt: skip

NOT_CLOSING = [
    "ache", "achhe", "haan", "ha", "hmm", "hanji", "yes", "no", "na", "না", "হ্যাঁ", "হুম", "हाँ", "जी", "है", "हम्म", "ha ache", "haan hai", "ji haan",
    "thik ache but one more thing", "ok book the appointment", "thik ache kobe asbo", "ok 10 am", "ok tomorrow", "okay what is the price", "achha thik ache ar ekta kotha chilo",
    "doctor sen ache", "thik ache doctor sen ke pabo", "ঠিক আছে কাল সকালে", "আচ্ছা ঠিক আছে ডাক্তার সেন কখন বসেন", "ठीक है कल सुबह", "अच्छा ठीक है डॉक्टर सेन कब बैठते हैं",
    "ami jante chai", "আমি এটা জানতে চাই", "ami ar ekta kotha jene nilam kintu", "मुझे पता नहीं", "ami eta chai", "আমি জানলাম না", "ami booking korbo", "ami ekhon booking korte chai",
    "ঠিক আছে বুক করে দিন", "ठीक है बुक कर दीजिए", "ok and also the report", "", "   ", "thank you but I have one question",
]  # fmt: skip


@pytest.mark.parametrize("text", CLOSING)
def test_a_bare_acknowledgement_is_a_goodbye(text):
    assert call_end.is_closing_acknowledgement(text, "en"), text


@pytest.mark.parametrize("text", NOT_CLOSING)
def test_anything_more_in_the_sentence_or_no_real_acknowledgement_is_not(text):
    assert not call_end.is_closing_acknowledgement(text, "en"), text


def test_the_existing_end_call_cues_are_unchanged():
    assert call_end.wants_to_end("end the call", "en")
    assert call_end.wants_to_end("কল কেটে দিন", "bn")
    assert call_end.wants_to_end("nothing else", "en", after_prompt=True)
    assert not call_end.wants_to_end("nothing else", "en")
