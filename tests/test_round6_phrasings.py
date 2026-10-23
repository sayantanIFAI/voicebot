"""Owner, 2026-10-23: phrasings a sweep of about 390 sentences found unrecognised, now matched -- in Bengali, Hindi and English, plain and romanized -- and the near
misses that must NOT match.

python -m pytest tests/test_round6_phrasings.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import billing_cues  # noqa: E402
from agent import early_routing as er  # noqa: E402

CASES = [
    # (cue function, early_routing kind, sentences)
    (
        "link",
        er.LINK_MISSING,
        [
            "link pai nai",
            "payment link pai nai",
            "botsapp e link ashe nai",
            "ami link pelam na",
            "amar link aseni",
            "link ashena",
            "link pai na",
            "whatsapp e link pacchi nai",
            "link nai aaya",
            "link na aaya",
        ],
    ),
    (
        "good",
        er.GOOD_DOCTOR,
        [
            "bhala doctor kaun sa hai",
            "sabse accha doctor kaun",
            "badhiya doctor kaun hai",
            "doctor bhala kaun hai",
            "भला डॉक्टर कौन है",
            "बेहतरीन डॉक्टर कौन सा है",
        ],
    ),
    (
        "weekdays",
        er.DOCTOR_WEEKDAYS,
        [
            "ki ki bare bosen",
            "ki ki din bose",
            "koyta din bose",
            "koi din bosen",
            "koy din boshen",
            "kitne din baithte hain",
            "kitne din milte hain",
            "কয়দিন বসেন",
            "কয় টা দিন বসে",
            "कितने दिन बैठते हैं",
            "कितने दिन मिलते हैं",
        ],
    ),
    (
        "absent",
        er.NEXT_IF_ABSENT,
        [
            "if the doctor is absent when is the next day",
            "in case doctor is on leave which day then",
            "when doctor is not available when next",
            "doctor na thakle next kobe",
            "doctor na pele kobe asbo",
            "dagtar na boshle porer din kobe",
            "aaj doctor na mile to agla din kab",
            "aaj doctor nahi mile to kab",
            "doctor na mile to next kab",
            "ডাক্তার না থাকলে কবে আসব",
            "ডাক্তার না পেলে পরের দিন কবে",
            "अगर डॉक्टर नहीं हैं तो कब आएं",
            "अगर डॉक्टर नहीं मिले तो अगला कब",
        ],
    ),
    (
        "cannotpay",
        er.CANNOT_PAY_ONLINE,
        [
            "phonepe will not work",
            "gpay won't work",
            "upi doesn't work",
            "paytm does not work",
            "phonepay wont go through",
        ],
    ),
    (
        "come",
        er.COME_TO_CLINIC,
        [
            "kya main clinic aakar book kar sakta hoon",
            "clinic jakar serial le sakta hoon",
            "counter pe aake naam likhwa sakta hoon",
            "book clinic par aakar kar sakte hain",
            "क्लिनिक आकर बुक कर सकते हैं",
        ],
    ),
    (
        "when",
        er.WHEN_TO_COME,
        ["koita r moddhe asbo", "koita moddhe asbo", "koyta shomoy asbo", "koita r shomoy ashbo", "koyta moddhe jabo"],
    ),
]

NOT = [
    "link ta khub bhalo", "amar link chai", "the link is open", "doctor sen kon din boshen", "bhalo achi", "phonepe ache", "upi is fine", "clinic kothay", "doctor bhala hai", "kitne baje aana hai",
    "ki khobor", "ki ki lagbe", "koyta baje", "link pathan", "aaj doctor hai", "doctor na", "ডাক্তার আছেন", "अगर आप चाहें", "doctor ke pass kab jaun", "booking kar do",
]  # fmt: skip


@pytest.mark.parametrize("name,kind,sentences", CASES)
def test_the_phrasing_is_now_recognised(name, kind, sentences):
    for s in sentences:
        got = er.classify(s, "en")
        assert got == kind, (s, got)


@pytest.mark.parametrize("sentence", NOT)
def test_a_near_miss_is_not_taken_for_one_of_them(sentence):
    new_kinds = {
        er.LINK_MISSING,
        er.GOOD_DOCTOR,
        er.DOCTOR_WEEKDAYS,
        er.NEXT_IF_ABSENT,
        er.CANNOT_PAY_ONLINE,
        er.COME_TO_CLINIC,
    }
    assert er.classify(sentence, "en") not in new_kinds, sentence


LINK_SPELLINGS = [
    "পেমেন্টের কোন লিঙ্ক পাইনি এখনও", "আমি হোয়াটসঅ্যাপে কোন পেমেন্ট লিঙ্ক পাইনি", "লিন্ক আসেনি", "লিঙ্ক আসছে না", "পেমেন্ট লিংক পাইনি", "ওয়াটসঅ্যাপে লিঙ্ক আসেনি",
    "लिन्क नहीं आया", "पेमेंट लिन्क नहीं मिला", "लिंक नहीं आया", "व्हाट्सएप पर लिन्क नहीं आ रहा",
]  # fmt: skip


@pytest.mark.parametrize("sentence", LINK_SPELLINGS)
def test_every_spelling_of_the_word_link_is_read_as_the_payment_link_missing(sentence):
    assert er.classify(sentence, "bn") == er.LINK_MISSING, sentence
    assert billing_cues.payment_link_missing(sentence, "bn")


def test_a_link_spelling_beside_no_complaint_is_left_alone():
    for sentence in ("লিঙ্কটা ভালো", "লিঙ্ক পাঠান", "लिन्क भेजिए", "link bhejiye"):
        assert er.classify(sentence, "bn") != er.LINK_MISSING, sentence
