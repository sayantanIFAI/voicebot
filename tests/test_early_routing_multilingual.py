"""The early-routing cues and their ORDER, in English, Hindi, Bengali and romanized Bengali (lead's list of failures: a health-package question answered with the
insurance FAQ, "money left to pay?" answered "I didn't understand", and direct-doctor / complaint requests not reaching a human).

Pure functions -- no voice stack. The same `early_routing.classify` is what main.py AND main_pcm.py call (tests/test_turn_routing_harness.py proves that by
driving both entry points).

    python -m pytest tests/test_early_routing_multilingual.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import early_routing as er  # noqa: E402
from agent import owner_answers  # noqa: E402

# ---------------------------------------------------------------------------------------------- the lead's own phrase lists
DUES = [
    ("Do I owe any money?", "en"),
    ("Do I have any money left to pay?", "en"),
    ("Is any payment pending?", "en"),
    ("Is there any balance left on my bill?", "en"),
    ("How much do I still owe?", "en"),
    ("My payment still shows as due.", "en"),
    ("I already paid, but the system shows pending.", "en"),
    ("Can you check my outstanding amount?", "en"),
    ("क्या मेरा कोई पैसा बाकी है?", "hi"),
    ("मुझे और कितने पैसे देने हैं?", "hi"),
    ("मेरा कोई पेमेंट बाकी है क्या?", "hi"),
    ("मेरे बिल में कितना बकाया है?", "hi"),
    ("मैंने पेमेंट कर दिया, फिर भी पेंडिंग दिख रहा है।", "hi"),
    ("मेरा बकाया पेमेंट चेक कर दो।", "hi"),
    ("আমার কি কোনো টাকা বাকি আছে?", "bn"),
    ("আমাকে কি আর টাকা দিতে হবে?", "bn"),
    ("আমার পেমেন্ট বাকি আছে কি?", "bn"),
    ("আমার বিলের কত টাকা বাকি?", "bn"),
    ("আমি পেমেন্ট করেছি, তবুও পেন্ডিং দেখাচ্ছে।", "bn"),
    ("আমার বকেয়া টাকা চেক করে দিন।", "bn"),
    ("Amar kono taka baki ache?", "bn"),
    ("Amake aro taka dite hobe?", "bn"),
    ("Amar payment pending ache?", "bn"),
    ("Amar bill-e koto taka baki?", "bn"),
    ("Payment korechi, tobu pending dekhachhe.", "bn"),
]
DOCTOR_DIRECT = [
    ("I want to speak directly to the doctor.", "en"),
    ("Can I talk to the doctor personally?", "en"),
    ("I don't want an appointment; I want to talk to the doctor.", "en"),
    ("Please connect me to the doctor.", "en"),
    ("I need to speak with the doctor myself.", "en"),
    ("मुझे डॉक्टर से सीधे बात करनी है।", "hi"),
    ("मुझे खुद डॉक्टर से बात करनी है।", "hi"),
    ("मुझे अपॉइंटमेंट नहीं, डॉक्टर से बात करनी है।", "hi"),
    ("मुझे डॉक्टर से कनेक्ट कर दीजिए।", "hi"),
    ("আমি সরাসরি ডাক্তারের সঙ্গে কথা বলতে চাই।", "bn"),
    ("আমি নিজে ডাক্তারের সঙ্গে কথা বলতে চাই।", "bn"),
    ("আমার অ্যাপয়েন্টমেন্ট নয়, ডাক্তারের সঙ্গে কথা বলা দরকার।", "bn"),
    ("আমাকে ডাক্তারের সঙ্গে কথা বলিয়ে দিন।", "bn"),
    ("Ami doctor-er sathe directly kotha bolte chai.", "bn"),
    ("Ami nije doctor-er sathe kotha bolbo.", "bn"),
    ("Appointment chai na, doctor-er sathe kotha bolte chai.", "bn"),
]
COMPLAINT = [
    ("I want to make a complaint.", "en"),
    ("I want to file a complaint.", "en"),
    ("I am unhappy with your staff.", "en"),
    ("I had a bad experience and want to complain.", "en"),
    ("Please register my complaint.", "en"),
    ("मुझे शिकायत करनी है।", "hi"),
    ("मैं शिकायत दर्ज कराना चाहता हूं।", "hi"),
    ("मुझे आपके स्टाफ से शिकायत है।", "hi"),
    ("मेरा अनुभव खराब रहा और मुझे शिकायत करनी है।", "hi"),
    ("আমি অভিযোগ জানাতে চাই।", "bn"),
    ("আমি একটা অভিযোগ করতে চাই।", "bn"),
    ("আপনাদের স্টাফের বিরুদ্ধে আমার অভিযোগ আছে।", "bn"),
    ("আমার খারাপ অভিজ্ঞতা হয়েছে, আমি অভিযোগ জানাতে চাই।", "bn"),
    ("Ami complaint korte chai.", "bn"),
    ("Ami ekta obhijog janate chai.", "bn"),
    ("Apnader staff-er biruddhe amar complaint ache.", "bn"),
]
HEALTH_PACKAGE = [
    ("Do you have any health packages?", "en"),
    ("Do you have health packages?", "en"),
    ("Do you offer a full body checkup package?", "en"),
    ("What is the price of your health package?", "en"),
    ("Is there any complete health checkup package?", "en"),
    ("Do you have a preventive health package?", "en"),
    ("क्या आपके यहां हेल्थ पैकेज मिलता है?", "hi"),
    ("फुल बॉडी चेकअप पैकेज है क्या?", "hi"),
    ("हेल्थ पैकेज का प्राइस कितना है?", "hi"),
    ("पूरा हेल्थ चेकअप पैकेज उपलब्ध है?", "hi"),
    ("আপনাদের এখানে হেলথ প্যাকেজ আছে?", "bn"),
    ("ফুল বডি চেকআপ প্যাকেজ আছে?", "bn"),
    ("হেলথ প্যাকেজের দাম কত?", "bn"),
    ("সম্পূর্ণ স্বাস্থ্য পরীক্ষার প্যাকেজ আছে?", "bn"),
    ("Apnader health package ache?", "bn"),
    ("Full body checkup package ache?", "bn"),
    ("Health package-er dam koto?", "bn"),
]
# genuine insurance / cashless questions: must keep using the existing insurance FAQ (classify returns None for them)
INSURANCE = [
    ("Do you accept cashless insurance?", "en"),
    ("Which insurance companies are accepted?", "en"),
    ("Is my insurance card accepted?", "en"),
    ("ক্যাশলেস ইনসিওরেন্স চলে?", "bn"),
    ("क्या आप कैशलेस इंश्योरेंस लेते हैं?", "hi"),
    ("Mediclaim cashless hobe?", "bn"),
]
# ordinary turns that must NOT be taken for any early request
ORDINARY = [
    ("I want to book a doctor", "en"),
    ("Which doctors are available?", "en"),
    ("When does Dr. Sen sit?", "en"),
    ("What is the price of a CBC test?", "en"),
    ("I will pay tomorrow", "en"),
    ("What payment methods do you accept?", "en"),
    ("The wait was a bit long today", "en"),
    ("No complaints, everything was fine", "en"),
    ("Nothing is due, I already paid", "en"),
    ("ডাক্তারের সাথে অ্যাপয়েন্টমেন্ট চাই", "bn"),
    ("ডাক্তার কখন বসেন", "bn"),
    ("কোনো অভিযোগ নেই, সব ঠিক আছে", "bn"),
    ("मुझे डॉक्टर के साथ अपॉइंटमेंट चाहिए", "hi"),
    ("डॉक्टर कब बैठते हैं", "hi"),
    ("कोई शिकायत नहीं है", "hi"),
    ("doctor ke sathe appointment chai", "bn"),
    ("", "en"),
]


@pytest.mark.parametrize("text,lang", DUES)
def test_payment_due_questions_are_recognised(text, lang):
    assert er.classify(text, lang) == er.DUES, text


@pytest.mark.parametrize("text,lang", DOCTOR_DIRECT)
def test_a_request_to_talk_to_the_doctor_himself_goes_to_a_human(text, lang):
    assert er.classify(text, lang) == er.DOCTOR_DIRECT, text
    assert er.HANDOFF_REASON[er.DOCTOR_DIRECT] == "wants_doctor_personally"


@pytest.mark.parametrize("text,lang", COMPLAINT)
def test_a_complaint_goes_to_a_human(text, lang):
    assert er.classify(text, lang) == er.COMPLAINT, text
    assert er.HANDOFF_REASON[er.COMPLAINT] == "complaint"


@pytest.mark.parametrize("text,lang", HEALTH_PACKAGE)
def test_a_health_package_question_is_caught_before_the_insurance_faq(text, lang):
    assert er.classify(text, lang) == er.HEALTH_PACKAGE, text


@pytest.mark.parametrize("text,lang", INSURANCE)
def test_genuine_insurance_questions_are_left_for_the_insurance_faq(text, lang):
    assert er.classify(text, lang) is None, text


@pytest.mark.parametrize("text,lang", ORDINARY)
def test_ordinary_turns_are_not_taken_for_an_early_request(text, lang):
    assert er.classify(text, lang) is None, text


def test_a_health_package_reply_is_the_fixed_no_packages_answer_in_every_language():
    for lang, words in (("en", ("does not offer", "health package")), ("bn", ("নেই",)), ("hi", ("नहीं है",))):
        reply = owner_answers.health_package_reply(lang)
        assert all(w in reply for w in words), (lang, reply)
        assert "insurance" not in reply.lower() and "cashless" not in reply.lower()


# ------------------------------------------------------------------------------------------------------------- the ORDER
def test_the_order_is_fixed_and_documented():
    assert er.ORDER == (
        er.PERSON,
        er.DOCTOR_DIRECT,
        er.COMPLAINT,
        er.LINK_MISSING,
        er.CANNOT_PAY_ONLINE,
        er.DUES,
        er.HEALTH_PACKAGE,
        er.RARE_TEST,
        er.REPORT_PICKUP,
        er.REFERRAL,
        er.GOOD_DOCTOR,
        er.DAY_AVAILABILITY,
        er.DOCTOR_WEEKDAYS,
        er.NEXT_IF_ABSENT,
        er.SITTING_HOURS,
        er.WHEN_TO_COME,
        er.SERIAL_NUMBER,
        er.COME_TO_CLINIC,
        er.WRITE_NAME,
    )


def test_a_person_request_beats_everything_else():
    assert er.classify("I want to talk to a person, I have a complaint", "en") == er.PERSON


def test_the_doctor_beats_a_complaint_and_a_complaint_beats_a_payment_question():
    assert er.classify("Please connect me to the doctor, I have a complaint", "en") == er.DOCTOR_DIRECT
    assert er.classify("I want to complain, my payment still shows pending", "en") == er.COMPLAINT


def test_a_payment_question_beats_a_health_package_question():
    assert er.classify("Is any payment pending for the health package?", "en") == er.DUES


def test_a_health_package_question_beats_a_report_question():
    assert er.classify("is my report ready, also do you have a health package", "en") == er.HEALTH_PACKAGE


def test_report_and_referral_still_route():
    assert er.classify("is my report ready", "en") == er.REPORT_PICKUP
    assert (
        er.classify("I'm calling from another branch, I want to pass a number for an appointment", "en") == er.REFERRAL
    )
