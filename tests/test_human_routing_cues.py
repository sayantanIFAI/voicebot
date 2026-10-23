"""New lead-requested human-routing/billing cues (KCD-lead-9/10/12): pure, deterministic, pre-model detectors --
same discipline and test style as test_disclosure_and_human_request.py's asks_for_a_person coverage.

    python -m pytest tests/test_human_routing_cues.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import (
    owner_answers,  # noqa: E402
    referral_cues,  # noqa: E402
    report_pickup,  # noqa: E402
)
from agent.billing_cues import asks_about_dues  # noqa: E402
from agent.human_request import wants_doctor_directly, wants_to_complain  # noqa: E402

# ------------------------------------------------------------------------------------------- wants_doctor_directly


@pytest.mark.parametrize(
    "text,lang",
    [
        ("I want to talk to the doctor personally", "en"),
        ("can you connect me to the doctor directly", "en"),
        ("I personally want to speak to the doctor", "en"),
        ("the doctor himself should talk to me", "en"),
        ("ডাক্তারের সাথে নিজে কথা বলতে চাই", "bn"),
        ("সরাসরি ডাক্তারের সঙ্গে কথা বলব", "bn"),
        ("मुझे डॉक्टर से खुद बात करनी है", "hi"),
        ("सीधे डॉक्टर से बात कराइए", "hi"),
    ],
)
def test_a_request_to_speak_to_the_doctor_personally_is_recognised(text, lang):
    assert wants_doctor_directly(text, lang), text


@pytest.mark.parametrize(
    "text,lang",
    [
        ("I want to book an appointment with the doctor", "en"),
        ("when is the doctor available", "en"),
        ("book me with the doctor", "en"),
        ("আমি ডাক্তারের সাথে অ্যাপয়েন্টমেন্ট চাই", "bn"),
        ("ডাক্তার কখন বসেন", "bn"),
        ("मुझे डॉक्टर के साथ अपॉइंटमेंट चाहिए", "hi"),
        ("", "en"),
    ],
)
def test_an_ordinary_booking_request_is_never_taken_as_wanting_the_doctor_personally(text, lang):
    assert not wants_doctor_directly(text, lang), text


# ------------------------------------------------------------------------------------------------- wants_to_complain


@pytest.mark.parametrize(
    "text,lang",
    [
        ("I want to make a complaint", "en"),
        ("I have a complaint about the service", "en"),
        ("I need to file a complaint", "en"),
        ("আমার একটা অভিযোগ আছে", "bn"),
        ("কমপ্লেইন করতে চাই", "bn"),
        ("मुझे एक शिकायत दर्ज करनी है", "hi"),
        ("कंप्लेंट करना चाहता हूं", "hi"),
    ],
)
def test_a_complaint_is_recognised_in_every_language(text, lang):
    assert wants_to_complain(text, lang), text


@pytest.mark.parametrize(
    "text,lang",
    [
        ("what is the price of a CBC test", "en"),
        ("no complaints, everything was fine", "en"),
        ("কোনো অভিযোগ নেই, সব ঠিক আছে", "bn"),
        ("कोई शिकायत नहीं है", "hi"),
        ("", "en"),
    ],
)
def test_an_ordinary_turn_or_an_explicit_no_complaint_does_not_fire(text, lang):
    assert not wants_to_complain(text, lang), text


# --------------------------------------------------------------------------------------------------- asks_about_dues


@pytest.mark.parametrize(
    "text,lang",
    [
        ("do I owe you anything", "en"),
        ("is there any pending payment on my account", "en"),
        ("how much do I owe", "en"),
        ("আমার কোনো বকেয়া আছে", "bn"),
        ("কত টাকা বাকি দিতে হবে", "bn"),
        ("मेरा कोई बकाया है क्या", "hi"),
        ("कितने पैसे बाकी देने हैं", "hi"),
    ],
)
def test_a_dues_question_is_recognised_in_every_language(text, lang):
    assert asks_about_dues(text, lang), text


@pytest.mark.parametrize(
    "text,lang",
    [
        ("what is the price of a CBC test", "en"),
        ("no, nothing is due, I already paid", "en"),
        ("কোনো বকেয়া নেই", "bn"),
        ("कोई पैसा बाकी नहीं", "hi"),
        ("", "en"),
    ],
)
def test_an_ordinary_turn_or_an_explicit_no_dues_does_not_fire(text, lang):
    assert not asks_about_dues(text, lang), text


# ------------------------------------------------------------------------------------- owner_answers.billing_dues_reply


def test_billing_dues_reply_mentions_whatsapp_in_every_language(monkeypatch):
    monkeypatch.setattr(owner_answers, "CLINIC_WHATSAPP_NUMBER", "")
    for lang in ("en", "bn", "hi"):
        said = owner_answers.billing_dues_reply(lang)
        assert said  # never empty
    assert "WhatsApp" in owner_answers.billing_dues_reply("en")
    assert "হোয়াটসঅ্যাপ" in owner_answers.billing_dues_reply("bn")
    assert "व्हाट्सऐप" in owner_answers.billing_dues_reply("hi")


def test_billing_dues_reply_includes_the_configured_number(monkeypatch):
    monkeypatch.setattr(owner_answers, "CLINIC_WHATSAPP_NUMBER", "9830000000")
    assert "9830000000" in owner_answers.billing_dues_reply("en")
    assert "9830000000" in owner_answers.billing_dues_reply("bn")


# ---------------------------------------------------------------------------------------------- report pickup (KCD-lead-8)


@pytest.mark.parametrize(
    "text,lang",
    [
        ("is my report ready", "en"),
        ("can I collect my report", "en"),
        ("when will my report be ready", "en"),
        ("রিপোর্টটা রেডি হয়েছে", "bn"),
        ("রিপোর্ট কালেক্ট করতে পারব", "bn"),
        ("मेरी रिपोर्ट तैयार है क्या", "hi"),
        ("रिपोर्ट कलेक्ट कर सकता हूं", "hi"),
    ],
)
def test_a_report_pickup_question_is_recognised(text, lang):
    assert report_pickup.asks_about_report_pickup(text, lang), text


@pytest.mark.parametrize(
    "text,lang",
    [
        ("what is the price of a CBC test", "en"),
        ("I want to book an appointment", "en"),
        ("", "en"),
    ],
)
def test_an_ordinary_turn_is_not_taken_as_a_report_question(text, lang):
    assert not report_pickup.asks_about_report_pickup(text, lang), text


def test_mentions_today_only_fires_on_an_explicit_today():
    assert report_pickup.mentions_today("is my report ready today", "en")
    assert report_pickup.mentions_today("আজ রিপোর্টটা পাব", "bn")
    assert report_pickup.mentions_today("आज रिपोर्ट मिल जाएगी", "hi")
    assert not report_pickup.mentions_today("is my report ready", "en")
    assert not report_pickup.mentions_today("is my report ready tomorrow", "en")
    assert not report_pickup.mentions_today("", "en")


def test_report_pickup_defer_reply_names_the_visit_day_not_a_guessed_status():
    for lang, words in (
        ("en", ("call us back", "day of your test")),
        ("bn", ("ফোন করে", "টেস্ট বা ডাক্তার")),
        ("hi", ("फ़ोन करके", "टेस्ट या डॉक्टर")),
    ):
        reply = report_pickup.report_pickup_defer_reply(lang)
        assert all(w in reply for w in words)


# -------------------------------------------------------------------------------------------- referral cues (KCD-lead-1)


@pytest.mark.parametrize(
    "text,lang",
    [
        ("I'm calling from another branch, I want to pass a number for an appointment", "en"),
        ("this is a referral from a different center", "en"),
        ("অন্য সেন্টার থেকে বলছি, একটা নম্বর দিতে চাই অ্যাপয়েন্টমেন্টের জন্য", "bn"),
        ("এটা একটা রেফারেল", "bn"),
        ("मैं दूसरी ब्रांच से बोल रहा हूं, एक नंबर देना चाहता हूं अपॉइंटमेंट के लिए", "hi"),
        ("यह एक रेफरल है", "hi"),
    ],
)
def test_an_inter_center_referral_call_is_recognised(text, lang):
    assert referral_cues.asks_to_pass_a_number(text, lang), text


@pytest.mark.parametrize(
    "text,lang",
    [
        ("I want to book an appointment with the doctor", "en"),
        (
            "I want to pass my number to you",
            "en",
        ),  # no "another branch" -- an ordinary patient leaving their own number
        ("অন্য সেন্টার থেকে ফোন করছি, রিপোর্ট কবে পাব", "bn"),  # another center mentioned, but not passing a number
        ("", "en"),
    ],
)
def test_an_ordinary_call_is_not_taken_as_a_referral(text, lang):
    assert not referral_cues.asks_to_pass_a_number(text, lang), text


def test_referral_ask_and_logged_replies_are_bilingual_and_distinct():
    for lang in ("en", "bn", "hi"):
        ask, logged = referral_cues.referral_ask_reply(lang), referral_cues.referral_logged_reply(lang)
        assert ask and logged and ask != logged
    assert "name" in referral_cues.referral_ask_reply("en") and "phone number" in referral_cues.referral_ask_reply("en")
    assert "noted" in referral_cues.referral_logged_reply(
        "en"
    ).lower() and "call back" in referral_cues.referral_logged_reply("en")
