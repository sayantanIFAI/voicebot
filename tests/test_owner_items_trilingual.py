"""Quick smart check of the owner's 2026-10-23 items in English, Bengali and Hindi -- script, romanized and mixed, plain "lame" sentences (sourav, 2026-10-24).

Two layers, both fast (a few seconds):
  1. ROUTING: one table of (sentence, language, expected request) through agent.early_routing.classify, the single order both entry points use;
  2. THE REAL TURN HANDLER: the reply the caller actually hears, including item 17 in the MIDDLE of a booking (collecting and confirming) and the "ok" goodbye.
Romanized sentences arrive from the recogniser as Latin letters, i.e. tagged English, so they are driven that way; Bengali / Hindi script is driven as bn / hi.
Items covered: 6 (link missing), 6-alt (ok = goodbye), 7, 8, 9, 11, 15, 16, 17, 21, 22, 24, 25. Items 18, 19, 20 have their own suites; 19 gets one spot check here.

    python -m pytest tests/test_owner_items_trilingual.py -q
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_orchestrator_booking_flow import FakeTools, env, m  # noqa: E402,F401

from agent import call_end, doctor_enquiry_cues  # noqa: E402
from agent import early_routing as E  # noqa: E402
from agent.booking_flow import new_state  # noqa: E402

ROUTING = [
    # item 6: payment link not received
    ("I did not get the payment link", "en", E.LINK_MISSING), ("amar payment link paini", "bn", E.LINK_MISSING), ("আমি পেমেন্ট লিংক পাইনি", "bn", E.LINK_MISSING),
    ("payment link nahi aaya", "hi", E.LINK_MISSING), ("मुझे पेमेंट लिंक नहीं मिला", "hi", E.LINK_MISSING),
    # item 7: rare test not in the table -> a person
    ("I need a TB test in the ear", "en", E.RARE_TEST), ("usg of thyroid gland korte chai", "bn", E.RARE_TEST), ("কানের টিবি টেস্ট হবে", "bn", E.RARE_TEST),
    ("थायराइड का अल्ट्रासाउंड करवाना है", "hi", E.RARE_TEST),
    # item 8: "good doctor" -> no name, ask the problem
    ("who is the best doctor", "en", E.GOOD_DOCTOR), ("valo doctor ke ache", "bn", E.GOOD_DOCTOR), ("ভালো ডাক্তার কে আছে", "bn", E.GOOD_DOCTOR),
    ("accha doctor kaun hai", "hi", E.GOOD_DOCTOR), ("अच्छा डॉक्टर कौन है", "hi", E.GOOD_DOCTOR),
    # item 9: which days he sits
    ("which days does doctor sen sit", "en", E.DOCTOR_WEEKDAYS), ("doctor sen kon kon din boshen", "bn", E.DOCTOR_WEEKDAYS),
    ("ডাক্তার সেন কোন কোন দিন বসেন", "bn", E.DOCTOR_WEEKDAYS), ("doctor sen kis kis din baithte hain", "hi", E.DOCTOR_WEEKDAYS),
    # item 11: not there today -> next date + others
    ("if doctor is not there today when next", "en", E.NEXT_IF_ABSENT), ("aj doctor na pele next kobe", "bn", E.NEXT_IF_ABSENT),
    ("aj daktar na pele next kobe", "bn", E.NEXT_IF_ABSENT), ("আজ ডাক্তার না পেলে পরে কবে", "bn", E.NEXT_IF_ABSENT), ("aaj doctor nahi mile to agla kab", "hi", E.NEXT_IF_ABSENT),
    # item 16: from what time to what time
    ("from what time to what time does doctor sen sit", "en", E.SITTING_HOURS), ("doctor sen kota theke kota bose", "bn", E.SITTING_HOURS),
    ("ডাক্তার সেন কটা থেকে কটা বসেন", "bn", E.SITTING_HOURS), ("doctor sen kitne baje se kitne baje tak baithte hain", "hi", E.SITTING_HOURS),
    # item 17: cannot pay online / come and book at the clinic
    ("I don't have PhonePe", "en", E.CANNOT_PAY_ONLINE), ("phonepe hobena", "bn", E.CANNOT_PAY_ONLINE), ("gpay hobe na amar", "bn", E.CANNOT_PAY_ONLINE),
    ("ফোনপে হবে না", "bn", E.CANNOT_PAY_ONLINE), ("upi nahi hai mere paas", "hi", E.CANNOT_PAY_ONLINE), ("फोनपे नहीं है", "hi", E.CANNOT_PAY_ONLINE),
    ("can I come to the clinic and book", "en", E.COME_TO_CLINIC), ("clinic e giye book korbo", "bn", E.COME_TO_CLINIC), ("ক্লিনিকে এসে বুক করা যাবে", "bn", E.COME_TO_CLINIC),
    ("clinic aake book kar sakte hain", "hi", E.COME_TO_CLINIC), ("क्लिनिक आकर बुक कर सकते हैं क्या", "hi", E.COME_TO_CLINIC),
    # item 21: serial number
    ("what is the serial number", "en", E.SERIAL_NUMBER), ("serial number ki", "bn", E.SERIAL_NUMBER), ("সিরিয়াল নম্বর কি", "bn", E.SERIAL_NUMBER),
    ("सीरियल नंबर क्या होता है", "hi", E.SERIAL_NUMBER),
    # item 22: a name to be written -> which doctor
    ("I want to write a name", "en", E.WRITE_NAME), ("bolchi ekta naam lekhanor chhilo", "bn", E.WRITE_NAME), ("मुझे एक नाम लिखवाना था", "hi", E.WRITE_NAME),
    # item 24: when to come
    ("when should I come", "en", E.WHEN_TO_COME), ("kokhon asbo", "bn", E.WHEN_TO_COME), ("কখন আসব", "bn", E.WHEN_TO_COME), ("कब आना है", "hi", E.WHEN_TO_COME),
    # item 25: that coming weekday
    ("is doctor sen available next sunday", "en", E.DAY_AVAILABILITY), ("samner robibar doctor sen ke pabo", "bn", E.DAY_AVAILABILITY),
    ("ei mongolbar doc sen ke pabo", "bn", E.DAY_AVAILABILITY), ("সামনের রবিবার ডাক্তার সেন কে পাব", "bn", E.DAY_AVAILABILITY),
    ("agle ravivar doctor sen milenge kya", "hi", E.DAY_AVAILABILITY),
]  # fmt: skip


@pytest.mark.parametrize("text,lang,expected", ROUTING, ids=[f"{l}:{t[:28]}" for t, l, _ in ROUTING])
def test_each_sentence_reaches_its_own_answer(text, lang, expected):
    assert E.classify(text, lang) == expected


@pytest.mark.parametrize("text", ["what is my serial number", "how much is the CBC", "doctor sen er chamber kothay", "report ready hoyeche"])
def test_ordinary_questions_are_not_swallowed_by_the_new_rules(text):
    assert E.classify(text, "en") not in (E.CANNOT_PAY_ONLINE, E.COME_TO_CLINIC, E.WRITE_NAME, E.RARE_TEST, E.GOOD_DOCTOR, E.LINK_MISSING)


@pytest.mark.parametrize("text,lang", [("achha thik ache", "bn"), ("achha thik hay", "bn"), ("ok", "en"), ("ঠিক আছে", "bn"), ("আচ্ছা ঠিক আছে", "bn"), ("accha theek hai", "hi"), ("ठीक है", "hi")])
def test_a_bare_acknowledgement_is_a_goodbye(text, lang):
    assert call_end.is_closing_acknowledgement(text, lang)


@pytest.mark.parametrize("text,lang", [("ok but what is the price of CBC", "en"), ("achha thik ache kintu doctor sen kokhon bose", "bn")])
def test_an_acknowledgement_with_a_real_question_is_not_a_goodbye(text, lang):
    assert not call_end.is_closing_acknowledgement(text, lang)


# ------------------------------------------------------------------------------------------------------ the replies the caller hears (real turn handler)
ANSWERS = [  # (language the recogniser tagged, sentence, a phrase the reply must contain)
    ("en", "phonepe hobena", "take help from someone who has WhatsApp and PhonePe"),
    ("en", "I don't have PhonePe", "cannot be registered without the payment"),
    ("bn", "ফোনপে হবে না", "পেমেন্ট ছাড়া নাম লেখানো যাবে না"),
    ("hi", "फोनपे नहीं है", "पेमेंट के बिना नाम नहीं लिखा जा सकता"),
    ("en", "clinic e giye book korbo", "you may not get the same date"),
    ("bn", "ক্লিনিকে এসে বুক করা যাবে", "একই তারিখ নাও পেতে পারেন"),
    ("hi", "क्लिनिक आकर बुक कर सकते हैं क्या", "उसी तारीख़ का स्लॉट न मिले"),
    ("en", "payment link paini", "network issue"),
    ("en", "serial number ki", "serial number on WhatsApp"),
    ("en", "bolchi ekta naam lekhanor chhilo", "Which doctor"),
    ("en", "kokhon asbo", "Which doctor"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("lang,text,must", ANSWERS, ids=[f"{l}:{t[:24]}" for l, t, _ in ANSWERS])
async def test_the_reply(env, m, lang, text, must):
    env.state["lang"] = lang
    env.session.lang = lang
    env.session.disclosed_langs.add(lang)
    out = " ".join(await env.say(text))
    assert must in out, out


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["collecting", "confirming"])
@pytest.mark.parametrize(
    "lang,text,must",
    [("en", "phonepe hobena", "WhatsApp and PhonePe"), ("bn", "জিপে হবে না আমার", "ফোনপে আছে"), ("hi", "यूपीआई नहीं है मेरे पास", "फोनपे है"), ("bn", "ক্লিনিকে এসে বুক করা যাবে", "একই তারিখ"), ("hi", "क्लिनिक आकर बुक कर सकते हैं क्या", "स्लॉट")],
)
async def test_item_17_in_the_middle_of_a_booking_is_answered_and_the_booking_is_kept(env, m, stage, lang, text, must):
    """The owner's doubt: 'is the reply correctly set WHILE BOOKING?' -- yes: answered in the caller's language, the open booking is untouched."""
    env.state["lang"] = lang
    env.session.lang = lang
    env.session.disclosed_langs.add(lang)
    st = new_state("book_appointment")
    st.stage = stage
    st.slots.update(doctor_name="Sen", date="2026-10-30", time_slot="10:00", patient_name="Rahul Dey", phone="9876543210", patient_age=45)
    env.session.booking = st
    out = " ".join(await env.say(text))
    assert must in out, out
    assert env.session.booking is st and st.stage == stage and not env.session.ws.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["achha thik ache", "achha thik hay", "ok", "accha theek hai"])
async def test_an_ok_with_nothing_open_says_goodbye_and_cuts_the_call(env, m, text):
    env.state["lang"] = "en"
    await env.say(text)
    assert env.session.ws.closed


GOODBYES = [("en", t) for t in ("fine", "thik ache", "theek hai", "ok thik ache", "alright")] + [
    ("bn", t) for t in ("আচ্ছা ঠিক আছে", "ঠিক আছে", "ওকে", "আচ্ছা", "ঠিক আছে বুঝেছি", "আচ্ছা বুঝলাম")
] + [("hi", t) for t in ("ठीक है", "अच्छा ठीक है", "ओके", "अच्छा", "जी ठीक है", "अच्छा समझ गई")]  # fmt: skip


@pytest.mark.asyncio
@pytest.mark.parametrize("lang,text", GOODBYES)
async def test_the_goodbye_cuts_the_call_in_every_language(env, m, lang, text):
    """Owner, item 6: 'achha thik ache' and the like, said alone, are answered with a goodbye in the caller's language and the call is cut at once."""
    env.state["lang"] = lang
    env.session.lang = lang
    env.session.disclosed_langs.add(lang)
    await env.say(text)
    assert env.session.ws.closed


@pytest.mark.asyncio
async def test_an_ok_in_the_middle_of_a_booking_does_not_cut_the_call(env, m):
    env.state["lang"] = "en"
    st = new_state("book_appointment")
    st.stage = "collecting"
    env.session.booking = st
    await env.say("ok")
    assert not env.session.ws.closed


@pytest.mark.asyncio
async def test_a_rare_test_goes_to_a_person(env, m):
    env.state["lang"] = "en"
    out = " ".join(await env.say("usg thyroid korbo"))
    assert "connect you to our staff" in out and env.session.ws.closed


def test_item_17_wording_is_complete_in_all_three_languages():
    for lang in ("bn", "hi", "en"):
        pay, clinic = doctor_enquiry_cues.cannot_pay_online_reply(lang), doctor_enquiry_cues.come_to_clinic_reply(lang)
        assert pay and clinic and pay != clinic
    assert "WhatsApp" in doctor_enquiry_cues.cannot_pay_online_reply("en") and "call again" in doctor_enquiry_cues.cannot_pay_online_reply("en")


def test_item_19_districts_and_police_stations_are_matched_by_sound():
    from _clinic_app import clinic_app

    with clinic_app() as (_app, client):
        for said, district in [("maldah", "Malda"), ("mursidabad", "Murshidabad"), ("hugli", "Hooghly"), ("dakkhin dinajpur", "Dakshin Dinajpur"), ("মালদা", "Malda"), ("রায়গঞ্জ", "Uttar Dinajpur")]:
            r = client.get("/api/v1/geo/match", params={"q": said}).json()
            assert r["matched"] and r["district"] == district, (said, r)
