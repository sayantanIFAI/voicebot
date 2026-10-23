"""Owner, 2026-10-23: the payment-link-missing reply and three doctor questions answered by rule (agent/doctor_enquiry_cues.py), in Bengali, Hindi and English.

python -m pytest tests/test_doctor_enquiry_cues.py -v
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import doctor_enquiry_cues as dec  # noqa: E402
from agent import early_routing as er  # noqa: E402
from agent import owner_answers  # noqa: E402

LINK = [
    "I did not get the payment link",
    "payment link not received",
    "I haven't received the link",
    "link nahi aaya",
    "payment link nhi mila",
    "link pai ni",
    "payment link ashe ni",
    "link aschhe na",
    "পেমেন্ট লিংক পাইনি",
    "লিংক আসেনি",
    "पेमेंट लिंक नहीं आया",
    "लिंक नहीं मिला",
]
GOOD = [
    "valo doctor ke ache",
    "which doctor is good",
    "best doctor",
    "bhalo daktar ke ache",
    "ভালো ডাক্তার কে আছে",
    "সবচেয়ে ভালো ডাক্তার",
    "अच्छा डॉक्टर कौन है",
    "कौन सा डॉक्टर अच्छा है",
    "accha doctor kaun hai",
]
DAYS = [
    "kon kon din boshen",
    "which days does he sit",
    "কোন কোন দিন বসেন",
    "कौन से दिन बैठते हैं",
    "किस किस दिन आते हैं",
    "kaun se din baithte hain",
]
NEXT = [
    "aj doctor na pele next kobe",
    "aj dagtar na pele next kobe",
    "ajke daktar na pele porer din kobe",
    "if not today when next",
    "আজ ডাক্তার না পেলে পরে কবে",
    "आज डॉक्टर नहीं मिले तो अगला कब",
    "aaj doctor nahi mile to agla kab",
]
ORDINARY = [
    "I want to book Dr Sen tomorrow",
    "what is the price of CBC",
    "when does doctor Sen sit tomorrow",
    "ডাক্তার সেন কখন বসেন",
    "payment link bhejo",
    "what days are you open",
    "आपका क्लिनिक कब खुलता है",
    "aj kono doctor ache?",
]


@pytest.mark.parametrize(
    "kind,texts",
    [(er.LINK_MISSING, LINK), (er.GOOD_DOCTOR, GOOD), (er.DOCTOR_WEEKDAYS, DAYS), (er.NEXT_IF_ABSENT, NEXT)],
)
def test_each_question_is_recognised_in_every_language(kind, texts):
    for t in texts:
        assert er.classify(t, "en") == kind, t


def test_ordinary_turns_are_not_taken_for_these_questions():
    for t in ORDINARY:
        assert er.classify(t, "en") not in (er.LINK_MISSING, er.GOOD_DOCTOR, er.DOCTOR_WEEKDAYS, er.NEXT_IF_ABSENT), t


def test_the_payment_link_reply_is_the_owners_sentence_in_every_language():
    assert "network issue" in owner_answers.payment_link_missing_reply(
        "en"
    ) and "fifteen minutes" in owner_answers.payment_link_missing_reply("en")
    assert "নেটওয়ার্ক" in owner_answers.payment_link_missing_reply(
        "bn"
    ) and "पंद्रह मिनट" in owner_answers.payment_link_missing_reply("hi")


def test_weekdays_are_said_as_day_names_never_dates():
    assert dec.weekdays_reply("সেন", [0, 2, 4], "18:00-20:00", "bn") == "সেন সোম বুধ শুক্র বারে বসেন। সময় 18:00-20:00।"
    assert dec.weekdays_reply("सेन", [0, 2, 4], None, "hi") == "सेन सोमवार, बुधवार और शुक्रवार को बैठते हैं।"
    assert dec.weekdays_reply("Sen", [0, 2, 4], None, "en") == "Sen sits on Monday, Wednesday and Friday."
    assert not any(ch.isdigit() for ch in dec.weekdays_reply("Sen", [1], None, "en"))


def test_a_good_doctor_question_names_no_one_but_asks_the_problem_and_lists_general_medicine():
    said = dec.good_doctor_reply(["Dr. A", "Dr. B"], None, "en")
    assert (
        "cannot say which doctor" in said
        and "problem" in said
        and "General Medicine" in said
        and "Dr. A, Dr. B" in said
    )
    assert "General Medicine" not in dec.good_doctor_reply([], None, "en")


def test_next_if_absent_gives_the_next_date_and_the_other_doctors():
    said = dec.next_if_absent_reply("Sen", "2099-01-05", ["Dr. B", "Dr. C"], "en")
    assert "2099-01-05" in said and "Dr. B, Dr. C" in said
    assert "Dr." not in dec.next_if_absent_reply("Sen", "2099-01-05", [], "en")


WHEN = [
    "When should I come?",
    "what time should I come",
    "what time do I need to reach",
    "what time to come for the doctor",
    "so when do I go to see him",
    "আমি কখন আসব?",
    "কটার সময় আসব",
    "কখন যাব ডাক্তার দেখাতে",
    "কটায় যেতে হবে",
    "কখন আসতে হবে",
    "मैं कब आऊं?",
    "कितने बजे आना है",
    "कब पहुंचना है",
    "डॉक्टर के पास कब जाना है",
    "kokhon asbo",
    "kokhon ashte hobe",
    "ami kokhon jabo",
    "kotay asbo",
    "kab aaun",
    "kitne baje aana hai",
    "kab aana hai",
]
SERIAL_Q = [
    "What is my serial number?",
    "what is the serial number",
    "when will I get my serial number",
    "where will I get the serial number",
    "I want my serial number",
    "how do I know my serial",
    "আমার সিরিয়াল নম্বর কত?",
    "সিরিয়াল নম্বর কখন পাব",
    "সিরিয়াল নম্বর কোথায় পাব",
    "আমার সিরিয়াল কত",
    "मेरा सीरियल नंबर क्या है?",
    "सीरियल नंबर कब मिलेगा",
    "सीरियल कहां मिलेगा",
    "amar serial number ki",
    "serial number kokhon pabo",
    "serial kothay pabo",
    "serial number kobe pabo",
    "mera serial number kya hai",
    "serial number kab milega",
]
KEEP_EXISTING = [  # a token NUMBER, or a push for an earlier one: owner_answers keeps its exact answers
    "when should I come, my token number is 20",
    "আমার টোকেন ২০ কখন আসব",
    "can I get a serial number if I come early, token push",
    "আমার সিরিয়াল ২০",
]


def test_when_to_come_is_recognised_in_every_language():
    for t in WHEN:
        assert er.classify(t, "en") == er.WHEN_TO_COME, t


def test_the_serial_number_question_is_recognised_in_every_language():
    for t in SERIAL_Q:
        assert er.classify(t, "en") == er.SERIAL_NUMBER, t


def test_token_number_sentences_and_opening_hours_keep_their_existing_answers():
    for t in KEEP_EXISTING + [
        "what time do you open",
        "clinic kokhon khola thake",
        "when does doctor Sen sit tomorrow",
    ]:
        assert er.classify(t, "en") not in (er.WHEN_TO_COME, er.SERIAL_NUMBER), t


def test_the_serial_number_reply_says_whatsapp_after_payment():
    assert "WhatsApp" in dec.serial_number_reply("en") and "pay" in dec.serial_number_reply("en")
    assert "হোয়াটসঅ্যাপে" in dec.serial_number_reply("bn") and "व्हाट्सऐप" in dec.serial_number_reply("hi")


# ------------------------------------------------------------------------------------------------------------ round 4 (owner, 2026-10-23): 17, 16, 22
CANNOT_PAY = [
    "PHONEPAY HOBENA",
    "UPI hobena",
    "GPAY hobena",
    "phonepe nei",
    "amar upi nei",
    "upi chole na",
    "ami online payment korte parbo na",
    "ফোনপে হবে না",
    "ইউপিআই নেই",
    "জিপে হবেনা",
    "ফোন পে নেই",
    "অনলাইনে পেমেন্ট করতে পারব না",
    "फोनपे नहीं है",
    "यूपीआई नहीं चलता",
    "ऑनलाइन पेमेंट नहीं हो पाएगा",
    "phonepe nahi hai",
    "gpay nhi hai",
    "I don't have PhonePe",
    "UPI is not available",
    "I cannot pay online",
]
CAN_PAY = [
    "I will pay by upi",
    "no problem I will pay by upi",
    "can I pay by gpay",
    "upi te dibo",
    "phonepe e payment korbo",
    "ফোনপে তে দেব",
    "यूपीआई से कर दूंगा",
    "not cash, upi",
]
CLINIC = [
    "can I come and book at the clinic",
    "I will come to the counter and book directly",
    "walk in booking possible",
    "clinic e giye book korte parbo",
    "counter e giye serial nebo",
    "sorasori clinic e asbo",
    "ক্লিনিকে এসে বুক করতে পারব",
    "কাউন্টারে গিয়ে সিরিয়াল নেব",
    "সরাসরি ক্লিনিকে গিয়ে",
    "क्लिनिक आकर बुक कर सकता हूं",
    "सीधे क्लिनिक आ जाऊं",
    "काउंटर पर जाकर नाम लिखवा दूंगा",
]
HOURS = [
    "doctor x kota theke kota bose",
    "kota theke kota bose",
    "kouta theke kota boshen",
    "kota thke kota bose",
    "kokhon theke kokhon bose",
    "kotokkhon porjonto bosen",
    "kota porjonto bose",
    "from what time to what time does he sit",
    "till what time does the doctor sit",
    "doctor timing",
    "what are the doctor hours",
    "কটা থেকে কটা বসেন",
    "ডাক্তার কটা থেকে কটা বসবেন",
    "কোটা থেকে কোটা বসে",
    "কখন থেকে কখন বসেন",
    "কটা পর্যন্ত বসেন",
    "কতক্ষণ বসেন",
    "ডাক্তারের সময় কত",
    "कितने बजे से कितने बजे तक बैठते हैं",
    "कब से कब तक बैठेंगे",
    "कितने बजे तक बैठते हैं",
    "डॉक्टर का टाइमिंग क्या है",
    "kitne baje se kitne baje tak",
    "kab se kab tak baithte hain",
]
CLINIC_HOURS = [
    "clinic kota theke kota khola",
    "what time do you open",
    "clinic kokhon khola",
    "ক্লিনিক কটা থেকে কটা খোলা",
    "क्लिनिक कब खुलता है",
]
NAME = [
    "bolchi ekta naam lekhanor chhilo",
    "ekta naam lekhate chai",
    "naam likhbo",
    "naam lekhabo",
    "put my name",
    "serial laganor chhilo",
    "আমি একটা নাম লেখাতে চাই",
    "নাম লেখানোর ছিল",
    "বলছি একটা নাম লেখানোর ছিল",
    "সিরিয়াল লাগাতে চাই",
    "नाम लिखवाना है",
    "एक नाम लिखवाना था",
    "सीरियल लगवाना है",
    "naam likhwana hai",
]
# an ordinary booking request (a doctor, an appointment, a serial) is NOT this: the booking flow asks its own questions
NOT_A_NAME_REQUEST = [
    "doctor ke sathe appointment chai",
    "I want an appointment",
    "serial nite chai",
    "appointment nite chai",
    "मुझे डॉक्टर के साथ अपॉइंटमेंट चाहिए",
    "सিরিয়াল নিতে চাই",
]
NAME_LEFT_ALONE = [
    "naam lekhano ache blood test er",
    "CBC test er serial nite chai",
    "রক্ত পরীক্ষার সিরিয়াল নিতে চাই",
    "রিপোর্টের নাম লেখা",
    "ডাক্তার কখন বসেন",
    "what is my name",
]


def test_item_17_cannot_pay_online_is_recognised_and_ordinary_payment_talk_is_not():
    for t in CANNOT_PAY:
        assert er.classify(t, "en") == er.CANNOT_PAY_ONLINE, t
    for t in CAN_PAY:
        assert er.classify(t, "en") != er.CANNOT_PAY_ONLINE, t
    assert (
        "WhatsApp" in dec.cannot_pay_online_reply("en")
        and "হোয়াটসঅ্যাপ" in dec.cannot_pay_online_reply("bn")
        and "व्हाट्सऐप" in dec.cannot_pay_online_reply("hi")
    )


def test_item_17_coming_to_the_clinic_to_book_is_recognised():
    for t in CLINIC:
        assert er.classify(t, "en") == er.COME_TO_CLINIC, t
    assert "may not get the same date" in dec.come_to_clinic_reply("en")


def test_item_16_from_and_to_hours_are_recognised_but_the_clinics_own_hours_are_not():
    for t in HOURS:
        assert er.classify(t, "en") == er.SITTING_HOURS, t
    for t in CLINIC_HOURS:
        assert er.classify(t, "en") != er.SITTING_HOURS, t
    assert (
        dec.sitting_hours_reply("Dr. Sen", "2099-01-05", "10:00", "12:00", "en")
        == "Doctor Sen sits on 2099-01-05 from 10:00 to 12:00."
    )
    assert "10:00 থেকে 12:00 পর্যন্ত" in dec.sitting_hours_reply("সেন", "2099-01-05", "10:00", "12:00", "bn")
    assert "10:00 से 12:00 तक" in dec.sitting_hours_reply("सेन", "2099-01-05", "10:00", "12:00", "hi")


def test_item_22_a_name_to_be_written_is_recognised_and_a_test_or_report_is_left_alone():
    for t in NAME:
        assert er.classify(t, "en") == er.WRITE_NAME, t
    for t in NOT_A_NAME_REQUEST:
        assert er.classify(t, "en") != er.WRITE_NAME, t
    for t in NAME_LEFT_ALONE:
        assert er.classify(t, "en") != er.WRITE_NAME, t


def test_item_18_stopping_after_two_digits_is_told_to_say_the_other_eight():
    from agent import booking_capture as bc

    for lang, word in (("en", "other eight"), ("hi", "आठ"), ("bn", "আটটা")):
        assert word in bc.prompt("phone_partial_two", lang, digits="9 8"), lang
    assert bc.prompt("phone_partial", "en") == "Okay, go on."  # a number said in groups keeps its short "ok": unchanged
