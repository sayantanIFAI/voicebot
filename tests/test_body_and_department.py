"""Owner, 2026-10-17: body-part pictures are asked back from the catalogue; department short forms; who is in today; no advice."""
import importlib
import pathlib
import sys

from agent import body_imaging as bi
from agent import dept_list as dl
from agent import reply_templates as rt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "clinic-api"))

CAT = [
    {"name": "Chest X-Ray (PA view)", "sample_type": "Imaging"},
    {"name": "Chest CT Scan", "sample_type": "Imaging"},
    {"name": "USG Whole Abdomen", "sample_type": "Imaging"},
    {"name": "2D Echocardiography", "sample_type": "Cardiac"},
]


def test_body_part_pictures_in_bengali_hindi_and_english():
    assert bi.detect("আমার বুকের ছবি করব") == "chest"
    assert bi.detect("buker chhobi korbo") == "chest"
    assert bi.detect("मेरी छाती का फोटो कराना है") == "chest"
    assert bi.detect("মাথার ছবি করব") == "head"
    assert bi.detect("peter chhobi korbo") == "abdomen"
    assert bi.detect("হাতের ছবি তুলব") == "hand"
    assert bi.detect("पैर की तस्वीर") == "leg"


def test_a_named_test_a_price_or_a_symptom_is_not_a_body_picture_request():
    for t in ["বুকের এক্স রে", "আমার বুকে ব্যথা", "chest x-ray price", "ডাক্তার দেখাব", "আমার বুকিং কনফার্ম করুন বুকের ছবি"]:
        assert bi.detect(t) is None, t


def test_a_picture_of_a_part_with_a_price_or_a_doctor_in_the_same_breath_is_still_a_picture_request():
    # SPEC CHANGE 2026-10-18: "পেটের ছবি তুলব, কোন টেস্ট মনে নাই, ডাক্তার বলেছিল, কখন আসব, কত টাকা পড়বে" -> the tests for that part, with their rates
    long = "আমার ওই পেটের ছবি তুলব কোন টেস্ট এ আমার মনে নাই পেটের ছবি তুলতে হবে ডাক্তার বলেছিল কখন আসবো কত টাকা পড়বে"
    assert bi.detect(long) == "abdomen" and bi.asks_price(long)
    assert bi.detect("বুকের ছবির দাম কত") == "chest" and bi.detect("হাতের ছবি তুলতে চাই কত লাগবে") == "hand"
    assert not bi.asks_price("মাথার ছবি তুলব")
    assert bi.listing("abdomen", [("পেটের আলট্রাসাউন্ড", 1500)], "bn") == "পেটের ছবির জন্য আমাদের এখানে এই টেস্টগুলো হয়: পেটের আলট্রাসাউন্ড 1500 টাকা।"
    assert "আমাদের এখানে ছবির যে টেস্টগুলো হয়" in bi.listing(None, [("বুকের এক্স-রে", None)], "bn")


def test_candidates_come_only_from_the_catalogue():
    assert bi.candidates("chest", CAT) == ["Chest X-Ray (PA view)", "Chest CT Scan"]
    assert bi.candidates("abdomen", CAT) == ["USG Whole Abdomen"]
    assert bi.candidates("hand", CAT) == []  # nothing for a hand: never invented


def test_the_question_asks_and_never_names_a_test_that_is_not_there():
    q = bi.question("chest", ["বুকের এক্স রে", "বুকের সিটি স্ক্যান"], "bn")
    assert "নাকি" in q and q.endswith("?")
    assert "listed right now" in bi.question("hand", [], "en") and "নেই" in bi.question("hand", [], "bn")


def test_the_doctor_list_question():
    assert dl.asks_doctor_list("who are the doctors in cardio")
    assert dl.asks_doctor_list("কার্ডিওতে কোন কোন ডাক্তার আছেন")
    assert dl.asks_doctor_list("कार्डियो में कौन कौन से डॉक्टर हैं")
    assert not dl.asks_doctor_list("Dr Sen কখন বসেন")


def test_department_short_forms():
    bs = importlib.import_module("booking_service")
    for q, dept in [("cardio", "Cardiology"), ("who are the doctors in ortho", "Orthopaedics"), ("gynae doctor", "Gynaecology & Obstetrics"), ("derma", "Dermatology"),
                    ("pedia", "Paediatrics"), ("gen med", "General Medicine"), ("endo", "Diabetology & Endocrinology"), ("কার্ডিও", "Cardiology"), ("ent doctor", "ENT")]:
        assert bs._alias_department(q) == dept, q
    assert bs._alias_department("what are your timings") is None
    assert bs._alias_department("recently") is None  # a short key inside a longer word is not a department


def test_the_department_answer_lists_today_and_asks_which_one_without_advice():
    res = {"matched": True, "department_name": "Cardiology", "doctors": ["ডাঃ সেন", "ডাঃ রায়", "ডাঃ দাস"], "doctors_today": ["ডাঃ সেন", "ডাঃ রায়", "ডাঃ দাস"]}
    bn = rt.department_route_reply(res, "বুকে ব্যথা", "bn")
    assert "আজ Cardiology বিভাগে তিনজন ডাক্তার বসছেন" in bn and bn.endswith("কাকে দেখাতে চান?")
    for advice in ("ভালো হবে", "টেস্ট", "ওষুধ"):
        assert advice not in bn
    en = rt.department_route_reply(res, "leg pain", "en")
    assert "Which one would you like to see?" in en and "best to see" not in en
    none_today = dict(res, doctors_today=[])
    assert "আজ কেউ বসছেন না" in rt.department_route_reply(none_today, "x", "bn")
