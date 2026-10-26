"""Owner, 2026-10-26: the Bengali / Hindi calls that went wrong, one test per fix. Real sentences from those calls."""

import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "clinic-api"))
sys.path.insert(0, os.path.join(HERE, ".."))


def test_phone_numbers_said_in_pairs_as_words():
    from agent import booking_capture as B

    assert B.read_digits("আটানব্বই ছত্রিশ সাতচল্লিশ একুশ পঞ্চাশ")[0] == "9836472150"
    assert B.read_digits("अट्ठांवे बांवे चौहत्तर पैंतीस इकसठ")[0] == "9892743561"
    assert B.read_digits("नंबर है अठानवे बानवे चौहत्तर पैतीस इकसठ")[0] == "9892743561"


def test_a_partial_whatsapp_number_is_added_up():
    from agent import booking_capture as B

    assert B.absorb("707972321", "पाँच", True).number == "7079723215"


@pytest.mark.parametrize(
    "text,dept",
    [
        ("बोल रहे थे की वो लड़की लोग का डॉक्टर् बैठता है क्या", "Gynaecology & Obstetrics"),
        ("बोल रहे थे की लड़की लोगों के लिए वो डॉक्टर् बैठता है", "Gynaecology & Obstetrics"),
        ("ইন্ট ডাক্তার আছে", "ENT"),
    ],
)
def test_lay_doctor_phrases(text, dept):
    import lay_departments

    assert lay_departments.resolve(text) == [dept]


def test_my_daughter_is_not_a_request_for_a_lady_doctor():
    import lay_departments

    assert "Gynaecology & Obstetrics" not in lay_departments.resolve("আমার মেয়ের জ্বর ডাক্তার")


@pytest.fixture(scope="module")
def client():
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, c):
        yield c


@pytest.mark.parametrize(
    "q",
    [
        "বলছি আমার কানে অনেক ব্যথা করছে তো আজকে কোন ডাক্তার বসছে",
        "বলছি আমার কানে অনেক ব্যাথা করছে তো আজকে কোনো ডাক্তার বসছে কি",
        "कान दर्द है कौन सा डॉक्टर बैठता है",
    ],
)
def test_ear_pain_is_routed_to_ent(client, q):
    r = client.get("/api/v1/departments/route", params={"query": q, "lang": "bn"}).json()
    assert r.get("matched") and r["department_name"] == "ENT", r


def test_eyes_have_no_department_and_the_agent_says_so():
    from agent import reply_templates as bn, reply_templates_i18n as i18n

    assert "চোখের কোনো বিভাগ" in bn.department_route_reply({"matched": False}, "আমার চোখটা অনেক জ্বালা করছে", "bn")
    assert "आँखों का कोई विभाग" in i18n.department_route_reply({"matched": False}, "मेरी आँख में जलन", "hi")
    assert "eye department" in i18n.department_route_reply({"matched": False}, "my eye burns", "en")
    assert "চোখ" not in bn.department_route_reply({"matched": False}, "পেটে ব্যথা", "bn")


@pytest.mark.parametrize("name", ["डॉक्टर्", "ডাক্তার", "doctor", "কোন ডাক্তার", "एक डॉक्टर"])
def test_the_word_doctor_alone_is_not_a_doctors_name(name):
    from agent import doctor_search

    assert doctor_search.is_generic_doctor_phrase(name)


@pytest.mark.parametrize("name", ["अमित कर", "Dr. Sen", "সঞ্জয় গুহ", "ইন্ট ডাক্তার"])
def test_a_real_name_is_not_generic(name):
    from agent import doctor_search

    assert not doctor_search.is_generic_doctor_phrase(name)


def test_a_doctor_is_chosen_with_the_bengali_of_ending():
    from agent import doctor_search as D

    docs = [["রঞ্জন পাল", "Dr. R. Pal"], ["সঞ্জয় গুহ", "Dr. S. Guha"]]
    assert D.choose_by_name("তাহলে ওই সঞ্জয় গুহার অ্যাপয়েন্টমেন্ট টা দেন কালকে", docs) == 1
    assert D.choose_by_name("রঞ্জন পাল আর সঞ্জয় গুহার মধ্যে কোন ডাক্তার ভালো", docs) is None  # both named: never a guess
    assert D.choose_by_name("বুক করে দিন", [["অমিত কর"], ["দেবাশীষ দাস"]]) is None  # "কর" alone is a common word


def test_girls_doctor_in_hindi_wants_a_woman():
    from agent import doctor_search as D

    assert D.wanted_sex("बोल रहे थे की लड़की लोग का डॉक्टर् बैठता है क्या") == "F"


def test_blood_and_body_lay_terms_in_hindi():
    from agent import lay_terms as L

    assert L.match_lay_term("वो ब्लड टेस्ट होता है क्या आप लोग के यहाँ पे") == L.BLOOD
    assert L.match_lay_term("वो माथा का करवाना था टेस्ट हो जायेगा आज") == L.HEAD
    assert L.match_lay_term("पेट का चेक उप करवाना था पेट का टेस्ट हो जाएगा आज") == L.ABDOMEN
    assert L.says_unsure_which_test("नहीं वो तो नहीं मालूम वो प्रेक्शन में कुछ लिखे हैं तो वो नहीं मालूम")
    assert L.says_unsure_which_test("ডাক্তার বলেছে, নামটা জানি না")
    assert not L.says_unsure_which_test("CBC")


def test_urine_in_hindi_is_a_known_test_name(client):
    names = {t["name"]: t for t in client.get("/api/v1/catalogue").json()["tests"]}
    aliases = names["Urine Routine Examination"]
    blob = " ".join(str(v) for v in aliases.values())
    assert "मूत्र का टेस्ट" in blob


def test_the_greeting_says_the_call_is_recorded_only_when_switched_on(monkeypatch):
    import importlib

    from agent import phrases

    assert "রেকর্ড" not in phrases.greeting_text("bn")
    monkeypatch.setenv("CALL_RECORDING_NOTICE", "1")
    assert "রেকর্ড" in phrases.greeting_text("bn")
    monkeypatch.delenv("CALL_RECORDING_NOTICE")
    importlib.reload(phrases)
