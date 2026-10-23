"""Department / specialty routing against a seeded clinic database (KCD-lead-A): a caller who asks for a SPECIALTY ("There is an orthopedic doctor") or describes
a symptom in everyday words must reach the department's REAL doctors -- for every department in the database, not just Orthopaedics -- and a department the
clinic does not have (eyes) must never be invented.

    python -m pytest tests/test_department_routing_multilingual.py -v
"""

import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


@pytest.fixture(scope="module")
def clinic():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)
    if CLINIC_API_DIR not in sys.path:
        sys.path.insert(0, CLINIC_API_DIR)
    for mod in (
        "main",
        "db",
        "models",
        "seed",
        "booking_service",
        "booking_migrate",
        "enquiry_migrate",
        "i18n_content",
    ):
        sys.modules.pop(mod, None)
    import seed as seed_mod

    seed_mod.seed()
    import booking_migrate

    booking_migrate.migrate_booking_schema()
    booking_migrate.finish_booking_schema_setup()
    import booking_service as bs
    import db as db_mod
    import models as m

    db = db_mod.SessionLocal()
    yield bs, db, m
    db.close()
    try:
        os.remove(db_path)
    except OSError:
        pass


ORTHO, ENT, DENTAL = "Orthopaedics", "ENT", "Dentistry"

# (sentence, language, expected department or None) -- the lead's lists, verbatim
SYMPTOMS = [
    ("I have pain in my bones.", "en", ORTHO),
    ("My leg hurts.", "en", ORTHO),
    ("My foot is broken.", "en", ORTHO),
    ("I fractured my leg.", "en", ORTHO),
    ("My friend had an accident and hurt his leg.", "en", ORTHO),
    ("I have ear pain.", "en", ENT),
    ("I cannot hear properly.", "en", ENT),
    ("I have a toothache.", "en", DENTAL),
    ("Which doctor should I see for my tooth?", "en", DENTAL),
    ("My eyes hurt.", "en", None),
    ("My vision is blurry.", "en", None),
    ("मेरी हड्डी में दर्द है।", "hi", ORTHO),
    ("मेरे पैर में दर्द है।", "hi", ORTHO),
    ("मेरा पैर टूट गया है।", "hi", ORTHO),
    ("मेरे पैर में फ्रैक्चर है।", "hi", ORTHO),
    ("मेरे दोस्त का एक्सीडेंट हुआ है और पैर में चोट लगी है।", "hi", ORTHO),
    ("मेरे कान में दर्द है।", "hi", ENT),
    ("मुझे ठीक से सुनाई नहीं देता।", "hi", ENT),
    ("मेरे दांत में दर्द है।", "hi", DENTAL),
    ("मेरी आंखों में दर्द है।", "hi", None),
    ("मुझे धुंधला दिखाई देता है।", "hi", None),
    ("আমার হাড়ে ব্যথা হচ্ছে।", "bn", ORTHO),
    ("আমার পায়ে ব্যথা।", "bn", ORTHO),
    ("আমার পা ভেঙে গেছে।", "bn", ORTHO),
    ("আমার পায়ে ফ্র্যাকচার হয়েছে।", "bn", ORTHO),
    ("আমার বন্ধুর অ্যাক্সিডেন্ট হয়েছে, পায়ে চোট লেগেছে।", "bn", ORTHO),
    ("আমার কানে ব্যথা।", "bn", ENT),
    ("আমি কানে ঠিকমতো শুনতে পাচ্ছি না।", "bn", ENT),
    ("আমার দাঁতে ব্যথা।", "bn", DENTAL),
    ("আমার চোখে ব্যথা।", "bn", None),
    ("আমি ঝাপসা দেখছি।", "bn", None),
    ("Amar har-e byatha.", "bn", ORTHO),
    ("Amar payer byatha.", "bn", ORTHO),
    ("Amar pa bhenge geche.", "bn", ORTHO),
    ("Amar paye fracture hoyeche.", "bn", ORTHO),
    ("Amar bondhur accident hoyeche.", "bn", ORTHO),
    ("Amar kane byatha.", "bn", ENT),
    ("Ami kane thik moto shunte pachhi na.", "bn", ENT),
    ("Amar daante byatha.", "bn", DENTAL),
    ("Amar chokhe byatha.", "bn", None),
    ("Ami jhapsa dekhchi.", "bn", None),
]


@pytest.mark.parametrize("text,lang,expected", SYMPTOMS)
def test_everyday_symptoms_reach_the_right_department_or_honestly_none(clinic, text, lang, expected):
    bs, db, m = clinic
    r = bs.route_department(db, text, lang)
    if expected is None:
        assert not r.get("matched"), (text, r)  # the clinic has no eye department: never invented
    else:
        assert r.get("matched") and r["department_name"] == expected, (text, r)
        assert r["doctors"], "the department's REAL doctors are listed"


SPECIALTY = [
    ("There is an orthopedic doctor.", "en", ORTHO),
    ("Orthopaedics Doctor", "en", ORTHO),
    ("I want to see an orthopaedic doctor.", "en", ORTHO),
    ("Do you have an orthopedic specialist?", "en", ORTHO),
    ("Which doctors are available in Orthopaedics?", "en", ORTHO),
    ("Do you have a dentist?", "en", DENTAL),
    ("Which doctor should I see for my tooth?", "en", DENTAL),
    ("Is there an ENT doctor?", "en", ENT),
    ("मुझे हड्डियों के डॉक्टर को दिखाना है।", "hi", ORTHO),
    ("ऑर्थोपेडिक विभाग में कौन-कौन से डॉक्टर हैं?", "hi", ORTHO),
    ("मुझे ऑर्थोपेडिक्स का डॉक्टर चाहिए।", "hi", ORTHO),
    ("दांत के लिए कौन से डॉक्टर हैं?", "hi", DENTAL),
    ("আমি অর্থোপেডিক ডাক্তার দেখাতে চাই।", "bn", ORTHO),
    ("অর্থোপেডিক বিভাগে কোন কোন ডাক্তার আছেন?", "bn", ORTHO),
    ("দাঁতের ডাক্তার কারা আছেন?", "bn", DENTAL),
    ("Ami orthopedic doctor dekhate chai.", "bn", ORTHO),
    ("Orthopedic-e kon kon doctor achen?", "bn", ORTHO),
]


@pytest.mark.parametrize("text,lang,expected", SPECIALTY)
def test_a_specialty_request_lists_that_departments_real_doctors(clinic, text, lang, expected):
    bs, db, m = clinic
    r = bs.department_doctors(db, text, lang)
    assert r.get("matched") and r["department_name"] == expected, (text, r)
    real = {
        bs._doctor_spoken_name(d, lang)
        for d in db.query(m.Doctor).join(m.Department).filter(m.Department.name == expected).all()
    }
    assert r["doctors"] and set(r["doctors"]) <= real, "only real doctors of that department, never invented"


def test_every_department_in_the_database_is_reachable_by_its_own_name(clinic):
    bs, db, m = clinic
    names = [d.name for d in db.query(m.Department).all()]
    assert len(names) >= 9
    for name in names:
        for form in (
            f"{name} doctor",
            f"There is a {name.lower()} doctor.",
            name.lower().replace("ae", "e") + " specialist",
        ):
            r = bs.department_doctors(db, form, "en")
            assert r.get("matched") and r["department_name"] == name, (form, r)


def test_a_department_with_no_alias_at_all_is_still_found_because_it_comes_from_the_database(clinic):
    bs, db, m = clinic
    dept = m.Department(name="Nephrology")
    db.add(dept)
    db.flush()
    db.add(m.Doctor(name="Dr. A. Kidney", qualifications="DM", department_id=dept.id))
    db.commit()
    for text in ("Is there a nephrology doctor?", "Nephrologist"):
        r = bs.department_doctors(db, text, "en")
        assert r.get("matched") and r["department_name"] == "Nephrology" and r["doctors"] == ["Dr. A. Kidney"], (
            text,
            r,
        )


def test_a_department_that_does_not_exist_is_never_invented(clinic):
    bs, db, m = clinic
    for text in ("Is there an eye doctor?", "ophthalmology doctor", "Do you have a psychiatrist?"):
        assert not bs.department_doctors(db, text, "en").get("matched"), text


def test_a_real_doctors_name_is_not_taken_for_a_department(clinic):
    bs, db, m = clinic
    for text in ("Dr. Sen", "when does doctor Mukherjee sit", "doctor Banerjee"):
        assert not bs.department_doctors(db, text, "en").get("matched"), text
