"""Owner's 2026-10-18 data-layer additions (clinic-api/clinic_extras.py and friends): doctor sex and the lady-doctor filter, the departments added for
routing, a doctor going on leave (appointments cancelled with the reason, apology queued, told later), the Rs 40 / 15-minute payment link."""

import datetime
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


@pytest.fixture()
def clinic():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)
    if CLINIC_API_DIR not in sys.path:
        sys.path.insert(0, CLINIC_API_DIR)
    for mod in ("main", "db", "models", "seed", "booking_service", "booking_migrate", "enquiry_migrate", "i18n_content", "clinic_extras", "payments"):
        sys.modules.pop(mod, None)
    import seed as seed_mod

    seed_mod.seed()
    import booking_migrate

    booking_migrate.migrate_booking_schema()
    booking_migrate.finish_booking_schema_setup()
    import booking_service as bs
    import clinic_extras
    import db as db_mod
    import models as m

    db = db_mod.SessionLocal()
    clinic_extras.apply_extras(db)
    yield bs, clinic_extras, db, m
    db.close()
    try:
        os.remove(db_path)
    except OSError:
        pass


def _next_weekday(wd: int) -> str:
    d = datetime.date.today() + datetime.timedelta(days=1)
    while d.weekday() != wd:
        d += datetime.timedelta(days=1)
    return d.isoformat()


def test_every_doctor_has_a_sex_and_the_extras_are_idempotent(clinic):
    bs, extras, db, m = clinic
    assert not db.query(m.Doctor).filter(m.Doctor.sex == "").all()
    again = extras.apply_extras(db)
    assert again == {"doctor_sex_set": 0, "departments": 0, "doctors": 0, "routes": 0, "relationships": 0}


def test_a_lady_gynaecologist_filters_to_women_only(clinic):
    bs, extras, db, m = clinic
    women = bs.department_doctors(db, "gynae", "en", sex="F")
    assert women["filtered"] and women["doctors"]
    assert all(s["sex"] == "F" for s in women["sitting"] or [])
    men = bs.department_doctors(db, "gynae", "en", sex="M")
    assert men["doctors"] == ["Dr. Alok Pathak"]
    assert not set(men["doctors"]) & set(women["doctors"])


def test_a_filter_that_leaves_nobody_says_so(clinic):
    bs, extras, db, m = clinic
    r = bs.department_doctors(db, "neurology", "en", sex="F")
    assert r["matched"] and r["doctors"] == []


def test_doctors_sitting_at_a_time_on_a_date(clinic):
    bs, extras, db, m = clinic
    monday = _next_weekday(0)
    dept = db.query(m.Department).filter_by(name="General Surgery").one()
    at_noon = bs.sitting_doctors(db, dept.id, "en", on_date=monday, at_time="12:00")
    assert [r["full_name"] for r in at_noon] == ["Dr. Sharmila Dasgupta"]  # Monday 11-13; the other surgeon sits Tue/Thu/Sat
    assert bs.sitting_doctors(db, dept.id, "en", on_date=monday, at_time="09:00") == []


def test_new_departments_and_hernia_routes_to_general_surgery(clinic):
    bs, extras, db, m = clinic
    r = bs.route_department(db, "hernia", "en")
    assert r["matched"] and r["department_name"] == "General Surgery"
    assert bs.route_department(db, "হার্নিয়া", "bn")["department_name"] == "General Surgery"
    for name in ("Neurology", "Neurosurgery", "Urology"):
        assert db.query(m.Department).filter_by(name=name).one()


def test_relationship_rules_carry_the_sex(clinic):
    bs, extras, db, m = clinic
    rules = {r["relationship"]: r for r in extras.relationship_rules(db)}
    assert rules["mother"]["sex"] == "F" and "মা" in rules["mother"]["keywords_bn"]
    assert rules["father"]["sex"] == "M"
    assert rules["child"]["sex"] == ""


def test_a_doctor_on_leave_cancels_the_booking_with_the_reason_and_an_apology(clinic):
    bs, extras, db, m = clinic
    doctor = db.query(m.Doctor).filter_by(name="Dr. K. Bhattacharya").one()
    date = _next_weekday(0)
    appt = m.Appointment(
        confirmation_id="KCD-TEST-LEAVE", doctor_id=doctor.id, date=date, time_slot="18:10", patient_name="Ravi", phone="9800000001",
        created_at=datetime.datetime.now(), status="confirmed",
    )
    db.add(appt)
    db.commit()
    out = extras.register_leave(db, doctor, date, date, return_date=None, queue_sms=bs.queue_sms)
    assert out["success"] and [c["confirmation_id"] for c in out["cancelled"]] == ["KCD-TEST-LEAVE"]
    db.refresh(appt)
    assert appt.status == "cancelled" and appt.cancel_reason == "doctor_leave" and appt.cancellation_charge_inr == 0
    sent = db.query(m.SmsOutbox).filter_by(related_confirmation_id="KCD-TEST-LEAVE").all()
    assert any("sorry" in s.message.lower() for s in sent) and any("দুঃখিত" in s.message for s in sent)
    told = bs.clinic_cancellations(db, phone="9800000001")
    assert told and told[0]["reason"] == "doctor_leave" and told[0]["confirmation_id"] == "KCD-TEST-LEAVE"
    assert bs.clinic_cancellations(db, phone="9111111111") == []


def test_the_payment_link_says_forty_rupees_within_fifteen_minutes(clinic):
    bs, extras, db, m = clinic
    doctor = db.query(m.Doctor).filter_by(name="Dr. K. Bhattacharya").one()
    db.add(
        m.Appointment(
            confirmation_id="KCD-TEST-PAY", doctor_id=doctor.id, date=_next_weekday(2), time_slot="18:10", patient_name="Ravi", phone="9800000002",
            created_at=datetime.datetime.now(), status="confirmed",
        )
    )
    db.commit()
    out = bs.send_payment_link(db, "KCD-TEST-PAY")
    assert out["amount_inr"] == 40 and out["expires_in_minutes"] == 15
    msg = db.query(m.SmsOutbox).filter_by(related_confirmation_id="KCD-TEST-PAY", template_key="payment_link").one().message
    assert "Rs 40" in msg and "15 minutes" in msg and "expires automatically" in msg and "15 মিনিটের" in msg
