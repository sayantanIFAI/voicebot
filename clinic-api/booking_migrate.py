"""In-place migration for Epic E26's booking-lifecycle columns.

Same discipline as seed.py's add_i18n_columns(): ALTER TABLE for any
column an existing `appointments` row is missing, run BEFORE the first
ORM query on that table (SQLAlchemy selects every mapped column, so an
old database fails "no such column" on even a plain count()). Brand NEW
tables (SlotLock, Patient, PatientProxy, TestBooking, SmsOutbox,
DraftBooking, DepartmentRoute) need no migration at all --
Base.metadata.create_all(engine), already called at startup, creates a
missing table but never touches an existing one, so it is safe to call
every boot alongside this.
"""
from __future__ import annotations

from db import SessionLocal, engine
from sqlalchemy import inspect, text

APPOINTMENT_BOOKING_COLUMNS: dict[str, str] = {
    "patient_id": "INTEGER",
    "caller_phone": "VARCHAR",
    "status": "VARCHAR NOT NULL DEFAULT 'confirmed'",
    "cancelled_at": "DATETIME",
    "cancellation_charge_inr": "INTEGER",
    "rescheduled_from_id": "INTEGER",
    "booking_group_id": "VARCHAR",
}

DOCTOR_BOOKING_COLUMNS: dict[str, str] = {
    # Needed to STATE a cancellation charge (KCD-372) instead of inventing
    # one -- 50% of this, within the charging window, per
    # booking_service.CANCELLATION_CHARGE_FRACTION. Fictional, same
    # convention as every other price in seed.py.
    "consultation_fee_inr": "INTEGER NOT NULL DEFAULT 500",
}


def add_appointment_booking_columns() -> list[str]:
    added = []
    insp = inspect(engine)
    have = {c["name"] for c in insp.get_columns("appointments")}
    with engine.begin() as conn:
        for col, decl in APPOINTMENT_BOOKING_COLUMNS.items():
            if col not in have:
                conn.execute(text(f"ALTER TABLE appointments ADD COLUMN {col} {decl}"))
                added.append(f"appointments.{col}")
    return added


def add_doctor_booking_columns() -> list[str]:
    added = []
    insp = inspect(engine)
    have = {c["name"] for c in insp.get_columns("doctors")}
    with engine.begin() as conn:
        for col, decl in DOCTOR_BOOKING_COLUMNS.items():
            if col not in have:
                conn.execute(text(f"ALTER TABLE doctors ADD COLUMN {col} {decl}"))
                added.append(f"doctors.{col}")
    return added


# Department -> fictional consultation fee, clinically plausible variation
# by specialty, same spirit as seed.py's LAB_TESTS pricing.
DEPARTMENT_FEE_INR: dict[str, int] = {
    "General Medicine": 500, "Cardiology": 900, "Gynaecology & Obstetrics": 700,
    "Orthopaedics": 700, "ENT": 500, "Dermatology": 600, "Paediatrics": 500,
    "Diabetology & Endocrinology": 800,
}


def backfill_doctor_fees() -> int:
    """Fills consultation_fee_inr only where it is still at the raw ALTER
    TABLE default (500) *and* the department has a more specific fee --
    never overwrites a fee a clinician has since edited by hand."""
    from models import Department, Doctor

    db = SessionLocal()
    filled = 0
    try:
        depts = {d.id: d.name for d in db.query(Department).all()}
        for doc in db.query(Doctor).filter(Doctor.consultation_fee_inr == 500).all():
            fee = DEPARTMENT_FEE_INR.get(depts.get(doc.department_id))
            if fee and fee != 500:
                doc.consultation_fee_inr = fee
                filled += 1
        db.commit()
    finally:
        db.close()
    return filled


# (department name, [Bengali keyword cues], [Hindi keyword cues], [English keyword cues])
# Clinician-approved, fictional, same convention as seed.py's FAQ_ENTRIES --
# administrative routing only, never a diagnosis (see
# reply_templates.department_route_reply's framing).
DEPARTMENT_ROUTE_KEYWORDS: list[tuple[str, list[str], list[str], list[str]]] = [
    ("Cardiology",
     ["বুকে ব্যথা", "বুক ধড়ফড়", "হার্টের সমস্যা", "হৃদযন্ত্র"],
     ["सीने में दर्द", "दिल की धड़कन", "दिल की बीमारी"],
     ["chest pain", "heart palpitation", "heart problem"]),
    ("Orthopaedics",
     ["হাড়ে ব্যথা", "গাঁটে ব্যথা", "কোমরে ব্যথা", "পিঠে ব্যথা"],
     ["हड्डी में दर्द", "जोड़ों में दर्द", "कमर दर्द", "पीठ दर्द"],
     ["bone pain", "joint pain", "back pain", "knee pain"]),
    ("ENT",
     ["কানে ব্যথা", "গলা ব্যথা", "নাক বন্ধ", "শুনতে অসুবিধা"],
     ["कान में दर्द", "गले में दर्द", "नाक बंद"],
     ["ear pain", "throat pain", "sore throat", "blocked nose"]),
    ("Dermatology",
     ["চামড়ায় সমস্যা", "ত্বকের সমস্যা", "চুলকানি", "র‍্যাশ"],
     ["त्वचा की समस्या", "खुजली", "चकत्ते"],
     ["skin problem", "itching", "rash"]),
    ("Paediatrics",
     ["বাচ্চার জ্বর", "শিশুর সমস্যা", "বাচ্চার সর্দি"],
     ["बच्चे को बुखार", "बच्चे की समस्या"],
     ["child fever", "baby problem", "infant"]),
    ("Gynaecology & Obstetrics",
     ["প্রেগন্যান্সি", "গর্ভাবস্থা", "মহিলাদের সমস্যা"],
     ["गर्भावस्था", "महिलाओं की समस्या"],
     ["pregnancy", "women's problem", "gynaecology"]),
    ("Diabetology & Endocrinology",
     ["সুগারের সমস্যা", "ডায়াবেটিস", "থাইরয়েড সমস্যা"],
     ["शुगर की समस्या", "डायबिटीज़", "थायराइड"],
     ["diabetes", "sugar problem", "thyroid problem"]),
    ("General Medicine",
     ["জ্বর", "সর্দি কাশি", "দুর্বলতা", "পেট খারাপ"],
     ["बुखार", "सर्दी खांसी", "कमज़ोरी", "पेट खराब"],
     ["fever", "cold and cough", "weakness", "stomach upset"]),
]


def seed_department_routes() -> int:
    """Idempotent: only inserts a department that has no route row yet, so
    a clinician's later edit to an existing row is never overwritten by a
    re-run -- same rule seed.backfill_i18n already applies to translated
    content."""
    from models import Department, DepartmentRoute

    db = SessionLocal()
    added = 0
    try:
        existing = {r.department_id for r in db.query(DepartmentRoute).all()}
        by_name = {d.name: d for d in db.query(Department).all()}
        for dept_name, kw_bn, kw_hi, kw_en in DEPARTMENT_ROUTE_KEYWORDS:
            dept = by_name.get(dept_name)
            if not dept or dept.id in existing:
                continue
            db.add(DepartmentRoute(
                department_id=dept.id,
                keywords_bn="|".join(kw_bn), keywords_hi="|".join(kw_hi), keywords_en="|".join(kw_en),
            ))
            added += 1
        db.commit()
    finally:
        db.close()
    return added


def migrate_booking_schema() -> dict:
    """One call, run at startup before any ORM query -- see main.py's
    startup handler. Safe to call every boot."""
    added_columns = add_appointment_booking_columns() + add_doctor_booking_columns()
    routes_added = seed_department_routes()
    fees_filled = backfill_doctor_fees()
    return {"columns_added": added_columns, "department_routes_added": routes_added,
            "doctor_fees_filled": fees_filled}
