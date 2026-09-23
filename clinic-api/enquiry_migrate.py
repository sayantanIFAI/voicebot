"""In-place migration for Epic E27's new LabTest columns.

Same discipline as booking_migrate.py: every brand-new TABLE (LabReport,
DoctorLeave, Package, WalkInPolicy, BillingLineItem, HomeCollectionCoverage,
InsurancePolicy, CallOutcome, CallbackRequest, DepartmentHours, ...) needs
no migration code at all -- Base.metadata.create_all(engine), already
called at startup, creates a missing table and never touches an existing
one. Only LabTest gained new COLUMNS on an existing table, which SQLite
cannot add without an explicit ALTER TABLE, run before any ORM query
against LabTest -- same "SQLAlchemy selects every mapped column" reason
booking_migrate.py's own columns exist.
"""
from __future__ import annotations

from sqlalchemy import inspect, text

from db import engine, SessionLocal

LAB_TEST_ENQUIRY_COLUMNS: dict[str, str] = {
    "fasting_hours": "INTEGER",
    "water_allowed_while_fasting": "BOOLEAN NOT NULL DEFAULT 1",
    "home_collection_eligible": "BOOLEAN NOT NULL DEFAULT 1",
    "prescription_required": "BOOLEAN NOT NULL DEFAULT 0",
    "prescription_note_bn": "VARCHAR NOT NULL DEFAULT ''",
    "prescription_note_hi": "VARCHAR NOT NULL DEFAULT ''",
    "prescription_note_en": "VARCHAR NOT NULL DEFAULT ''",
}


def add_enquiry_columns() -> list[str]:
    added = []
    insp = inspect(engine)
    have = {c["name"] for c in insp.get_columns("lab_tests")}
    with engine.begin() as conn:
        for col, decl in LAB_TEST_ENQUIRY_COLUMNS.items():
            if col not in have:
                conn.execute(text(f"ALTER TABLE lab_tests ADD COLUMN {col} {decl}"))
                added.append(f"lab_tests.{col}")
    return added


# Per-test structured prep facts, keyed by the exact LAB_TESTS name from
# seed.py. Only tests with a genuine requirement get an override; every
# other test keeps the ALTER TABLE defaults (no fasting, home-collection
# eligible, no prescription needed) -- same "only list the exceptions"
# convention seed.py's own PREP_OVERRIDES already uses.
FASTING_HOURS_OVERRIDES: dict[str, tuple[int, bool]] = {   # name -> (hours, water_allowed)
    "Blood Sugar Fasting": (8, True),
    "Lipid Profile": (11, True),
    "Kidney Function Test (KFT)": (7, True),
    "Liver Function Test (LFT)": (8, True),
}

# Samples that physically cannot survive a courier trip, or need
# same-visit equipment -- ineligible for home collection regardless of
# the general default.
HOME_COLLECTION_INELIGIBLE = {
    "ECG", "2D Echocardiography", "TMT (Treadmill Test)",
    "USG Whole Abdomen", "USG Pregnancy Profile", "Chest X-Ray (PA view)", "Pap Smear",
}

# Tests a walk-in caller cannot simply request without a doctor's note.
PRESCRIPTION_REQUIRED = {
    "HIV Test (ELISA)", "HBsAg", "HCV",
}
_PRESCRIPTION_NOTE = (
    "এই টেস্টের জন্য ডাক্তারের প্রেসক্রিপশন লাগবে। প্রেসক্রিপশনের ছবি হোয়াটসঅ্যাপে পাঠাতে পারেন।",
    "इस टेस्ट के लिए डॉक्टर का पर्चा चाहिए। पर्चे की फोटो व्हाट्सऐप पर भेज सकते हैं।",
    "A doctor's prescription is required for this test. You can send a photo of it over WhatsApp.",
)


def backfill_enquiry_facts(db=None) -> dict:
    """Idempotent, never overwrites a value a clinician has since edited
    by hand -- same rule seed.backfill_i18n() already applies."""
    from models import LabTest

    added = add_enquiry_columns()
    own = db is None
    db = db or SessionLocal()
    filled = 0
    try:
        for t in db.query(LabTest).all():
            if t.fasting_hours is None and t.name in FASTING_HOURS_OVERRIDES:
                hours, water_ok = FASTING_HOURS_OVERRIDES[t.name]
                t.fasting_hours, t.water_allowed_while_fasting = hours, water_ok
                filled += 1
            if t.name in HOME_COLLECTION_INELIGIBLE and t.home_collection_eligible:
                t.home_collection_eligible = False
                filled += 1
            if t.name in PRESCRIPTION_REQUIRED and not t.prescription_required:
                t.prescription_required = True
                t.prescription_note_bn, t.prescription_note_hi, t.prescription_note_en = _PRESCRIPTION_NOTE
                filled += 1
        db.commit()
    finally:
        if own:
            db.close()
    return {"columns_added": added, "values_filled": filled}
