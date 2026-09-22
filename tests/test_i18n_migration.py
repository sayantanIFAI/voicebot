"""A database created BEFORE the Hindi/English columns must still boot.

Reproduces the real failure: clinic-api's startup ran `count()` on lab_tests
first, and SQLAlchemy selects every mapped column, so the pod's existing
clinic.db raised "no such column: lab_tests.aliases_hi" and the service never
came up. Uses the old schema built by raw SQL, not the current models, so the
test cannot pass by accident."""
import importlib
import os
import sys

import pytest

CLINIC_API = os.path.join(os.path.dirname(__file__), "..", "clinic-api")


@pytest.fixture()
def old_db(tmp_path, monkeypatch):
    path = tmp_path / "old.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{path}")
    monkeypatch.syspath_prepend(CLINIC_API)
    for mod in ("db", "models", "seed", "i18n_content", "booking_service", "booking_migrate"):
        sys.modules.pop(mod, None)
    import sqlite3
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL UNIQUE);
        CREATE TABLE doctors (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL, qualifications VARCHAR NOT NULL,
            aliases_bn VARCHAR NOT NULL DEFAULT '', department_id INTEGER NOT NULL);
        CREATE TABLE lab_tests (id INTEGER PRIMARY KEY, name VARCHAR NOT NULL UNIQUE,
            aliases_bn VARCHAR NOT NULL DEFAULT '', rate_inr INTEGER NOT NULL, sample_type VARCHAR NOT NULL,
            report_time_hours INTEGER NOT NULL, fasting_required BOOLEAN NOT NULL DEFAULT 0,
            prep_instructions_bn VARCHAR NOT NULL DEFAULT '');
        CREATE TABLE faqs (id INTEGER PRIMARY KEY, topic VARCHAR NOT NULL UNIQUE,
            keywords_bn VARCHAR NOT NULL DEFAULT '', answer_bn VARCHAR NOT NULL);
        INSERT INTO departments VALUES (1, 'General Medicine');
        INSERT INTO doctors VALUES (1, 'Dr. A. Sen', 'MBBS', 'সেন', 1);
        INSERT INTO lab_tests VALUES (1, 'Uric Acid', 'ইউরিক অ্যাসিড', 250, 'Blood', 12, 0, 'বাংলা');
        INSERT INTO lab_tests VALUES (2, 'Blood Sugar Fasting', 'সুগার ফাস্টিং', 120, 'Blood', 4, 1, 'বাংলা');
        INSERT INTO faqs VALUES (1, 'hours', 'সময়', 'বাংলা উত্তর');
    """)
    con.commit()
    con.close()
    yield path
    for mod in ("db", "models", "seed", "i18n_content", "booking_service", "booking_migrate"):
        sys.modules.pop(mod, None)


def test_old_database_gains_columns_and_translations_without_losing_rows(old_db):
    seed = importlib.import_module("seed")
    models = importlib.import_module("models")
    db_mod = importlib.import_module("db")

    added = seed.add_i18n_columns()
    assert set(added) == {"doctors.aliases_hi", "lab_tests.aliases_hi", "lab_tests.prep_instructions_hi",
                          "lab_tests.prep_instructions_en", "faqs.answer_hi", "faqs.answer_en"}

    # Epic E26 added its own mapped columns to Doctor/Appointment (see
    # clinic-api/booking_migrate.py). SQLAlchemy selects every mapped
    # column on any query, so this real production startup order --
    # create_all() for brand-new tables, then EVERY ALTER-TABLE migration,
    # before the first ORM query -- is required here too, not just in
    # clinic-api/main.py's own startup handler.
    booking_migrate = importlib.import_module("booking_migrate")
    models.Base.metadata.create_all(db_mod.engine)
    booking_migrate.migrate_booking_schema()

    db = db_mod.SessionLocal()
    try:
        assert db.query(models.LabTest).count() == 2      # the query that used to crash at boot
        result = seed.backfill_i18n(db)
        assert result["columns_added"] == []               # already added; idempotent
        uric = db.query(models.LabTest).filter_by(name="Uric Acid").one()
        assert uric.aliases_hi == "यूरिक एसिड"
        assert uric.prep_instructions_bn == "বাংলা"        # existing Bengali untouched
        assert "No special preparation" in uric.prep_instructions_en
        sugar = db.query(models.LabTest).filter_by(name="Blood Sugar Fasting").one()
        assert "आठ घंटे" in sugar.prep_instructions_hi
        assert db.query(models.Doctor).one().aliases_hi == "सेन"
        faq = db.query(models.FAQ).one()
        assert "8 AM to 8 PM" in faq.answer_en and "सुबह आठ बजे" in faq.answer_hi
        assert faq.answer_bn == "বাংলা উত্তর"
    finally:
        db.close()


def test_backfill_never_overwrites_a_hand_edited_value(old_db):
    seed = importlib.import_module("seed")
    models = importlib.import_module("models")
    db_mod = importlib.import_module("db")
    booking_migrate = importlib.import_module("booking_migrate")
    seed.add_i18n_columns()
    # backfill_i18n() below also queries Doctor, which needs Epic E26's
    # columns to exist too -- see the sibling test's comment.
    models.Base.metadata.create_all(db_mod.engine)
    booking_migrate.migrate_booking_schema()
    db = db_mod.SessionLocal()
    try:
        t = db.query(models.LabTest).filter_by(name="Uric Acid").one()
        t.prep_instructions_en = "Edited by the clinic."
        db.commit()
        seed.backfill_i18n(db)
        db.refresh(t)
        assert t.prep_instructions_en == "Edited by the clinic."
    finally:
        db.close()
