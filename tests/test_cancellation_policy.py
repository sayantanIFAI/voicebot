"""KCD-488: "Cancellation applies versioned policy, never improvisation."

Window rules and refund eligibility now come from clinic-api/models.py's
CancellationPolicy table (versioned, dated) instead of booking_service's
old flat CANCELLATION_FREE_WINDOW_HOURS/CANCELLATION_CHARGE_FRACTION
constants, and the version actually applied is recorded on the
Appointment row it cancelled -- see booking_service.active_cancellation_
policy and Appointment.cancellation_policy_version's own docstrings.

    python -m pytest tests/test_cancellation_policy.py -v
"""
import datetime
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


@pytest.fixture()
def clinic_modules():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)
    if CLINIC_API_DIR not in sys.path:
        sys.path.insert(0, CLINIC_API_DIR)
    for mod in ("main", "db", "models", "seed", "booking_service", "booking_migrate",
                "enquiry_migrate", "i18n_content"):
        sys.modules.pop(mod, None)

    import seed as seed_mod
    seed_mod.seed()
    import booking_migrate
    booking_migrate.migrate_booking_schema()
    booking_migrate.finish_booking_schema_setup()
    import booking_service as bs
    import db as db_mod
    import models as m

    yield bs, db_mod, m

    try:
        os.remove(db_path)
    except OSError:
        pass


def _next_weekday(target_weekday: int) -> str:
    d = datetime.date.today() + datetime.timedelta(days=1)
    while d.weekday() != target_weekday:
        d += datetime.timedelta(days=1)
    return d.isoformat()


def _doctor(db, m, weekday: int = 0):
    row = db.query(m.DoctorSchedule).filter_by(weekday=weekday).first()
    return db.get(m.Doctor, row.doctor_id)


def test_a_default_policy_is_seeded_matching_the_old_hardcoded_behaviour(clinic_modules):
    # finish_booking_schema_setup() (called by the fixture, same as every
    # real boot) must leave exactly the values the old flat constants
    # used to hardcode, so shipping this story changes nothing about
    # today's actual charges.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        policies = db.query(m.CancellationPolicy).all()
        assert len(policies) == 1
        p = policies[0]
        assert p.version == 1
        assert p.free_window_hours == 24
        assert p.charge_percent == 50
        assert p.refund_eligible is True
    finally:
        db.close()


def test_seeding_is_idempotent_and_never_overwrites_an_edited_row(clinic_modules):
    bs, db_mod, m = clinic_modules
    import booking_migrate
    db = db_mod.SessionLocal()
    try:
        p = db.query(m.CancellationPolicy).filter_by(version=1).one()
        p.free_window_hours = 48   # a clinician's own edit
        db.commit()
    finally:
        db.close()

    added = booking_migrate.seed_default_cancellation_policy()
    assert added is False

    db = db_mod.SessionLocal()
    try:
        p = db.query(m.CancellationPolicy).filter_by(version=1).one()
        assert p.free_window_hours == 48, "a re-run must not clobber a hand-edited policy"
    finally:
        db.close()


def test_cancellation_records_which_policy_version_applied(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        far = datetime.date.today() + datetime.timedelta(days=10)
        while far.weekday() != (doc.schedule[0].weekday if doc.schedule else 0):
            far += datetime.timedelta(days=1)
        date = far.isoformat()
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "X", "111", "111")

        result = bs.cancel_appointment(db, booked["confirmation_id"])
        assert result["success"] is True

        appt = db.query(m.Appointment).filter_by(confirmation_id=booked["confirmation_id"]).one()
        assert appt.cancellation_policy_version == 1
    finally:
        db.close()


def test_a_future_dated_policy_version_does_not_apply_before_its_effective_date(clinic_modules):
    # A clinician schedules a rule CHANGE (e.g. moving to a 72h free
    # window) for a future date without touching any row already applied
    # to a past cancellation -- KCD-488's whole point.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        future_effective = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()
        db.add(m.CancellationPolicy(
            version=2, effective_from=future_effective,
            free_window_hours=72, charge_percent=25, refund_eligible=True,
        ))
        db.commit()

        # Today, version 1 (the only one already in force) still applies.
        active_today = bs.active_cancellation_policy(db)
        assert active_today.version == 1

        # As of a date on/after version 2's effective_from, version 2 wins.
        as_of_future = datetime.date.fromisoformat(future_effective) + datetime.timedelta(days=1)
        active_later = bs.active_cancellation_policy(db, as_of=as_of_future)
        assert active_later.version == 2
    finally:
        db.close()


def test_a_non_refund_eligible_policy_charges_the_full_consultation_fee(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        # Replace version 1 with a non-refundable policy. The appointment
        # below is 10 days out -- far beyond any free window -- so a
        # non-zero charge proves refund_eligible=False overrides the
        # window rather than merely applying inside it.
        db.query(m.CancellationPolicy).delete()
        db.add(m.CancellationPolicy(
            version=3, effective_from="2020-01-01",
            free_window_hours=24, charge_percent=50, refund_eligible=False,
        ))
        db.commit()

        doc = _doctor(db, m)
        far = datetime.date.today() + datetime.timedelta(days=10)
        while far.weekday() != (doc.schedule[0].weekday if doc.schedule else 0):
            far += datetime.timedelta(days=1)
        date = far.isoformat()
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "X", "111", "111")

        appt = db.query(m.Appointment).filter_by(confirmation_id=booked["confirmation_id"]).one()
        charge, policy = bs.cancellation_charge(db, appt)
        doctor = db.get(m.Doctor, doc.id)
        assert policy.version == 3
        assert charge == (doctor.consultation_fee_inr or 0)
    finally:
        db.close()
