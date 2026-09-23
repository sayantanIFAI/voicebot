"""Unit tests for clinic-api/booking_service.py's core lifecycle:
confirm, cancel, reschedule, multi-test booking, proxy authorisation,
conflict detection, and the SMS placeholder -- covers KCD-371, 372, 364,
365, 374, 377, 378 at the data-layer level. Conversation-turn behaviour
(what the caller hears) is covered separately in tests/test_booking_flow.py
and the i18n reply-template tests.

    python -m pytest tests/test_booking_service.py -v
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
    for mod in ("main", "db", "models", "seed", "booking_service", "booking_migrate", "i18n_content"):
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
    d = datetime.date.today()
    while d.weekday() != target_weekday:
        d += datetime.timedelta(days=1)
    return d.isoformat()


def _doctor(db, m, weekday: int = 0):
    """A doctor who actually sits on `weekday`, so booking calls don't
    depend on which SHIFT_TEMPLATES rotation seed.py happened to assign."""
    row = (db.query(m.DoctorSchedule).filter_by(weekday=weekday).first())
    return db.get(m.Doctor, row.doctor_id)


def test_confirm_books_and_creates_a_patient_and_a_self_proxy(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        assert hold["success"]

        result = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot,
                                     "Ravi Das", "9800000001", "9800000001")
        assert result["success"] and result["confirmation_id"].startswith("KCD-")

        patient = db.query(m.Patient).filter_by(name="Ravi Das", phone="9800000001").one()
        proxy = db.query(m.PatientProxy).filter_by(patient_id=patient.id).one()
        assert proxy.verified_by == "self"

        sms = db.query(m.SmsOutbox).filter_by(related_confirmation_id=result["confirmation_id"]).one()
        assert sms.status == "queued"          # never claims "sent"
    finally:
        db.close()


def test_confirm_with_a_different_caller_phone_records_a_proxy(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        result = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot,
                                     "Anita Sen", "9800000002", caller_phone="9800000099",
                                     patient_age=62, relationship_label="son")
        assert result["success"]
        patient = db.query(m.Patient).filter_by(name="Anita Sen", phone="9800000002").one()
        proxy = db.query(m.PatientProxy).filter_by(patient_id=patient.id, caller_phone="9800000099").one()
        assert proxy.relationship_label == "son"
        assert proxy.verified_by == "relationship_stated"

        # KCD-364's stricter disclosure rule: relationship alone is not
        # enough to read the record back; the age must also match.
        assert bs.authorize_disclosure(db, patient, "9800000099") is False
        assert bs.authorize_disclosure(db, patient, "9800000099", confirmed_age=62) is True
    finally:
        db.close()


def test_expired_hold_cannot_be_confirmed(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        lock = db.get(m.SlotLock, (doc.id, date, slot))
        lock.hold_expires_at = datetime.datetime.now() - datetime.timedelta(seconds=1)
        db.commit()

        result = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "X", "1", "1")
        assert result == {"success": False, "reason": "hold_expired"}
        # and the slot is free again for someone else
        assert slot in bs.available_slots(db, doc.id, date)
    finally:
        db.close()


def test_cancel_outside_the_charging_window_is_free(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        # Force a date well outside the 24h window regardless of "today".
        far = (datetime.date.today() + datetime.timedelta(days=10))
        while far.weekday() != (doc.schedule[0].weekday if doc.schedule else 0):
            far += datetime.timedelta(days=1)
        date = far.isoformat()
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "X", "111", "111")

        result = bs.cancel_appointment(db, booked["confirmation_id"])
        assert result == {"success": True, "confirmation_id": booked["confirmation_id"], "charge_inr": 0}
        # the slot must be bookable again
        assert slot in bs.available_slots(db, doc.id, date)
    finally:
        db.close()


def test_cancel_inside_the_charging_window_requires_explicit_confirmation(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        # hold_slot/confirm_booking do not themselves validate against
        # DoctorSchedule (that happens one layer up, when a slot is
        # offered) -- so a slot just over an hour from now, well inside
        # the 24h charging window, is deterministic regardless of which
        # weekday this doctor sits or what time "now" happens to be.
        soon = datetime.datetime.now() + datetime.timedelta(hours=1)
        date, slot = soon.date().isoformat(), soon.strftime("%H:%M")
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "X", "111", "111")

        first = bs.cancel_appointment(db, booked["confirmation_id"])
        assert first["success"] is False and first["reason"] == "charge_confirmation_required"
        assert first["charge_inr"] == round(doc.consultation_fee_inr * 0.5)

        second = bs.cancel_appointment(db, booked["confirmation_id"], confirm_charge=True)
        assert second["success"] is True and second["charge_inr"] == first["charge_inr"]
    finally:
        db.close()


def test_reschedule_moves_the_appointment_and_frees_the_old_slot(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slots = bs.available_slots(db, doc.id, date)
        old_slot, new_slot = slots[0], slots[1]
        hold = bs.hold_slot(db, doc.id, date, old_slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, old_slot, "X", "111", "111")

        result = bs.reschedule_appointment(db, booked["confirmation_id"], date, new_slot)
        assert result["success"] and result["time_slot"] == new_slot
        assert old_slot in bs.available_slots(db, doc.id, date)
        assert new_slot not in bs.available_slots(db, doc.id, date)

        old_appt = db.query(m.Appointment).filter_by(confirmation_id=booked["confirmation_id"]).one()
        assert old_appt.status == "rescheduled"
    finally:
        db.close()


def test_reschedule_onto_a_taken_slot_leaves_the_original_intact(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slots = bs.available_slots(db, doc.id, date)
        mine, contested = slots[0], slots[1]
        hold = bs.hold_slot(db, doc.id, date, mine)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, mine, "X", "111", "111")
        # someone else takes the slot we're about to try to move onto
        other_hold = bs.hold_slot(db, doc.id, date, contested)
        bs.confirm_booking(db, other_hold["hold_token"], doc.id, date, contested, "Y", "222", "222")

        result = bs.reschedule_appointment(db, booked["confirmation_id"], date, contested)
        assert result["success"] is False and result["reason"] == "slot_taken"

        still = db.query(m.Appointment).filter_by(confirmation_id=booked["confirmation_id"]).one()
        assert still.status == "confirmed" and still.time_slot == mine
        assert mine not in bs.available_slots(db, doc.id, date)  # original booking still holds it
    finally:
        db.close()


def test_earliest_available_finds_the_soonest_free_slot(clinic_modules):
    # KCD-360: caller asks for the earliest available appointment.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        result = bs.earliest_available(db, doc.id, datetime.date.today())
        assert result is not None
        assert result["date"] and result["time_slot"]
        # what it found must actually be free right now
        assert result["time_slot"] in bs.available_slots(db, doc.id, result["date"])

        # once that exact slot is taken, the search must skip past it
        hold = bs.hold_slot(db, doc.id, result["date"], result["time_slot"])
        bs.confirm_booking(db, hold["hold_token"], doc.id, result["date"], result["time_slot"],
                            "Earliest Taker", "666", "666")
        second = bs.earliest_available(db, doc.id, datetime.date.today())
        assert second is not None
        assert not (second["date"] == result["date"] and second["time_slot"] == result["time_slot"])
    finally:
        db.close()


def test_earliest_available_returns_none_when_nothing_is_free_in_the_horizon(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        # A doctor who never sits (no DoctorSchedule rows at all, simulated
        # by asking about a horizon of zero days) has nothing to offer --
        # must say so plainly, never guess a date.
        doc = _doctor(db, m)
        result = bs.earliest_available(db, doc.id, datetime.date.today(), horizon_days=0)
        assert result is None
    finally:
        db.close()


def test_lookup_bookings_finds_by_phone_and_by_confirmation_id(clinic_modules):
    # KCD-373: caller asks what they have booked.
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot,
                                     "Lookup Patient", "777", "777")

        by_phone = bs.lookup_bookings(db, phone="777")
        assert len(by_phone) == 1 and by_phone[0]["confirmation_id"] == booked["confirmation_id"]

        by_id = bs.lookup_bookings(db, confirmation_id=booked["confirmation_id"])
        assert len(by_id) == 1 and by_id[0]["date"] == date

        assert bs.lookup_bookings(db, phone="no-such-number") == []

        # cancelled bookings must not be offered as "what you have booked"
        bs.cancel_appointment(db, booked["confirmation_id"], confirm_charge=True)
        assert bs.lookup_bookings(db, phone="777") == []
    finally:
        db.close()


def test_conflict_detection_finds_the_same_patient_double_booked(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "Same Patient", "333", "333")

        conflict = bs.find_conflict(db, "333", date, slot)
        assert conflict is not None and conflict["date"] == date
        assert bs.find_conflict(db, "not-a-real-number", date, slot) is None
    finally:
        db.close()


def test_multi_test_booking_and_add_test(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        tests = db.query(m.LabTest).limit(2).all()
        date = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
        result = bs.book_tests(db, [t.id for t in tests], date, "Multi Test Patient", "444", "444")
        assert result["success"] and len(result["test_names"]) == 2
        assert result["total_rate_inr"] == sum(t.rate_inr for t in tests)

        third = db.query(m.LabTest).offset(2).first()
        added = bs.add_test_to_booking(db, result["confirmation_id"], third.name)
        assert added["success"]

        group_rows = db.query(m.TestBooking).filter_by(
            booking_group_id=db.query(m.TestBooking).filter_by(
                confirmation_id=result["confirmation_id"]).first().booking_group_id).all()
        assert len(group_rows) == 3

        dup = bs.add_test_to_booking(db, result["confirmation_id"], third.name)
        assert dup == {"success": False, "reason": "already_booked"}
    finally:
        db.close()


def test_resend_confirmation_is_rate_limited(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slot = bs.available_slots(db, doc.id, date)[0]
        hold = bs.hold_slot(db, doc.id, date, slot)
        booked = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, "X", "5550001", "5550001")

        first = bs.resend_confirmation(db, booked["confirmation_id"])
        assert first["success"] and first["sent_to_last4"] == "0001"
        second = bs.resend_confirmation(db, booked["confirmation_id"])
        assert second == {"success": False, "reason": "rate_limited"}
    finally:
        db.close()


def test_department_routing_matches_and_flags_ambiguity(clinic_modules):
    bs, db_mod, _m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        result = bs.route_department(db, "amar buke khub betha hocche", "en")
        assert result == {"matched": False}   # no English keyword hit -- transliteration is out of scope here

        result = bs.route_department(db, "I have chest pain", "en")
        assert result["matched"] and result["department_name"] == "Cardiology"

        result = bs.route_department(db, "বুকে ব্যথা", "bn")
        assert result["matched"] and result["department_name"] == "Cardiology"
    finally:
        db.close()


def test_draft_booking_round_trip_and_expiry(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        bs.save_draft(db, "9990001", "call-1", '{"doctor_name": "Sen"}')
        found = bs.find_draft(db, "9990001")
        assert found and found["slots_json"] == '{"doctor_name": "Sen"}'

        row = db.query(m.DraftBooking).filter_by(caller_phone="9990001").one()
        row.expires_at = datetime.datetime.now() - datetime.timedelta(seconds=1)
        db.commit()
        assert bs.find_draft(db, "9990001") is None

        bs.save_draft(db, "9990002", "call-2", "{}")
        bs.clear_draft(db, "9990002")
        assert bs.find_draft(db, "9990002") is None
    finally:
        db.close()
