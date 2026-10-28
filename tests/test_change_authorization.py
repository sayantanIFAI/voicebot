"""P0 (review 2026-10-28): cancelling or moving a booking needs the confirmation id AND a phone number that belongs to the booking.

    python -m pytest tests/test_change_authorization.py -q
"""

import datetime
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app  # noqa: E402


def _book(phone):
    import booking_service as bs
    from db import SessionLocal
    from models import Doctor, DoctorSchedule

    db = SessionLocal()
    try:
        today = datetime.date.today()
        for sched in db.query(DoctorSchedule).all():
            doctor = db.get(Doctor, sched.doctor_id)
            for i in range(1, 4):  # from tomorrow: late in the evening every slot of today has passed
                day = today + datetime.timedelta(days=i)
                if day.weekday() != sched.weekday:
                    continue
                slots = bs.available_slots(db, doctor.id, day.isoformat())
                if not slots:
                    continue
                held = bs.hold_slot(db, doctor.id, day.isoformat(), slots[0])
                done = bs.confirm_booking(db, held["hold_token"], doctor.id, day.isoformat(), slots[0], "Pt One", phone, phone, 30)
                assert done["success"]
                return done["confirmation_id"]
        raise AssertionError("no bookable slot in the next three days")
    finally:
        db.close()


def test_the_owner_number_may_cancel_and_a_stranger_may_not():
    with clinic_app(sample_patients=False) as (_app, client):
        cid = _book("9811100001")
        # no number at all
        assert client.post("/api/v1/bookings/cancel", json={"confirmation_id": cid, "confirm_charge": True}, headers={"Idempotency-Key": "a1"}).json() == {"success": False, "reason": "phone_required"}
        # somebody else's number
        r = client.post("/api/v1/bookings/cancel", json={"confirmation_id": cid, "confirm_charge": True, "caller_phone": "9822200002"}, headers={"Idempotency-Key": "a2"}).json()
        assert r == {"success": False, "reason": "not_authorized"}
        # the booking is untouched, and the refusal is in the audit log
        from db import SessionLocal
        from models import Appointment, AuditLog

        db = SessionLocal()
        try:
            assert db.query(Appointment).filter_by(confirmation_id=cid).first().status == "confirmed"
            assert db.query(AuditLog).filter_by(action="booking.change_denied").count() == 1
        finally:
            db.close()
        # the owner's number, written differently (+91, spaces), is accepted
        ok = client.post("/api/v1/bookings/cancel", json={"confirmation_id": cid, "confirm_charge": True, "caller_phone": "+91 98111 00001"}, headers={"Idempotency-Key": "a3"}).json()
        assert ok["success"] is True


def test_a_stranger_cannot_move_a_booking_either():
    with clinic_app(sample_patients=False) as (_app, client):
        cid = _book("9811100003")
        far = (datetime.date.today() + datetime.timedelta(days=1)).isoformat()
        r = client.post("/api/v1/bookings/reschedule", json={"confirmation_id": cid, "new_date": far, "new_time_slot": "10:00", "caller_phone": "9833300004"}, headers={"Idempotency-Key": "b1"}).json()
        assert r == {"success": False, "reason": "not_authorized"}


def test_repeated_wrong_numbers_lock_the_booking():
    with clinic_app(sample_patients=False) as (_app, client):
        cid = _book("9811100005")
        reasons = []
        for i in range(7):
            reasons.append(
                client.post("/api/v1/bookings/cancel", json={"confirmation_id": cid, "confirm_charge": True, "caller_phone": f"98777000{i:02d}"}, headers={"Idempotency-Key": f"c{i}"}).json()["reason"]
            )
        assert reasons[:5] == ["not_authorized"] * 5 and reasons[5:] == ["too_many_attempts"] * 2
        # even the right number is held back while the lock lasts (it blocks changes only; the booking is untouched)
        r = client.post("/api/v1/bookings/cancel", json={"confirmation_id": cid, "confirm_charge": True, "caller_phone": "9811100005"}, headers={"Idempotency-Key": "c9"}).json()
        assert r["reason"] == "too_many_attempts"


def test_an_unknown_id_is_still_just_not_found():
    with clinic_app(sample_patients=False) as (_app, client):
        r = client.post("/api/v1/bookings/cancel", json={"confirmation_id": "KCD-NOPE", "caller_phone": "9811100009"}, headers={"Idempotency-Key": "d1"}).json()
        assert r == {"success": False, "reason": "not_found"}
