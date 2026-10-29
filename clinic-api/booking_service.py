"""Booking-lifecycle business logic for Epic E26 (Booking, Rescheduling
and Cancellation).

Kept separate from clinic-api/main.py's route handlers so the
atomicity-critical parts -- hold_slot, confirm_booking,
reschedule_appointment -- can be unit tested directly, including under
real concurrency, without a FastAPI TestClient in the loop.

Every function takes an open `db: Session` and returns a plain dict --
same contract as clinic-api/main.py's existing endpoints -- and never
raises for a normal not-found/taken/policy-blocked outcome. Only
IntegrityError from a genuine slot race is caught here, in hold_slot;
anything else propagates, matching agent/tools_client.py's ToolCallError
convention on the caller side.

No fact in here is ever composed by an LLM: prices, charges and slot
availability come from this module's own deterministic policy constants
and the database, the same discipline agent/reply_templates.py already
applies to test prices and doctor schedules.
"""

from __future__ import annotations

import datetime

import clinic_clock  # the clinic's own (Kolkata) clock, not the server's
import os
import re
import uuid

import lay_departments  # sourav, 2026-10-24: lay doctor phrases -> department, and the paediatric age limit (lay_departments.py)
import booking_log  # sourav, 2026-10-24: a readable record of every confirmed booking (booking_log.py); never allowed to break a booking
import notifications
import payments
from disclosure import with_notice as _with_disclosure_notice
from models import (
    Appointment,
    AuditLog,
    CancellationPolicy,
    Department,
    DepartmentRoute,
    Doctor,
    DoctorSchedule,
    DraftBooking,
    LabTest,
    Patient,
    PatientProxy,
    SlotLock,
    SmsOutbox,
    TestBooking,
)
from phrase_proximity import phrase_matches
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

HOLD_TTL_SECONDS = 900  # sourav, 2026-10-24: was 90, then 300 -- both still expired live on a slow multi-turn call (name/
# age/address/slot confirmation over several turns; a tester pausing between turns easily spends minutes). Two separate
# calls observed live expired their hold before confirm and left the booking unresolved (callback_requests
# "unresolved:booking_not_finished"). 900s (15 min) is generous for a real conversation pace while still bounded, so a
# dropped call does not hold a slot forever.
DRAFT_BOOKING_TTL_MINUTES = 30
SMS_RESEND_MIN_INTERVAL_S = 60
# A repeat of the same message to the same number for the same booking within this many seconds is one message.
SMS_DEDUP_WINDOW_S = 60


def _now() -> datetime.datetime:
    return clinic_clock.now()


def _confirmation_id(date: str) -> str:
    # 8 hex chars (32 bits, ~4.3 billion per date) rather than 4 (16 bits,
    # 65536): Epic E26 added /api/v1/bookings/lookup?confirmation_id=...,
    # an unauthenticated endpoint (this service has no auth layer at all
    # yet -- Epic E14) that returns patient name, doctor and time for a
    # match. 4 hex chars made that a realistically enumerable guess
    # against one date's bookings; 8 does not, without removing the
    # underlying "add real authentication" gap this alone cannot close.
    return f"KCD-{date.replace('-', '')}-{uuid.uuid4().hex[:8].upper()}"


SLOT_STEP_MIN = int(os.environ.get("SLOT_STEP_MIN", "10"))  # owner, 2026-10-17: one booking per 10-minute slot (env override: the older test-suite pins a 15-minute grid)


def _generate_slots(start: str, end: str, step_min: int = SLOT_STEP_MIN) -> list[str]:
    t = datetime.datetime.strptime(start, "%H:%M")
    end_t = datetime.datetime.strptime(end, "%H:%M")
    out = []
    while t < end_t:
        out.append(t.strftime("%H:%M"))
        t += datetime.timedelta(minutes=step_min)
    return out


def _slot_minutes(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _alias(aliases: str | None) -> str | None:
    return aliases.split("|")[0] if aliases else None


# ============================================================= holds


def _release_expired_hold(db: Session, doctor_id: int, date: str, time_slot: str) -> None:
    # CodeRabbit-flagged, real race: sqlite3's legacy transaction mode
    # (db.py does not override it) does not BEGIN on a plain SELECT, so
    # two concurrent callers can both read the same expired row before
    # either writes. A read-then-delete-by-primary-key would let caller B
    # delete caller A's REPLACEMENT hold (inserted between B's read and
    # B's delete) instead of the expired one B actually read -- both
    # hold_slot calls report success, and A's later confirm_booking then
    # fails with hold_expired on a hold that should still be live. The
    # fix is a single DELETE whose WHERE clause re-checks status/expiry
    # at delete time, so a hold someone else already replaced is simply
    # not matched and stays in place; the duplicate insert that follows
    # in hold_slot then correctly reports slot_taken.
    db.query(SlotLock).filter(
        SlotLock.doctor_id == doctor_id,
        SlotLock.date == date,
        SlotLock.time_slot == time_slot,
        SlotLock.status == "held",
        SlotLock.hold_expires_at < _now(),
    ).delete()
    db.flush()


def hold_slot(db: Session, doctor_id: int, date: str, time_slot: str) -> dict:
    """Atomically claim a slot. SlotLock's own primary key IS
    (doctor_id, date, time_slot); SQLite serializes writers and enforces
    that key, so of any number of simultaneous callers, exactly one INSERT
    here can ever succeed -- see models.SlotLock's docstring. Everyone
    else gets success=False, reason="slot_taken", not a crash."""
    _release_expired_hold(db, doctor_id, date, time_slot)
    token = uuid.uuid4().hex
    expires = _now() + datetime.timedelta(seconds=HOLD_TTL_SECONDS)
    db.add(
        SlotLock(
            doctor_id=doctor_id,
            date=date,
            time_slot=time_slot,
            status="held",
            hold_token=token,
            hold_expires_at=expires,
        )
    )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return {"success": False, "reason": "slot_taken"}
    return {"success": True, "hold_token": token, "expires_at": expires.isoformat()}


def _taken_slots(db: Session, doctor_id: int, date: str) -> set[str]:
    now = _now()
    return {
        r.time_slot
        for r in db.query(SlotLock).filter_by(doctor_id=doctor_id, date=date).all()
        if r.status == "confirmed" or (r.status == "held" and r.hold_expires_at and r.hold_expires_at >= now)
    }


def available_slots(db: Session, doctor_id: int, date: str) -> list[str]:
    target = datetime.date.fromisoformat(date)
    sched = db.query(DoctorSchedule).filter_by(doctor_id=doctor_id, weekday=target.weekday()).first()
    if not sched:
        return []
    all_slots = _generate_slots(sched.start_time, sched.end_time)
    taken = _taken_slots(db, doctor_id, date)
    free = [s for s in all_slots if s not in taken]
    now = _now()
    if target == now.date():
        # CodeRabbit-flagged, real bug: a same-day slot earlier than the
        # current time was offered and bookable -- hold_booking's own
        # date check is date-level only ("target < today"), never
        # time-of-day, so "today at 10am" stayed offerable at 3pm.
        now_minutes = now.hour * 60 + now.minute
        free = [s for s in free if _slot_minutes(s) > now_minutes]
    # KCD-lead-14: a per-weekday cap on patients taken, separate from the time-grid's own slot count. NULL (every
    # row until a clinician sets a real number) changes nothing -- same grid-only list as before this column existed.
    if sched.max_bookings_per_day is not None:
        remaining_capacity = sched.max_bookings_per_day - len(taken)
        free = free[: max(remaining_capacity, 0)]
    return free


def earliest_available(db: Session, doctor_id: int, from_date: datetime.date, horizon_days: int = 14) -> dict | None:
    for i in range(horizon_days):
        d = (from_date + datetime.timedelta(days=i)).isoformat()
        free = available_slots(db, doctor_id, d)
        if free:
            return {"date": d, "time_slot": free[0], "alternatives": free[1:3]}
    return None


def nearest_alternatives(db: Session, doctor_id: int, date: str, time_slot: str) -> list[dict]:
    """KCD-361's exact shape: the nearest other time the same day, plus
    the same time on the nearest other day -- never a slot that is not
    actually free at the moment of asking."""
    out = []
    same_day = [s for s in available_slots(db, doctor_id, date) if s != time_slot]
    if same_day:
        nearest = min(same_day, key=lambda s: abs(_slot_minutes(s) - _slot_minutes(time_slot)))
        out.append({"date": date, "time_slot": nearest})
    start = datetime.date.fromisoformat(date)
    for i in range(1, 8):
        d = (start + datetime.timedelta(days=i)).isoformat()
        free = available_slots(db, doctor_id, d)
        if time_slot in free:
            out.append({"date": d, "time_slot": time_slot})
            break
        if free:
            out.append({"date": d, "time_slot": free[0]})
            break
    return out


# ========================================================== patients

# Mirrors agent/booking_flow.py's effective_phone() sentinel by VALUE,
# not by import -- clinic-api is a separate service this repo does not
# own (see this module's own docstring / CLAUDE.md's service-boundary
# discipline), the same reason agent/phonetic_match.py and
# clinic-api/phonetic_match.py are deliberately duplicated rather than
# shared.
NOT_PROVIDED_PHONE = "not_provided"


def find_or_create_patient(db: Session, name: str, phone: str, age: int | None = None, address: str = "", sex: str = "") -> Patient:
    """CodeRabbit-flagged, real bug: the "not_provided" sentinel (a
    caller who declined to give any phone number) used to be looked up
    like a real phone value -- two DIFFERENT patients who share a name
    and BOTH decline a number would silently become the SAME Patient
    row, cross-contaminating whatever booking history/proxy
    authorisation is attached to it. A declined phone always creates a
    fresh Patient instead of searching for one."""
    if phone == NOT_PROVIDED_PHONE:
        p = Patient(name=name, phone=phone, age=age, address=address or "", sex=sex or "", created_at=_now())
        db.add(p)
        db.flush()
        return p
    p = db.query(Patient).filter_by(name=name, phone=phone).first()
    if p:
        if age and not p.age:
            p.age = age
        if address and not p.address:
            p.address = address
        if sex and not p.sex:
            p.sex = sex
        return p
    p = Patient(name=name, phone=phone, age=age, address=address or "", sex=sex or "", created_at=_now())
    db.add(p)
    db.flush()
    return p


def set_patient_senior(db: Session, phone: str, senior: bool, caller_phone: str | None = None) -> int:
    """KCD-084: remember (or clear) that callers using this phone number are
    served in senior mode. Applies to every Patient row with that phone --
    the number is what a returning caller can be recognised by. Returns how
    many rows changed. A declined number is never a key (it would flag
    every stranger who withheld one).

    Each row is changed only if `caller_phone` (defaulting to `phone`, as in
    lookup_bookings) is authorised for that patient by authorize_disclosure:
    self, or a verified proxy. Until real CallerID reaches this API the
    default means the number the caller SAID, so this is exactly as strong
    as the booking lookups -- it closes the door for a bridge that supplies a
    real caller_phone, and it is stated here so it is not mistaken for more."""
    if not phone or phone == NOT_PROVIDED_PHONE:
        return 0
    asking_as = caller_phone or phone
    rows = [p for p in db.query(Patient).filter(Patient.phone == phone).all() if authorize_disclosure(db, p, asking_as)]
    changed = 0
    for p in rows:
        if bool(p.senior_mode) != bool(senior):
            p.senior_mode = bool(senior)
            changed += 1
    if changed:
        db.commit()
    return changed


def get_patient_senior(db: Session, phone: str, caller_phone: str | None = None) -> bool:
    if not phone or phone == NOT_PROVIDED_PHONE:
        return False
    asking_as = caller_phone or phone
    return any(
        authorize_disclosure(db, p, asking_as)
        for p in db.query(Patient).filter(Patient.phone == phone, Patient.senior_mode.is_(True)).all()
    )


def record_proxy(db: Session, patient: Patient, caller_phone: str, relationship_label: str, verified_by: str) -> None:
    existing = db.query(PatientProxy).filter_by(patient_id=patient.id, caller_phone=caller_phone).first()
    if existing:
        return
    db.add(
        PatientProxy(
            patient_id=patient.id,
            caller_phone=caller_phone,
            relationship_label=relationship_label,
            verified_by=verified_by,
            created_at=_now(),
        )
    )


def authorize_disclosure(db: Session, patient: Patient, caller_phone: str, confirmed_age: int | None = None) -> bool:
    """Whether an EXISTING record may be read out to this caller (KCD-373,
    KCD-378) or modified by them. Booking a brand-new appointment never
    needs this check -- only looking up or changing one that was not made
    from this same phone."""
    if caller_phone == patient.phone:
        return True
    proxy = db.query(PatientProxy).filter_by(patient_id=patient.id, caller_phone=caller_phone).first()
    if not proxy:
        return False
    if proxy.verified_by == "dob_confirmed":
        return True
    if proxy.verified_by == "relationship_stated" and confirmed_age is not None and patient.age == confirmed_age:
        proxy.verified_by = "dob_confirmed"
        db.commit()
        return True
    return False


# ======================================================== appointments


def _verify_appointment_persisted(db: Session, appointment_id: int, expected: dict) -> bool:
    """KCD-486: "the write is verified in the system of record with the
    expected values before it is verbalised" -- re-reads the row AFTER
    commit and checks it, rather than trusting that db.commit() raising
    nothing means the caller's specific values actually landed. Against
    this prototype's single-process SQLite this will realistically never
    fail (a successful commit() IS durable here) -- but that is exactly
    why this check belongs at this layer rather than being skipped:
    swapping the real hospital/lab system in behind this same function
    (this module's own docstring's stated migration path) is precisely
    the case where "the commit call didn't raise" and "the system of
    record now reflects it" can come apart, e.g. an async-replicated
    write or a queue-backed integration. Uses a FRESH query rather than
    the in-memory `appt` object, which would just echo back the same
    Python values that were assigned before commit and verify nothing.
    """
    db.expire_all()
    row = db.get(Appointment, appointment_id)
    if row is None:
        return False
    return all(getattr(row, field) == value for field, value in expected.items())


def next_serial(db: Session, doctor_id: int, date: str) -> int:
    """sourav, 2026-10-24: the serial number the NEXT confirmed booking of this doctor on this date gets: one above the highest already issued. Why (owner): for now, as a simulation,
    every booking generates its serial in the appointments table at booking time, and "if I book today what will my serial be?" is answered with exactly this number.
    Cancelled bookings do not count (their number is free again only if they were the last; a gap is harmless)."""
    top = (
        db.query(func.max(Appointment.token_number))
        .filter(Appointment.doctor_id == doctor_id, Appointment.date == date, Appointment.status == "confirmed")
        .scalar()
    )
    return (top or 0) + 1


def confirm_booking(
    db: Session,
    hold_token: str,
    doctor_id: int,
    date: str,
    time_slot: str,
    patient_name: str,
    phone: str,
    caller_phone: str,
    patient_age: int | None = None,
    relationship_label: str = "self",
    patient_address: str = "",
    patient_sex: str = "",
) -> dict:
    lock = db.get(SlotLock, (doctor_id, date, time_slot))
    # IDEMPOTENT on the hold token: a retried request (a dropped connection, a repeated "yes") for a
    # slot this SAME hold already confirmed returns the booking that exists -- the same confirmation
    # number, no second appointment and no false "hold expired" for a booking that did succeed.
    if lock and lock.status == "confirmed" and lock.hold_token == hold_token and lock.appointment_id:
        appt = db.get(Appointment, lock.appointment_id)
        if appt is not None and appt.status == "confirmed" and appt.phone == phone:
            doctor = db.get(Doctor, doctor_id)
            return {
                "success": True,
                "confirmation_id": appt.confirmation_id,
                "doctor_name": doctor.full_name or doctor.name,
                "doctor_name_bn": doctor.full_name_bn or _alias(doctor.aliases_bn),
                "doctor_name_hi": doctor.full_name_hi or _alias(doctor.aliases_hi),
                "date": date,
                "time_slot": time_slot,
                "replayed": True,
            }
    if not lock or lock.status != "held" or lock.hold_token != hold_token:
        return {"success": False, "reason": "hold_expired"}
    if lock.hold_expires_at and lock.hold_expires_at < _now():
        db.delete(lock)
        db.commit()
        return {"success": False, "reason": "hold_expired"}

    doctor = db.get(Doctor, doctor_id)
    patient = find_or_create_patient(db, patient_name, phone, patient_age, patient_address, patient_sex)
    verified_by = "self" if caller_phone == phone else "relationship_stated"
    record_proxy(db, patient, caller_phone, relationship_label, verified_by)

    confirmation_id = _confirmation_id(date)
    appt = Appointment(
        confirmation_id=confirmation_id,
        doctor_id=doctor_id,
        date=date,
        time_slot=time_slot,
        patient_name=patient_name,
        phone=phone,
        created_at=_now(),
        patient_id=patient.id,
        caller_phone=caller_phone,
        status="confirmed",
        token_number=next_serial(db, doctor_id, date),  # sourav, 2026-10-24: simulated serial generated with the booking (paid_at, set by mark_paid, is what releases it to the caller)
    )
    db.add(appt)
    db.flush()
    lock.status = "confirmed"
    lock.hold_expires_at = None
    lock.appointment_id = appt.id
    db.commit()

    # KCD-486: a confirmation number is spoken only once the write is
    # verified -- an unverified write produces a distinct outcome (never
    # a false "success", and never the ordinary tool_failure apology
    # either) so main.py can put the caller on hold and escalate instead
    # of reading out a reference that may not be real.
    if not _verify_appointment_persisted(
        db,
        appt.id,
        {
            "confirmation_id": confirmation_id,
            "doctor_id": doctor_id,
            "date": date,
            "time_slot": time_slot,
            "patient_name": patient_name,
            "phone": phone,
            "status": "confirmed",
        },
    ):
        return {"success": False, "reason": "write_unverified", "confirmation_id": confirmation_id}

    queue_sms(
        db,
        phone,
        "booking_confirmed",
        f"Your appointment with {doctor.name} on {date} at {time_slot} is confirmed. Ref: {confirmation_id}.",
        confirmation_id,
    )

    booking_log.append(db, confirmation_id)  # sourav, 2026-10-24: one line in booking_log.csv (best effort, swallows its own errors)
    return {
        "success": True,
        "confirmation_id": confirmation_id,
        "doctor_name": doctor.full_name or doctor.name,
        "doctor_name_bn": doctor.full_name_bn or _alias(doctor.aliases_bn),
        "doctor_name_hi": doctor.full_name_hi or _alias(doctor.aliases_hi),
        "date": date,
        "time_slot": time_slot,
    }


def find_conflict(db: Session, patient_phone: str, date: str, time_slot: str) -> dict | None:
    """KCD-378: the SAME patient already has a confirmed appointment at
    this date and time. Checked against the hospital system (this table),
    never a local cache, so it is current at the moment of the new
    booking."""
    existing = (
        db.query(Appointment).filter_by(phone=patient_phone, date=date, time_slot=time_slot, status="confirmed").first()
    )
    if not existing:
        return None
    doctor = db.get(Doctor, existing.doctor_id)
    return {
        "confirmation_id": existing.confirmation_id,
        "doctor_name": (doctor.full_name or doctor.name) if doctor else None,
        "date": existing.date,
        "time_slot": existing.time_slot,
    }


def active_cancellation_policy(db: Session, as_of: datetime.date | None = None) -> CancellationPolicy | None:
    """KCD-488: the policy row IN FORCE at `as_of` (default: today) --
    versioned configuration with effective dates, never a flat constant a
    developer would have to redeploy to change. Whichever row has the
    LATEST `effective_from` that is not after `as_of` wins, so adding a
    new version with a future effective_from schedules a change without
    touching any row already applied to a past cancellation."""
    as_of_iso = (as_of or _now().date()).isoformat()
    return (
        db.query(CancellationPolicy)
        .filter(CancellationPolicy.effective_from <= as_of_iso)
        .order_by(CancellationPolicy.effective_from.desc(), CancellationPolicy.version.desc())
        .first()
    )


def cancellation_charge(db: Session, appt: Appointment) -> tuple[int, CancellationPolicy | None]:
    """-> (charge_inr, policy_used). policy_used is None only if the
    cancellation_policies table has no row at all -- seed_default_
    cancellation_policy() backfills one on every boot, so this is an
    infrastructure gap to fix, not a normal outcome; treated as "no
    charge" rather than raising, so a missing policy row fails open on
    price (never silently overcharges) but is loud in the return shape
    for whoever calls this to notice."""
    # The rule in force when the caller CANCELS (today), not on the day of the
    # appointment: a rule change scheduled for a date before the appointment
    # must not be applied to a cancellation made before it takes effect, and
    # cancellation_policy_version must record the rule that actually applied.
    policy = active_cancellation_policy(db)
    if not policy:
        return 0, None
    doctor = db.get(Doctor, appt.doctor_id)
    fee = doctor.consultation_fee_inr or 0
    # refund_eligible=False marks a NON-REFUNDABLE policy: the full fee is
    # charged however much notice was given, so free_window_hours and
    # charge_percent (which only describe the refundable case) are
    # deliberately ignored -- checked FIRST, otherwise a long-notice
    # cancellation would fall through the free window and this flag
    # would never affect anything.
    if not policy.refund_eligible:
        return fee, policy
    appt_dt = datetime.datetime.combine(
        datetime.date.fromisoformat(appt.date),
        datetime.datetime.strptime(appt.time_slot, "%H:%M").time(),
    )
    if appt_dt - _now() >= datetime.timedelta(hours=policy.free_window_hours):
        return 0, policy
    return round(fee * policy.charge_percent / 100), policy


# ---------------------------------------------------------------------------------------------------------------------------------------------
# WHO MAY CHANGE A BOOKING (review 2026-10-28, P0). A confirmation id alone used to be enough to cancel or move any booking: 8 hex characters, said aloud, texted and printed, is a
# reference, not a credential. Now a change needs the id AND a phone number that belongs to the booking (the number it was made from, its contact number, the patient's number, or a
# proxy already verified for that patient). There is no caller ID in this stack, so the number is still something the caller SAYS: it removes the one-factor path (a guessed, overheard
# or leaked id), not a determined impersonator who also knows the phone number. A second factor (date of birth, or a code sent to the number on file once SMS is real) is the next step.
# Every refusal is written to audit_log, and repeated refusals lock the booking and the number for a while (the lock only blocks changes; the booking itself is untouched).
# ---------------------------------------------------------------------------------------------------------------------------------------------
CHANGE_REQUIRES_PHONE = os.environ.get("CHANGE_REQUIRES_PHONE", "1") != "0"
CHANGE_MAX_FAILS = int(os.environ.get("CHANGE_MAX_FAILS", "5"))
CHANGE_LOCK_MINUTES = int(os.environ.get("CHANGE_LOCK_MINUTES", "15"))


def _phone_key(value: str | None) -> str:
    """The last ten digits of a phone number, '' when there are fewer (so 'not_provided' never matches anything)."""
    digits = re.sub(r"\D", "", value or "")
    return digits[-10:] if len(digits) >= 10 else ""


def _recent_change_denials(db: Session, needle: str) -> int:
    since = _now() - datetime.timedelta(minutes=CHANGE_LOCK_MINUTES)
    return (
        db.query(AuditLog)
        .filter(AuditLog.action == "booking.change_denied", AuditLog.at >= since, AuditLog.detail_json.like(f"%{needle}%"))
        .count()
    )


def check_change_authorized(db: Session, confirmation_id: str, caller_phone: str | None, call_id: str | None = None) -> dict | None:
    """None when this caller may change the booking; otherwise the refusal to return ({"success": False, "reason": ...}). An unknown id is left to the caller's own not_found."""
    if not CHANGE_REQUIRES_PHONE:
        return None
    appt = db.query(Appointment).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    if appt is None:
        return None
    who = _phone_key(caller_phone)
    if not who:
        return {"success": False, "reason": "phone_required"}
    if _recent_change_denials(db, f'"confirmation_id": "{confirmation_id}"') >= CHANGE_MAX_FAILS or _recent_change_denials(db, f'"caller": "{who}"') >= CHANGE_MAX_FAILS:
        _audit_denied(db, appt, confirmation_id, who, call_id, "locked")
        return {"success": False, "reason": "too_many_attempts"}
    patient = db.get(Patient, appt.patient_id) if appt.patient_id else None
    own = {_phone_key(appt.phone), _phone_key(getattr(appt, "caller_phone", None)), _phone_key(patient.phone if patient else None)} - {""}
    if who in own:
        return None
    if patient is not None and authorize_disclosure(db, patient, caller_phone or ""):
        return None
    _audit_denied(db, appt, confirmation_id, who, call_id, "denied")
    return {"success": False, "reason": "not_authorized"}


def _audit_denied(db: Session, appt, confirmation_id: str, who: str, call_id: str | None, outcome: str) -> None:
    import json

    db.add(
        AuditLog(
            at=_now(),
            call_id=call_id,
            patient_id=appt.patient_id,
            actor="agent",
            action="booking.change_denied",
            outcome=outcome,
            detail_json=json.dumps({"confirmation_id": confirmation_id, "caller": who}),
        )
    )
    db.commit()


def cancel_appointment(db: Session, confirmation_id: str, confirm_charge: bool = False) -> dict:
    appt = db.query(Appointment).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    if not appt:
        return {"success": False, "reason": "not_found"}
    charge, policy = cancellation_charge(db, appt)
    if charge > 0 and not confirm_charge:
        # Stated BEFORE it is applied (KCD-372) -- the caller must say yes
        # a second time, with the amount already in their ear, before
        # anything is written.
        return {"success": False, "reason": "charge_confirmation_required", "charge_inr": charge}

    lock = db.get(SlotLock, (appt.doctor_id, appt.date, appt.time_slot))
    if lock:
        db.delete(lock)
    appt.status = "cancelled"
    appt.cancelled_at = _now()
    appt.cancellation_charge_inr = charge
    # KCD-488: which rule produced this charge, recorded on the row it
    # applied to -- so a later audit sees the actual rule in force at
    # cancellation time, not just today's policy (which may since differ).
    appt.cancellation_policy_version = policy.version if policy else None
    db.commit()

    queue_sms(
        db,
        appt.phone,
        "booking_cancelled",
        f"Your appointment ({confirmation_id}) has been cancelled."
        + (f" A charge of Rs.{charge} applies." if charge else ""),
        confirmation_id,
    )
    return {"success": True, "confirmation_id": confirmation_id, "charge_inr": charge}


def reschedule_appointment(db: Session, confirmation_id: str, new_date: str, new_time_slot: str) -> dict:
    """Holds the NEW slot first; only once that succeeds is the old one
    touched at all. A failed hold returns immediately with the original
    appointment completely untouched -- KCD-371's "a failed swap leaves
    the original intact"."""
    appt = db.query(Appointment).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    if not appt:
        return {"success": False, "reason": "not_found"}

    hold = hold_slot(db, appt.doctor_id, new_date, new_time_slot)
    if not hold["success"]:
        return {
            "success": False,
            "reason": "slot_taken",
            "alternative_slots": available_slots(db, appt.doctor_id, new_date)[:3],
        }

    doctor = db.get(Doctor, appt.doctor_id)
    new_confirmation_id = _confirmation_id(new_date)
    new_appt = Appointment(
        confirmation_id=new_confirmation_id,
        doctor_id=appt.doctor_id,
        date=new_date,
        time_slot=new_time_slot,
        patient_name=appt.patient_name,
        phone=appt.phone,
        created_at=_now(),
        patient_id=appt.patient_id,
        caller_phone=appt.caller_phone,
        status="confirmed",
        rescheduled_from_id=appt.id,
    )
    db.add(new_appt)
    db.flush()

    new_lock = db.get(SlotLock, (appt.doctor_id, new_date, new_time_slot))
    new_lock.status, new_lock.hold_expires_at, new_lock.appointment_id = "confirmed", None, new_appt.id

    old_lock = db.get(SlotLock, (appt.doctor_id, appt.date, appt.time_slot))
    if old_lock:
        db.delete(old_lock)
    appt.status = "rescheduled"
    db.commit()

    # KCD-486, same discipline as confirm_booking: verify before speaking
    # the new reference number.
    if not _verify_appointment_persisted(
        db,
        new_appt.id,
        {
            "confirmation_id": new_confirmation_id,
            "doctor_id": appt.doctor_id,
            "date": new_date,
            "time_slot": new_time_slot,
            "status": "confirmed",
        },
    ):
        return {"success": False, "reason": "write_unverified", "confirmation_id": new_confirmation_id}

    queue_sms(
        db,
        appt.phone,
        "booking_rescheduled",
        f"Your appointment with {doctor.name} has been moved to {new_date} at {new_time_slot}. "
        f"New ref: {new_confirmation_id}.",
        new_confirmation_id,
    )

    return {
        "success": True,
        "confirmation_id": new_confirmation_id,
        "doctor_name": doctor.full_name or doctor.name,
        "date": new_date,
        "time_slot": new_time_slot,
    }


def lookup_bookings(
    db: Session,
    phone: str | None = None,
    confirmation_id: str | None = None,
    name: str | None = None,
    upcoming_only: bool = True,
    caller_phone: str | None = None,
) -> list[dict]:
    """SECURITY (CWE-639/IDOR, CodeRabbit-flagged): every row this used to
    return on a bare phone/name search, unauthenticated -- anyone who
    spoke a phone number, guessed or overheard, could hear a stranger's
    doctor/date/time/name. authorize_disclosure() already existed for
    exactly this (its own docstring says so) but was never called
    anywhere. Fixed here, the one place every phone/confirmation_id/name
    lookup in the codebase funnels through.

    A caller who supplies the exact confirmation_id already holds the
    bearer token for that one row (unchanged from before -- 8 random hex
    chars is the existing access-control model for that path, per
    _confirmation_id's own comment) and skips the extra check. A caller
    searching by PHONE or NAME ALONE must additionally be authorized for
    EACH row: `caller_phone` (defaulting to `phone`, since every existing
    call site already treats the phone it searches by as the asking
    caller's own number) is checked against the row's patient via
    authorize_disclosure -- the same self/proxy/dob-confirmed tiers
    booking time already builds and records, now actually load-bearing.
    A name-only search with no phone at all now discloses nothing: a
    name alone was never a meaningful access-control check to begin
    with."""
    q = db.query(Appointment).filter(Appointment.status == "confirmed")
    if confirmation_id:
        q = q.filter(Appointment.confirmation_id == confirmation_id)
    if phone:
        q = q.filter((Appointment.phone == phone) | (Appointment.caller_phone == phone))
    if name:
        q = q.filter(Appointment.patient_name.ilike(f"%{name}%"))
    if upcoming_only:
        q = q.filter(Appointment.date >= _now().date().isoformat())
    rows = q.order_by(Appointment.date, Appointment.time_slot).all()

    require_auth = not confirmation_id
    asking_as = caller_phone or phone

    out = []
    for a in rows:
        if require_auth:
            patient = db.get(Patient, a.patient_id) if a.patient_id else None
            if not patient or not asking_as or not authorize_disclosure(db, patient, asking_as):
                continue
        doctor = db.get(Doctor, a.doctor_id)
        out.append(
            {
                "confirmation_id": a.confirmation_id,
                "doctor_name": (doctor.full_name or doctor.name) if doctor else None,
                "date": a.date,
                "time_slot": a.time_slot,
                "patient_name": a.patient_name,
            }
        )
    return out


# ============================================================ tests


def book_tests(
    db: Session,
    test_ids: list[int],
    date: str,
    patient_name: str,
    phone: str,
    caller_phone: str,
    patient_age: int | None = None,
    relationship_label: str = "self",
) -> dict:
    patient = find_or_create_patient(db, patient_name, phone, patient_age)
    verified_by = "self" if caller_phone == phone else "relationship_stated"
    record_proxy(db, patient, caller_phone, relationship_label, verified_by)

    group_id = uuid.uuid4().hex[:10]
    confirmation_id = _confirmation_id(date)
    booked, total, combined_prep = [], 0, []
    for tid in test_ids:
        t = db.get(LabTest, tid)
        if not t:
            continue
        db.add(
            TestBooking(
                booking_group_id=group_id,
                confirmation_id=confirmation_id,
                lab_test_id=t.id,
                patient_id=patient.id,
                patient_name=patient_name,
                date=date,
                phone=phone,
                caller_phone=caller_phone,
                status="confirmed",
                created_at=_now(),
            )
        )
        booked.append(t.name)
        total += t.rate_inr
        if t.prep_instructions_bn:
            combined_prep.append(t.prep_instructions_bn)
    if not booked:
        return {"success": False, "reason": "no_valid_tests"}
    db.commit()

    queue_sms(
        db,
        phone,
        "tests_confirmed",
        f"Your tests ({', '.join(booked)}) on {date} are confirmed. Total Rs.{total}. Ref: {confirmation_id}.",
        confirmation_id,
    )

    booking_log.append(db, confirmation_id)  # sourav, 2026-10-24: the lab booking's line in booking_log.csv
    return {
        "success": True,
        "confirmation_id": confirmation_id,
        "test_names": booked,
        "total_rate_inr": total,
        "date": date,
        "combined_prep": " ".join(combined_prep),
    }


def add_test_to_booking(db: Session, confirmation_id: str, test_name: str) -> dict:
    existing = db.query(TestBooking).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    if not existing:
        return {"success": False, "reason": "not_found"}
    test = db.query(LabTest).filter(func.lower(LabTest.name) == test_name.lower()).first()
    if not test:
        return {"success": False, "reason": "test_not_found"}
    already = {
        tb.lab_test_id
        for tb in db.query(TestBooking).filter_by(booking_group_id=existing.booking_group_id, status="confirmed").all()
    }
    if test.id in already:
        return {"success": False, "reason": "already_booked"}
    db.add(
        TestBooking(
            booking_group_id=existing.booking_group_id,
            confirmation_id=confirmation_id,
            lab_test_id=test.id,
            patient_id=existing.patient_id,
            patient_name=existing.patient_name,
            date=existing.date,
            phone=existing.phone,
            caller_phone=existing.caller_phone,
            status="confirmed",
            created_at=_now(),
        )
    )
    db.commit()
    return {"success": True, "confirmation_id": confirmation_id, "test_name": test.name, "date": existing.date}


# ========================================================= notifications


def send_payment_link(db: Session, confirmation_id: str) -> dict:
    """Queue a payment-link message for a CONFIRMED DOCTOR appointment. Only a doctor appointment is payable by
    phone (CLAUDE.md's truth boundary already keeps money out of the model's hands; this keeps it out of a lab test's
    hands too -- a lab test's confirmation_id belongs to TestBooking, not Appointment, so it is never found here and
    never gets a link. A lab test is paid and booked at the counter, never over the call.

    The link is an EXPLICIT PLACEHOLDER on an RFC 2606 reserved, never-resolvable domain -- the same "log exactly
    what WOULD be sent, never claim it was" discipline as queue_sms itself, which this calls. No payment gateway is
    integrated; wiring one means building the real link in a new clinic-api/payments.py and calling it from here."""
    appt = db.query(Appointment).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    if not appt:
        return {"success": False, "reason": "not_found"}
    if appt.phone == NOT_PROVIDED_PHONE:
        return {"success": False, "reason": "no_phone_on_file"}
    doctor = db.get(Doctor, appt.doctor_id)
    # Owner, 2026-10-18: Rs 40 holds the booking, payable within 15 minutes, after which the link expires by itself. The doctor's visit fee is paid at the clinic.
    fee = payments.BOOKING_FEE_INR
    minutes = payments.LINK_EXPIRY_MINUTES
    link = payments.create_payment_link(confirmation_id, fee, minutes)["url"]
    message = (
        f"{(doctor.full_name or doctor.name) if doctor else 'Doctor'} on {appt.date}. Pay Rs {fee} within {minutes} minutes to hold your booking, "
        f"otherwise this link expires automatically: {link} "
        f"-- আপনার বুকিং ধরে রাখতে {minutes} মিনিটের মধ্যে {fee} টাকা পেমেন্ট করুন, নাহলে লিংকটি নিজে থেকেই বাতিল হয়ে যাবে। Booking ID follows once paid."
    )
    # One message on each channel, each its own row in the outbox and each handed to the provider hook (agent says: SMS and WhatsApp).
    result = queue_sms(db, appt.phone, "payment_link", message, confirmation_id)
    wa = queue_sms(db, appt.phone, "payment_link_whatsapp", message, confirmation_id)
    for channel, res in (("sms", result), ("whatsapp", wa)):
        if res.get("queued") and not res.get("duplicate"):
            delivery = notifications.send(channel, appt.phone, message, confirmation_id)
            row = db.get(SmsOutbox, res["id"])
            if row is not None and delivery["status"] in ("sent", "failed"):
                row.status = delivery["status"]  # stays "queued" while the placeholder is in place
                db.commit()
    out = {"success": result["queued"], "amount_inr": fee, "expires_in_minutes": minutes, "payment_link": link, "confirmation_id": confirmation_id, "channels": ["sms", "whatsapp"]}
    if not result["queued"]:
        out["reason"] = result.get("reason", "unknown")
    return out


def send_booking_confirmation_whatsapp(db: Session, confirmation_id: str) -> dict:
    """KCD-lead-2: the warm/friendly WhatsApp confirmation message (the lead's chosen Template B), sent once a
    doctor appointment is CONFIRMED -- separate from send_payment_link's own message above (sent earlier in the
    same flow, about the payment hold, not the appointment itself). Same placeholder discipline: queue_sms stays
    status="queued" until a real provider is wired in (clinic-api/notifications.py); nothing here ever claims a
    message reached a caller's phone."""
    appt = db.query(Appointment).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    if not appt:
        return {"success": False, "reason": "not_found"}
    if appt.phone == NOT_PROVIDED_PHONE:
        return {"success": False, "reason": "no_phone_on_file"}
    doctor = db.get(Doctor, appt.doctor_id)
    doctor_en = (doctor.full_name or doctor.name) if doctor else "Doctor"
    doctor_bn = (doctor.full_name_bn or doctor.name) if doctor else "ডাক্তার"
    department = db.get(Department, doctor.department_id) if doctor else None
    dept_suffix = f" ({department.name})" if department else ""
    message = (
        f"Hi {appt.patient_name}, your appointment with {doctor_en}{dept_suffix} is booked for {appt.date}, {appt.time_slot}. "
        f"Please arrive 10 minutes early. Booking ID: {confirmation_id}. "
        f"-- হ্যালো {appt.patient_name}, {doctor_bn}{dept_suffix}-এর সাথে আপনার অ্যাপয়েন্টমেন্ট {appt.date}, {appt.time_slot}-এ বুক হয়েছে। "
        f"১০ মিনিট আগে আসবেন। বুকিং আইডি: {confirmation_id}।"
    )
    result = queue_sms(db, appt.phone, "booking_confirmation_whatsapp", message, confirmation_id)
    if result.get("queued") and not result.get("duplicate"):
        delivery = notifications.send("whatsapp", appt.phone, message, confirmation_id)
        row = db.get(SmsOutbox, result["id"])
        if row is not None and delivery["status"] in ("sent", "failed"):
            row.status = delivery["status"]  # stays "queued" while the placeholder is in place
            db.commit()
    out = {"success": result["queued"], "confirmation_id": confirmation_id, "channel": "whatsapp"}
    if not result["queued"]:
        out["reason"] = result.get("reason", "unknown")
    return out


def queue_sms(
    db: Session, to_phone: str, template_key: str, message: str, related_confirmation_id: str | None = None
) -> dict:
    """The open integration placeholder. Logs exactly what WOULD be sent,
    in the shape a real provider needs, and returns status="queued" --
    never "sent". Wiring a real SMS/WhatsApp provider means implementing
    send() in a new clinic-api/notifications.py and calling it from here;
    nothing else in this module needs to change, and nothing anywhere else
    claims a message reached a caller's phone until that exists.

    CodeRabbit-flagged: a caller who declined to give any phone number
    (NOT_PROVIDED_PHONE) has nowhere to send anything -- queuing a row
    addressed to the literal string "not_provided" is not merely useless,
    it is a fake record of an outbound message that was never actually
    addressable. Skipped, with a distinct, honest outcome."""
    if to_phone == NOT_PROVIDED_PHONE:
        return {"queued": False, "reason": "no_phone_on_file"}
    # KCD-353: a text channel states it is automated, every time (idempotent).
    message = _with_disclosure_notice(message)
    # The same notice to the same number for the same booking inside the window is a retry, not a second message.
    twin = (
        db.query(SmsOutbox)
        .filter_by(
            to_phone=to_phone,
            template_key=template_key,
            message=message,
            related_confirmation_id=related_confirmation_id,
        )
        .order_by(SmsOutbox.created_at.desc())
        .first()
    )
    if twin is not None and (_now() - twin.created_at).total_seconds() < SMS_DEDUP_WINDOW_S:
        return {"queued": True, "id": twin.id, "duplicate": True}
    row = SmsOutbox(
        to_phone=to_phone,
        template_key=template_key,
        message=message,
        status="queued",
        related_confirmation_id=related_confirmation_id,
        created_at=_now(),
    )
    db.add(row)
    db.commit()
    return {"queued": True, "id": row.id}


def resend_confirmation(db: Session, confirmation_id: str) -> dict:
    appt = db.query(Appointment).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    to_phone = message = None
    if appt:
        to_phone = appt.phone
        message = f"Your appointment ({confirmation_id}) is confirmed for {appt.date} at {appt.time_slot}."
    else:
        tb = db.query(TestBooking).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
        if tb:
            to_phone = tb.phone
            message = f"Your test booking ({confirmation_id}) is confirmed for {tb.date}."
    if not to_phone:
        return {"success": False, "reason": "not_found"}
    if to_phone == NOT_PROVIDED_PHONE:
        # CodeRabbit-flagged: to_phone[-4:] on the literal sentinel string
        # produced the nonsense "ided" (the last 4 characters of
        # "not_provided") as if it were a real phone's last 4 digits --
        # confusing at best, and could read as a real (wrong) number.
        return {"success": False, "reason": "no_phone_on_file"}

    last = (
        db.query(SmsOutbox)
        .filter_by(related_confirmation_id=confirmation_id, template_key="resend")
        .order_by(SmsOutbox.created_at.desc())
        .first()
    )
    if last and (_now() - last.created_at).total_seconds() < SMS_RESEND_MIN_INTERVAL_S:
        return {"success": False, "reason": "rate_limited"}

    queue_sms(db, to_phone, "resend", message, confirmation_id)
    return {"success": True, "sent_to_last4": to_phone[-4:]}


# ========================================================== department routing


MAX_SUGGESTED_DOCTORS = 3


def _doctor_spoken_name(d: Doctor, lang: str) -> str:
    """The short name in a script this caller's TTS voice can actually say. "Dr. S. Mukherjee" is Latin --
    the bn/hi vocoders drop Latin script outright (agent/pronunciation.py, agent/tts.py KCD-455), and a live
    call on 2026-10-12 showed exactly that: the reply text was correct but the doctor's name came out as
    silence, three separate "unspeakable spans blocked this reply" log lines (one per word, since clause_split
    also breaks on the periods in "Dr." and "S."). full_name_bn / full_name_hi are native-script, no "Dr."
    prefix to strip -- falling back to the Latin short name only if a localisation is somehow missing, which
    is still better than crashing even though it reproduces the silent-name bug for that one doctor."""
    if lang == "bn":
        return d.full_name_bn or d.name
    if lang == "hi":
        return d.full_name_hi or d.name
    return d.full_name or d.name  # the whole name in English too


def _department_doctor_names(
    db: Session, department_id: int, lang: str = "bn", limit: int = MAX_SUGGESTED_DOCTORS
) -> list[str]:
    """The real, currently-seeded doctors in a department, spoken-name form -- never a name the model made up
    (CLAUDE.md's truth boundary): this is a live query, the same table booking_service's own doctor search reads."""
    return [
        _doctor_spoken_name(d, lang)
        for d in db.query(Doctor).filter_by(department_id=department_id).order_by(Doctor.id).limit(limit).all()
    ]


# Short forms a caller uses for a department ("cardio", "ortho", "gynae", "derma", "pedia", "gen med", "হার্টের ডাক্তার") -> the department's own name.
# Owner, 2026-10-17: "who are the doctors in cardio" means Cardiology. An ASCII key must be a whole word; a Bengali/Hindi key is a substring.
_DEPT_ALIASES: dict[str, str] = {
    "cardio": "Cardiology", "cardiac": "Cardiology", "cardiologist": "Cardiology", "heart": "Cardiology",
    "কার্ডিও": "Cardiology", "হার্ট": "Cardiology", "হৃদরোগ": "Cardiology", "कार्डियो": "Cardiology", "हार्ट": "Cardiology", "दिल": "Cardiology",
    "ortho": "Orthopaedics", "orthopedic": "Orthopaedics", "orthopedics": "Orthopaedics", "orthopaedic": "Orthopaedics", "bone": "Orthopaedics",
    "অর্থো": "Orthopaedics", "হাড়ের": "Orthopaedics", "हड्डी": "Orthopaedics", "ऑर्थो": "Orthopaedics",
    "gynae": "Gynaecology & Obstetrics", "gyne": "Gynaecology & Obstetrics", "gynec": "Gynaecology & Obstetrics", "gynaec": "Gynaecology & Obstetrics",
    "gynecology": "Gynaecology & Obstetrics", "gynaecology": "Gynaecology & Obstetrics", "gyn": "Gynaecology & Obstetrics", "obs": "Gynaecology & Obstetrics",
    "obstetrics": "Gynaecology & Obstetrics", "গাইনি": "Gynaecology & Obstetrics", "स्त्री रोग": "Gynaecology & Obstetrics", "गायनी": "Gynaecology & Obstetrics",
    "derma": "Dermatology", "dermato": "Dermatology", "skin": "Dermatology", "চর্ম": "Dermatology", "ত্বকের": "Dermatology", "स्किन": "Dermatology", "त्वचा": "Dermatology",
    "pedia": "Paediatrics", "peds": "Paediatrics", "paeds": "Paediatrics", "pediatric": "Paediatrics", "pediatrics": "Paediatrics", "paediatric": "Paediatrics",
    "child": "Paediatrics", "children": "Paediatrics", "শিশু": "Paediatrics", "বাচ্চার": "Paediatrics", "बच्चों": "Paediatrics", "बाल रोग": "Paediatrics",
    "diabetes": "Diabetology & Endocrinology", "diabetology": "Diabetology & Endocrinology", "endo": "Diabetology & Endocrinology",
    "endocrine": "Diabetology & Endocrinology", "endocrinology": "Diabetology & Endocrinology", "ডায়াবেটিস": "Diabetology & Endocrinology", "मधुमेह": "Diabetology & Endocrinology",
    "dental": "Dentistry", "dentist": "Dentistry", "teeth": "Dentistry", "tooth": "Dentistry", "দাঁতের": "Dentistry", "दांत": "Dentistry",
    "gen med": "General Medicine", "general medicine": "General Medicine", "physician": "General Medicine", "medicine": "General Medicine",
    "ent": "ENT", "ear nose throat": "ENT", "নাক কান গলা": "ENT", "नाक कान गला": "ENT",
    # Owner, 2026-10-18: specialist titles callers actually say, and the departments added for routing
    "gynecologist": "Gynaecology & Obstetrics", "gynaecologist": "Gynaecology & Obstetrics", "gynaeco": "Gynaecology & Obstetrics", "obstetrician": "Gynaecology & Obstetrics",
    "গাইনোকোলজিস্ট": "Gynaecology & Obstetrics", "গাইনো": "Gynaecology & Obstetrics", "স্ত্রীরোগ": "Gynaecology & Obstetrics", "স্ত্রী রোগ": "Gynaecology & Obstetrics", "গায়নোকোলজিস্ট": "Gynaecology & Obstetrics",
    "गायनेकोलॉजिस्ट": "Gynaecology & Obstetrics", "गायनो": "Gynaecology & Obstetrics",
    "cardiologist": "Cardiology", "কার্ডিওলজিস্ট": "Cardiology", "हृदय रोग": "Cardiology",
    "general surgeon": "General Surgery", "general surgery": "General Surgery", "surgeon": "General Surgery", "surgery": "General Surgery",
    "জেনারেল সার্জন": "General Surgery", "জেনারেল সার্জারি": "General Surgery", "সার্জন": "General Surgery", "সার্জারি": "General Surgery", "সার্জেন": "General Surgery",
    "जनरल सर्जन": "General Surgery", "सर्जन": "General Surgery", "सर्जरी": "General Surgery",
    "neurologist": "Neurology", "neurology": "Neurology", "নিউরোলজিস্ট": "Neurology", "নিউরোলজি": "Neurology", "न्यूरोलॉजिस्ट": "Neurology",
    "neurosurgeon": "Neurosurgery", "neurosurgery": "Neurosurgery", "neuro surgeon": "Neurosurgery", "নিউরোসার্জন": "Neurosurgery", "নিউরো সার্জন": "Neurosurgery", "न्यूरोसर्जन": "Neurosurgery",
    "urologist": "Urology", "urology": "Urology", "ইউরোলজিস্ট": "Urology", "ইউরোলজি": "Urology", "यूरोलॉजिस्ट": "Urology",
    "orthopedician": "Orthopaedics", "paediatrician": "Paediatrics", "pediatrician": "Paediatrics", "dermatologist": "Dermatology", "endocrinologist": "Diabetology & Endocrinology", "diabetologist": "Diabetology & Endocrinology",
    # KCD-lead-A: plural / "-ics" / "-ology" forms and specialist titles a caller says. The ASCII lookahead above refuses "orthopaedic" inside
    # "orthopaedics", which is exactly how the department's own name is said ("There is an orthopaedics doctor"). Departments that exist in the
    # database but have no entry here are still found by resolve_department_by_name() below, so a new department needs no alias to be reachable.
    "orthopaedics": "Orthopaedics", "orthopaedician": "Orthopaedics", "bones": "Orthopaedics", "bone doctor": "Orthopaedics", "bone specialist": "Orthopaedics",
    "অর্থোপেডিক": "Orthopaedics", "অর্থোপেডিক্স": "Orthopaedics", "হাড়": "Orthopaedics", "ऑर्थोपेडिक": "Orthopaedics", "ऑर्थोपेडिक्स": "Orthopaedics", "हड्डियों": "Orthopaedics", "हड्डी के": "Orthopaedics",
    "cardiology": "Cardiology", "dermatology": "Dermatology", "paediatrics": "Paediatrics", "dentistry": "Dentistry", "toothache": "Dentistry", "tooth doctor": "Dentistry",
    "দাঁত": "Dentistry", "दांतों": "Dentistry", "ear doctor": "ENT", "ear specialist": "ENT", "throat doctor": "ENT", "কানের": "ENT", "कान के": "ENT",
    "general physician": "General Medicine", "family doctor": "General Medicine", "gp": "General Medicine",
    # Real call, 2026-10-29: "নিউরো মেডিসিন" ("neuro medicine") is how a lot of callers say "neurologist" (the
    # non-surgical, medicine side of neuro care, as opposed to a neurosurgeon) -- not a department in its own right,
    # but a real, common way of naming this one. "medicine" alone still means General Medicine (unchanged, sorted by
    # key length above so the longer, more specific phrase always wins). Includes the real call's own dropped-vowel
    # misspelling ("नूरो" for "न्यूरो") and the matching Bengali/romanized forms.
    "neuro medicine": "Neurology", "neuro medicines": "Neurology", "neuromedicine": "Neurology", "nuero medicine": "Neurology",
    "nuro medicine": "Neurology", "newro medicine": "Neurology", "niuro medicine": "Neurology",
    "নিউরো মেডিসিন": "Neurology", "নিউরোমেডিসিন": "Neurology", "নিউরো মেডিসিনের": "Neurology", "নুরো মেডিসিন": "Neurology",
    "न्यूरो मेडिसिन": "Neurology", "न्यूरोमेडिसिन": "Neurology", "नूरो मेडिसिन": "Neurology", "नुरो मेडिसिन": "Neurology", "न्यूरो मेडिसिन्स": "Neurology",
    # Real call, 2026-10-29: "psychiatrist"/"psychiatry" itself is already found by resolve_department_by_name() below
    # (it matches the department's own English name by word-stem) -- these are the Bengali/Hindi lay phrases that
    # function never sees, since it only matches the department row's own (English) name text.
    "psychiatrist": "Psychiatry", "psychiatric": "Psychiatry", "psychologist": "Psychiatry", "mental health doctor": "Psychiatry", "mind doctor": "Psychiatry",
    "মানসিক রোগের ডাক্তার": "Psychiatry", "মানসিক ডাক্তার": "Psychiatry", "মনোরোগ বিশেষজ্ঞ": "Psychiatry", "মনোরোগ ডাক্তার": "Psychiatry", "সাইকিয়াট্রিস্ট": "Psychiatry",
    "मानसिक रोग के डॉक्टर": "Psychiatry", "मानसिक डॉक्टर": "Psychiatry", "मनोचिकित्सक": "Psychiatry", "मनोरोग विशेषज्ञ": "Psychiatry", "साइकियाट्रिस्ट": "Psychiatry",
}


# Owner, 2026-10-28 (real call): "...একটা জেনারেল মেডিসিন এর একটা ডাক্তার দেখাতে চাই আজকে কাকে পাবো" -- the caller NAMED the department, but only English and a few Bengali forms were known, so the agent asked
# "what is the problem?". Every department's own Bengali and Hindi display name (the words the agent itself says) now resolves to it.
_DEPT_ALIASES.update(
    {
        "জেনারেল মেডিসিন": "General Medicine", "जनरल मेडिसिन": "General Medicine", "জেনারেল ফিজিশিয়ান": "General Medicine", "जनरल फिजिशियन": "General Medicine", "জেনারেল ফিজিসিয়ান": "General Medicine",
        "কার্ডিওলজি": "Cardiology", "कार्डियोलॉजी": "Cardiology",
        "গাইনোকোলজি": "Gynaecology & Obstetrics", "गायनेकोलॉजी": "Gynaecology & Obstetrics", "গাইনি": "Gynaecology & Obstetrics", "गायनी": "Gynaecology & Obstetrics",
        "অর্থোপেডিক্স": "Orthopaedics", "ऑर्थोपेडिक्स": "Orthopaedics", "অর্থোপেডিক": "Orthopaedics",
        "গ্যাস্ট্রো": "Gastroenterology", "গ্যাস্ট্রোএন্টারোলজি": "Gastroenterology", "গ্যাস্ট্রোএন্টারোলজিস্ট": "Gastroenterology", "गैस्ट्रो": "Gastroenterology", "gastro": "Gastroenterology",
        "gastroenterologist": "Gastroenterology", "gastroenterology": "Gastroenterology", "gastrologist": "Gastroenterology",
        "ডারমেটোলজিস্ট": "Dermatology", "ডার্মাটোলজিস্ট": "Dermatology", "ডার্মেটোলজিস্ট": "Dermatology", "ডারমাটোলজিস্ট": "Dermatology", "ডারমেটোলজি": "Dermatology", "ডার্মেটোলজি": "Dermatology",
        "চর্মরোগ": "Dermatology", "স্কিনের ডাক্তার": "Dermatology", "স্কিন ডাক্তার": "Dermatology", "ত্বকের ডাক্তার": "Dermatology", "चमड़ी": "Dermatology", "डर्मेटोलॉजिस्ट": "Dermatology", "स्किन डॉक्टर": "Dermatology",
        "ডার্মাটোলজি": "Dermatology", "डर्मेटोलॉजी": "Dermatology", "ডার্মাটোলজি": "Dermatology",
        "পেডিয়াট্রিক্স": "Paediatrics", "पीडियाट्रिक्स": "Paediatrics", "পেডিয়াট্রিক": "Paediatrics",
        "ডায়াবেটিস ও এন্ডোক্রিনোলজি": "Diabetology & Endocrinology", "এন্ডোক্রিনোলজি": "Diabetology & Endocrinology", "एंडोक्रिनोलॉजी": "Diabetology & Endocrinology",
        "ডেন্টিস্ট্রি": "Dentistry", "डेंटिस्ट्री": "Dentistry",
        "জেনারেল সার্জারি": "General Surgery", "जनरल सर्जरी": "General Surgery",
        "নিউরোলজি": "Neurology", "न्यूरोलॉजी": "Neurology",
        "নিউরোসার্জারি": "Neurosurgery", "न्यूरोसर्जरी": "Neurosurgery",
        "ইউরোলজি": "Urology", "यूरोलॉजी": "Urology",
        "ই এন টি": "ENT", "ई एन टी": "ENT",
    }
)


def _alias_department(q: str) -> str | None:
    """The department a short form in `q` stands for, or None. Longest key first, so 'general medicine' beats 'medicine'."""
    import re

    # sourav, 2026-10-24: lay phrases first ("haarer doctor" = the bone doctor, "mathar doctor" = neurology ...), see lay_departments.py. Only a NEW phrase can match here,
    # so every earlier alias keeps resolving exactly as before.
    lay = lay_departments.resolve(q)
    if lay:
        return lay[0]
    for key in sorted(_DEPT_ALIASES, key=len, reverse=True):
        if key.isascii():
            if re.search(rf"(?<![a-z]){re.escape(key)}(?![a-z])", q):
                return _DEPT_ALIASES[key]
        elif key in q:
            return _DEPT_ALIASES[key]
    return None


_DEPT_NAME_STOP = frozenset({"and", "the", "general", "of"})


def _dept_fold(s: str) -> str:
    """Lower-case, British->American spelling folded ("Orthopaedics" == "orthopedics"), punctuation to spaces."""
    import re

    s = (s or "").lower().replace("ae", "e").replace("oe", "e")
    return " ".join(re.sub(r"[^a-z0-9ঀ-৿ऀ-ॿ ]+", " ", s).split())


def resolve_department_by_name(db: Session, text: str) -> Department | None:
    """KCD-lead-A: the department a free-text request names, from the DEPARTMENTS TABLE ITSELF -- not from a hand-kept alias list -- so every department
    in the clinic's database (and any added later) is reachable by its own name in any common spelling or ending: "orthopaedic doctor",
    "Orthopaedics Doctor", "orthopedics", "neurologist" all find their department. Each significant word of a department's name is matched as a
    word prefix (its first max(5, len-3) letters), so "-ics / -ic / -ist / -ology" endings do not matter; a very short name (ENT) must be a whole word.
    Returns the single department that matches, or None when none or more than one does (never a guess between two)."""
    import re

    q = _dept_fold(text)
    if not q:
        return None
    hits: list[Department] = []
    for d in db.query(Department).order_by(Department.id).all():
        words = [w for w in _dept_fold(d.name).split() if w not in _DEPT_NAME_STOP and len(w) >= 3]
        for w in words:
            if len(w) <= 4:
                ok = re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", q) is not None
            else:
                stem = w[: max(5, len(w) - 3)]
                ok = re.search(rf"(?<![a-z]){re.escape(stem)}[a-z]*", q) is not None
            if ok:
                hits.append(d)
                break
    return hits[0] if len(hits) == 1 else None


def _chamber_over_today(on_date: str | None, end_time: str | None) -> bool:
    """True when `on_date` is TODAY and the chamber's end time has already passed: "today 10 to 12" is no longer a sitting at 3 pm (owner, 2026-10-24: the closed chamber was still listed)."""
    if not on_date or not end_time:
        return False
    now = clinic_clock.now()
    if on_date != now.date().isoformat():
        return False
    return _minutes(end_time) <= now.hour * 60 + now.minute


def _department_doctors_today(db: Session, department_id: int, lang: str = "bn") -> list[str]:
    """The doctors of the department who sit on today's weekday (the clinic's day: the server runs on Asia/Kolkata time) and whose chamber has not closed yet."""
    wd = clinic_clock.today().weekday()
    today_iso = clinic_clock.today().isoformat()
    out = []
    for d in db.query(Doctor).filter_by(department_id=department_id).order_by(Doctor.id).all():
        sched = db.query(DoctorSchedule).filter_by(doctor_id=d.id, weekday=wd).first()
        if sched and not _chamber_over_today(today_iso, sched.end_time):
            out.append(_doctor_spoken_name(d, lang))
    return out


# a broken / sprained / slipped bone word, or plain hand pain: with the bone doctor already a match, no choice between the bone doctor and the ordinary one is offered
_BROKEN_BONE = re.compile(
    r"ভেঙে|ভেঙ্গে|ভাঙা|ভাঙ্গা|ফ্র্যাকচার|প্লাস্টার|মচক|স্লিপ|ডিস্ক|কলার ?বোন|স্পাইন|মেরুদণ্ড|হাতে? ব্যথা|"
    r"टूट|फ्रैक्चर|प्लास्टर|मोच|स्लिप|डिस्क|कॉलर|रीढ़|हाथ (?:में )?दर्द|"
    r"fractur|broke|plaster|sprain|slip(?:ped)? dis[ck]|collar ?bone|spinal|spine|dislocat|hand pain|hat[eu]? byatha|haat byatha",
    re.I,
)


# a cut / split / scraped place ("মাথা ফেটে গেছে", "হাত কেটে গেছে", "सिर फट गया", "kete geche"): the ordinary doctor, even when a body word beside it ("মাথা") would send the caller to a specialist
_CUT_WOUND = re.compile(
    r"(?:ফেটে|কেটে|ছড়ে|কাটা ছেঁড়া)\s*(?:গেছে|গিয়েছে|গেল|গিয়ে|যাওয়া)|(?:कट|फट|छिल)\s*(?:गया|गई|गयी|गए)|सिर\s+(?:फट|कट)|"
    r"(?:ph?ete|fete|kete|kate)\s+(?:gech?he|geche|gelo|gese)|split open|got cut|cut my (?:head|hand|leg|finger)|(?:head|hand|leg) is cut",
    re.I,
)


def route_department(db: Session, query_text: str, lang: str, age: int | None = None) -> dict:
    """Symptom text -> department, plus (KCD-lay-terms) the real doctors in it, so the agent can name someone
    instead of only the department. `phrase_matches` tolerates a word or two inserted inside a keyword phrase --
    "বুকে ব্যথা" (chest pain) must still match "বুকে খুব ব্যথা" (chest REALLY hurts), an ordinary way to say it."""
    q = query_text.lower()
    matches: list[int] = []
    for r in db.query(DepartmentRoute).all():
        # KCD-lead-A: every language's keywords are checked, the caller's own language first in meaning only -- the call's sticky reply language says nothing
        # about which script THIS sentence came back in (agent/main.py's own comment on language_of_script), and romanized Bengali ("amar payer byatha") lives
        # in the English column. Strictly more permissive than checking one column: a phrase that matched before still matches.
        kws = "|".join(k for k in (r.keywords_bn, r.keywords_hi, r.keywords_en) if k)
        if any(kw and phrase_matches(kw.lower(), q) for kw in kws.split("|")):
            matches.append(r.department_id)
    matches = list(dict.fromkeys(matches))
    # sourav, 2026-10-24: (1) lay doctor phrases ("peter doctor", "mathar doctor", "bachhader haarer doctor" ...) -- a specific one overrides the symptom rows, an ordinary-
    # physician one ("pain er doctor") is used only when nothing else matched; (2) a patient of 13 or more is never sent to the paediatrician. See lay_departments.py.
    # owner, 2026-10-28: a broken-bone word beside "hand pain" is the bone doctor, not a choice between the bone doctor and the ordinary one
    if len(matches) > 1 and _BROKEN_BONE.search(q):
        ortho = db.query(Department).filter_by(name="Orthopaedics").first()
        if ortho is not None and ortho.id in matches:
            matches = [ortho.id]
    if _CUT_WOUND.search(q):
        gm = db.query(Department).filter_by(name=lay_departments.GENERAL_MEDICINE).first()
        if gm is not None:
            dept = gm
            return {
                "matched": True,
                "department_name": dept.name,
                "department_id": dept.id,
                "doctors": _department_doctor_names(db, dept.id, lang),
                "doctors_today": _department_doctors_today(db, dept.id, lang),
            }
    lay_names = lay_departments.resolve(q, age)
    if lay_names:
        by_name = {d.name: d.id for d in db.query(Department).all()}
        lay_ids = [by_name[n] for n in lay_names if n in by_name]
        strong = any(not weak for weak in lay_departments.lay_hits(q).values())
        if lay_ids and (strong or not matches):
            matches = list(dict.fromkeys(lay_ids))
    if age is not None and matches:
        by_id = {d.id: d.name for d in db.query(Department).all()}
        names = lay_departments.apply_age_rule([by_id.get(m, "") for m in matches], age)
        back = {n: i for i, n in by_id.items()}
        matches = list(dict.fromkeys(back[n] for n in names if n in back)) or matches
    if not matches:
        return {"matched": False}
    if len(matches) > 1:
        depts = [db.get(Department, m) for m in matches[:2]]
        return {
            "matched": False,
            "ambiguous": True,
            "candidates": [d.name for d in depts],
            "doctors_by_department": {d.name: _department_doctor_names(db, d.id, lang) for d in depts},
        }
    dept = db.get(Department, matches[0])
    return {
        "matched": True,
        "department_name": dept.name,
        "department_id": dept.id,
        "doctors": _department_doctor_names(db, dept.id, lang),
        "doctors_today": _department_doctors_today(db, dept.id, lang),
    }


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def sitting_doctors(
    db: Session, department_id: int, lang: str = "bn", sex: str | None = None, on_date: str | None = None, at_time: str | None = None
) -> list[dict]:
    """The doctors of a department who match what the caller asked: a lady / gentleman doctor (`sex` "F" | "M"), sitting on `on_date`
    (not on leave that day), and -- when `at_time` ("HH:MM") is given -- whose chamber hours contain that time. Owner, 2026-10-18:
    "whoever is in X department and sits tomorrow at 11". Each row says when the doctor sits that day."""
    from models import DoctorLeave

    wd = datetime.date.fromisoformat(on_date).weekday() if on_date else None
    out = []
    for d in db.query(Doctor).filter_by(department_id=department_id).order_by(Doctor.id).all():
        if sex and d.sex != sex:
            continue
        start = end = None
        if wd is not None:
            sched = db.query(DoctorSchedule).filter_by(doctor_id=d.id, weekday=wd).first()
            if not sched:
                continue
            if any(lv.start_date <= on_date <= lv.end_date for lv in db.query(DoctorLeave).filter_by(doctor_id=d.id).all()):
                continue
            start, end = sched.start_time, sched.end_time
            if _chamber_over_today(on_date, end):
                continue  # today, and the chamber is already closed
            if at_time and not (_minutes(start) <= _minutes(at_time) < _minutes(end)):
                continue
        out.append(
            {"name": _doctor_spoken_name(d, lang), "full_name": d.full_name or d.name, "sex": d.sex, "start": start, "end": end, "visit_fee_inr": d.consultation_fee_inr}
        )
    return out


def doctors_with_free_slots(
    db: Session, department_id: int, date: str, lang: str = "bn", exclude_doctor_id: int | None = None
) -> list[dict]:
    """KCD-lead-6: OTHER doctors in a department who actually have a bookable slot left on `date` -- sitting,
    not on leave, AND not already full (time-grid or the new max_bookings_per_day cap). Deliberately separate
    from sitting_doctors() above, which only checks the schedule/leave and does NOT look at available_slots(),
    so it would still list a doctor who is rostered on but has nothing left -- exactly the doctor a caller whose
    OWN doctor is full that day must not be offered. Used when a caller is told their doctor has no slot left
    that day and asks for the same day anyway, instead of only being offered a later day for the same doctor."""
    from models import DoctorLeave

    try:
        wd = datetime.date.fromisoformat(date).weekday()
    except ValueError:
        return []
    out = []
    for d in db.query(Doctor).filter_by(department_id=department_id).order_by(Doctor.id).all():
        if exclude_doctor_id and d.id == exclude_doctor_id:
            continue
        sched = db.query(DoctorSchedule).filter_by(doctor_id=d.id, weekday=wd).first()
        if not sched:
            continue
        if any(lv.start_date <= date <= lv.end_date for lv in db.query(DoctorLeave).filter_by(doctor_id=d.id).all()):
            continue
        free = available_slots(db, d.id, date)
        if not free:
            continue
        out.append(
            {
                "doctor_id": d.id,
                "name": _doctor_spoken_name(d, lang),
                "full_name": d.full_name or d.name,
                "chamber_hours": f"{sched.start_time}-{sched.end_time}",
                "free_slots": free,
                "visit_fee_inr": d.consultation_fee_inr,
            }
        )
    return out


def department_doctors(
    db: Session,
    department_name: str,
    lang: str = "bn",
    limit: int = MAX_SUGGESTED_DOCTORS,
    sex: str | None = None,
    on_date: str | None = None,
    at_time: str | None = None,
) -> dict:
    """KCD-dentistry: the real doctors in a NAMED department. Two callers, both real:
    (1) no usable symptom at all (route_department had nothing to match) and no doctor name either -- a
        general checkup first, same real-doctor-not-a-guess discipline as route_department.
    (2) a caller who names a DEPARTMENT/specialty instead of a doctor ("ENT doctor", "I need a dentist") --
        the LLM has nowhere else to put that but doctor_name, since it is never allowed to guess a department
        from a symptom itself (CLAUDE.md's truth boundary); this is deliberately case-insensitive and tolerant
        of a word or two around the department's own name ("ENT doctor" must still match "ENT") for exactly
        that reason -- a real doctor's name never collides with one of this clinic's ~9 department names."""
    q = department_name.strip().lower()
    alias = _alias_department(q)
    if alias:
        q = alias.lower()
    # KCD-lead-A: the EXACT name wins before any looser match. Before this, the alias for "dentist" -> q = "dentistry" was tested against every department with
    # a plain substring match in database order, and the three-letter name "ENT" is a substring of "dentistry": every Dentistry request was answered with
    # the ENT doctors. A name of four letters or fewer must be a whole word.
    depts = db.query(Department).all()

    def _named_in(name: str, text: str) -> bool:
        n = name.lower()
        if len(n) <= 4:
            import re

            return re.search(rf"(?<![a-z]){re.escape(n)}(?![a-z])", text) is not None
        return phrase_matches(n, text)

    dept = (
        next((d for d in depts if d.name.lower() == q), None)
        or next((d for d in depts if _named_in(d.name, q)), None)
        or resolve_department_by_name(db, q)
    )
    if not dept:
        return {"matched": False}
    out = {
        "matched": True,
        "department_name": dept.name,
        "doctors": _department_doctor_names(db, dept.id, lang, limit),
        "doctors_today": _department_doctors_today(db, dept.id, lang),
    }
    if sex or on_date or at_time:
        # a filtered search lists EVERY match (not the first three), and says plainly when the filter left nobody
        matches = sitting_doctors(db, dept.id, lang, sex, on_date, at_time)
        out.update({"filtered": True, "sex": sex, "on_date": on_date, "at_time": at_time, "sitting": matches, "doctors": [m["name"] for m in matches]})
        if sex and not on_date:
            # "doctors_today" must obey the same filter: a lady doctor asked for is never answered with today's gentlemen
            today_iso = clinic_clock.today().isoformat()
            out["doctors_today"] = [m["name"] for m in sitting_doctors(db, dept.id, lang, sex, today_iso)]
    return out


def clinic_cancellations(db: Session, phone: str | None = None, confirmation_id: str | None = None) -> list[dict]:
    """Bookings the CLINIC cancelled (a doctor on leave) for this phone / confirmation id, newest first -- so a caller who asks about
    one is told what happened, with an apology, instead of "no booking found"."""
    q = db.query(Appointment).filter(Appointment.status == "cancelled", Appointment.cancel_reason == "doctor_leave")
    if confirmation_id:
        q = q.filter(Appointment.confirmation_id == confirmation_id)
    elif phone:
        q = q.filter((Appointment.phone == phone) | (Appointment.caller_phone == phone))
    else:
        return []
    out = []
    for a in q.order_by(Appointment.cancelled_at.desc()).limit(3).all():
        doctor = db.get(Doctor, a.doctor_id)
        out.append(
            {
                "confirmation_id": a.confirmation_id,
                "date": a.date,
                "time_slot": a.time_slot,
                "reason": a.cancel_reason,
                "doctor_name": (doctor.full_name or doctor.name) if doctor else "",
                "doctor_name_bn": (doctor.full_name_bn if doctor else ""),
                "doctor_name_hi": (doctor.full_name_hi if doctor else ""),
            }
        )
    return out


# ============================================================= drafts


def save_draft(db: Session, caller_phone: str, call_id: str, slots_json: str) -> None:
    existing = db.query(DraftBooking).filter_by(caller_phone=caller_phone).first()
    expires = _now() + datetime.timedelta(minutes=DRAFT_BOOKING_TTL_MINUTES)
    if existing:
        existing.call_id, existing.slots_json = call_id, slots_json
        existing.updated_at, existing.expires_at = _now(), expires
    else:
        db.add(
            DraftBooking(
                caller_phone=caller_phone, call_id=call_id, slots_json=slots_json, updated_at=_now(), expires_at=expires
            )
        )
    db.commit()


def find_draft(db: Session, caller_phone: str) -> dict | None:
    row = db.query(DraftBooking).filter_by(caller_phone=caller_phone).first()
    if not row or row.expires_at < _now():
        return None
    return {"call_id": row.call_id, "slots_json": row.slots_json, "updated_at": row.updated_at.isoformat()}


def clear_draft(db: Session, caller_phone: str) -> None:
    db.query(DraftBooking).filter_by(caller_phone=caller_phone).delete()
    db.commit()


# ============================================================= serial (token) numbers -- paid, but never got it on WhatsApp
# sourav, 2026-10-24: "I called, booked, paid, and the serial number never came on WhatsApp." The agent confirms the caller's mobile number, then asks THIS module what the
# database holds for it and tells the caller from that -- the serial is never composed by the model (CLAUDE.md's one rule). Payment is SIMULATED: no gateway is wired
# (clinic-api/payments.py is a placeholder), so `mark_paid` below is what a payment confirmation would call -- it stamps `paid_at` and issues the serial in the order
# bookings are PAID (the owner's rule, models.Appointment.token_number). Only doctor appointments have a serial; a lab test is paid at the counter.


def mark_paid(db: Session, confirmation_id: str) -> dict:
    """sourav, 2026-10-24: record that a CONFIRMED doctor appointment's booking fee was paid and give it its serial. Why: the stand-in for a payment gateway's "paid" callback, so the
    whole paid-but-no-WhatsApp path can run end to end before a real gateway exists (tests, the sample patients, a demo). Flow: find the confirmed appointment; if already paid
    return its serial unchanged (idempotent -- paying twice never issues a second serial); else stamp paid_at and take the next number for that doctor and date, one above the
    highest serial already issued to a confirmed booking of the same doctor and day."""
    appt = db.query(Appointment).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    if appt is None:
        return {"success": False, "reason": "not_found"}
    if appt.paid_at is not None and appt.token_number is not None:
        return {"success": True, "already_paid": True, "confirmation_id": confirmation_id, "serial_number": appt.token_number}
    appt.paid_at = _now()
    if appt.token_number is None:
        appt.token_number = next_serial(db, appt.doctor_id, appt.date)
    db.commit()
    return {"success": True, "already_paid": False, "confirmation_id": confirmation_id, "serial_number": appt.token_number}


def serial_status(db: Session, phone: str) -> dict:
    """sourav, 2026-10-24: what the database holds about the serial number(s) of the upcoming doctor appointments booked from / for this mobile number. Why: answers "I paid but
    did not get my serial number" from the system of record. Flow: the same rows and the same disclosure rule as lookup_bookings (an upcoming confirmed appointment whose patient
    this number is authorised for -- never a stranger's booking); each row says whether it was paid and, if so, its serial. Nearest three only: a caller does not need ten read out."""
    today = _now().date().isoformat()
    rows = (
        db.query(Appointment)
        .filter(Appointment.status == "confirmed", Appointment.date >= today)
        .filter((Appointment.phone == phone) | (Appointment.caller_phone == phone))
        .order_by(Appointment.date, Appointment.time_slot)
        .all()
    )
    out = []
    for a in rows:
        patient = db.get(Patient, a.patient_id) if a.patient_id else None
        if not patient or not authorize_disclosure(db, patient, phone):
            continue
        doctor = db.get(Doctor, a.doctor_id)
        out.append(
            {
                "confirmation_id": a.confirmation_id,
                "doctor_name": (doctor.full_name or doctor.name) if doctor else None,
                "date": a.date,
                "time_slot": a.time_slot,
                "patient_name": a.patient_name,
                "paid": a.paid_at is not None,
                "serial_number": a.token_number if a.paid_at is not None else None,
            }
        )
        if len(out) == 3:
            break
    return {"found": bool(out), "bookings": out}


def resend_serial(db: Session, phone: str, confirmation_id: str) -> dict:
    """sourav, 2026-10-24: put a paid booking's serial number on WhatsApp again. Why: the caller says it never arrived. Flow: the booking must be one this number is allowed to hear
    about (serial_status) AND paid with a serial issued; the message goes to the number ON THE BOOKING (the one the first message went to) and the reply says which one it ends in,
    so a caller can notice a wrongly-typed number -- the likeliest reason it never came. Same placeholder discipline as every message here: queue_sms records it as "queued" and
    nothing claims it reached the phone until a real provider is wired in (notifications.py)."""
    mine = {b["confirmation_id"]: b for b in serial_status(db, phone)["bookings"]}
    row = mine.get(confirmation_id)
    if row is None:
        return {"success": False, "reason": "not_found"}
    if not row["paid"] or row["serial_number"] is None:
        return {"success": False, "reason": "not_paid"}
    appt = db.query(Appointment).filter_by(confirmation_id=confirmation_id, status="confirmed").first()
    if appt is None or appt.phone == NOT_PROVIDED_PHONE:
        return {"success": False, "reason": "no_phone_on_file"}
    message = (
        f"Hi {appt.patient_name}, your serial number for {row['doctor_name'] or 'the doctor'} on {appt.date} is {row['serial_number']}. Booking ID: {confirmation_id}. "
        f"-- হ্যালো {appt.patient_name}, {appt.date} তারিখে আপনার সিরিয়াল নম্বর {row['serial_number']}। বুকিং আইডি: {confirmation_id}।"
    )
    result = queue_sms(db, appt.phone, "serial_number_whatsapp", message, confirmation_id)
    if result.get("queued") and not result.get("duplicate"):
        delivery = notifications.send("whatsapp", appt.phone, message, confirmation_id)
        sms_row = db.get(SmsOutbox, result["id"])
        if sms_row is not None and delivery["status"] in ("sent", "failed"):
            sms_row.status = delivery["status"]  # stays "queued" while the placeholder is in place
            db.commit()
    out = {"success": bool(result.get("queued")), "channel": "whatsapp", "confirmation_id": confirmation_id, "sent_to_last4": appt.phone[-4:]}
    if not result.get("queued"):
        out["reason"] = result.get("reason", "unknown")
    return out
