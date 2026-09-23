"""Clinic data service -- implements the exact 3-endpoint contract
agent/tools_client.py in the voice agent already expects. Backed by
PostgreSQL, seeded with dummy departments/doctors/schedules/tests via
seed.py.

Matching is deliberately simple (ILIKE + difflib) for this prototype --
production callers slurring "লিপিড প্রোফাইল" through a phone mic deserve
something closer to voicerx/glossary.py's phonetic-fold gazetteer, not a
plain substring match. Flagged here rather than silently left as if this
were already that robust.
"""
from __future__ import annotations

import datetime
import difflib
import logging
import unicodedata
import uuid

import booking_service as bs
import enquiry_service as eq
from db import SessionLocal, get_db
from fastapi import Depends, FastAPI, Query
from models import FAQ, Appointment, Department, Doctor, DoctorSchedule, LabTest
from phonetic_match import phonetic_key, phonetic_match
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

app = FastAPI(title="Kolkata Care Diagnostics -- Clinic Data API (dummy)")

SLOT_STEP_MIN = 15


@app.on_event("startup")
def _ensure_seeded():
    """Create the schema, and seed it only if it is EMPTY.

    The catalogue is derived data -- 8 departments, 32 doctors, 34 tests,
    all defined in seed.py -- so regenerating it costs nothing and removes
    the manual reseed step that every pod restart used to require.

    Guarded on emptiness because seed() itself is destructive (drop_all
    then create_all). Running it unconditionally at startup would wipe
    every appointment booked since the last boot, turning a convenience
    into data loss.
    """
    from db import engine
    from models import Base, LabTest
    Base.metadata.create_all(engine)
    from seed import add_i18n_columns
    add_i18n_columns()   # before ANY ORM query: an old database lacks the new columns

    # Epic E26 (booking/reschedule/cancel): new columns on `appointments`
    # and `doctors`, and the department-routing seed table. MUST also run
    # before any ORM query that touches those tables -- same reason as
    # add_i18n_columns above, and it caught a REAL bug here: backfill_i18n()
    # below queries Doctor, which SQLAlchemy selects every mapped column
    # of, so it crashed on a database migrated only as far as
    # add_i18n_columns(). create_all() already created every brand-new
    # table (SlotLock, Patient, PatientProxy, TestBooking, SmsOutbox,
    # DraftBooking, DepartmentRoute) a moment ago; this only ALTERs what
    # already existed.
    from booking_migrate import migrate_booking_schema
    logging.getLogger("clinic-api").info("booking schema migration: %s", migrate_booking_schema())

    # Epic E27 (information and enquiry): new columns on `lab_tests`.
    # Column-ALTER only, same "before any ORM query" reasoning, same
    # class of bug if this ran after the LabTest.count() query below
    # instead of before it.
    from enquiry_migrate import add_enquiry_columns
    logging.getLogger("clinic-api").info("enquiry schema migration: %s", add_enquiry_columns())

    db = SessionLocal()
    try:
        if db.query(LabTest).count() == 0:
            logging.getLogger("clinic-api").info("empty database -- seeding catalogue")
            from seed import seed
            seed()
        else:
            logging.getLogger("clinic-api").info("catalogue already present, not reseeding")
            from seed import backfill_i18n
            logging.getLogger("clinic-api").info("i18n backfill: %s", backfill_i18n(db))

        # Only now do departments/doctors/tests definitely have rows --
        # either seed() just created them, or they already existed.
        # Calling either of these any earlier seeds zero rows on a
        # brand-new database, the exact bug finish_booking_schema_setup's
        # own docstring documents.
        from booking_migrate import finish_booking_schema_setup
        logging.getLogger("clinic-api").info(
            "booking schema setup (routes/fees): %s", finish_booking_schema_setup())
        from enquiry_migrate import backfill_enquiry_facts, seed_enquiry_demo_data
        logging.getLogger("clinic-api").info(
            "enquiry facts backfill: %s", backfill_enquiry_facts(db))
        logging.getLogger("clinic-api").info(
            "enquiry demo data: %s", seed_enquiry_demo_data(db))
    finally:
        db.close()


@app.get("/api/health")
def health(db: Session = Depends(get_db)):
    return {
        "status": "ok",
        "departments": db.query(Department).count(),
        "doctors": db.query(Doctor).count(),
        "lab_tests": db.query(LabTest).count(),
    }


# =============================================================================
# Tool 1: GET /api/v1/tests/search?name=...
# =============================================================================
def _first_alias_bn(aliases_bn: str) -> str | None:
    """The spoken form. The reply the caller HEARS is synthesized by a
    Bengali-only tokenizer that silently drops Latin script, so returning
    only `t.name` ("Uric Acid") means the caller is read a price with the
    test name missing from the sentence. Every row is seeded with at least
    one Bengali alias for exactly this reason -- see seed.py."""
    for alias in (aliases_bn or "").split("|"):
        if alias.strip():
            return alias.strip()
    return None


def _test_reply_dict(t: LabTest) -> dict:
    return {
        "found": True, "test_name": t.name, "test_name_bn": _first_alias_bn(t.aliases_bn),
        "test_name_hi": _first_alias_bn(t.aliases_hi),
        "rate_inr": t.rate_inr,
        "sample_type": t.sample_type, "report_time_hours": t.report_time_hours,
    }


@app.get("/api/v1/catalogue")
def catalogue(db: Session = Depends(get_db)):
    """Every test and doctor with their Bengali aliases, in one call.

    Exists for the voice agent's deterministic fast path: matching a
    caller's words against a 74-row catalogue is a local string operation,
    but only if the caller HAS the catalogue. Fetching it once at startup
    turns "which test did they say" from a 7B-model inference into a
    microsecond comparison -- see agent/fast_path.py.
    """
    return {
        "tests": [
            {"name": t.name,
             "aliases_bn": [a for a in (t.aliases_bn or "").split("|") if a],
             "aliases_hi": [a for a in (t.aliases_hi or "").split("|") if a]}
            for t in db.query(LabTest).all()
        ],
        "doctors": [
            {"name": d.name,
             "surname": d.name.split()[-1],
             "aliases_bn": [a for a in (d.aliases_bn or "").split("|") if a],
             "aliases_hi": [a for a in (d.aliases_hi or "").split("|") if a]}
            for d in db.query(Doctor).all()
        ],
        # FAQ topics + their keyword sets, for FastPath.FAQCatalogue --
        # same "fetch the small table once, match locally" shape as tests
        # and doctors above. NOT the answer text itself: the answer is
        # still fetched live via /api/v1/faq on every turn, the same
        # discipline test_rate and doctor_availability already apply, so a
        # cached routing decision can never serve a stale FAQ answer.
        "faq_topics": [
            {"topic": f.topic,
             "keywords_bn": [k for k in (f.keywords_bn or "").split("|") if k]}
            for f in db.query(FAQ).all()
        ],
    }


# Words that carry no test identity when a caller wraps a test name in a
# sentence: "lipid profile TEST" must match "Lipid Profile". Bengali/Hindi
# spellings of "test" included.
_GENERIC_WORDS = {"test", "tests", "the", "a", "of", "for", "price", "rate",
                  "টেস্ট", "টেস্টের", "टेस्ट", "जांच"}


def _norm_name(s: str) -> str:
    """Comparison form for a caller-said test name.

    Lowercases, drops generic words, and removes the Devanagari nukta
    (U+093C): the Hindi ASR emits both "प्रोफाइल" and "प्रोफ़ाइल" for the same
    spoken word, and NFC keeps the nukta as a separate mark, so removing it
    makes the two spellings equal. Never fuzzy -- a wrong test's real price is
    the failure this file exists to prevent, so a near-miss is offered as a
    suggestion to the caller, not silently accepted."""
    s = unicodedata.normalize("NFC", s).lower().replace("\u093c", "").replace("-", " ")
    toks = [t.strip("?.,;:!'\"()") for t in s.split()]
    return " ".join(t for t in toks if t and t not in _GENERIC_WORDS)


def _find_test(db: Session, name: str) -> LabTest | None:
    """The exact/Bengali-alias/fuzzy cascade search_test() and the prep
    endpoint both need -- factored out so "which test did they mean" has
    exactly one implementation, not two that can drift apart."""
    exact = db.query(LabTest).filter(func.lower(LabTest.name).contains(name.lower())).first()
    if exact:
        return exact

    all_tests = db.query(LabTest).all()
    for t in all_tests:
        aliases = [a for a in (t.aliases_bn + "|" + (t.aliases_hi or "")).split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            return t

    # Normalised, two-way containment against the English name and every
    # alias: the caller's phrase may contain the test name ("lipid profile
    # test") or be a part of it, in any script.
    q = _norm_name(name)
    if q:
        for t in all_tests:
            forms = [t.name] + [a for a in (t.aliases_bn + "|" + (t.aliases_hi or "")).split("|") if a]
            for form in forms:
                n = _norm_name(form)
                if n and (q in n or n in q):
                    return t

    # KCD-434: a romanised spelling of a Bengali/Hindi term (or the
    # reverse -- a Bengali/Hindi ASR engine transliterating an English
    # loanword into its own script) folds to the same consonant skeleton
    # as the canonical entry even though no character-level match exists.
    # Gated on a minimum key length so a short, generic fold ("test" ->
    # "3" after generic-word stripping leaves nothing, or a two-letter
    # leftover) can never stand in for a real match.
    qk = phonetic_key(name)
    if len(qk) >= 3:
        for t in all_tests:
            forms = [t.name] + [a for a in (t.aliases_bn + "|" + (t.aliases_hi or "")).split("|") if a]
            for form in forms:
                if phonetic_key(form) == qk:
                    return t
    return None


def _find_test_candidates(db: Session, name: str) -> list[LabTest]:
    """KCD-446: every test matching `name` at the FIRST cascade tier that
    produces any match at all (exact substring, then alias, then
    normalised containment) -- the same first three tiers _find_test
    uses, but collecting every match at the winning tier instead of
    silently returning the first one.

    Deliberately stops BEFORE _find_test's fourth, phonetic-fold tier
    (KCD-434): that tier is a loose, best-effort fallback for
    mishearings and romanised spellings, and its short folded keys are
    short precisely because they are generic -- gating IT on "more than
    one match" would flag routine romanised lookups (e.g. "sibisi" for
    "সিবিসি") as ambiguous and break KCD-434's own resolution, which
    this function's callers fall through to _find_test for unchanged
    when tiers 1-3 find nothing here."""
    all_tests = db.query(LabTest).all()

    exact = [t for t in all_tests if name.lower() in t.name.lower()]
    if exact:
        return exact

    alias_matches = []
    for t in all_tests:
        aliases = [a for a in (t.aliases_bn + "|" + (t.aliases_hi or "")).split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            alias_matches.append(t)
    if alias_matches:
        return alias_matches

    q = _norm_name(name)
    norm_matches = []
    if q:
        for t in all_tests:
            forms = [t.name] + [a for a in (t.aliases_bn + "|" + (t.aliases_hi or "")).split("|") if a]
            if any(_norm_name(f) and (q in _norm_name(f) or _norm_name(f) in q) for f in forms):
                norm_matches.append(t)
    if norm_matches:
        return norm_matches

    return []


def _test_suggestions(db: Session, name: str) -> list[str]:
    all_tests = db.query(LabTest).all()
    candidates = []
    for t in all_tests:
        candidates.append(t.name)
        candidates.extend(a for a in (t.aliases_bn + "|" + (t.aliases_hi or "")).split("|") if a)
    suggestions = difflib.get_close_matches(name, candidates, n=3, cutoff=0.5)
    alias_to_name = {a: t.name for t in all_tests
                     for a in (t.aliases_bn + "|" + (t.aliases_hi or "")).split("|") if a}
    return list(dict.fromkeys(alias_to_name.get(s, s) for s in suggestions))


@app.get("/api/v1/tests/search")
def search_test(name: str = Query(...), db: Session = Depends(get_db)):
    # English substring match, then Bengali-script match -- covers callers
    # who say the test name in English/transliterated form, then the
    # actual common case. A caller saying "ইউরিক এসিড" was matched against
    # nothing before the Bengali path existed: the DB only stored the
    # English name "Uric Acid", and Bengali script shares zero characters
    # with Latin script, so substring AND fuzzy matching against the
    # English column alone can NEVER succeed on Bengali input, regardless
    # of how close the pronunciation is.
    # KCD-446: two or three catalogue rows matching equally well is a
    # genuinely different outcome from "nothing matched" -- silently
    # picking one (what _find_test does, correctly, for booking) would
    # risk quoting the price of a DIFFERENT test than the one meant.
    candidates = _find_test_candidates(db, name)
    if len(candidates) > 1:
        return {"found": False, "query": name, "ambiguous": True,
                "did_you_mean": [t.name for t in candidates[:3]]}

    found = candidates[0] if candidates else _find_test(db, name)
    if found:
        return _test_reply_dict(found)

    # Fuzzy fallback -- try both the English name and every Bengali alias,
    # so suggestions are useful regardless of which script the caller used.
    return {"found": False, "query": name, "did_you_mean": _test_suggestions(db, name)}


# =============================================================================
# Tool 4: GET /api/v1/tests/prep?name=...
# =============================================================================
def _by_lang(lang: str, bn: str, hi: str, en: str) -> str:
    """Requested-language text, falling back to Bengali (the system of
    record) if a translation is missing rather than returning an empty
    string the caller would hear as silence."""
    return {"hi": hi, "en": en}.get(lang) or bn


def _test_prep_reply_dict(t: LabTest, lang: str = "bn") -> dict:
    return {
        "found": True, "test_name": t.name, "test_name_bn": _first_alias_bn(t.aliases_bn),
        "test_name_hi": _first_alias_bn(t.aliases_hi),
        "fasting_required": t.fasting_required, "lang": lang,
        "prep_instructions": _by_lang(lang, t.prep_instructions_bn,
                                      t.prep_instructions_hi, t.prep_instructions_en),
    }


@app.get("/api/v1/tests/prep")
def test_prep(name: str = Query(...), lang: str = Query("bn"), db: Session = Depends(get_db)):
    """Same entity resolution as /tests/search, different fact -- kept as a
    separate endpoint rather than folding prep fields into every search
    response, because prep instructions are Tier-3 "approved content"
    (Blueprint 2.2) conceptually distinct from the Tier-1 price/turnaround
    facts search_test returns, and the two may end up backed by different
    systems of record later."""
    candidates = _find_test_candidates(db, name)   # KCD-446: see search_test's own comment
    if len(candidates) > 1:
        return {"found": False, "query": name, "ambiguous": True,
                "did_you_mean": [t.name for t in candidates[:3]]}

    found = candidates[0] if candidates else _find_test(db, name)
    if found:
        return _test_prep_reply_dict(found, lang)
    return {"found": False, "query": name, "did_you_mean": _test_suggestions(db, name)}


# =============================================================================
# Tool 2: GET /api/v1/doctors/availability?name=...&date=YYYY-MM-DD (optional)
# =============================================================================
# FUZZY_SURNAME_FLOOR -- LOW confidence, reasoned not measured (no real call
# audio to calibrate against yet, unlike voicerx/gate.py's SIMILARITY_FLOOR).
#
# This exists because of a bug caught in local testing: matching the raw
# query against the full formatted name ("Dr. A. Sen") let a query for
# "Doctor Nobody" fuzzy-match "Dr. N. Roy" at ratio 0.522 -- HIGHER than the
# ratio for a real garbled name against its own doctor ("sen" vs "Dr. A. Sen"
# scores only 0.462, because SequenceMatcher penalizes the length mismatch
# against the "Dr. X." prefix on both sides, so short queries and wrong
# queries land in the same range). That is this system's own small version
# of the "Naloxone" bug: confidently answering with the wrong doctor's real
# schedule instead of saying "not found".
#
# Fix: match against the SURNAME only, which cleanly separates the two
# cases in testing -- genuine garbles (e.g. "mukharji" vs "Mukherjee")
# scored 0.70-0.80; unrelated queries (e.g. "doctor nobody" vs "Roy")
# scored <=0.44. 0.60 sits in the gap. Recalibrate once real call audio
# exists, the same way gate.py's floors were tightened from real samples.
FUZZY_SURNAME_FLOOR = 0.60

# KCD-436: a mishearing severe enough to drop the character ratio below
# FUZZY_SURNAME_FLOOR can still be the right doctor if the CONSONANT
# SKELETON matches (agent/phonetic_match.py) -- e.g. an aspirated/
# unaspirated swap or a cross-script transliteration variant. Used only
# as a SECOND, independent gate on top of phonetic_match(), not instead
# of the character floor: this floor sits just above the 0.44 the
# documented "doctor nobody" vs "Roy" false positive scored, so that
# regression stays blocked even with phonetic matching turned on (its own
# phonetic keys don't match, either -- belt and suspenders).
PHONETIC_ASSISTED_FLOOR = 0.45


def _find_doctor(db: Session, name: str) -> Doctor | None:
    # English substring match (e.g. "Sen", "Dr Sen").
    exact = db.query(Doctor).filter(func.lower(Doctor.name).contains(name.lower())).first()
    if exact:
        return exact

    all_doctors = db.query(Doctor).all()

    # Bengali-script exact match -- a real caller says "ডক্টর সেন", which
    # shares no characters with the Latin "Dr. A. Sen" stored as the
    # canonical name. Same root cause and same fix as search_test()'s
    # aliases_bn check.
    for d in all_doctors:
        aliases = [a for a in (d.aliases_bn + "|" + (d.aliases_hi or "")).split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            return d

    # Fuzzy fallback, against BOTH the English surname and the Bengali
    # alias(es) -- garbled ASR output can land on either script depending
    # on what the caller actually said and how the decoder heard it.
    best_doctor, best_ratio, best_phonetic = None, 0.0, False
    for d in all_doctors:
        candidates = [d.name.split()[-1].lower()] + [
            a for a in (d.aliases_bn + "|" + (d.aliases_hi or "")).split("|") if a]
        for c in candidates:
            ratio = difflib.SequenceMatcher(None, name.lower(), c.lower()).ratio()
            if ratio > best_ratio:
                best_doctor, best_ratio, best_phonetic = d, ratio, phonetic_match(name, c)

    if best_ratio >= FUZZY_SURNAME_FLOOR:
        return best_doctor
    if best_ratio >= PHONETIC_ASSISTED_FLOOR and best_phonetic:
        return best_doctor
    return None


# Same ratio floor already used to ACCEPT a single fuzzy match above
# (FUZZY_SURNAME_FLOOR) -- reused here as the bar for COLLECTING a name
# into the ambiguity set, so "which doctors are even in the running" and
# "would _find_doctor have accepted this one alone" stay the same
# question asked twice, never two different bars that could disagree.
def _find_doctor_candidates(db: Session, name: str) -> list[Doctor]:
    """Every doctor whose surname or alias clears FUZZY_SURNAME_FLOOR
    against `name`, not just the single best one -- so two similarly-
    spelled doctors (e.g. two surnames both folding close to a garbled
    query) surface as a genuine "which one" choice (KCD-446's doctor-side
    counterpart) instead of _find_doctor silently picking whichever one
    happened to score a hair higher. Exact/alias matches never reach this
    function -- _find_doctor already returns on those, same as
    _find_test_candidates never needing to consider _find_test's own
    exact/alias tiers."""
    all_doctors = db.query(Doctor).all()
    scored = []
    for d in all_doctors:
        candidates = [d.name.split()[-1].lower()] + [
            a for a in (d.aliases_bn + "|" + (d.aliases_hi or "")).split("|") if a]
        best_ratio = max(
            (difflib.SequenceMatcher(None, name.lower(), c.lower()).ratio() for c in candidates),
            default=0.0,
        )
        if best_ratio >= FUZZY_SURNAME_FLOOR:
            scored.append((best_ratio, d))
    if not scored:
        return []
    top = max(r for r, _ in scored)
    # Only doctors within a hair of the top score are genuinely "in the
    # running" -- a query that clearly favours one doctor over another
    # (both above the floor, but one far ahead) is not an ambiguous case,
    # it is a confident match with a distant runner-up.
    return [d for r, d in scored if top - r <= 0.05]


def _doctor_suggestions(db: Session, name: str) -> list[str]:
    all_doctors = db.query(Doctor).all()
    candidates = []
    for d in all_doctors:
        candidates.append(d.name)
        candidates.extend(a for a in (d.aliases_bn + "|" + (d.aliases_hi or "")).split("|") if a)
    suggestions = difflib.get_close_matches(name, candidates, n=3, cutoff=0.5)
    alias_to_name = {a: d.name for d in all_doctors
                     for a in (d.aliases_bn + "|" + (d.aliases_hi or "")).split("|") if a}
    return list(dict.fromkeys(alias_to_name.get(s, s) for s in suggestions))


def _schedule_for_weekday(db: Session, doctor_id: int, weekday: int) -> DoctorSchedule | None:
    return db.query(DoctorSchedule).filter_by(doctor_id=doctor_id, weekday=weekday).first()


def _next_available_date(db: Session, doctor_id: int, from_date: datetime.date,
                          horizon_days: int = 14) -> str | None:
    for offset in range(horizon_days):
        d = from_date + datetime.timedelta(days=offset)
        if _schedule_for_weekday(db, doctor_id, d.weekday()):
            return d.isoformat()
    return None


@app.get("/api/v1/doctors/availability")
def doctor_availability(name: str = Query(...), date: str | None = Query(None),
                         db: Session = Depends(get_db)):
    # Doctor-side counterpart of KCD-446: two similarly-spelled doctors
    # tying on the fuzzy floor is a genuinely different outcome from
    # "nothing matched" -- silently picking one (what _find_doctor does,
    # correctly, when there is no tie) would risk reading out a DIFFERENT
    # doctor's real schedule, the CLAUDE.md "Doctor Nobody" class of bug.
    candidates = _find_doctor_candidates(db, name)
    if len(candidates) > 1:
        return {"found": False, "query": name, "ambiguous": True,
                "did_you_mean": [d.name for d in candidates[:3]]}

    doctor = candidates[0] if candidates else _find_doctor(db, name)
    if not doctor:
        return {"found": False, "query": name, "did_you_mean": _doctor_suggestions(db, name)}

    today = datetime.date.today()

    if date:
        try:
            target = datetime.date.fromisoformat(date)
        except ValueError:
            return {"found": False, "query": name}
        sched = _schedule_for_weekday(db, doctor.id, target.weekday())
        if sched:
            return {
                "found": True, "doctor_name": doctor.name,
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
                "doctor_name_hi": _first_alias_bn(doctor.aliases_hi), "date": target.isoformat(),
                "available": True, "chamber_hours": f"{sched.start_time}-{sched.end_time}",
                "next_available_date": None,
            }
        next_date = _next_available_date(db, doctor.id, target + datetime.timedelta(days=1))
        return {
            "found": True, "doctor_name": doctor.name,
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
                "doctor_name_hi": _first_alias_bn(doctor.aliases_hi), "date": target.isoformat(),
            "available": False, "chamber_hours": None, "next_available_date": next_date,
        }

    # No date given -> "when is this doctor next available"
    next_date = _next_available_date(db, doctor.id, today)
    if not next_date:
        return {
            "found": True, "doctor_name": doctor.name,
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
                "doctor_name_hi": _first_alias_bn(doctor.aliases_hi), "date": None,
            "available": False, "chamber_hours": None, "next_available_date": None,
        }
    sched = _schedule_for_weekday(db, doctor.id, datetime.date.fromisoformat(next_date).weekday())
    return {
        "found": True, "doctor_name": doctor.name,
                "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
                "doctor_name_hi": _first_alias_bn(doctor.aliases_hi), "date": next_date,
        "available": True, "chamber_hours": f"{sched.start_time}-{sched.end_time}",
        "next_available_date": None,
    }


# =============================================================================
# Tool 3: POST /api/v1/appointments
# =============================================================================
class BookingRequest(BaseModel):
    doctor_name: str
    date: str
    time_slot: str
    patient_name: str
    phone: str


def _generate_slots(start: str, end: str, step_min: int = SLOT_STEP_MIN) -> list[str]:
    t = datetime.datetime.strptime(start, "%H:%M")
    end_t = datetime.datetime.strptime(end, "%H:%M")
    slots = []
    while t < end_t:
        slots.append(t.strftime("%H:%M"))
        t += datetime.timedelta(minutes=step_min)
    return slots


@app.post("/api/v1/appointments")
def book_appointment(req: BookingRequest, db: Session = Depends(get_db)):
    """This is the ORIGINAL, pre-Epic-E26 booking endpoint, kept for
    backward compatibility. CodeRabbit-flagged, real bug: it used to
    create an Appointment row directly, with its own "taken" check
    against nothing but a raw Appointment query (which did not even
    filter by status, so a CANCELLED row here blocked the slot forever)
    and total invisibility to models.SlotLock -- the actual atomicity
    guard hold_slot()/confirm_booking() give the newer /api/v1/bookings/*
    flow (KCD-376). A slot booked through this endpoint was silently NOT
    reflected in available_slots() (which only reads SlotLock), so the
    voice agent could offer and double-book it, and conversely a slot
    HELD by the voice agent was invisible here too. Fixed by routing
    through the SAME hold_slot()/confirm_booking() primitives instead of
    maintaining a second, unguarded booking path -- one atomicity
    guarantee, not two that can silently disagree."""
    doctor = _find_doctor(db, req.doctor_name)
    if not doctor:
        return {"success": False, "reason": "doctor_not_found"}

    try:
        target = datetime.date.fromisoformat(req.date)
    except ValueError:
        return {"success": False, "reason": "missing_field"}

    sched = _schedule_for_weekday(db, doctor.id, target.weekday())
    if not sched:
        # Doctor doesn't sit that day at all -- not in the caller-facing
        # reason enum reply_templates.booking_reply() specifically handles,
        # so it falls to that function's generic "couldn't book" message,
        # which remains true and safe rather than a false "slot taken".
        return {"success": False, "reason": "doctor_not_available_that_day"}

    valid_slots = _generate_slots(sched.start_time, sched.end_time)
    if req.time_slot not in valid_slots:
        return {"success": False, "reason": "slot_taken", "alternative_slots": valid_slots[:3]}

    hold = bs.hold_slot(db, doctor.id, req.date, req.time_slot)
    if not hold["success"]:
        return {"success": False, "reason": "slot_taken",
                "alternative_slots": bs.available_slots(db, doctor.id, req.date)[:3]}

    # 8 hex chars, not 4 -- see booking_service._confirmation_id's comment:
    # this ID is now also accepted by /api/v1/bookings/lookup, unauthenticated.
    result = bs.confirm_booking(db, hold["hold_token"], doctor.id, req.date, req.time_slot,
                                 req.patient_name, req.phone, req.phone)
    if not result["success"]:
        return {"success": False, "reason": "slot_taken",
                "alternative_slots": bs.available_slots(db, doctor.id, req.date)[:3]}

    return {
        "success": True, "confirmation_id": result["confirmation_id"],
        "doctor_name": doctor.name,
        "doctor_name_bn": _first_alias_bn(doctor.aliases_bn),
                "doctor_name_hi": _first_alias_bn(doctor.aliases_hi), "date": req.date, "time_slot": req.time_slot,
    }


# =============================================================================
# Tool 5: GET /api/v1/faq?topic=...
# =============================================================================
@app.get("/api/v1/faq")
def faq_answer(topic: str = Query(...), lang: str = Query("bn"), db: Session = Depends(get_db)):
    """Looked up by TOPIC KEY, not free text -- the caller-facing entity
    resolution (which topic did they mean) already happened in
    FastPath.FAQCatalogue against the keyword sets from /api/v1/catalogue.
    This endpoint's only job is "give me the current answer for this
    topic", fetched live on every turn for the same reason test_rate and
    doctor_availability never trust a cached VALUE -- only a cached
    ROUTING decision."""
    row = db.query(FAQ).filter_by(topic=topic).first()
    if not row:
        return {"found": False, "topic": topic}
    return {"found": True, "topic": row.topic, "lang": lang,
            "answer": _by_lang(lang, row.answer_bn, row.answer_hi, row.answer_en)}


# =============================================================================
# Epic E26: booking, rescheduling and cancellation
#
# Two-phase everywhere a slot is claimed (hold, then confirm) so the actual
# concurrency guard -- booking_service.hold_slot()'s atomic INSERT into
# SlotLock -- runs BEFORE the caller has to speak a patient name and phone
# number, not after. See booking_service.py's module docstring and
# models.SlotLock's for the full reasoning (KCD-376).
# =============================================================================
class HoldRequest(BaseModel):
    doctor_name: str
    date: str
    time_slot: str


def _validate_date_not_past(date_str: str) -> dict | None:
    """Shared by every booking endpoint that takes a caller-supplied
    date. Returns an error dict in hold_booking's own shape, or None
    when the date is at least parseable and not in the past."""
    try:
        target = datetime.date.fromisoformat(date_str)
    except ValueError:
        return {"success": False, "reason": "invalid_date"}
    if target < datetime.date.today():
        # KCD-363: never book a past date, and never silently roll it
        # forward -- the caller is told plainly (main.py's reply template)
        # and offered the same weekday next week.
        return {"success": False, "reason": "date_in_past"}
    return None


def _validate_doctor_slot(db: Session, doctor: Doctor, date_str: str, time_slot: str) -> dict | None:
    """CodeRabbit-flagged: this used to live only inside hold_booking,
    which reschedule_booking never called -- it went straight to
    bs.reschedule_appointment(), which calls hold_slot(), a bare
    ATOMICITY primitive that deliberately does not validate against the
    doctor's schedule at all (see its own docstring). A malformed date
    (a bad LLM extraction) or a genuinely past date could reach a
    "successful" reschedule to a broken state, confirmed to the caller,
    then crash something unrelated later (e.g. cancellation_charge()
    parsing that date/time). Extracted so every endpoint that resolves a
    date+slot against a specific doctor applies the SAME checks."""
    err = _validate_date_not_past(date_str)
    if err:
        return err
    target = datetime.date.fromisoformat(date_str)
    sched = _schedule_for_weekday(db, doctor.id, target.weekday())
    if not sched:
        return {"success": False, "reason": "doctor_not_available_that_day"}
    valid_slots = _generate_slots(sched.start_time, sched.end_time)
    if time_slot not in valid_slots:
        return {"success": False, "reason": "invalid_slot", "valid_slots": valid_slots}
    if target == datetime.date.today():
        # CodeRabbit-flagged, real bug: _validate_date_not_past is
        # DATE-level only ("target < today"), so a same-day slot earlier
        # than right now was still requestable and holdable directly
        # (available_slots() no longer OFFERS it -- see that function's
        # own fix -- but nothing stopped a caller from asking for it by
        # name anyway).
        now = datetime.datetime.now()
        try:
            slot_h, slot_m = (int(x) for x in time_slot.split(":"))
        except ValueError:
            return {"success": False, "reason": "invalid_slot", "valid_slots": valid_slots}
        if slot_h * 60 + slot_m <= now.hour * 60 + now.minute:
            return {"success": False, "reason": "invalid_slot",
                    "valid_slots": [s for s in valid_slots
                                    if int(s.split(":")[0]) * 60 + int(s.split(":")[1]) > now.hour * 60 + now.minute]}
    return None


@app.post("/api/v1/bookings/hold")
def hold_booking(req: HoldRequest, db: Session = Depends(get_db)):
    doctor = _find_doctor(db, req.doctor_name)
    if not doctor:
        return {"success": False, "reason": "doctor_not_found"}
    err = _validate_doctor_slot(db, doctor, req.date, req.time_slot)
    if err:
        return err

    result = bs.hold_slot(db, doctor.id, req.date, req.time_slot)
    if not result["success"]:
        result["alternative_slots"] = [a["time_slot"] for a in
                                        bs.nearest_alternatives(db, doctor.id, req.date, req.time_slot)]
    else:
        result["doctor_id"] = doctor.id
        result["doctor_name"] = doctor.name
    return result


class ConfirmRequest(BaseModel):
    hold_token: str
    doctor_id: int
    date: str
    time_slot: str
    patient_name: str
    phone: str
    caller_phone: str
    patient_age: int | None = None
    relationship: str = "self"


@app.post("/api/v1/bookings/confirm")
def confirm_booking_endpoint(req: ConfirmRequest, db: Session = Depends(get_db)):
    return bs.confirm_booking(db, req.hold_token, req.doctor_id, req.date, req.time_slot,
                               req.patient_name, req.phone, req.caller_phone,
                               req.patient_age, req.relationship)


class RescheduleRequest(BaseModel):
    confirmation_id: str
    new_date: str
    new_time_slot: str


@app.post("/api/v1/bookings/reschedule")
def reschedule_booking(req: RescheduleRequest, db: Session = Depends(get_db)):
    appt = db.query(Appointment).filter_by(confirmation_id=req.confirmation_id, status="confirmed").first()
    if not appt:
        return {"success": False, "reason": "not_found"}
    doctor = db.get(Doctor, appt.doctor_id)
    err = _validate_doctor_slot(db, doctor, req.new_date, req.new_time_slot)
    if err:
        return err
    return bs.reschedule_appointment(db, req.confirmation_id, req.new_date, req.new_time_slot)


class CancelRequest(BaseModel):
    confirmation_id: str
    confirm_charge: bool = False


@app.post("/api/v1/bookings/cancel")
def cancel_booking(req: CancelRequest, db: Session = Depends(get_db)):
    return bs.cancel_appointment(db, req.confirmation_id, req.confirm_charge)


@app.get("/api/v1/bookings/lookup")
def lookup_booking(phone: str | None = Query(None), confirmation_id: str | None = Query(None),
                    name: str | None = Query(None), db: Session = Depends(get_db)):
    if not (phone or confirmation_id):
        return {"found": False, "bookings": []}
    rows = bs.lookup_bookings(db, phone=phone, confirmation_id=confirmation_id, name=name)
    return {"found": bool(rows), "bookings": rows}


@app.get("/api/v1/bookings/conflict")
def booking_conflict(phone: str = Query(...), date: str = Query(...), time_slot: str = Query(...),
                      db: Session = Depends(get_db)):
    conflict = bs.find_conflict(db, phone, date, time_slot)
    return {"conflict": conflict is not None, "existing": conflict}


class TestsBookingRequest(BaseModel):
    test_names: list[str]
    date: str
    patient_name: str
    phone: str
    caller_phone: str
    patient_age: int | None = None
    relationship: str = "self"


@app.post("/api/v1/bookings/tests")
def book_tests_endpoint(req: TestsBookingRequest, db: Session = Depends(get_db)):
    err = _validate_date_not_past(req.date)
    if err:
        return err
    ids, not_found = [], []
    for name in req.test_names:
        t = _find_test(db, name)
        (ids if t else not_found).append(t.id if t else name)
    if not ids:
        return {"success": False, "reason": "no_valid_tests", "not_found": not_found}
    result = bs.book_tests(db, ids, req.date, req.patient_name, req.phone, req.caller_phone,
                            req.patient_age, req.relationship)
    result["not_found"] = not_found
    return result


class AddTestRequest(BaseModel):
    confirmation_id: str
    test_name: str


@app.post("/api/v1/bookings/add-test")
def add_test_endpoint(req: AddTestRequest, db: Session = Depends(get_db)):
    return bs.add_test_to_booking(db, req.confirmation_id, req.test_name)


@app.get("/api/v1/doctors/earliest")
def doctor_earliest(name: str = Query(...), db: Session = Depends(get_db)):
    doctor = _find_doctor(db, name)
    if not doctor:
        return {"found": False, "query": name}
    result = bs.earliest_available(db, doctor.id, datetime.date.today())
    if not result:
        return {"found": True, "available": False, "doctor_name": doctor.name}
    return {"found": True, "available": True, "doctor_name": doctor.name,
            "doctor_name_bn": _first_alias_bn(doctor.aliases_bn), **result}


@app.get("/api/v1/departments/route")
def department_route(query: str = Query(...), lang: str = Query("bn"), db: Session = Depends(get_db)):
    return bs.route_department(db, query, lang)


class SmsRequest(BaseModel):
    to: str
    message: str
    template_key: str = "generic"
    related_confirmation_id: str | None = None


@app.post("/api/v1/notifications/sms")
def send_sms(req: SmsRequest, db: Session = Depends(get_db)):
    """The open placeholder the caller-confirmation flow already writes
    to via booking_service.queue_sms(). Exposed as its own endpoint too,
    so an external system (or a future clinic-api/notifications.py) has
    one clear place to either call in, or be wired up as the thing THIS
    function calls out to. `status` is always "queued": nothing in this
    codebase claims a message reached a phone until a real provider is
    plugged in here."""
    return bs.queue_sms(db, req.to, req.template_key, req.message, req.related_confirmation_id)


@app.post("/api/v1/bookings/resend")
def resend_booking_confirmation(confirmation_id: str, db: Session = Depends(get_db)):
    return bs.resend_confirmation(db, confirmation_id)


class DraftRequest(BaseModel):
    caller_phone: str
    call_id: str
    slots_json: str


@app.post("/api/v1/bookings/draft")
def save_draft_endpoint(req: DraftRequest, db: Session = Depends(get_db)):
    bs.save_draft(db, req.caller_phone, req.call_id, req.slots_json)
    return {"saved": True}


@app.get("/api/v1/bookings/draft")
def get_draft_endpoint(phone: str = Query(...), db: Session = Depends(get_db)):
    draft = bs.find_draft(db, phone)
    return {"found": draft is not None, "draft": draft}


# =============================================================================
# Epic E27: information and enquiry
# =============================================================================

@app.get("/api/v1/doctors/{doctor_name}/leave")
def doctor_leave_endpoint(doctor_name: str, date: str = Query(...), db: Session = Depends(get_db)):
    doctor = _find_doctor(db, doctor_name)
    if not doctor:
        return {"found": False, "query": doctor_name}
    leave = eq.doctor_leave_on(db, doctor.id, date)
    return {"found": True, "doctor_name": doctor.name, "on_leave": leave is not None, "leave": leave}


class PrepMergeRequest(BaseModel):
    test_names: list[str]


@app.post("/api/v1/tests/prep/merge")
def prep_merge_endpoint(req: PrepMergeRequest, db: Session = Depends(get_db)):
    ids, not_found = [], []
    for name in req.test_names:
        t = _find_test(db, name)
        (ids if t else not_found).append(t.id if t else name)
    result = eq.merge_prep_instructions(db, ids)
    result["not_found"] = not_found
    return result


@app.get("/api/v1/packages/{name}")
def package_endpoint(name: str, db: Session = Depends(get_db)):
    return eq.compare_package_vs_separate(db, name)


@app.get("/api/v1/walk-in")
def walk_in_endpoint(department: str | None = Query(None), test_name: str | None = Query(None),
                      db: Session = Depends(get_db)):
    dept_id = None
    if department:
        dept = db.query(Department).filter(Department.name.ilike(f"%{department}%")).first()
        dept_id = dept.id if dept else None
    test_id = None
    if test_name:
        t = _find_test(db, test_name)
        test_id = t.id if t else None
    return eq.walk_in_policy(db, department_id=dept_id, lab_test_id=test_id)


@app.get("/api/v1/billing/outstanding")
def billing_outstanding_endpoint(phone: str = Query(...), db: Session = Depends(get_db)):
    return eq.outstanding_balance(db, phone)


@app.get("/api/v1/home-collection/eligibility")
def home_collection_endpoint(test_name: str = Query(...), postal_code: str = Query(...),
                              db: Session = Depends(get_db)):
    t = _find_test(db, test_name)
    if not t:
        return {"found": False, "query": test_name}
    return eq.home_collection_eligibility(db, t.id, postal_code)


@app.get("/api/v1/insurance/coverage")
def insurance_coverage_endpoint(policy_number: str = Query(...), test_name: str | None = Query(None),
                                 db: Session = Depends(get_db)):
    test_id = None
    if test_name:
        t = _find_test(db, test_name)
        test_id = t.id if t else None
    return eq.check_insurance_coverage(db, policy_number, test_id)


@app.get("/api/v1/tests/{test_name}/prescription-requirement")
def prescription_requirement_endpoint(test_name: str, lang: str = Query("bn"), db: Session = Depends(get_db)):
    t = _find_test(db, test_name)
    if not t:
        return {"found": False, "query": test_name}
    return eq.prescription_requirement(db, t.id, lang)


class OutOfScopeRequest(BaseModel):
    call_id: str
    caller_question: str
    reason_code: str = "out_of_scope"


@app.post("/api/v1/calls/out-of-scope")
def out_of_scope_endpoint(req: OutOfScopeRequest, db: Session = Depends(get_db)):
    return eq.record_out_of_scope(db, req.call_id, req.caller_question, req.reason_code)


class CallbackRequestBody(BaseModel):
    phone: str
    call_id: str
    requested_window: str
    reason: str = ""


@app.post("/api/v1/callbacks")
def callback_endpoint(req: CallbackRequestBody, db: Session = Depends(get_db)):
    return eq.request_callback(db, req.phone, req.call_id, req.requested_window, req.reason)


@app.get("/api/v1/reports/status")
def report_status_endpoint(confirmation_id: str = Query(...), db: Session = Depends(get_db)):
    return eq.report_status(db, confirmation_id)


class ReportOTPRequest(BaseModel):
    confirmation_id: str
    phone: str


@app.post("/api/v1/reports/request-otp")
def report_request_otp_endpoint(req: ReportOTPRequest, db: Session = Depends(get_db)):
    return eq.request_report_otp(db, req.confirmation_id, req.phone)


class ReportDeliverRequest(BaseModel):
    confirmation_id: str
    otp_code: str


@app.post("/api/v1/reports/deliver")
def report_deliver_endpoint(req: ReportDeliverRequest, db: Session = Depends(get_db)):
    return eq.deliver_report(db, req.confirmation_id, req.otp_code)


@app.get("/api/v1/departments/{department_name}/hours")
def department_hours_endpoint(department_name: str, lang: str = Query("bn"), db: Session = Depends(get_db)):
    dept = db.query(Department).filter(Department.name.ilike(f"%{department_name}%")).first()
    if not dept:
        return {"found": False, "query": department_name}
    hours = eq.department_hours(db, dept.id, lang)
    return hours or {"found": False}
