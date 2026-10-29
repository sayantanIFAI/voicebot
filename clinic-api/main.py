"""Clinic data service -- implements the exact 3-endpoint contract
agent/tools_client.py in the voice agent already expects. Backed by
PostgreSQL, seeded with dummy departments/doctors/schedules/tests via
seed.py.

Resolving a name is deliberately simple and exact (written-form containment, see _find_test and
_find_doctor). "Who might the caller have meant" -- the suggestions read back to a caller who mispronounced
a test or a doctor -- comes from a phonetic-fold gazetteer with an index (gazetteer.py, KCD-096), built once
per catalogue and looked up, not scanned per request. A suggestion is never a resolution.
"""

from __future__ import annotations

import datetime

import clinic_clock  # the clinic's own (Kolkata) clock, not the server's
import difflib
import json
import logging
import os
import secrets
import time
import unicodedata

import agent_messages as am
import booking_service as bs
import enquiry_service as eq
import gazetteer as gz
import patient_context as pc
import registry as reg
from db import SessionLocal, get_db
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from idempotency import idempotent
from idempotency import set_key_from_header as set_idempotency_key
from models import FAQ, Appointment, AuditLog, Department, Doctor, DoctorSchedule, LabTest
from i18n_content import TEST_ALIASES_EN
import synonyms_extra as sx  # five more lay/clinical/loan-word names per test per language
import spelled_forms as sf  # every test code spelled out in Bengali/Hindi letter names, generated from the code
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

app = FastAPI(title="Kolkata Care Diagnostics -- Clinic Data API (dummy)")


@app.middleware("http")
async def require_service_token(request: Request, call_next):
    """Service-to-service authentication for everything under /api/v1/ (an external review: patient
    data was served to anyone who could reach the port).

    Enforced when CLINIC_API_TOKEN is set, or when CLINIC_API_REQUIRE_TOKEN=1 (then a missing token
    refuses every request -- fail closed). With neither, the API runs open as before and logs that
    at startup; that is only acceptable on a closed dev network. This is a shared secret between
    the orchestrator and this API: it authenticates the CALLER OF THE API, not the phone caller,
    and is not a substitute for verifying a patient's identity (agent/identity.py). Read per
    request so it can be rotated without a code change; compared in constant time."""
    token = os.environ.get("CLINIC_API_TOKEN", "")
    set_idempotency_key(request.headers.get("idempotency-key"))  # for the write endpoints (idempotency.py)
    if request.url.path.startswith("/api/v1/") and (token or os.environ.get("CLINIC_API_REQUIRE_TOKEN") == "1"):
        supplied = request.headers.get("authorization", "")
        if not token or not secrets.compare_digest(supplied.encode(), f"Bearer {token}".encode()):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


SLOT_STEP_MIN = int(os.environ.get("SLOT_STEP_MIN", "10"))  # owner, 2026-10-17: one booking per 10-minute slot (same env as booking_service.SLOT_STEP_MIN)


@app.on_event("startup")
def _warn_if_open():
    if not os.environ.get("CLINIC_API_TOKEN") and os.environ.get("CLINIC_API_REQUIRE_TOKEN") != "1":
        logging.getLogger("clinic-api").warning(
            "CLINIC_API_TOKEN is not set: /api/v1/* is served WITHOUT authentication. "
            "Acceptable only on a closed development network."
        )


@app.on_event("startup")
def _ensure_seeded():
    """Create the schema, and seed it only if it is EMPTY.

    The catalogue is derived data -- 9 departments, 36 doctors, 34 tests,
    all defined in seed.py -- so regenerating it costs nothing and removes
    the manual reseed step that every pod restart used to require. (A
    department added to seed.py AFTER this database was first seeded, e.g.
    Dentistry, is not covered by this guard -- see booking_migrate.
    seed_new_departments, called every boot regardless of emptiness.)

    Guarded on emptiness because seed() itself is destructive (drop_all
    then create_all). Running it unconditionally at startup would wipe
    every appointment booked since the last boot, turning a convenience
    into data loss.
    """
    import db as _db_module
    from db import engine
    from models import Base, LabTest

    if _db_module.USING_CLIENT_DB and not _db_module.CLIENT_DB_INIT:
        # sourav, 2026-10-24: the CLIENT's database is connected (client_database.env). Their data is read as it is: nothing is created, altered, seeded or dropped
        # unless CLIENT_DB_INIT=1 is set once on purpose. python tools/check_client_db.py says what their database still lacks.
        logging.getLogger("clinic-api").info("client database in use: schema and seed steps skipped (set CLIENT_DB_INIT=1 once to prepare it)")
        return
    Base.metadata.create_all(engine)
    from seed import add_i18n_columns

    add_i18n_columns()  # before ANY ORM query: an old database lacks the new columns

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
    from enquiry_migrate import add_callback_request_columns, add_enquiry_columns

    logging.getLogger("clinic-api").info("enquiry schema migration: %s", add_enquiry_columns())
    logging.getLogger("clinic-api").info("callback request schema migration: %s", add_callback_request_columns())

    db = SessionLocal()
    try:
        if db.query(LabTest).count() == 0:
            logging.getLogger("clinic-api").info("empty database -- seeding catalogue")
            from seed import seed

            seed()
        else:
            logging.getLogger("clinic-api").info("catalogue already present, not reseeding")
            from seed import backfill_doctor_aliases, backfill_faq_keywords, backfill_i18n, backfill_test_alias_keywords, backfill_test_days

            logging.getLogger("clinic-api").info("i18n backfill: %s", backfill_i18n(db))
            logging.getLogger("clinic-api").info("test alias keywords backfilled: %s", backfill_test_alias_keywords())
            logging.getLogger("clinic-api").info("test days backfilled: %s", backfill_test_days())  # sourav, 2026-10-24
            logging.getLogger("clinic-api").info("FAQ keywords backfilled: %s", backfill_faq_keywords())
            logging.getLogger("clinic-api").info("doctor aliases backfilled: %s", backfill_doctor_aliases())  # owner, 2026-10-27

        # Only now do departments/doctors/tests definitely have rows --
        # either seed() just created them, or they already existed.
        # Calling either of these any earlier seeds zero rows on a
        # brand-new database, the exact bug finish_booking_schema_setup's
        # own docstring documents.
        from booking_migrate import finish_booking_schema_setup

        logging.getLogger("clinic-api").info("booking schema setup (routes/fees): %s", finish_booking_schema_setup())
        import clinic_extras

        logging.getLogger("clinic-api").info("owner extras (sex, departments, routes, relationships): %s", clinic_extras.apply_extras(db))
        import geo_service

        logging.getLogger("clinic-api").info("geography tables (districts, police stations, blocks, places): %s inserted", geo_service.seed_geo(db))
        from patient_seed import seed_patient_context

        logging.getLogger("clinic-api").info("agent messages: %s inserted", am.seed_defaults(db))
        if os.environ.get("CLINIC_SEED_SAMPLE_PATIENTS", "1") == "1":
            logging.getLogger("clinic-api").info("sample patient data: %s", seed_patient_context(db))
        from enquiry_migrate import backfill_enquiry_facts, seed_enquiry_demo_data

        logging.getLogger("clinic-api").info("enquiry facts backfill: %s", backfill_enquiry_facts(db))
        logging.getLogger("clinic-api").info("enquiry demo data: %s", seed_enquiry_demo_data(db))
        import booking_log

        from db import engine as _engine

        logging.getLogger("clinic-api").info("booking_log view (sourav, 2026-10-24): %s", booking_log.ensure_view(_engine))
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
        "found": True,
        "test_name": t.name,
        "test_name_bn": _first_alias_bn(t.aliases_bn),
        "test_name_hi": _first_alias_bn(t.aliases_hi),
        "rate_inr": t.rate_inr,
        "sample_type": t.sample_type,
        "report_time_hours": t.report_time_hours,
        "prescription_required": bool(t.prescription_required),
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
            {
                "name": t.name,
                "aliases_bn": _with_spelled([a for a in (t.aliases_bn or "").split("|") if a], sf.forms_bn(t.name) + sx.extra(t.name, "bn")),
                "aliases_hi": _with_spelled([a for a in (t.aliases_hi or "").split("|") if a], sf.forms_hi(t.name) + sx.extra(t.name, "hi")),
                "aliases_en": _with_spelled(list(TEST_ALIASES_EN.get(t.name, [])), sx.extra(t.name, "en")),
                # The lay-term fast path (agent/lay_terms.py) reads this to answer "রক্ত পরীক্ষা" ("blood test") with
                # the REAL list of blood tests, off the same cached catalogue fetch every other lookup already uses --
                # never a second endpoint, never a hardcoded list that could drift from what the lab actually runs.
                "sample_type": t.sample_type,
            }
            for t in db.query(LabTest).all()
        ],
        "doctors": [
            {
                "name": d.name,
                "full_name": d.full_name or d.name,
                "full_name_bn": d.full_name_bn,
                "full_name_hi": d.full_name_hi,
                "surname": d.name.split()[-1],
                "department": d.department.name if d.department is not None else None,
                "aliases_bn": [a for a in (d.aliases_bn or "").split("|") if a],
                "aliases_hi": [a for a in (d.aliases_hi or "").split("|") if a],
            }
            for d in db.query(Doctor).all()
        ],
        # FAQ topics + their keyword sets, for FastPath.FAQCatalogue --
        # same "fetch the small table once, match locally" shape as tests
        # and doctors above. NOT the answer text itself: the answer is
        # still fetched live via /api/v1/faq on every turn, the same
        # discipline test_rate and doctor_availability already apply, so a
        # cached routing decision can never serve a stale FAQ answer.
        "faq_topics": [
            {
                "topic": f.topic,
                "keywords_bn": [k for k in (f.keywords_bn or "").split("|") if k],
                "keywords_hi": [k for k in (f.keywords_hi or "").split("|") if k],
                "keywords_en": [k for k in (f.keywords_en or "").split("|") if k],
            }
            for f in db.query(FAQ).all()
        ],
    }


# Words that carry no test identity when a caller wraps a test name in a
# sentence: "lipid profile TEST" must match "Lipid Profile". Bengali/Hindi
# spellings of "test" included.
_GENERIC_WORDS = {"test", "tests", "the", "a", "of", "for", "price", "rate", "টেস্ট", "টেস্টের", "टेस्ट", "जांच"}


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


def _with_spelled(aliases: list[str], spelled: list[str]) -> list[str]:
    return aliases + [s for s in spelled if s not in aliases]


def _alias_blob(t: LabTest) -> str:
    """sourav, 2026-10-14: every alias of a test as one "|"-joined string: Bengali + Hindi (database columns) + the
    English table in i18n_content.TEST_ALIASES_EN. Why: the name search used to build `t.aliases_bn + "|" +
    t.aliases_hi` in five places, so English lay words ("typhoid test") could never match; this is the one place that
    knows all three, so every place that matches a name sees the same forms."""
    return (
        (t.aliases_bn or "") + "|" + (t.aliases_hi or "") + "|" + "|".join(TEST_ALIASES_EN.get(t.name, []))
        + "|" + "|".join(sf.forms_bn(t.name) + sf.forms_hi(t.name) + sx.extra(t.name, "bn") + sx.extra(t.name, "hi") + sx.extra(t.name, "en"))
    )


def _find_test(db: Session, name: str) -> LabTest | None:
    """The exact / alias / containment cascade search_test() and the prep endpoint both need --
    factored out so "which test did they mean" has exactly one implementation.

    RESOLVES ONLY ON A WRITTEN-FORM MATCH. A sound-alike is never a resolution: "CBC" and "CRP"
    are one phoneme apart, and a price or a preparation instruction for the wrong one is a
    confident wrong answer that the database will happily supply. Phonetic matches come back
    only as suggestions (`_phonetic_test_matches`, `_test_suggestions`) for the caller to confirm."""
    exact = db.query(LabTest).filter(func.lower(LabTest.name).contains(name.lower())).first()
    if exact:
        return exact

    all_tests = db.query(LabTest).all()
    for t in all_tests:
        aliases = [a for a in _alias_blob(t).split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            return t

    # Normalised, two-way containment against the English name and every
    # alias: the caller's phrase may contain the test name ("lipid profile
    # test") or be a part of it, in any script.
    q = _norm_name(name)
    if q:
        for t in all_tests:
            forms = [t.name] + [a for a in _alias_blob(t).split("|") if a]
            for form in forms:
                n = _norm_name(form)
                if n and (q in n or n in q):
                    return t

    return None


# ------------------------------------------------------------ the gazetteer (KCD-096)

# The gazetteer is built from every row once and reused; this is how long a built one is trusted before the
# catalogue is read again. A change in the NUMBER of rows rebuilds it at once (see _catalogue_stamp); an edit to
# a single alias shows up within this many seconds. REASONED: catalogue edits are rare and a suggestion that lags
# an alias edit by half a minute costs nothing, whereas rebuilding on every request would be the scan again.
GAZETTEER_TTL_S = 30.0
_gazetteers: dict[tuple, tuple[float, tuple, gz.Gazetteer]] = {}


def invalidate_gazetteers() -> None:
    """Drop every built gazetteer. Call after writing catalogue rows or aliases so the next lookup sees them."""
    _gazetteers.clear()


def _catalogue_stamp(db: Session, model) -> tuple:
    return (db.query(func.count(model.id)).scalar(), db.query(func.max(model.id)).scalar())


def _test_forms(t: LabTest) -> list[str]:
    """Every way a test is written or said: its name, the name without its bracket, the bracketed code
    ("Complete Blood Count (CBC)" is called "CBC"), and each Bengali and Hindi alias."""
    forms = [t.name]
    if "(" in t.name and ")" in t.name:
        forms += [t.name.split("(")[0].strip(), t.name.split("(")[1].split(")")[0].strip()]
    forms += [a for a in _alias_blob(t).split("|") if a.strip()]
    return [f for f in forms if f]


def _doctor_forms(d: Doctor) -> list[str]:
    """A doctor is called by surname or by an alias; the initials in the formal name ("Dr. S. Mukherjee")
    are not something a caller says and only add noise to a similarity comparison."""
    forms = [d.name.split()[-1]] + [a for a in (d.aliases_bn + "|" + (d.aliases_hi or "")).split("|") if a.strip()]
    # The whole name in every script, so "Ashok Sen" / "অশোক সেন" reaches one doctor and not every Sen.
    forms += [f for f in (_strip_title(d.full_name), d.full_name_bn, d.full_name_hi) if f]
    return forms


def _strip_title(name: str) -> str:
    return " ".join(w for w in (name or "").split() if w.strip(".,").lower() not in _TITLE_WORDS)


def _gazetteer(db: Session, kind: str) -> gz.Gazetteer:
    model, forms_of, drop = (
        (LabTest, _test_forms, _GENERIC_WORDS) if kind == "test" else (Doctor, _doctor_forms, _TITLE_WORDS)
    )
    key = (kind, id(db.get_bind()))
    stamp = _catalogue_stamp(db, model)
    now = time.monotonic()
    cached = _gazetteers.get(key)
    if cached and cached[1] == stamp and now - cached[0] < GAZETTEER_TTL_S:
        return cached[2]
    entries = [(row.name, form) for row in db.query(model).all() for form in forms_of(row)]
    built = gz.Gazetteer(entries, drop_words=drop)
    _gazetteers[key] = (now, stamp, built)
    return built


def _phonetic_test_matches(db: Session, name: str) -> list[LabTest]:
    """Tests that sound like `name`, from the gazetteer's sound tiers (KCD-434/096): a romanised Bengali or
    Hindi spelling, or a loanword written in another script. Suggestions only -- what the caller MIGHT have
    meant; the agent asks, it does not act on them."""
    names = [s.canonical for s in _gazetteer(db, "test").suggest(name, limit=3) if s.basis.startswith("sound")]
    by_name = {t.name: t for t in db.query(LabTest).filter(LabTest.name.in_(names)).all()} if names else {}
    return [by_name[n] for n in names if n in by_name]


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
        aliases = [a for a in _alias_blob(t).split("|") if a]
        if any(name in alias or alias in name for alias in aliases):
            alias_matches.append(t)
    if alias_matches:
        return alias_matches

    q = _norm_name(name)
    norm_matches = []
    if q:
        for t in all_tests:
            forms = [t.name] + [a for a in _alias_blob(t).split("|") if a]
            if any(_norm_name(f) and (q in _norm_name(f) or _norm_name(f) in q) for f in forms):
                norm_matches.append(t)
    if norm_matches:
        return norm_matches

    return []


def _first_alias(csv: str | None) -> str | None:
    return next((a for a in (csv or "").split("|") if a.strip()), None)


def _scripted(db: Session, body: dict, model) -> dict:
    """Add the same suggestions in Bengali and Hindi script (`did_you_mean_bn`, `did_you_mean_hi`).
    A Bengali or Hindi voice cannot say a Latin name, so the reply for that language uses these; the
    English canonical names stay in `did_you_mean`. A single suggestion that is not an exact match is
    marked `needs_confirmation` so the agent asks and follows through on a yes."""
    names = body.get("did_you_mean") or []
    if not names:
        return body
    rows = {r.name: r for r in db.query(model).filter(model.name.in_(names)).all()}
    body["did_you_mean_bn"] = [_first_alias(rows[n].aliases_bn) or n for n in names if n in rows]
    body["did_you_mean_hi"] = [_first_alias(rows[n].aliases_hi) or n for n in names if n in rows]
    if len(names) == 1 and not body.get("ambiguous"):
        body["needs_confirmation"] = True
    return body


def _test_suggestions(db: Session, name: str) -> list[str]:
    return [sg.canonical for sg in _gazetteer(db, "test").suggest(name, limit=3)]


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
        return _scripted(
            db,
            {"found": False, "query": name, "ambiguous": True, "did_you_mean": [t.name for t in candidates[:3]]},
            LabTest,
        )

    found = candidates[0] if candidates else _find_test(db, name)
    if found:
        return _test_reply_dict(found)

    # Fuzzy fallback -- try both the English name and every Bengali alias,
    # so suggestions are useful regardless of which script the caller used.
    return _scripted(db, {"found": False, "query": name, "did_you_mean": _test_suggestions(db, name)}, LabTest)


@app.get("/api/v1/tests/availability")
def test_availability(name: str = Query(...), date: str | None = Query(None), db: Session = Depends(get_db)):
    """sourav, 2026-10-24: the weekdays a lab test is done on (models.LabTest.available_days), and -- when `date` (ISO) is given -- whether it is done on that date and the
    next date it is. Why: the voice agent answers "is the CT scan done on Thursday?" / "which days?" from this, never from the model. Flow: the same name matching as
    /tests/search (two equally good matches = ambiguous, none = not found with suggestions); then the column is read through clinic-api/lab_days.py."""
    import lab_days

    candidates = _find_test_candidates(db, name)
    if len(candidates) > 1:
        return _scripted(db, {"found": False, "query": name, "ambiguous": True, "did_you_mean": [t.name for t in candidates[:3]]}, LabTest)
    found = candidates[0] if candidates else _find_test(db, name)
    if not found:
        return _scripted(db, {"found": False, "query": name, "did_you_mean": _test_suggestions(db, name)}, LabTest)
    days = lab_days.parse_days(found.available_days)
    out = {
        **_test_reply_dict(found),
        "available_days": days,
        "closed_days": lab_days.closed_days(days),
        "every_day": len(days) == 7,
        "date": None,
        "available_on_date": None,
        "next_available_date": None,
    }
    if date:
        try:
            when = datetime.date.fromisoformat(date)
        except ValueError:
            raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD")
        out["date"] = when.isoformat()
        out["available_on_date"] = lab_days.is_open_on(days, when)
        out["next_available_date"] = lab_days.next_open_date(days, when, include_from=False).isoformat() if not out["available_on_date"] else when.isoformat()
    return out


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
        "found": True,
        "test_name": t.name,
        "test_name_bn": _first_alias_bn(t.aliases_bn),
        "test_name_hi": _first_alias_bn(t.aliases_hi),
        "fasting_required": t.fasting_required,
        "lang": lang,
        "prep_instructions": _by_lang(lang, t.prep_instructions_bn, t.prep_instructions_hi, t.prep_instructions_en),
    }


@app.get("/api/v1/tests/prep")
def test_prep(name: str = Query(...), lang: str = Query("bn"), db: Session = Depends(get_db)):
    """Same entity resolution as /tests/search, different fact -- kept as a
    separate endpoint rather than folding prep fields into every search
    response, because prep instructions are Tier-3 "approved content"
    (Blueprint 2.2) conceptually distinct from the Tier-1 price/turnaround
    facts search_test returns, and the two may end up backed by different
    systems of record later."""
    candidates = _find_test_candidates(db, name)  # KCD-446: see search_test's own comment
    if len(candidates) > 1:
        return _scripted(
            db,
            {"found": False, "query": name, "ambiguous": True, "did_you_mean": [t.name for t in candidates[:3]]},
            LabTest,
        )

    found = candidates[0] if candidates else _find_test(db, name)
    if found:
        return _test_prep_reply_dict(found, lang)
    return _scripted(db, {"found": False, "query": name, "did_you_mean": _test_suggestions(db, name)}, LabTest)


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


# Honorifics stripped from what the caller said before it is compared with a doctor's name.
_TITLE_WORDS = frozenset({"dr", "doctor", "doc", "ডাক্তার", "ডক্টর", "ডাঃ", "डॉक्टर", "डॉ", "डाक्टर"})

# Owner, 2026-10-28: "Sir" is said before a name, after it, after the full name, or beside "Dr": "Dr Amit Sir", "Amit Sir", "Amit Kumar Sir", "Amit Kumar Saha Sir", "Sir Amit"; in Bengali "স্যার/সার/সির",
# in Hindi "सर/सिर", and the recogniser spells it any of these ways. None of them is part of a name. Stripped from what was said wherever it stands (never when nothing else is left).
_HONORIFICS = frozenset(
    "sir sar saar sahab saheb sahib sab madam maam mam ji babu mahashay mahasay "
    "স্যার স্যর সার সির সাহেব সাব ম্যাডাম ম্যাম বাবু মহাশয় জি "
    "सर सिर साहब साहेब सहाब मैडम मैम बाबू महोदय जी".split()
)


def _bare_word(w: str) -> str:
    return w.strip(".,!?।").replace("\u094d", "").replace("\u09cd", "").lower()  # no dots, no halant


def _strip_honorifics(name: str) -> str:
    """`name` without titles ("Dr", "ডাক্তার", "डॉक्टर") and honorifics ("Sir", "স্যার", "सर") wherever they stand. The original is returned when nothing else would be left."""
    stop = {_bare_word(w) for w in _HONORIFICS | _TITLE_WORDS}
    kept = [w for w in (name or "").split() if _bare_word(w) and _bare_word(w) not in stop]
    return " ".join(kept) if kept else (name or "")


# A name fragment shorter than this is not matched as a substring of a doctor's name (REASONED).
MIN_SUBSTRING_CHARS = 3


def _bn_vowel_variants(s: str) -> set[str]:
    """Real call 2026-10-12: "ডাক্তার মুখার্জির সাথে" ("with Doctor Mukherjee's...") was extracted as doctor_name
    "মুখার্জি" -- the LLM copies the caller's words verbatim (SLOT RULES: never normalize), and Bengali's own
    possessive declension shortens a name stem's final "ী" to "ি" before the (here already-stripped) case
    suffix "-র" -- but the doctor was seeded under the dictionary surname "মুখার্জী" (long ী). Same length, one
    character different at the very end, so neither direction's substring check in the alias loop below ever
    matched, and a real, unambiguous doctor came back "not found". Trying both vowel forms of the WORD'S LAST
    character is still an exact, character-for-character written-form comparison afterwards (CLAUDE.md "Doctor
    Nobody": never a sound-alike guess) -- it just stops a single grammatical vowel shortening from hiding an
    otherwise-exact match."""
    out = {s}
    if s.endswith("ি"):  # ি (short i)
        out.add(s[:-1] + "ী")  # ী (long i)
    elif s.endswith("ী"):
        out.add(s[:-1] + "ি")
    # Real call 2026-10-18: the recogniser wrote "বিশ্বজিত্ সেন" (ত + hasanta) for the seeded "বিশ্বজিৎ সেন" (the letter ৎ): the same sound, a different
    # code point, so the whole name was not recognised and "Arindam or Biswajit?" was asked. Both spellings are tried for a word-final ত্/ৎ.
    import re as _re

    for v in list(out):
        out.add(_re.sub(r"ত্(?=\s|$)", "ৎ", v))
        out.add(v.replace("ৎ", "ত্"))
    return out


def _doctor_exact_matches(db: Session, name: str) -> list[Doctor]:
    """Doctors whose WRITTEN name or alias contains what the caller said ("Sen", "Dr Sen", a Bengali
    or Hindi alias). Several matches are returned as several: choosing among them is the caller's job."""
    name = _strip_honorifics(name)
    needle_tokens = [w for w in (x.strip(".,") for x in (name or "").lower().split()) if w and w not in _TITLE_WORDS]
    needle = " ".join(needle_tokens)
    if not needle:
        return []
    doctors = db.query(Doctor).all()
    name_variants = _bn_vowel_variants(name or "")

    def tokens(d):
        # the short name's words and the whole name's words ("dr", "a", "sen" and "ashok"): both are written forms
        return [t.strip(".,").lower() for t in (d.name + " " + (d.full_name or "")).split()]

    # A whole name token that IS what was said ("Sen" is Dr. Sen's surname) outranks a mere substring
    # of another name ("Sen" inside "Sengupta"): it is the written form, not a coincidence of letters.
    whole = [d for d in doctors if all(w in tokens(d) for w in needle_tokens)]
    if whole:
        return whole
    # The whole name in Bengali or Devanagari ("অশোক সেন") is more specific than the surname alias every Sen shares:
    # when it was said in full, that doctor and no other.
    said_in_full = [
        d
        for d in doctors
        if any(fn and any(fn in nv for nv in name_variants) for fn in (d.full_name_bn, d.full_name_hi))
    ]
    if said_in_full:
        return said_in_full
    # sourav, 2026-10-14: an alias that IS what was said ("সেন" / "सेन" = Dr. Sen) outranks an alias that merely CONTAINS it
    # ("সেনগুপ্ত" = Dr. Sengupta). Same rule as the Latin-script "whole token" check above; without it every Bengali/Hindi "Sen"
    # question came back "which one -- Sen or Sengupta?" although the caller had named Dr. Sen exactly. A surname two doctors
    # really share (রায়) still matches both here, so it still asks "which one?".
    exact_alias = [
        d
        for d in doctors
        if any(a.strip() and a.strip() in name_variants for a in (d.aliases_bn + "|" + (d.aliases_hi or "")).split("|"))
    ]
    if exact_alias:
        return exact_alias
    out = []
    for d in doctors:
        aliases = [
            a
            for a in (
                d.aliases_bn + "|" + (d.aliases_hi or "") + "|" + (d.full_name_bn or "") + "|" + (d.full_name_hi or "")
            ).split("|")
            if a
        ]
        # a one- or two-letter fragment ("ry") sits inside far too many names to identify any of them
        if (len(needle) >= MIN_SUBSTRING_CHARS and needle in d.name.lower()) or any(
            (nv in a) or (a in nv and _given_name_fits(d, nv, a)) for a in aliases for nv in name_variants
        ):
            out.append(d)
    return out


_NAME_NOISE = frozenset("সির সার সাব সাহেব স্যর sir सिर सर साहब ডাঃ ডা ডাক্তার ডক্টর ডাক্তর স্যার ম্যাডাম বাবু কে কেই এর র কে দেখাব দেখাবো দেখাতে ডা. डॉ डॉक्टर जी साहब से को".split())


def _given_name_fits(d: Doctor, said: str, alias: str) -> bool:
    """Real call 2026-10-28: "শুভদীপা মণ্ডল" was answered as Dr. Anirban Mondal because the surname alias "মণ্ডল" sits INSIDE what was said. The surname alone matching does not make a
    doctor the one named: when the caller said MORE than the surname, every other word (the given name) must be a near-match for a word of this doctor's own name, otherwise this is a doctor
    that does not exist, and the caller hears "no doctor of that name" with the real suggestions (Doctor Nobody)."""
    def _bare(w: str) -> str:
        return w.strip(".,").replace("्", "").replace("্", "").lower()  # no dots, no halant: "डॉक्टर्" is "डॉक्टर"

    noise = {_bare(n) for n in _NAME_NOISE}
    extra = [
        w
        for w in said.replace(alias, " ").split()
        if _bare(w) and _bare(w) not in noise and not any(difflib.SequenceMatcher(None, _bare(w), n).ratio() >= 0.8 for n in noise if len(n) >= 3)
    ]
    if not extra:
        return True
    own = [t.strip(".,").lower() for t in ((d.full_name_bn or "") + " " + (d.full_name_hi or "") + " " + d.name + " " + (d.full_name or "")).split()]
    own = [t for t in own if t]
    return all(any(difflib.SequenceMatcher(None, w.lower(), t).ratio() >= 0.72 for t in own) for w in extra)


def _doctor_choices(db: Session, name: str, doctors: list[Doctor], near: bool = False) -> dict:
    """The "which one?" answer: every doctor in the running, each named so the caller can tell them apart -- by the
    WHOLE name (and in Bengali / Devanagari script for a voice that cannot say a Latin one), never by the short name two
    doctors can share ("Dr. A. Sen" for Ashok Sen and Abhishek Sen). If two still read the same, their department is
    added. Ordered by id so the question is the same every time it is asked."""
    ordered = sorted(doctors, key=lambda d: d.id)[:4]
    depts = {d.id: (db.get(Department, d.department_id).name if d.department_id else "") for d in ordered}

    def label(d: Doctor, script: str) -> str:
        if script == "bn":
            return d.full_name_bn or _first_alias_bn(d.aliases_bn) or d.full_name or d.name
        if script == "hi":
            return d.full_name_hi or _first_alias_bn(d.aliases_hi) or d.full_name or d.name
        return d.full_name or d.name

    body = {"found": False, "query": name, "ambiguous": True}
    for script, key in (("en", "did_you_mean"), ("bn", "did_you_mean_bn"), ("hi", "did_you_mean_hi")):
        labels = [label(d, script) for d in ordered]
        if len(set(labels)) < len(labels):  # still alike: say which department
            labels = [f"{lab} ({depts[d.id]})" if depts[d.id] else lab for lab, d in zip(labels, ordered)]
        body[key] = labels
    if near:
        body["needs_confirmation"] = True
    return body


def _doctor_ref(db: Session, d: Doctor) -> str:
    """What a reply calls this doctor: the short name, unless another doctor has the same short name ("Dr. A. Sen" twice),
    in which case the whole name -- a name that can be looked up again must point at one doctor."""
    # Owner, 2026-10-17: ALWAYS the whole name ("Dr. Arindam Sen"), even when only one doctor has that surname -- the caller learns who they are booking.
    return d.full_name or d.name


def _ambiguous_doctor(db: Session, name: str) -> dict | None:
    """The "which one?" body when the name fits more than one doctor (two Sens, "Dr. A. Sen" for two A. Sens), else None."""
    matches = _doctor_exact_matches(db, name)
    return _doctor_choices(db, name, matches) if len(matches) > 1 else None


def _find_doctor(db: Session, name: str) -> Doctor | None:
    """The ONE doctor the caller unambiguously named, or None.

    Resolves only on a unique written-form match. A fuzzy or phonetic near-match is NEVER a
    resolution (OpenAI review; CLAUDE.md "Doctor Nobody"): it answers "who might they have meant",
    not "who is it", and reading a different doctor's real schedule is worse than asking again.
    Those come back through `_doctor_suggestions` and the caller confirms. Two doctors matching
    the same words (e.g. "Sen" for Dr. Sen and Dr. Sengupta) is also None here -- the availability
    endpoint reports it as ambiguous instead of picking whichever the database returned first."""
    matches = _doctor_exact_matches(db, name)
    return matches[0] if len(matches) == 1 else None


# Same ratio floor already used to ACCEPT a single fuzzy match above
# (FUZZY_SURNAME_FLOOR) -- reused here as the bar for COLLECTING a name
# into the ambiguity set, so "which doctors are even in the running" and
# "would _find_doctor have accepted this one alone" stay the same
# question asked twice, never two different bars that could disagree.
def _find_doctor_candidates(db: Session, name: str) -> list[Doctor]:
    """Every doctor whose surname or alias clears FUZZY_SURNAME_FLOOR against `name`, not just the single best
    one -- so two similarly-spelled doctors surface as a genuine "which one" choice (KCD-446's doctor-side
    counterpart) instead of one being picked because it scored a hair higher. Exact/alias matches never reach
    this function -- _doctor_exact_matches already returned on those.

    The floor and the "within a hair of the top" rule are unchanged; what changed (KCD-096) is which doctors
    they are applied to: the gazetteer's shortlist, not every row. The floor is a character similarity of at
    least 0.60, and every form that clears it shares character pairs with the query, so it is on the shortlist."""
    name = _strip_honorifics(name)
    shortlisted = {c for c, _f, _r in _gazetteer(db, "doctor").neighbours(name)}
    if not shortlisted:
        return []
    scored = []
    for d in db.query(Doctor).filter(Doctor.name.in_(shortlisted)).all():
        candidates = _doctor_forms(d)
        best_ratio = max(
            (difflib.SequenceMatcher(None, name.lower(), c.lower()).ratio() for c in candidates),
            default=0.0,
        )
        if best_ratio >= FUZZY_SURNAME_FLOOR:
            scored.append((best_ratio, d))
    if not scored:
        return []
    top = max(r for r, _ in scored)
    # Only doctors within a hair of the top score are genuinely "in the running" -- a query that clearly
    # favours one doctor over another is not an ambiguous case, it is a confident match with a distant runner-up.
    return [d for r, d in scored if top - r <= 0.05]


def _looks_like_a_name(candidate: str) -> bool:
    """CLAUDE.md "Doctor Nobody" guard, at the data layer: a suggestion is spoken aloud as "did you mean
    <this>?" straight from the Doctor.name column, so a stray non-name value edited into that column (a
    qualification string, a pasted fragment) must never reach a caller's ear just because it cleared the
    fuzzy-match floor. Real doctor names here are always properly-cased ("Dr. Samir Sen"); a token that is
    not capitalised is never one, and is the cheapest signal that the column holds something else."""
    words = [w for w in _strip_title(candidate).split() if w]
    return bool(words) and all(w[:1].isupper() for w in words)


def _doctor_suggestions(db: Session, name: str) -> list[str]:
    """Who the caller MIGHT have meant: sound-alikes (including one- and two-consonant surnames such as Sen,
    Das, Roy) and near-spellings, best first (KCD-436/096). Never acted on -- see _find_doctor."""
    name = _strip_honorifics(name)
    return [sg.canonical for sg in _gazetteer(db, "doctor").suggest(name, limit=3) if _looks_like_a_name(sg.canonical)]


def _schedule_for_weekday(db: Session, doctor_id: int, weekday: int) -> DoctorSchedule | None:
    return db.query(DoctorSchedule).filter_by(doctor_id=doctor_id, weekday=weekday).first()


def _sched_still_open(sched: DoctorSchedule, d: datetime.date) -> bool:
    """False only for TODAY, once the chamber's own end time has already passed -- same class of bug
    booking_service.available_slots() was already fixed for ("today at 10am" stayed offerable at 3pm), never
    applied to this endpoint. Real call 2026-10-24, evening: "Dr. X sits today, 10:00-12:00" was still said
    well after 12:00, instead of going straight to the next real day."""
    if d != clinic_clock.now().date():
        return True
    h, m = sched.end_time.split(":")
    return (int(h) * 60 + int(m)) > (clinic_clock.now().hour * 60 + clinic_clock.now().minute)


def _next_available_date(db: Session, doctor_id: int, from_date: datetime.date, horizon_days: int = 14) -> str | None:
    for offset in range(horizon_days):
        d = from_date + datetime.timedelta(days=offset)
        sched = _schedule_for_weekday(db, doctor_id, d.weekday())
        if sched and _sched_still_open(sched, d):
            return d.isoformat()
    return None


@app.get("/api/v1/doctors/availability")
def doctor_availability(name: str = Query(...), date: str | None = Query(None), db: Session = Depends(get_db)):
    # Doctor-side counterpart of KCD-446: two similarly-spelled doctors
    # tying on the fuzzy floor is a genuinely different outcome from
    # "nothing matched" -- silently picking one (what _find_doctor does,
    # correctly, when there is no tie) would risk reading out a DIFFERENT
    # doctor's real schedule, the CLAUDE.md "Doctor Nobody" class of bug.
    exact = _doctor_exact_matches(db, name)
    if len(exact) > 1:
        return _doctor_choices(db, name, exact)
    doctor = exact[0] if exact else None
    if not doctor:
        near = _find_doctor_candidates(db, name)
        if len(near) > 1:
            return _doctor_choices(db, name, near, near=True)
        return _scripted(
            db,
            {"found": False, "query": name, "did_you_mean": _doctor_suggestions(db, name), "needs_confirmation": True},
            Doctor,
        )

    today = clinic_clock.today()

    if date:
        try:
            target = datetime.date.fromisoformat(date)
        except ValueError:
            return {"found": False, "query": name}
        sched = _schedule_for_weekday(db, doctor.id, target.weekday())
        if sched and _sched_still_open(sched, target):
            return {
                "found": True,
                "doctor_name": _doctor_ref(db, doctor),
                "doctor_name_bn": (doctor.full_name_bn or _first_alias_bn(doctor.aliases_bn)),
                "doctor_name_hi": (doctor.full_name_hi or _first_alias_bn(doctor.aliases_hi)),
                "date": target.isoformat(),
                "available": True,
                "chamber_hours": f"{sched.start_time}-{sched.end_time}",
                "next_available_date": None,
            }
        next_date = _next_available_date(db, doctor.id, target + datetime.timedelta(days=1))
        return {
            "found": True,
            "doctor_name": _doctor_ref(db, doctor),
            "doctor_name_bn": (doctor.full_name_bn or _first_alias_bn(doctor.aliases_bn)),
            "doctor_name_hi": (doctor.full_name_hi or _first_alias_bn(doctor.aliases_hi)),
            "date": target.isoformat(),
            "available": False,
            "chamber_hours": None,
            "next_available_date": next_date,
        }

    # No date given -> "when is this doctor next available"
    next_date = _next_available_date(db, doctor.id, today)
    if not next_date:
        return {
            "found": True,
            "doctor_name": _doctor_ref(db, doctor),
            "doctor_name_bn": (doctor.full_name_bn or _first_alias_bn(doctor.aliases_bn)),
            "doctor_name_hi": (doctor.full_name_hi or _first_alias_bn(doctor.aliases_hi)),
            "date": None,
            "available": False,
            "chamber_hours": None,
            "next_available_date": None,
        }
    sched = _schedule_for_weekday(db, doctor.id, datetime.date.fromisoformat(next_date).weekday())
    return {
        "found": True,
        "doctor_name": _doctor_ref(db, doctor),
        "doctor_name_bn": (doctor.full_name_bn or _first_alias_bn(doctor.aliases_bn)),
        "doctor_name_hi": (doctor.full_name_hi or _first_alias_bn(doctor.aliases_hi)),
        "date": next_date,
        "available": True,
        "chamber_hours": f"{sched.start_time}-{sched.end_time}",
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
@idempotent("appointments")
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
    amb = _ambiguous_doctor(db, req.doctor_name)
    if amb is not None:
        return {
            "success": False,
            "reason": "doctor_ambiguous",
            **{k: v for k, v in amb.items() if k.startswith("did_you_mean")},
        }
    doctor = _find_doctor(db, req.doctor_name)
    if not doctor:
        return {
            "success": False,
            "reason": "doctor_not_found",
            "did_you_mean": _doctor_suggestions(db, req.doctor_name),
        }

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
        return {
            "success": False,
            "reason": "slot_taken",
            "alternative_slots": bs.available_slots(db, doctor.id, req.date)[:3],
        }

    # 8 hex chars, not 4 -- see booking_service._confirmation_id's comment:
    # this ID is now also accepted by /api/v1/bookings/lookup, unauthenticated.
    result = bs.confirm_booking(
        db, hold["hold_token"], doctor.id, req.date, req.time_slot, req.patient_name, req.phone, req.phone
    )
    if not result["success"]:
        return {
            "success": False,
            "reason": "slot_taken",
            "alternative_slots": bs.available_slots(db, doctor.id, req.date)[:3],
        }

    # sourav, 2026-10-24: no booking_log.append() here -- bs.confirm_booking() already logs it (booking_service.py);
    # a second call here was a real duplicate-row bug caught live (two identical lines per booking in booking_log.csv).
    return {
        "success": True,
        "confirmation_id": result["confirmation_id"],
        "doctor_name": _doctor_ref(db, doctor),
        "doctor_name_bn": (doctor.full_name_bn or _first_alias_bn(doctor.aliases_bn)),
        "doctor_name_hi": (doctor.full_name_hi or _first_alias_bn(doctor.aliases_hi)),
        "date": req.date,
        "time_slot": req.time_slot,
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
    return {
        "found": True,
        "topic": row.topic,
        "lang": lang,
        "answer": _by_lang(lang, row.answer_bn, row.answer_hi, row.answer_en),
    }


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
    if target < clinic_clock.today():
        # KCD-363: never book a past date, and never silently roll it
        # forward -- the caller is told plainly (main.py's reply template)
        # and offered the same weekday next week.
        return {"success": False, "reason": "date_in_past"}
    return None


# sourav, 2026-10-24 (owner): a doctor appointment can be taken at most 3 days ahead -- today and the next three days; a later date is refused and the caller is asked to call
# again nearer the day. Applies to booking (hold) and rescheduling (both go through _validate_doctor_slot); lab tests are not affected. BOOKING_MAX_ADVANCE_DAYS changes it.
MAX_ADVANCE_DAYS = int(os.environ.get("BOOKING_MAX_ADVANCE_DAYS", "3"))


def _first_sitting_next_month(db: Session | None, doctor: Doctor | None) -> dict | None:
    """Owner, 2026-10-28: a caller who asks for a date in NEXT month is told the first day next month this doctor sits (and the hours), nothing more. The first date on or after the 1st
    of next month with a schedule row and no leave. None when there is no doctor or no such day within 62 days."""
    if db is None or doctor is None:
        return None
    today = clinic_clock.today()
    first = (today.replace(day=1) + datetime.timedelta(days=32)).replace(day=1)
    from models import DoctorLeave

    leaves = db.query(DoctorLeave).filter_by(doctor_id=doctor.id).all()
    for i in range(62):
        day = first + datetime.timedelta(days=i)
        sched = _schedule_for_weekday(db, doctor.id, day.weekday())
        if not sched:
            continue
        if any(lv.start_date <= day.isoformat() <= lv.end_date for lv in leaves):
            continue
        return {"date": day.isoformat(), "start": sched.start_time, "end": sched.end_time}
    return None


def _validate_not_too_far_ahead(date_str: str, db: Session | None = None, doctor: Doctor | None = None) -> dict | None:
    """sourav, 2026-10-24: the booking-window rule. Returns the refusal (reason "too_far_ahead", with the limit and the last date that can be booked today) or None. Flow: the
    date is compared with today + MAX_ADVANCE_DAYS; a past or unparseable date is not this check's business (the check before it already answered)."""
    try:
        target = datetime.date.fromisoformat(date_str)
    except ValueError:
        return None
    last = clinic_clock.today() + datetime.timedelta(days=MAX_ADVANCE_DAYS)
    if target > last:
        out = {"success": False, "reason": "too_far_ahead", "max_advance_days": MAX_ADVANCE_DAYS, "last_bookable_date": last.isoformat()}
        today = clinic_clock.today()
        if (target.year, target.month) > (today.year, today.month) and (target.year * 12 + target.month) - (today.year * 12 + today.month) == 1:
            nxt = _first_sitting_next_month(db, doctor)
            if nxt:
                out["next_month_first"] = nxt
        return out
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
    err = _validate_not_too_far_ahead(date_str, db, doctor)
    if err:
        return err
    target = datetime.date.fromisoformat(date_str)
    sched = _schedule_for_weekday(db, doctor.id, target.weekday())
    if not sched:
        return {"success": False, "reason": "doctor_not_available_that_day"}
    valid_slots = _generate_slots(sched.start_time, sched.end_time)
    if time_slot not in valid_slots:
        return {"success": False, "reason": "invalid_slot", "valid_slots": valid_slots}
    if target == clinic_clock.today():
        # CodeRabbit-flagged, real bug: _validate_date_not_past is
        # DATE-level only ("target < today"), so a same-day slot earlier
        # than right now was still requestable and holdable directly
        # (available_slots() no longer OFFERS it -- see that function's
        # own fix -- but nothing stopped a caller from asking for it by
        # name anyway).
        now = clinic_clock.now()
        try:
            slot_h, slot_m = (int(x) for x in time_slot.split(":"))
        except ValueError:
            return {"success": False, "reason": "invalid_slot", "valid_slots": valid_slots}
        if slot_h * 60 + slot_m <= now.hour * 60 + now.minute:
            return {
                "success": False,
                "reason": "invalid_slot",
                "valid_slots": [
                    s
                    for s in valid_slots
                    if int(s.split(":")[0]) * 60 + int(s.split(":")[1]) > now.hour * 60 + now.minute
                ],
            }
    return None


@app.post("/api/v1/bookings/hold")
@idempotent("bookings.hold")
def hold_booking(req: HoldRequest, db: Session = Depends(get_db)):
    amb = _ambiguous_doctor(db, req.doctor_name)
    if amb is not None:
        return {
            "success": False,
            "reason": "doctor_ambiguous",
            **{k: v for k, v in amb.items() if k.startswith("did_you_mean")},
        }
    doctor = _find_doctor(db, req.doctor_name)
    if not doctor:
        return {
            "success": False,
            "reason": "doctor_not_found",
            "did_you_mean": _doctor_suggestions(db, req.doctor_name),
        }
    err = _validate_doctor_slot(db, doctor, req.date, req.time_slot)
    if err:
        return err

    result = bs.hold_slot(db, doctor.id, req.date, req.time_slot)
    if not result["success"]:
        nearest = bs.nearest_alternatives(db, doctor.id, req.date, req.time_slot)
        result["alternative_slots"] = [a["time_slot"] for a in nearest]
        # sourav, 2026-10-16: the same offers WITH their dates (one may be on another day), so "the first one" said by the caller
        # maps to the exact slot the clinic offered. Additive: alternative_slots is unchanged for every existing reader.
        result["alternative_options"] = [{"date": a["date"], "time_slot": a["time_slot"]} for a in nearest]
    else:
        result["doctor_id"] = doctor.id
        result["doctor_name"] = _doctor_ref(db, doctor)
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
    patient_address: str = ""
    patient_sex: str = ""


@app.post("/api/v1/bookings/confirm")
@idempotent("bookings.confirm")
def confirm_booking_endpoint(req: ConfirmRequest, db: Session = Depends(get_db)):
    result = bs.confirm_booking(
        db,
        req.hold_token,
        req.doctor_id,
        req.date,
        req.time_slot,
        req.patient_name,
        req.phone,
        req.caller_phone,
        req.patient_age,
        req.relationship,
        req.patient_address,
        req.patient_sex,
    )
    # sourav, 2026-10-24: no booking_log.append() here -- bs.confirm_booking() already logs it (booking_service.py);
    # a second call here was a real duplicate-row bug caught live (two identical lines per booking in booking_log.csv).
    return result


class RescheduleRequest(BaseModel):
    confirmation_id: str
    new_date: str
    new_time_slot: str
    caller_phone: str | None = None  # a change needs the id AND a phone number that belongs to the booking (booking_service.check_change_authorized)
    call_id: str | None = None


@app.post("/api/v1/bookings/reschedule")
@idempotent("bookings.reschedule")
def reschedule_booking(req: RescheduleRequest, db: Session = Depends(get_db)):
    denied = bs.check_change_authorized(db, req.confirmation_id, req.caller_phone, req.call_id)
    if denied:
        return denied
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
    caller_phone: str | None = None  # see RescheduleRequest
    call_id: str | None = None


@app.post("/api/v1/bookings/cancel")
@idempotent("bookings.cancel")
def cancel_booking(req: CancelRequest, db: Session = Depends(get_db)):
    denied = bs.check_change_authorized(db, req.confirmation_id, req.caller_phone, req.call_id)
    if denied:
        return denied
    return bs.cancel_appointment(db, req.confirmation_id, req.confirm_charge)


@app.get("/api/v1/bookings/lookup")
def lookup_booking(
    phone: str | None = Query(None),
    confirmation_id: str | None = Query(None),
    name: str | None = Query(None),
    db: Session = Depends(get_db),
):
    if not (phone or confirmation_id):
        return {"found": False, "bookings": []}
    rows = bs.lookup_bookings(db, phone=phone, confirmation_id=confirmation_id, name=name)
    return {"found": bool(rows), "bookings": rows}


class SeniorModeRequest(BaseModel):
    phone: str
    senior: bool = True
    # The number the call actually came from, when the bridge knows it.
    # Defaults to `phone` (as lookup_booking does) until real CallerID exists.
    caller_phone: str | None = None


@app.post("/api/v1/patients/senior")
@idempotent("patients.senior")
def set_patient_senior(req: SeniorModeRequest, db: Session = Depends(get_db)):
    """KCD-084: persist the delivery mode against the patient. Boolean only."""
    return {"updated": bs.set_patient_senior(db, req.phone, req.senior, req.caller_phone)}


@app.get("/api/v1/patients/senior")
def get_patient_senior(phone: str = Query(...), caller_phone: str | None = Query(None), db: Session = Depends(get_db)):
    return {"senior": bs.get_patient_senior(db, phone, caller_phone)}


@app.get("/api/v1/bookings/conflict")
def booking_conflict(
    phone: str = Query(...), date: str = Query(...), time_slot: str = Query(...), db: Session = Depends(get_db)
):
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
@idempotent("bookings.tests")
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
    result = bs.book_tests(
        db, ids, req.date, req.patient_name, req.phone, req.caller_phone, req.patient_age, req.relationship
    )
    result["not_found"] = not_found
    return result


class AddTestRequest(BaseModel):
    confirmation_id: str
    test_name: str


@app.post("/api/v1/bookings/add-test")
@idempotent("bookings.add_test")
def add_test_endpoint(req: AddTestRequest, db: Session = Depends(get_db)):
    return bs.add_test_to_booking(db, req.confirmation_id, req.test_name)


@app.get("/api/v1/doctors/earliest")
def doctor_earliest(name: str = Query(...), db: Session = Depends(get_db)):
    amb = _ambiguous_doctor(db, name)
    if amb is not None:
        return amb
    doctor = _find_doctor(db, name)
    if not doctor:
        return {"found": False, "query": name}
    result = bs.earliest_available(db, doctor.id, clinic_clock.today())
    if not result:
        return {"found": True, "available": False, "doctor_name": _doctor_ref(db, doctor)}
    return {
        "found": True,
        "available": True,
        "doctor_name": _doctor_ref(db, doctor),
        "doctor_name_bn": (doctor.full_name_bn or _first_alias_bn(doctor.aliases_bn)),
        **result,
    }


@app.get("/api/v1/doctors/free-slots")
def doctor_free_slots(name: str = Query(...), date: str = Query(...), db: Session = Depends(get_db)):
    """The slots of one doctor on one date that are still free (not booked, not held, not already past), and every valid slot of that day.
    The voice agent picks the time for the caller from this (agent/slot_proposal.py) instead of asking for one."""
    amb = _ambiguous_doctor(db, name)
    if amb is not None:
        return amb
    doctor = _find_doctor(db, name)
    if not doctor:
        return {"found": False, "query": name}
    try:
        target = datetime.date.fromisoformat(date)
    except ValueError:
        return {"found": False, "query": name}
    sched = _schedule_for_weekday(db, doctor.id, target.weekday())
    leave = eq.doctor_leave_on(db, doctor.id, target.isoformat())  # not consulting that day: nothing is offered, and the return date is told
    if leave:
        sched = None
    return {
        "found": True,
        "doctor_name": _doctor_ref(db, doctor),
        "doctor_name_bn": (doctor.full_name_bn or _first_alias_bn(doctor.aliases_bn)),
        "doctor_name_hi": (doctor.full_name_hi or _first_alias_bn(doctor.aliases_hi)),
        "date": target.isoformat(),
        "on_leave": bool(leave),
        "leave": leave,
        "sits": bool(sched),
        "chamber_hours": f"{sched.start_time}-{sched.end_time}" if sched else None,
        "valid_slots": _generate_slots(sched.start_time, sched.end_time) if sched else [],
        "free_slots": bs.available_slots(db, doctor.id, target.isoformat()) if sched else [],
        # when nothing is left on this date (the sitting hours are over, or every slot is taken): the next day with a free slot
        "next": bs.earliest_available(db, doctor.id, target + datetime.timedelta(days=1)),
    }


@app.get("/api/v1/doctors/weekdays")
def doctor_weekdays(name: str = Query(...), db: Session = Depends(get_db)):
    """The WEEKDAYS a doctor sits (0 = Monday), with each day's hours -- the weekly pattern from the schedule table, never a date. The agent says them as
    "সোম বুধ শুক্র বারে বসেন" (agent/doctor_enquiry_cues.weekdays_reply)."""
    amb = _ambiguous_doctor(db, name)
    if amb is not None:
        return amb
    doctor = _find_doctor(db, name)
    if not doctor:
        return {"found": False, "query": name}
    rows = db.query(DoctorSchedule).filter_by(doctor_id=doctor.id).order_by(DoctorSchedule.weekday).all()
    return {
        "found": True,
        "doctor_name": _doctor_ref(db, doctor),
        "doctor_name_bn": (doctor.full_name_bn or _first_alias_bn(doctor.aliases_bn)),
        "doctor_name_hi": (doctor.full_name_hi or _first_alias_bn(doctor.aliases_hi)),
        "weekdays": [r.weekday for r in rows],
        "hours": {str(r.weekday): f"{r.start_time}-{r.end_time}" for r in rows},
    }


@app.get("/api/v1/geo/match")
def geo_match(q: str = Query(..., min_length=1), db: Session = Depends(get_db)):
    """Item 19: which district / police station / block / place of the geo_* tables a spoken or written phrase names, and the police station that gives
    (a place's is its parent row). Candidates only; nothing is resolved that the tables do not hold (geo_service.py)."""
    import geo_service

    return geo_service.match_geo(db, q)


@app.get("/api/v1/doctors/department-alternatives")
def doctor_department_alternatives(name: str = Query(...), date: str = Query(...), lang: str = Query("bn"), db: Session = Depends(get_db)):
    """KCD-lead-6: when `name`'s own day is full, other doctors in the SAME department who actually have a free
    slot on `date` -- for the caller who insists on the same day instead of the next day this doctor sits."""
    amb = _ambiguous_doctor(db, name)
    if amb is not None:
        return amb
    doctor = _find_doctor(db, name)
    if not doctor:
        return {"found": False, "query": name}
    try:
        datetime.date.fromisoformat(date)
    except ValueError:
        return {"found": False, "query": name}
    dept = db.get(Department, doctor.department_id)
    return {
        "found": True,
        "department_name": dept.name if dept else None,
        "date": date,
        "alternatives": bs.doctors_with_free_slots(db, doctor.department_id, date, lang, exclude_doctor_id=doctor.id),
    }


@app.get("/api/v1/doctors/department")
def doctor_department(name: str = Query(...), db: Session = Depends(get_db)):
    """sourav, 2026-10-24: the department a named doctor sits in -- the agent needs it for ONE rule: a patient of 13 or more is not booked with the paediatrician."""
    exact = _doctor_exact_matches(db, name)
    near = [] if exact else _find_doctor_candidates(db, name)
    found = exact if len(exact) == 1 else (near if len(near) == 1 else [])
    if not found:
        return {"found": False}
    dept = db.get(Department, found[0].department_id)
    return {"found": True, "department_name": dept.name if dept else None}


@app.get("/api/v1/departments/route")
def department_route(query: str = Query(...), lang: str = Query("bn"), age: int | None = Query(None, ge=0, le=130), db: Session = Depends(get_db)):
    # sourav, 2026-10-24: `age` (optional) -- a patient of 13 or more is never routed to the paediatrician (lay_departments.PAEDIATRIC_MAX_AGE)
    return bs.route_department(db, query, lang, age)


@app.get("/api/v1/departments/doctors")
def department_doctors(
    name: str = Query(...),
    lang: str = Query("bn"),
    sex: str | None = Query(None),
    on_date: str | None = Query(None),
    at_time: str | None = Query(None),
    db: Session = Depends(get_db),
):
    return bs.department_doctors(db, name, lang, sex=sex if sex in ("F", "M") else None, on_date=on_date, at_time=at_time)


@app.get("/api/v1/relationships")
def relationships(db: Session = Depends(get_db)):
    """Who is who: the words for each relationship and the sex it implies ("my mother" -> a woman)."""
    import clinic_extras

    return {"rules": clinic_extras.relationship_rules(db)}


@app.get("/api/v1/doctors/visit")
def doctor_visit(name: str = Query(...), db: Session = Depends(get_db)):
    """What a doctor's visit costs ("ভিজিট"). Read from the doctors table, never said from memory."""
    amb = _ambiguous_doctor(db, name)
    if amb is not None:
        return amb
    doctor = _find_doctor(db, name)
    if not doctor:
        return {"found": False, "query": name}
    return {
        "found": True,
        "doctor_name": _doctor_ref(db, doctor),
        "doctor_name_bn": (doctor.full_name_bn or _first_alias_bn(doctor.aliases_bn)),
        "doctor_name_hi": (doctor.full_name_hi or _first_alias_bn(doctor.aliases_hi)),
        "visit_fee_inr": doctor.consultation_fee_inr,
    }


class DoctorLeaveRequest(BaseModel):
    doctor_name: str
    start_date: str
    end_date: str
    return_date: str | None = None


@app.post("/api/v1/doctors/leave")
@idempotent("doctors.leave")
def doctor_leave_register(req: DoctorLeaveRequest, db: Session = Depends(get_db)):
    """The doctor will not consult from start_date to end_date: the leave is recorded, the confirmed appointments in the range are cancelled
    (reason doctor_leave, no charge) and an apology is queued for each patient. A caller who asks later is told, sympathetically."""
    import clinic_extras

    doctor = _find_doctor(db, req.doctor_name)
    if not doctor:
        return {"success": False, "reason": "doctor_not_found"}
    try:
        return clinic_extras.register_leave(db, doctor, req.start_date, req.end_date, req.return_date, queue_sms=bs.queue_sms)
    except ValueError:
        return {"success": False, "reason": "invalid_date"}


@app.get("/api/v1/bookings/cancelled-by-clinic")
def cancelled_by_clinic(phone: str | None = Query(None), confirmation_id: str | None = Query(None), db: Session = Depends(get_db)):
    rows = bs.clinic_cancellations(db, phone=phone, confirmation_id=confirmation_id)
    return {"found": bool(rows), "bookings": rows}


class SmsRequest(BaseModel):
    to: str
    message: str
    template_key: str = "generic"
    related_confirmation_id: str | None = None


@app.post("/api/v1/notifications/sms")
@idempotent("notifications.sms")
def send_sms(req: SmsRequest, db: Session = Depends(get_db)):
    """The open placeholder the caller-confirmation flow already writes
    to via booking_service.queue_sms(). Exposed as its own endpoint too,
    so an external system (or a future clinic-api/notifications.py) has
    one clear place to either call in, or be wired up as the thing THIS
    function calls out to. `status` is always "queued": nothing in this
    codebase claims a message reached a phone until a real provider is
    plugged in here."""
    return bs.queue_sms(db, req.to, req.template_key, req.message, req.related_confirmation_id)


class ResendRequest(BaseModel):
    confirmation_id: str


@app.post("/api/v1/bookings/resend")
@idempotent("bookings.resend")
def resend_booking_confirmation(
    req: ResendRequest | None = None, confirmation_id: str | None = Query(None), db: Session = Depends(get_db)
):
    """The confirmation id comes in the body (ResendRequest); the old `?confirmation_id=` form is still accepted."""
    cid = req.confirmation_id if req is not None else confirmation_id
    if not cid:
        raise HTTPException(status_code=422, detail="confirmation_id is required")
    return bs.resend_confirmation(db, cid)


class DraftRequest(BaseModel):
    caller_phone: str
    call_id: str
    slots_json: str


@app.post("/api/v1/bookings/draft")
@idempotent("bookings.draft")
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
    amb = _ambiguous_doctor(db, doctor_name)
    if amb is not None:
        return amb
    doctor = _find_doctor(db, doctor_name)
    if not doctor:
        return {"found": False, "query": doctor_name}
    leave = eq.doctor_leave_on(db, doctor.id, date)
    return {"found": True, "doctor_name": _doctor_ref(db, doctor), "on_leave": leave is not None, "leave": leave}


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
def walk_in_endpoint(
    department: str | None = Query(None), test_name: str | None = Query(None), db: Session = Depends(get_db)
):
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
def home_collection_endpoint(test_name: str = Query(...), postal_code: str = Query(...), db: Session = Depends(get_db)):
    t = _find_test(db, test_name)
    if not t:
        return {"found": False, "query": test_name}
    return eq.home_collection_eligibility(db, t.id, postal_code)


@app.get("/api/v1/insurance/coverage")
def insurance_coverage_endpoint(
    policy_number: str = Query(...), test_name: str | None = Query(None), db: Session = Depends(get_db)
):
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
@idempotent("calls.out_of_scope")
def out_of_scope_endpoint(req: OutOfScopeRequest, db: Session = Depends(get_db)):
    return eq.record_out_of_scope(db, req.call_id, req.caller_question, req.reason_code)


class CallbackRequestBody(BaseModel):
    phone: str
    call_id: str
    requested_window: str
    reason: str = ""
    call_summary: str = ""


@app.post("/api/v1/callbacks")
@idempotent("callbacks")
def callback_endpoint(req: CallbackRequestBody, db: Session = Depends(get_db)):
    return eq.request_callback(db, req.phone, req.call_id, req.requested_window, req.reason, req.call_summary)


class ReferralRequest(BaseModel):
    patient_name: str = ""
    phone: str = ""
    doctor_name: str = ""
    symptom_description: str = ""
    notes: str = ""


@app.post("/api/v1/referrals")
def referral_endpoint(req: ReferralRequest):
    """KCD-lead-1: staff at another center passing along a patient's number for a later outbound call --
    logged to a CSV worklist (referral_log.py), not a database table; no DB session needed."""
    import referral_log

    return referral_log.log_referral(req.patient_name, req.phone, req.doctor_name, req.symptom_description, req.notes)


class PaymentLinkRequest(BaseModel):
    confirmation_id: str


@app.post("/api/v1/payments/send-link")
@idempotent("payments.send_link")
def send_payment_link_endpoint(req: PaymentLinkRequest, db: Session = Depends(get_db)):
    """Only a CONFIRMED DOCTOR appointment's confirmation_id ever gets a link back -- a lab test is paid and booked
    at the counter, never over the call (bs.send_payment_link's own docstring)."""
    return bs.send_payment_link(db, req.confirmation_id)


@app.get("/api/v1/bookings/serial")
def booking_serial_status(phone: str = Query(...), db: Session = Depends(get_db)):
    """sourav, 2026-10-24: "I paid but my serial number never came on WhatsApp" -- what the database holds for this mobile number: each upcoming doctor appointment, whether it was
    paid, and its serial number if so (booking_service.serial_status). Read-only; the same disclosure rule as /bookings/lookup."""
    return bs.serial_status(db, phone)


@app.get("/api/v1/bookings/next-serial")
def booking_next_serial(name: str = Query(...), date: str | None = Query(None), db: Session = Depends(get_db)):
    """sourav, 2026-10-24: "if I book today, what will my serial number be?" -- the number the next booking of this doctor on this date would get (booking_service.next_serial; serials are
    simulated, generated with each booking). Read-only. A doctor who does not sit that day gets no number (available False) -- a serial is never promised for a day he is not in his chamber."""
    exact = _doctor_exact_matches(db, name)
    doctor = exact[0] if len(exact) == 1 else (None if exact else _find_doctor(db, name))
    if not doctor:
        return {"found": False, "query": name}
    try:
        target = datetime.date.fromisoformat(date) if date else clinic_clock.today()
    except ValueError:
        return {"found": False, "query": name}
    sched = _schedule_for_weekday(db, doctor.id, target.weekday())
    out = {
        "found": True,
        "doctor_name": _doctor_ref(db, doctor),
        "doctor_name_bn": (doctor.full_name_bn or _first_alias_bn(doctor.aliases_bn)),
        "doctor_name_hi": (doctor.full_name_hi or _first_alias_bn(doctor.aliases_hi)),
        "date": target.isoformat(),
        "available": bool(sched),
        "chamber_hours": f"{sched.start_time}-{sched.end_time}" if sched else None,
        "next_serial": bs.next_serial(db, doctor.id, target.isoformat()) if sched else None,
    }
    if not sched:
        out["next_available_date"] = _next_available_date(db, doctor.id, target + datetime.timedelta(days=1))
    return out


class SerialResendRequest(BaseModel):
    phone: str
    confirmation_id: str


@app.post("/api/v1/bookings/serial/resend")
@idempotent("bookings.serial_resend")
def booking_serial_resend(req: SerialResendRequest, db: Session = Depends(get_db)):
    """sourav, 2026-10-24: queue the serial number to the booking's WhatsApp number again (booking_service.resend_serial). Placeholder channel: recorded as queued, never claimed delivered."""
    return bs.resend_serial(db, req.phone, req.confirmation_id)


class BookingConfirmationWhatsAppRequest(BaseModel):
    confirmation_id: str


@app.post("/api/v1/bookings/send-whatsapp-confirmation")
@idempotent("bookings.send_whatsapp_confirmation")
def send_booking_confirmation_whatsapp_endpoint(req: BookingConfirmationWhatsAppRequest, db: Session = Depends(get_db)):
    """KCD-lead-2: the warm/friendly WhatsApp confirmation template, sent once a doctor appointment is confirmed."""
    return bs.send_booking_confirmation_whatsapp(db, req.confirmation_id)


@app.get("/api/v1/reports/status")
def report_status_endpoint(confirmation_id: str = Query(...), db: Session = Depends(get_db)):
    return eq.report_status(db, confirmation_id)


class ReportOTPRequest(BaseModel):
    confirmation_id: str
    phone: str


@app.post("/api/v1/reports/request-otp")
@idempotent("reports.request_otp")
def report_request_otp_endpoint(req: ReportOTPRequest, db: Session = Depends(get_db)):
    return eq.request_report_otp(db, req.confirmation_id, req.phone)


class ReportDeliverRequest(BaseModel):
    confirmation_id: str
    otp_code: str


@app.post("/api/v1/reports/deliver")
@idempotent("reports.deliver")
def report_deliver_endpoint(req: ReportDeliverRequest, db: Session = Depends(get_db)):
    return eq.deliver_report(db, req.confirmation_id, req.otp_code)


@app.get("/api/v1/departments/{department_name}/hours")
def department_hours_endpoint(department_name: str, lang: str = Query("bn"), db: Session = Depends(get_db)):
    dept = db.query(Department).filter(Department.name.ilike(f"%{department_name}%")).first()
    if not dept:
        return {"found": False, "query": department_name}
    hours = eq.department_hours(db, dept.id, lang)
    return hours or {"found": False}


# =============================================================================
# Epic E33: patient context and history (clinic-api/patient_context.py)
# =============================================================================


@app.get("/api/v1/calls/satisfaction/summary")
def satisfaction_summary(
    since: str | None = Query(None, description="ISO date; calls that started on or after it"),
    language: str | None = Query(None, description="bn | hi | en: the call's first language"),
    db: Session = Depends(get_db),
):
    """How the calls went, from the implicit happiness score (agent/call_score.py): scored calls, their mean and
    median, the share in each band, the same by language, and what pulled scores down most. A PROXY built from behaviour,
    not from anything callers said; no call, phone number or text is returned."""
    from models import CallRecord

    q = db.query(CallRecord)
    if since:
        try:
            q = q.filter(CallRecord.started_at >= datetime.datetime.fromisoformat(since))
        except ValueError:
            raise HTTPException(status_code=422, detail="since must be an ISO date") from None
    rows = q.all()
    if language:
        rows = [r for r in rows if (r.languages or "").split(",")[0] == language]

    def block(subset):
        scored = sorted(r.satisfaction_score for r in subset if r.satisfaction_score is not None)
        bands = {
            b: sum(1 for r in subset if r.satisfaction_score is not None and r.satisfaction_band == b)
            for b in ("happy", "neutral", "unhappy", "very_unhappy")
        }
        return {
            "calls": len(subset),
            "scored": len(scored),
            "mean": round(sum(scored) / len(scored), 1) if scored else None,
            "median": scored[len(scored) // 2] if scored else None,
            "bands": bands,
            "share_unhappy": round((bands["unhappy"] + bands["very_unhappy"]) / len(scored), 3) if scored else None,
        }

    by_language: dict[str, list] = {}
    for r in rows:
        by_language.setdefault((r.languages or "").split(",")[0] or "unknown", []).append(r)
    costs: dict[str, list[int]] = {}
    unscored: dict[str, int] = {}
    for r in rows:
        try:
            data = json.loads(r.satisfaction_json or "{}")
        except ValueError:
            data = {}
        if r.satisfaction_score is None:
            unscored[data.get("reason") or "no_score"] = unscored.get(data.get("reason") or "no_score", 0) + 1
            continue
        for name, pts in data.get("reasons", []):
            if pts < 0:
                costs.setdefault(name, []).append(pts)
    top = sorted(
        ({"signal": n, "calls": len(v), "points_lost": -sum(v)} for n, v in costs.items()),
        key=lambda d: (-d["points_lost"], d["signal"]),
    )[:8]
    return {
        "basis": "behaviour",
        "version": "implicit-v1",
        "overall": block(rows),
        "by_language": {lang: block(rs) for lang, rs in sorted(by_language.items())},
        "top_causes": top,
        "unscored_reasons": unscored,
    }


class CallEventRequest(BaseModel):
    seq: int
    kind: str
    payload: dict = Field(default_factory=dict)
    caller_phone: str | None = None


@app.post("/api/v1/calls/{call_id}/events")
def call_event(call_id: str, req: CallEventRequest, db: Session = Depends(get_db)):
    """KCD-501: idempotent by (call_id, seq)."""
    return pc.record_call_event(db, call_id, req.seq, req.kind, req.payload, req.caller_phone)


@app.get("/api/v1/calls/{call_id}")
def call_record(call_id: str, db: Session = Depends(get_db)):
    rec = pc.get_call_record(db, call_id)
    return {"found": rec is not None, "record": rec}


@app.get("/api/v1/patients/identify")
def patient_identify(phone: str = Query(...), db: Session = Depends(get_db)):
    """KCD-494: existence and count only -- never a name."""
    return pc.identify(db, phone)


class ResolveRequest(BaseModel):
    phone: str
    name: str
    age: int | None = None


@app.post("/api/v1/patients/resolve")
def patient_resolve(req: ResolveRequest, db: Session = Depends(get_db)):
    return pc.resolve_named(db, req.phone, req.name, req.age)


@app.get("/api/v1/patients/{patient_ref}/timeline")
def patient_timeline(
    patient_ref: int, caller_phone: str = Query(...), call_id: str = Query("unknown"), db: Session = Depends(get_db)
):
    """KCD-493/495: authorised, audited, and free of result values. Refused unless THIS CALL has
    passed the security questions for THIS patient (POST /api/v1/patients/verify): the server decides,
    not the caller."""
    return pc.timeline(db, patient_ref, caller_phone, call_id, verified=reg.is_verified(db, call_id, patient_ref))


@app.get("/api/v1/patients/{patient_ref}/preferences")
def patient_preferences(patient_ref: int, caller_phone: str = Query(...), db: Session = Depends(get_db)):
    return pc.get_preferences(db, patient_ref, caller_phone)


class PreferencesRequest(BaseModel):
    caller_phone: str
    branch: str | None = None
    collection_address: str | None = None
    delivery_channel: str | None = None
    accessibility_mode: str | None = None
    language_bias: str | None = None


@app.post("/api/v1/patients/{patient_ref}/preferences")
@idempotent("patients.preferences")
def set_patient_preferences(patient_ref: int, req: PreferencesRequest, db: Session = Depends(get_db)):
    fields = req.model_dump(exclude={"caller_phone"})
    return pc.set_preferences(db, patient_ref, req.caller_phone, **fields)


@app.get("/api/v1/patients/{patient_ref}/test-status")
def patient_test_status(
    patient_ref: int,
    test_name: str = Query(...),
    caller_phone: str = Query(...),
    call_id: str = Query("unknown"),
    db: Session = Depends(get_db),
):
    """KCD-498: when a test was last performed and whether it is due -- authorised and audited
    like the timeline, and carrying no result."""
    patient = db.get(pc.Patient, patient_ref)
    if patient is None or not reg.is_verified(db, call_id, patient_ref):
        pc._audit(db, call_id, patient_ref, "test_status_denied", [])
        return {"success": False, "reason": "not_verified"}
    test = db.query(LabTest).filter(LabTest.name.ilike(test_name)).first()
    if test is None:
        return {"success": False, "reason": "unknown_test"}
    pc._audit(db, call_id, patient_ref, "test_status_read", [test.name])
    return {"success": True, "test_name": test.name, **pc.test_status(db, patient_ref, test.id)}


@app.get("/api/v1/continuity")
def continuity_endpoint(
    caller_phone: str = Query(...),
    call_id: str | None = Query(None),
    patient_ref: int | None = Query(None),
    db: Session = Depends(get_db),
):
    """KCD-496: what was left unfinished, the last three interactions on this number, and the last
    calls cached for ONE DAY. The draft's details are for the agent to speak only after verification."""
    return pc.continuity(db, caller_phone, call_id, patient_id=patient_ref)


# ------------------------------------------------ security questions, find, messages, audit


class VerifyRequest(BaseModel):
    call_id: str
    patient_ref: int
    patient_id: str | None = None  # the id on the patient's card
    dob: str | None = None  # ISO date, parsed by the agent
    name: str | None = None
    address: str | None = None
    caller_phone: str | None = None


@app.post("/api/v1/patients/verify")
def patient_verify(req: VerifyRequest, db: Session = Depends(get_db)):
    """KCD-495: check the caller's answers against the registry. Two matching facts, one of them
    strong (patient id or date of birth). The reply never says which answer was wrong."""
    return reg.verify(
        db,
        req.call_id,
        req.patient_ref,
        {"patient_id": req.patient_id, "dob": req.dob, "name": req.name, "address": req.address},
        req.caller_phone,
    )


class FindRequest(BaseModel):
    call_id: str
    patient_id: str | None = None
    dob: str | None = None
    name: str | None = None
    address: str | None = None


@app.post("/api/v1/patients/find")
def patient_find(req: FindRequest, db: Session = Depends(get_db)):
    """KCD-497: find a patient by a patient id, or a date of birth with a name, when the phone number
    found nobody. Returns an opaque reference only; finding is not verifying."""
    return reg.find_by_details(
        db, req.call_id, {"patient_id": req.patient_id, "dob": req.dob, "name": req.name, "address": req.address}
    )


@app.get("/api/v1/agent/messages")
def agent_messages(lang: str | None = Query(None), db: Session = Depends(get_db)):
    """KCD-353/500/513: the operator-editable wording, read by the agent at the start of a call."""
    return am.get_messages(db, lang)


class MessageRequest(BaseModel):
    key: str
    lang: str
    text: str
    updated_by: str = "operator"
    active: bool = True


@app.put("/api/v1/agent/messages")
def agent_message_set(req: MessageRequest, db: Session = Depends(get_db)):
    """Change what callers hear, without a deploy. Behind the service token like every /api/v1 route;
    check the wording first with tools/check_messages.py."""
    return am.set_message(db, req.key, req.lang, req.text, req.updated_by, req.active)


@app.get("/api/v1/audit")
def audit_log_read(
    call_id: str | None = Query(None),
    patient_ref: int | None = Query(None),
    action: str | None = Query(None),
    limit: int = Query(100, le=500),
    db: Session = Depends(get_db),
):
    """The audit trail, newest first, filtered. Read-only."""
    q = db.query(AuditLog)
    if call_id:
        q = q.filter(AuditLog.call_id == call_id)
    if patient_ref is not None:
        q = q.filter(AuditLog.patient_id == patient_ref)
    if action:
        q = q.filter(AuditLog.action == action)
    rows = q.order_by(AuditLog.at.desc(), AuditLog.id.desc()).limit(limit).all()
    return {
        "entries": [
            {
                "at": r.at.isoformat(),
                "call_id": r.call_id,
                "patient_ref": r.patient_id,
                "actor": r.actor,
                "action": r.action,
                "outcome": r.outcome,
                "detail": json.loads(r.detail_json),
            }
            for r in rows
        ]
    }


class BookingSearchRequest(BaseModel):
    caller_phone: str
    phone: str | None = None
    name: str | None = None
    approx_date: str | None = None
    date_window_days: int = 3
    test_name: str | None = None
    branch: str | None = None


@app.post("/api/v1/bookings/search")
def bookings_search(req: BookingSearchRequest, db: Session = Depends(get_db)):
    """KCD-497: several matches are returned as several, never collapsed to the likeliest."""
    return pc.search_bookings(
        db,
        req.caller_phone,
        phone=req.phone,
        name=req.name,
        approx_date=req.approx_date,
        date_window_days=req.date_window_days,
        test_name=req.test_name,
        branch=req.branch,
    )
