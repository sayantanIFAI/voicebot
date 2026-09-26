"""Node 2: intent + slot extraction via Qwen2.5, over Ollama.

Deliberately NOT vLLM, and deliberately NOT the model's native tool-calling
format. Two reasons, both grounded in what's already running on this
account rather than assumed:

1. VRAM. vLLM serving Qwen2.5-7B-Instruct at fp16 needs ~17GB just for the
   model (measured: https://www.spheron.network/tools/gpu-recommender/
   Qwen/Qwen2.5-7B-Instruct/). The already-deployed pipeline runs the same
   model through Ollama as a Q4_K_M GGUF -- 4.68GB on disk -- specifically
   so IndicConformer (~1.2GB VRAM) and this service's TTS model can share
   the same 24GB card comfortably. Introducing vLLM here would mean running
   two different serving stacks for the same model on the same box, for no
   benefit at this call volume.

2. voicerx/extract.py already hardened a JSON-mode + strict-schema pattern
   against a *reproduced* failure: a looser prompt caused the model to
   invent a drug name ("Naloxone") that was never said. Ollama's function
   tool_calls) is comparatively unproven on this stack. The pattern below
   -- classify intent, extract slots as literal spans, then have CODE (not
   the model) fill the reply from the API response -- is the same
   discipline, applied to prices and appointment slots instead of drugs.
   Quoting a wrong price with total confidence is this system's version of
   that bug, so the model is never allowed to state a number on its own;
   see main.py's _compose_reply().
"""

from __future__ import annotations

import datetime
import json
import time
import urllib.request

from agent.enquiry_followup import ENQUIRY_INTENTS

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

VALID_INTENTS = {
    "test_rate",
    "doctor_availability",
    "book_appointment",
    "test_prep",
    "clinic_faq",
    "smalltalk",
    "unclear",
    # Epic E26 -- booking, rescheduling and cancellation
    "book_test",
    "reschedule_appointment",
    "cancel_appointment",
    "lookup_booking",
    "add_test_booking",
    "resend_confirmation",
    "department_query",
}

# The FAQ topic keys FastPath.FAQCatalogue matches locally against
# /api/v1/catalogue's faq_topics. Kept here too, as a fixed enum for the
# model's OWN classification when fast_path abstains on a paraphrase its
# keyword table doesn't cover -- the model still only ever picks a TOPIC
# KEY, never composes the answer itself (agent/reply_templates.py fetches
# and speaks the real one). If clinic-api's FAQ table grows, update both
# this tuple and seed.py's FAQ_ENTRIES together.
FAQ_TOPICS = (
    "hours",
    "location",
    "payment_methods",
    "insurance",
    "parking",
    "report_collection",
    "contact_number",
    "home_collection",
)

SYSTEM_PROMPT_TEMPLATE = """You are the intent-and-slot extractor for a diagnostic clinic's phone assistant. You will be given ONE caller utterance in {language_name}, transcribed by automatic speech recognition from live phone audio -- it may contain ASR errors, missing punctuation, or code-switched English words written in the caller's own script.

Today's date is {today_iso} ({today_weekday}), Asia/Kolkata.

YOUR ONLY JOB is to classify intent and pull out slots that are LITERALLY present in the utterance. You do NOT know test prices, doctor schedules, test preparation instructions, or clinic FAQ answers -- do not guess or state any of those; that data comes from a separate lookup after you run.

INTENTS (exactly one):
- "test_rate": caller is asking the price/rate of a diagnostic test.
- "doctor_availability": caller is asking whether/when a named doctor is available.
- "book_appointment": caller wants to book a NEW appointment with a doctor.
- "book_test": caller wants to book a lab test (or several), as opposed to just asking its price or prep.
- "reschedule_appointment": caller wants to MOVE an appointment they already have to a different date/time. Fill "confirmation_id" if they state one, otherwise leave it null -- the system looks it up by phone instead.
- "cancel_appointment": caller wants to cancel an appointment or test booking they already have.
- "lookup_booking": caller is asking what they already have booked ("what's my appointment", "do I have anything booked").
- "add_test_booking": caller wants to add a test to a booking they already have, rather than book a new, separate one.
- "resend_confirmation": caller wants the confirmation message sent again (they lost it, didn't get it, etc.).
- "department_query": caller describes a symptom or problem and needs to be routed to the right department, without naming a doctor or department themselves. Fill "symptom_description" with what they said, verbatim.
- "test_prep": caller is asking how to prepare for a test (fasting, before/after instructions).
- "clinic_faq": caller is asking a general clinic question with no specific test or doctor -- hours, location, payment methods, insurance, parking, report collection, contact number, or home sample collection. Fill "faq_topic" with exactly one of: {faq_topics}. If the question doesn't clearly match one of those topics, use "unclear" instead of guessing a topic.
- "smalltalk": greeting, thanks, or anything with no clinic-data lookup needed. You MAY write a short, warm reply yourself for this case only, in {language_name}, in that language's own script.
- "unclear": you cannot confidently tell what the caller wants, or the utterance is empty/garbled ASR noise.

SECOND QUESTION IN THE SAME TURN:
A caller sometimes asks two separate, answerable questions in one breath
("what's the CBC rate, and is Dr. Sen available tomorrow?"). When -- and
ONLY when -- the utterance clearly contains a SECOND, DISTINCT question,
fill "secondary_intent" and "secondary_slots" the same way you filled
"intent" and "slots" for the first one. Leave both null for an ordinary
single-question turn; never split one question into two, and never use
this for anything except "test_rate", "doctor_availability", "test_prep",
"clinic_faq" or "department_query" -- if the second thing the caller said
is a booking/reschedule/cancel action or is itself unclear, leave
"secondary_intent" null rather than guessing.

SLOT RULES:
- Only fill a slot if the caller's words support it. Leave it null rather than inferring.
- "date" / "new_date": resolve relative time words (Bengali আজ/কাল/পরশু and "আগামী <weekday>", Hindi आज/कल/परसों and "अगले <weekday>", English today/tomorrow/day after tomorrow/"this <weekday>"/"next <weekday>") to an ISO yyyy-mm-dd using today's date above. If no date is mentioned for an availability/booking request, leave it null -- do not assume "today". If the resolved date is clearly in the past, still return it as stated -- the system rejects and corrects past dates itself; you must never silently roll a date forward.
- "test_name" / "doctor_name": copy the term as the caller said it (in {language_name} script, or English if they said it in English), do not translate or normalize it -- the lookup service handles matching.
- "test_names": for "book_test", a list of every test name mentioned this turn, each copied as the caller said it, same rule as "test_name".
- "faq_topic": only for "clinic_faq" -- one of the fixed topic keys above, never free text.
- "phone": the number to send the confirmation to, only if explicitly spoken, digits only.
- "contact_phone": a phone number the caller gives that is DIFFERENT from the number they are calling from, for the confirmation message -- only when they say so explicitly (e.g. "send it to a different number").
- "relationship": only when the caller says who the patient is to them and it is not themselves, e.g. "mother", "son", "wife" -- copy the relationship word as said, in {language_name}. Leave null if they are booking for themselves.
- "patient_age": only if an age is explicitly spoken, as a number.
- "confirmation_id" / "new_time_slot": only if explicitly spoken/known this turn.
- "spelled_letters": if the caller is spelling a name letter by letter (e.g. "R, A, V, I"), a list of the individual letters in order, lowercase. Otherwise null.
- "symptom_description": only for "department_query" -- the caller's own words describing the problem, never your own paraphrase or a diagnosis.
- Never invent a patient name, phone number, confirmation number, or date that was not said.

Output ONLY a single valid JSON object, no other text, in exactly this shape:
{{
  "intent": "test_rate" | "doctor_availability" | "book_appointment" | "book_test" | "reschedule_appointment" | "cancel_appointment" | "lookup_booking" | "add_test_booking" | "resend_confirmation" | "department_query" | "test_prep" | "clinic_faq" | "smalltalk" | "unclear",
  "slots": {{
    "test_name": string or null,
    "test_names": array of strings or null,
    "doctor_name": string or null,
    "date": string or null,
    "time_slot": string or null,
    "new_date": string or null,
    "new_time_slot": string or null,
    "confirmation_id": string or null,
    "patient_name": string or null,
    "patient_age": number or null,
    "phone": string or null,
    "contact_phone": string or null,
    "relationship": string or null,
    "spelled_letters": array of strings or null,
    "symptom_description": string or null,
    "faq_topic": string or null
  }},
  "secondary_intent": "test_rate" | "doctor_availability" | "test_prep" | "clinic_faq" | "department_query" | null,
  "secondary_slots": {{
    "test_name": string or null,
    "doctor_name": string or null,
    "date": string or null,
    "faq_topic": string or null,
    "symptom_description": string or null
  }} or null,
  "direct_reply_bn": string or null
}}

"direct_reply_bn" must be null for every intent except "smalltalk" -- for every other intent, the reply is composed later from real clinic data, not from you."""


class ExtractionError(Exception):
    pass


def _call_ollama(prompt: str, timeout_s: int = 90) -> str:
    # 90s, not 20s: a cold-loaded Qwen2.5:7b (Ollama unloaded it after its
    # default 5-minute idle timeout) measured at 47s just to answer "Say
    # OK" on this pod. The real fix is OLLAMA_KEEP_ALIVE keeping the model
    # resident (see setup docs) so this path is rarely hit in practice --
    # this margin is a backstop for whenever it still is.
    payload = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            # Keep Qwen resident. Ollama unloads a model after 5 idle minutes and a
            # cold 7B load measured 74 s on this pod -- the first caller after any
            # quiet spell would wait that long for one intent.
            "keep_alive": -1,
            "options": {"temperature": 0.0},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


def _normalize_list_slot(slots: dict, key: str) -> None:
    """CodeRabbit-flagged, real bug: Ollama's JSON mode enforces valid
    JSON syntax, not this schema's shape -- a documented failure mode for
    a single-item list is the model collapsing it to a bare string
    ("test_names": "CBC" instead of ["CBC"]). agent/booking_flow.merge_slots
    does `for t in incoming_tests: ...` on whatever this slot holds; over
    a STRING that iterates its individual CHARACTERS, each one then
    treated as an entered test name -- exactly the class of silent,
    confident-but-wrong fact CLAUDE.md's truth boundary exists to catch,
    here self-inflicted by a schema-shape slip rather than a bad lookup.

    A bare string is wrapped as a single-element list (test_names: one
    test was meant) or split into characters (spelled_letters: the slot
    IS the individual letters, so a collapsed string's characters ARE
    the letters). Anything else that is not a list of strings is dropped
    to null rather than guessed."""
    val = slots.get(key)
    if val is None:
        return
    if isinstance(val, str):
        slots[key] = [val] if key == "test_names" else list(val)
        return
    if isinstance(val, list) and all(isinstance(v, str) for v in val):
        return
    slots[key] = None


def _normalize_age(value) -> int | None:
    """The schema says "number or null", and a model may return "72" or
    "seventy" or 7.2e1. Only a whole number in a human range survives; anything
    else is None -- a wrong-typed age must not reach code that compares it."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, str):
        try:
            value = int(value.strip())
        except ValueError:
            return None
    if isinstance(value, int) and 0 < value < 130:
        return value
    return None


def _validate(data: dict) -> tuple[bool, list[str]]:
    errors = []
    if data.get("intent") not in VALID_INTENTS:
        errors.append(f"invalid intent: {data.get('intent')!r}")
    slots = data.get("slots")
    if not isinstance(slots, dict):
        errors.append("slots: expected object")
    else:
        for key in (
            "test_name",
            "test_names",
            "doctor_name",
            "date",
            "time_slot",
            "new_date",
            "new_time_slot",
            "confirmation_id",
            "patient_name",
            "patient_age",
            "phone",
            "contact_phone",
            "relationship",
            "spelled_letters",
            "symptom_description",
            "faq_topic",
        ):
            if key not in slots:
                errors.append(f"slots.{key}: missing")
    if data.get("intent") != "smalltalk" and data.get("direct_reply_bn") not in (None, ""):
        # Not fatal -- just strip it. The model overstepping here is the
        # exact failure mode this schema exists to prevent (see module
        # docstring), so we defend in code rather than trust a retry to fix it.
        data["direct_reply_bn"] = None
    if isinstance(slots, dict) and slots.get("faq_topic") not in (None, *FAQ_TOPICS):
        # The model invented a topic key outside the fixed enum. Not fatal
        # either: null it so main.py's missing-slot path asks the caller
        # to repeat, rather than passing an unknown key to clinic-api's
        # /api/v1/faq, which would just 404 -- same "the model proposes,
        # code decides" discipline as direct_reply_bn above.
        slots["faq_topic"] = None

    if isinstance(slots, dict):
        _normalize_list_slot(slots, "test_names")
        _normalize_list_slot(slots, "spelled_letters")
        slots["patient_age"] = _normalize_age(slots.get("patient_age"))

    # KCD-395: a second question in the same turn. Optional and defended
    # the same way as direct_reply_bn/faq_topic above -- an older cached
    # intent (from before this field existed) or a model that omits it
    # entirely is just "no second question", never a hard failure of the
    # whole extraction. Restricted to the stateless, single-tool-call
    # intents (agent/enquiry_followup.ENQUIRY_INTENTS) -- a booking action
    # or "unclear" as a secondary_intent is defensively dropped here too,
    # not just described-away in the prompt, because a model ignoring an
    # instruction is exactly the failure mode this schema-level check
    # exists to catch instead of trust.
    secondary_intent = data.get("secondary_intent")
    if secondary_intent not in (None, *ENQUIRY_INTENTS):
        secondary_intent = None
    secondary_slots = data.get("secondary_slots")
    if secondary_intent is None or not isinstance(secondary_slots, dict):
        secondary_intent, secondary_slots = None, None
    elif secondary_slots.get("faq_topic") not in (None, *FAQ_TOPICS):
        # Same guard as the primary slots' faq_topic above -- an invented
        # secondary topic must not reach _tools.get_faq, which would 404
        # and surface as tool_failure instead of the ordinary unclear/
        # missing-slot prompt the caller actually needs here.
        secondary_slots["faq_topic"] = None
    data["secondary_intent"] = secondary_intent
    data["secondary_slots"] = secondary_slots

    return (
        len([e for e in errors if "missing" not in e or "intent" in e or "slots: expected" in e]) == 0
        and "slots" in data,
        errors,
    )


_LANGUAGE_NAMES = {"bn": "Bengali", "hi": "Hindi", "en": "English"}

# KCD-465: "a deadline budget rather than a retry count bounds every
# stage... no path can leave the caller waiting past the total budget."
# REASONED, not measured against real telephony: agent/asr.py's own
# module docstring measures a cold Qwen load at ~74s, and this session's
# earlier work margined _call_ollama's per-attempt timeout at 90s for
# exactly that case -- but 90s (let alone up to 3 attempts of it) is
# nowhere near a bound a caller on the line should ever wait behind.
# DEFAULT_DEADLINE_S is the whole-operation ceiling every retry shares;
# a caller hits the spoken "llm_failure" apology at a predictable time
# instead of an unbounded one, exactly the acceptance criterion's own
# wording. Recalibrate against real telephony latency once measured.
DEFAULT_DEADLINE_S = 12.0
_MAX_ATTEMPTS_BACKSTOP = 5  # defence in depth only -- see extract_intent's docstring


def extract_intent(
    transcript_bn: str, max_retries: int = 2, lang: str = "bn", deadline_s: float = DEFAULT_DEADLINE_S
) -> tuple[dict, dict]:
    """Returns (parsed JSON dict, diagnostics dict).

    `transcript_bn` keeps its historical name for callers; it is the caller's
    utterance in `lang`. Only the prompt's language wording changes -- the
    schema, the slot rules and the never-state-a-fact rule are identical
    for every language.

    `max_retries` is kept only as a defensive attempt-count backstop
    (capped at _MAX_ATTEMPTS_BACKSTOP regardless of its value) against a
    pathological case where every attempt somehow returns instantly --
    `deadline_s` is what actually bounds how long this function may run,
    and each attempt's own network timeout is clamped to whatever of the
    deadline remains, so one slow attempt cannot by itself consume the
    whole budget meant to cover retries too."""
    now = datetime.datetime.now()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        today_iso=now.strftime("%Y-%m-%d"),
        today_weekday=now.strftime("%A"),
        faq_topics=", ".join(FAQ_TOPICS),
        language_name=_LANGUAGE_NAMES.get(lang, "Bengali"),
    )
    prompt = (
        f"{system_prompt}\n\nCALLER UTTERANCE ({_LANGUAGE_NAMES.get(lang, 'Bengali')}, "
        f"ASR output):\n{transcript_bn}\n\nJSON:"
    )

    diagnostics = {"attempts": 0, "total_time_s": 0.0, "errors": [], "deadline_s": deadline_s}
    last_error: Exception | ExtractionError = ExtractionError("no attempt was made")
    start = time.time()
    max_attempts = min(max_retries + 2, _MAX_ATTEMPTS_BACKSTOP)

    for attempt in range(1, max_attempts + 1):
        remaining = deadline_s - (time.time() - start)
        if remaining <= 0:
            break
        diagnostics["attempts"] = attempt
        t0 = time.time()
        try:
            raw = _call_ollama(prompt, timeout_s=remaining)
            diagnostics["total_time_s"] += time.time() - t0
            data = json.loads(raw)
            ok, errors = _validate(data)
            if not ok:
                raise ValueError(f"schema validation failed: {errors}")
            return data, diagnostics
        except Exception as e:  # noqa: BLE001 - retry on anything, log it
            diagnostics["total_time_s"] += time.time() - t0
            last_error = e
            diagnostics["errors"].append(f"attempt {attempt}: {type(e).__name__}: {e}")

    raise ExtractionError(
        f"intent extraction did not complete within its {deadline_s}s deadline "
        f"({diagnostics['attempts']} attempt(s), last error: {last_error})"
    )
