"""Rural lay terms for what a test is, matched deterministically to a CATEGORY -- never to a specific test name.

A caller who has never heard "CBC" or "USG" describes what the doctor wants checked, not the test's clinical name:
"rokto porikkha" (blood test), "peter chhobi" (a picture of the stomach), "mathar chhobi" (a picture of the head).
This module recognises the CATEGORY the phrase describes; agent/reply_templates.py's lay_term_reply() (or main.py,
for the blood list) turns that into an answer built from the REAL catalogue -- this file never states a test exists.

    BLOOD    -- every test the catalogue marks sample_type == "Blood"; ask which one (main.py lists them)
    ABDOMEN  -- resolves to the catalogue's Imaging test(s) whose name mentions the abdomen (today: USG Whole
                Abdomen); state it directly, there is nothing to ask
    HEAD     -- the catalogue has NO head/brain imaging test today (checked against the seeded data: only Chest
                X-Ray and USG Whole Abdomen exist under "Imaging"). Naming "brain scan" or "brain MRI" here would be
                exactly the fabricated fact CLAUDE.md's truth boundary exists to prevent -- a clinic that does not
                offer a service must never be told it does. So this category resolves to NOTHING; the caller is told
                to send the prescription so a colleague can check what the doctor actually wants (see main.py).

Phrases are matched in Bengali script (what a real caller's ASR transcript looks like), Devanagari, and the
phonetic-romanised Latin spelling the person who wrote this table typed it in -- kept because rural callers on a
poor line, or a WhatsApp/SMS channel, sometimes come through in Latin script too. REASONED, not measured: this list
has not been checked against a real call corpus; grow it from real transcripts once they exist (docs/transcription-
rulebook.md), and confirm with the clinic which imaging tests actually exist before this category list grows past
BLOOD/ABDOMEN (docs/REMINDERS-if-client-asks.md).
"""

from __future__ import annotations

import re
import unicodedata

BLOOD = "blood"
ABDOMEN = "abdomen"
HEAD = "head"
# OWNER'S INSTRUCTION, 2026-10-12: a caller who does not know the test name at all is first asked which of
# these three BROAD sample types it is -- URINE lists every catalogue test with sample_type == "Urine" (same
# "list from the real catalogue, never invent one" pattern BLOOD already uses); IMAGING is the generic category
# for a caller who says "scan"/"x-ray"/"ultrasound" without naming ABDOMEN specifically -- it lists every
# catalogue test with sample_type == "Imaging" (ABDOMEN/HEAD stay as they were: more specific, resolved
# differently). Both handled in main.py the same way BLOOD already is.
URINE = "urine"
IMAGING = "imaging"

# Phrase -> category. Substring match on the normalised text (see normalise()); a phrase is chosen to be distinctive
# enough that it does not fire on an unrelated sentence ("chhobi" alone is too short -- "peter chhobi"/"mathar
# chhobi" are not).
_PHRASES: dict[str, str] = {
    # ---- blood ----
    "রক্ত পরীক্ষা": BLOOD,
    "রক্তের পরীক্ষা": BLOOD,
    "রক্ত টেস্ট": BLOOD,
    "রক্ত দেওয়া": BLOOD,
    "রক্তের টেস্ট": BLOOD,
    "खून की जांच": BLOOD,
    "खून टेस्ट": BLOOD,
    "रक्त जांच": BLOOD,
    "rokto porikkha": BLOOD,
    "rokto porikha": BLOOD,
    "rokter test": BLOOD,
    "rokto test": BLOOD,
    "khoon ki jaanch": BLOOD,
    "khun ki jaanch": BLOOD,
    # English had no entries of its own at all (every other form here is bn/hi/phonetic-romanised) -- added for
    # parity, 2026-10-12.
    "blood test": BLOOD,
    "blood check": BLOOD,
    "blood work": BLOOD,
    # NOTE: there is deliberately NO unconditional "urine" entry here (unlike BLOOD, which has many different
    # blood tests so a bare mention is genuinely ambiguous). The catalogue has essentially one urine test
    # ("Urine Routine Examination"), so intercepting "urine test" here would hijack a caller directly asking
    # its price into an unhelpful "which urine test do you mean" -- confirmed by a real test regression
    # (tests/test_live_call_fixes.py::test_when_only_the_name_was_not_trusted_the_agent_asks_for_the_name[en],
    # 2026-10-12): "what is the price of a urine test" must reach the normal test_rate/entity-confirmation
    # flow, not this module. URINE is still a valid category -- see _CATEGORY_ANSWER_PHRASES below -- just only
    # ever reached as the ANSWER to "is it blood, urine, or imaging?", a context where the caller has already
    # said they do not know a specific name, so there is nothing left to hijack.
    # ---- abdomen (USG whole abdomen) ----
    "পেটের ছবি": ABDOMEN,
    "পেটের পরীক্ষা": ABDOMEN,
    "পেটের আলট্রাসাউন্ড": ABDOMEN,
    "পেট স্ক্যান": ABDOMEN,
    "पेट की तस्वीर": ABDOMEN,
    "पेट की जांच": ABDOMEN,
    "peter chhobi": ABDOMEN,
    "peter porikkha": ABDOMEN,
    "peter porikha": ABDOMEN,
    "pete ki jaanch": ABDOMEN,
    "pet ki jaanch": ABDOMEN,
    "pet ki tasvir": ABDOMEN,
    "abdomen scan": ABDOMEN,
    "stomach picture": ABDOMEN,
    "stomach scan": ABDOMEN,
    # ---- head ----
    "মাথার ছবি": HEAD,
    "মাথার পরীক্ষা": HEAD,
    "মাথা স্ক্যান": HEAD,
    "सिर की तस्वीर": HEAD,
    "सिर की जांच": HEAD,
    "mathar chhobi": HEAD,
    "mathar porikkha": HEAD,
    "mathar porikha": HEAD,
    "matha scan": HEAD,
    "sar ki jaanch": HEAD,
    "head scan": HEAD,
    "head picture": HEAD,
}

# OWNER'S INSTRUCTION, 2026-10-12: generic imaging words ("scan", "x-ray", "ultrasound") are deliberately NOT in
# _PHRASES above: unlike every phrase there, these are short enough to collide with a REAL test name a caller
# might say directly -- the catalogue's own "Chest X-Ray (PA view)" has the Bengali alias "বুকের এক্স-রে", which
# contains "এক্স-রে" as a plain substring. _PHRASES is checked on EVERY turn unconditionally (main.py), so a bare
# "x-ray" entry there would hijack a direct "what's the price of a chest x-ray" query before it ever reached the
# real lookup -- exactly the kind of regression this instruction said not to cause. These are only ever checked
# by match_category_answer() below, used ONLY as the answer to the "is it blood, urine, or imaging?" question
# this module's caller (main.py) asks after a caller has already said they do not know the test's name -- a
# context where "scan"/"x-ray" meaning the broad category, not a specific test, is the only sensible reading.
_CATEGORY_ANSWER_PHRASES: dict[str, str] = {
    "রক্ত": BLOOD,
    "ব্লাড": BLOOD,
    "खून": BLOOD,
    "रक्त": BLOOD,
    "ब्लड": BLOOD,
    "blood": BLOOD,
    "পেশাব": URINE,
    "প্রস্রাব": URINE,
    "पेशाब": URINE,
    "मूत्र": URINE,
    "urine": URINE,
    "স্ক্যান": IMAGING,
    "এক্স-রে": IMAGING,
    "এক্সরে": IMAGING,
    "আল্ট্রাসাউন্ড": IMAGING,
    "স্ক্যানিং": IMAGING,
    "ইমেজিং": IMAGING,
    "स्कैन": IMAGING,
    "एक्स रे": IMAGING,
    "एक्सरे": IMAGING,
    "अल्ट्रासाउंड": IMAGING,
    "इमेजिंग": IMAGING,
    "scan": IMAGING,
    "x-ray": IMAGING,
    "x ray": IMAGING,
    "xray": IMAGING,
    "ultrasound": IMAGING,
    "imaging": IMAGING,
}


def match_category_answer(text: str) -> str | None:
    """BLOOD / URINE / IMAGING if `text` answers "is it blood, urine, or imaging?", else None. Separate from
    match_lay_term() on purpose -- see the module-level comment above _CATEGORY_ANSWER_PHRASES for why these
    particular short words are unsafe to check on every turn."""
    norm = normalise(text)
    if not norm:
        return None
    hits = [(len(phrase), cat) for phrase, cat in _CATEGORY_ANSWER_PHRASES.items() if normalise(phrase) in norm]
    return max(hits)[1] if hits else None


# Words a caller wraps a phrase in that carry no category of their own -- stripped before matching so "আমার একটা
# পেটের ছবি করাতে হবে" still contains the bare phrase "পেটের ছবি".
_STRIP = frozenset({"amar", "amake", "ekta", "korte", "korate", "hobe", "hocche", "chai", "please", "mera", "mujhe"})

_BENGALI_RANGE = re.compile(r"[ঀ-৿]")
_DEVANAGARI_RANGE = re.compile(r"[ऀ-ॿ]")


def normalise(text: str) -> str:
    """NFC, lower case, punctuation to spaces -- the same comparison form agent/gazetteer.py uses, kept separate
    (not imported) because this file matches whole PHRASES, not names, and never needs gazetteer's fuzzy tiers: a
    lay term is either recognisably one of the phrases above or it is not, never a sound-alike guess."""
    t = unicodedata.normalize("NFC", text or "").lower()
    kept = "".join(
        ch if (ch.isalnum() or unicodedata.category(ch) in ("Mn", "Mc") or ch.isspace()) else " " for ch in t
    )
    words = [w for w in kept.split() if w not in _STRIP]
    return " ".join(words)


def match_lay_term(text: str) -> str | None:
    """BLOOD / ABDOMEN / HEAD / URINE if `text` contains one of the recognised phrases, else None. The longest
    matching phrase wins when more than one appears (a transcript naming two categories in one sentence is rare
    enough that picking the more specific -- usually longer -- phrase is the safer default; a caller who meant
    both will be asked again on the next turn regardless, since only one category is ever acted on per turn)."""
    norm = normalise(text)
    if not norm:
        return None
    hits = [(len(phrase), cat) for phrase, cat in _PHRASES.items() if normalise(phrase) in norm]
    return max(hits)[1] if hits else None


# OWNER'S INSTRUCTION, 2026-10-12: "wo test ka naam toh nhi pta" / "doctor ne bola tha ek test ka naam kya tha pta
# nahi" -- a caller who was just asked which test, and does not know its name at all (not even a category).
# Deterministic phrase matching, same reasoning as every other detector in this codebase (agent/call_end.py,
# agent/booking_flow.py's classify_yes_no): zero-latency, no model, and "did the caller say they don't know" is
# exactly the kind of cheap substring question that does not need one. main.py uses this ONLY as a fallback when
# match_lay_term() above found no category AND the caller was just asked for a test name -- it is never checked
# on an ordinary turn, so it cannot misfire on an unrelated "I don't know" said about something else entirely.
_DOES_NOT_KNOW_PHRASES: tuple[str, ...] = (
    # Bengali
    "নাম জানি না",
    "নাম মনে নেই",
    "নাম বলতে পারছি না",
    "টেস্টের নাম জানি না",
    "ঠিক জানি না",
    "ki test bola hoyni",
    # Hindi
    "नाम नहीं पता",
    "नाम नहीं जानता",
    "नाम नहीं जानती",
    "नाम याद नहीं",
    "टेस्ट का नाम नहीं पता",
    "ठीक से पता नहीं",
    "naam pata nahi",
    "naam nahi pata",
    # English
    "don't know the name",
    "dont know the name",
    "don't know the test name",
    "dont know the test name",
    "do not know the name",
    "do not know the test name",
    "not sure what test",
    "not sure which test",
    "no idea what test",
    "doctor didn't tell me the name",
    "doctor did not tell me the name",
)


def expresses_not_knowing_test_name(text: str) -> bool:
    """True if `text` says, in any of the three languages, "I don't know the test's name" -- see the module-level
    comment above _DOES_NOT_KNOW_PHRASES for the scope this is meant to be used in."""
    norm = normalise(text)
    return bool(norm) and any(normalise(p) in norm for p in _DOES_NOT_KNOW_PHRASES)
