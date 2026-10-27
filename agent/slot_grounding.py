"""Drop a test or doctor name the model made up, before anything is looked up or priced.

Real call 2026-10-14: the caller said "reply in english". The intent model (Qwen2.5-7B, run on the pod against that
exact sentence) answered `intent=test_rate, test_names=["CBC", "ECG"]` -- two tests nobody mentioned, lifted from the
examples in its own prompt. The code then did what it is built to do with a test list: priced both from the clinic
catalogue and said "CBC 400, ECG 300, total 700". Every number was true; the question was not asked.

This is the code half of CLAUDE.md's one rule: the model may decide what the caller wants, it may never put a fact in
play on its own. A slot is supposed to be a LITERAL span of what was said (agent/llm.py: "do not translate or normalize
it"). This module checks the cheapest version of that claim and nothing more.

THE RULE for a TEST name, and why it is this loose: dropped only when the caller's sentence gives it NO support at
all, i.e. when ALL of these hold:
  * the sentence contains no word about tests or prices at all ("test", "rate", "total", "scan", ...), and
  * the sentence names no test the catalogue knows, by any of its spoken forms, and
  * the model's name does not literally appear in the sentence (or nearly: a one-letter ASR difference).
A garbled name inside a real question ("ক্বক র সাথে উরিক এসিদ টেস্ট ... টোটাল কত") has test words in it, so the model's
reading of it is kept exactly as before: this module only ever removes names from sentences that have no test content
whatsoever, which is where the model is guessing rather than reading.

THE RULE for a DOCTOR name is narrower, on real evidence (2026-10-27) that the test rule's "a context word is
enough" leg is not: "ডাক্তার সেন" (two real doctors share the surname) and "ডক্টর ঋষা" (a recogniser's miss of a real
doctor's name) both came back with a DIFFERENT, unrelated doctor's real schedule confidently reported as this one's
-- the word "doctor" is in nearly every doctor sentence there is, so that leg alone validated any name the model
felt like filling in. A doctor name is kept only when the sentence literally names it (or nearly), OR the catalogue
itself recognised it unambiguously (`names_a_doctor` with no `doctor_candidates`), OR -- when the catalogue's
mention WAS ambiguous -- the model's claim shares a real name-token with one of the specific candidates the
catalogue offered (`doctor_candidates`; see _matches_any_candidate). A plain cross-script reading of the one doctor
actually named ("ডক্টর সেন" -> "Dr. Sen") still survives every one of these; a different doctor the catalogue never
offered as a candidate does not, however plausible-sounding, catalogue-real and politely phrased the answer is.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

# A sentence containing any of these is, at least loosely, ABOUT tests / prices, so the model's reading is left alone.
_TEST_CONTEXT = (
    "test", "টেস্ট", "टेस्ट", "পরীক্ষা", "परीक्षा", "जांच", "जाँच", "scan", "স্ক্যান", "स्कैन", "x-ray", "xray",
    "এক্সরে", "एक्सरे", "profile", "প্রোফাইল", "प्रोफाइल", "sample", "স্যাম্পল", "सैंपल",
    "rate", "রেট", "रेट", "দাম", "দর", "কত", "টাকা", "price", "cost", "fee", "fees", "charge", "how much", "total",
    "টোটাল", "टोटल", "कीमत", "कितना", "कितने", "भाव", "पैसे", "खर्च", "फीस", "মোট", "कुल",
    "prep", "preparation", "fasting", "ফাস্টিং", "फास्टिंग", "প্রস্তুতি", "तैयारी",
    "report", "রিপোর্ট", "रिपोर्ट", "book", "বুক", "बुक", "pathology", "lab",
)
_DOCTOR_CONTEXT = (
    "doctor", "dr", "ডাক্তার", "ডক্টর", "डॉक्टर", "डाक्टर", "डॉ", "specialist", "physician", "consultant",
    "appointment", "অ্যাপয়েন্টমেন্ট", "अपॉइंटमेंट", "available", "chamber", "চেম্বার", "चेंबर", "book", "বুক", "बुक",
)
# Owner, 2026-10-27 (real call): "চুয়াল্লিশ সিআইটি রোড ..." (44 CIT Road ...), answering "what is the address?",
# came back from the model as patient_age=44 -- the house number in the address, not an age -- and overwrote the
# real age already on the booking, asked the caller to confirm it, and never looked at the address at all. Same
# root cause as doctor_name/test_name needing a context word (has_doctor_context/has_test_context) before the
# model's reading is trusted: a real age answer always names what it is ("বছর"/"বয়স"/"years"/"old"); a bare
# number inside an unrelated sentence does not.
_AGE_CONTEXT = (
    "বয়স", "বয়েস", "বছর", "বছরের", "age", "aged", "years", "year", "yrs", "yr", "old",
    "उम्र", "उमर", "आयु", "साल", "वर्ष",
)

_PUNCT = re.compile(r"[.,;:!?\"'()\[\]{}।॥\-–—_/]+")


def _norm(text: str) -> str:
    s = unicodedata.normalize("NFC", text or "").lower().replace("़", "").replace("়", "")
    return " ".join(_PUNCT.sub(" ", s).split())


def _has_any(norm_text: str, cues: tuple[str, ...]) -> bool:
    padded = f" {norm_text} "
    for cue in cues:
        c = _norm(cue)
        if c.isascii():
            if f" {c} " in padded or (len(c) > 3 and c in norm_text):  # whole word; longer stems also inside a word
                return True
        elif c in norm_text:
            return True
    return False


# Real call 2026-10-17: the model returned test_names=["ক্রিকাসিড", "র", "ওই", "থাইরাইড টেস্ট"] -- "র" is a letter and "ওই" means "that".
# Looked up as tests, "র" even matched ESR / Blood Sugar by sound and polluted the reply. A word of one or two letters, or a
# function word, or the bare word "test", is never a test name.
_NOT_A_TEST_NAME = frozenset(
    "র ওই এই আর যে তো টা টি এটা ওটা সেটা আমাকে আমার আপনাদের আপনার আছে হয় হবে কত টাকা টোটাল মোট টেস্ট টেস্টটা টেস্টের পরীক্ষা "
    "ये वो और का की के है हैं कितना टोटल कुल टेस्ट आपके मेरा "
    "the and of is it this that test tests total price rate cost".split()
)


def is_plausible_test_name(name) -> bool:
    if not isinstance(name, str):
        return False
    n = _norm(name)
    if n in _NOT_A_TEST_NAME:
        return False
    letters = [c for c in n if c.isalpha()]
    return len(letters) >= 3


def has_test_context(text: str) -> bool:
    return _has_any(_norm(text), _TEST_CONTEXT)


def has_doctor_context(text: str) -> bool:
    return _has_any(_norm(text), _DOCTOR_CONTEXT)


def has_age_context(text: str) -> bool:
    return _has_any(_norm(text), _AGE_CONTEXT)


# real calls, 2026-10-27: "ডাক্তার সেন" (two real doctors share the surname "Sen" / "সেন") was answered with a THIRD,
# unrelated doctor's real schedule -- not a transliteration of "Sen", a different surname entirely ("Chanda" on one
# attempt, "Ray" on another). "ডক্টর ঋষা" (the recogniser's miss of "ত্রিষা"/"Trisha") got a FOURTH unrelated
# doctor's schedule the same way. `has_doctor_context` alone ("the sentence says the word doctor") was enough to
# wave the model's claim through in both cases -- true of nearly every doctor sentence there is, so it was no
# check at all. The fix below is catalogue-driven, not a list of names: when the catalogue itself names the real
# doctors a shared surname could mean (`doctor_candidates`), the model's claim must be one of them (by surname/
# given-name overlap, so a plain cross-script reading like "ডক্টর সেন" -> "Dr. Sen" still survives) or literally
# in the sentence -- never a different name the catalogue never offered as a candidate.
_DOCTOR_TITLE_TOKENS = frozenset({"dr", "doctor", "doc", "dr.", "ডাক্তার", "ডক্টর", "ডাঃ", "डॉक्टर", "डॉ", "डाक्टर"})


def _name_tokens(s: str) -> set[str]:
    return {t for t in _norm(s or "").split() if t not in _DOCTOR_TITLE_TOKENS and len(t) > 1}


def _matches_any_candidate(value: str, candidates: tuple[str, ...]) -> bool:
    """Does `value` (what the model claims) share a real name-token -- surname or given name, any script -- with
    at least one of the candidates the catalogue itself offered for the surname/alias the caller actually said?
    A plain cross-script reading of the SAME doctor (token "sen" in both) matches; an unrelated name the model
    invented (no shared token with any candidate) does not."""
    v_tokens = _name_tokens(value)
    if not v_tokens:
        return False
    return any(v_tokens & _name_tokens(c) for c in candidates if c)


def _literally_in(value: str, norm_text: str) -> bool:
    """The model's name appears in the sentence, or is within one or two letters of a stretch of it."""
    v = _norm(value)
    if not v:
        return True  # nothing to check
    if v in norm_text:
        return True
    v_tokens = v.split()
    t_tokens = norm_text.split()
    n = len(v_tokens)
    for size in {max(1, n - 1), n, n + 1}:
        for i in range(0, max(0, len(t_tokens) - size) + 1):
            window = " ".join(t_tokens[i : i + size])
            if window and difflib.SequenceMatcher(None, v, window).ratio() >= 0.8:
                return True
    return False


def _clean_names(values, norm_text: str):
    kept = [v for v in values if isinstance(v, str) and _literally_in(v, norm_text)]
    return kept, [v for v in values if isinstance(v, str) and v not in kept]


def ground_slots(
    text: str,
    slots: dict | None,
    *,
    names_a_test: bool = False,
    names_a_doctor: bool = False,
    doctor_candidates: tuple[str, ...] = (),
    trust_age: bool = False,
):
    """(slots, dropped). `names_a_test` / `names_a_doctor`: the catalogue found a test / doctor in `text` by one of its
    spoken forms (the caller of this function asks agent.transcript_rules.AliasTable). `doctor_candidates`: when that
    doctor mention was AMBIGUOUS (a surname/alias two or more real doctors share, e.g. two Sens), the real doctors the
    catalogue itself says it could be -- the model's claimed doctor_name must be one of them (see _matches_any_candidate)
    or literally in the sentence; a name the catalogue never offered as a candidate is dropped regardless of any other
    signal, so the turn asks "which one?" instead of confidently reporting a doctor nobody asked about (CLAUDE.md
    "Doctor Nobody"). `trust_age`: the age in `slots` was set deterministically (agent/booking_capture.py's own
    read-back/confirm flow), not guessed by the model, so the age-context check below -- there to catch the MODEL
    misreading an unrelated number as an age -- does not apply to it; a caller confirming a read-back with a bare
    "yes" (no "বয়স"/"বছর" in that one word either) must not have the age it just confirmed thrown away by the same
    check that protects against the model's mistake. Never raises; never edits the dict it is given."""
    if not isinstance(slots, dict) or not slots:
        return slots, []
    norm_text = _norm(text)
    out = dict(slots)
    dropped: list[str] = []

    # a junk "name" is dropped whatever the sentence is about
    for key in ("test_name", "test_names"):
        value = out.get(key)
        if isinstance(value, str) and not is_plausible_test_name(value):
            dropped.append(value)
            out[key] = None
        elif isinstance(value, list):
            good = [v for v in value if is_plausible_test_name(v)]
            if len(good) != len(value):
                dropped.extend(v for v in value if v not in good)
                out[key] = good or None

    if not (names_a_test or has_test_context(text)):
        for key in ("test_name", "test_names"):
            value = out.get(key)
            if isinstance(value, str):
                if not _literally_in(value, norm_text):
                    dropped.append(value)
                    out[key] = None
            elif isinstance(value, list):
                kept, gone = _clean_names(value, norm_text)
                if gone:
                    dropped.extend(gone)
                    out[key] = kept or None
    value = out.get("doctor_name")
    if isinstance(value, str):
        if doctor_candidates:
            drop = not _matches_any_candidate(value, doctor_candidates) and not _literally_in(value, norm_text)
        else:
            drop = not names_a_doctor and not _literally_in(value, norm_text)
        if drop:
            dropped.append(value)
            out["doctor_name"] = None

    if not trust_age and out.get("patient_age") is not None and not has_age_context(text):
        dropped.append(str(out["patient_age"]))
        out["patient_age"] = None

    # Owner, 2026-10-27: a patient_name has no catalogue to ground against (unlike a test or doctor name), so the
    # model guessing one from a long, confusing or rambling sentence that never actually named anyone went
    # uncaught -- the booking then believed the name step was already done and skipped straight to asking for the
    # phone number. Same literal-or-near-literal rule as doctor_name, no context exemption needed: a real name the
    # caller gave is, by construction, present in their own sentence (fuzzy ASR noise included); one the model
    # invented is not.
    value = out.get("patient_name")
    if isinstance(value, str) and not _literally_in(value, norm_text):
        dropped.append(value)
        out["patient_name"] = None
    return out, dropped
