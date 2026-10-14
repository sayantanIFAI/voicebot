"""Drop a test or doctor name the model made up, before anything is looked up or priced.

Real call 2026-10-14: the caller said "reply in english". The intent model (Qwen2.5-7B, run on the pod against that
exact sentence) answered `intent=test_rate, test_names=["CBC", "ECG"]` -- two tests nobody mentioned, lifted from the
examples in its own prompt. The code then did what it is built to do with a test list: priced both from the clinic
catalogue and said "CBC 400, ECG 300, total 700". Every number was true; the question was not asked.

This is the code half of CLAUDE.md's one rule: the model may decide what the caller wants, it may never put a fact in
play on its own. A slot is supposed to be a LITERAL span of what was said (agent/llm.py: "do not translate or normalize
it"). This module checks the cheapest version of that claim and nothing more.

THE RULE, and why it is this loose: a name from the model is dropped only when the caller's sentence gives it NO
support at all, i.e. when ALL of these hold:
  * the sentence contains no word about tests or prices at all ("test", "rate", "total", "scan", ...), and
  * the sentence names no test (or doctor) the catalogue knows, by any of its spoken forms, and
  * the model's name does not literally appear in the sentence (or nearly: a one-letter ASR difference).
A garbled name inside a real question ("ক্বক র সাথে উরিক এসিদ টেস্ট ... টোটাল কত") has test words in it, so the model's
reading of it is kept exactly as before: this module only ever removes names from sentences that have no test content
whatsoever, which is where the model is guessing rather than reading.
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


def has_test_context(text: str) -> bool:
    return _has_any(_norm(text), _TEST_CONTEXT)


def has_doctor_context(text: str) -> bool:
    return _has_any(_norm(text), _DOCTOR_CONTEXT)


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


def ground_slots(text: str, slots: dict | None, *, names_a_test: bool = False, names_a_doctor: bool = False):
    """(slots, dropped). `names_a_test` / `names_a_doctor`: the catalogue found a test / doctor in `text` by one of its
    spoken forms (the caller of this function asks agent.transcript_rules.AliasTable). Never raises; never edits the
    dict it is given."""
    if not isinstance(slots, dict) or not slots:
        return slots, []
    norm_text = _norm(text)
    out = dict(slots)
    dropped: list[str] = []

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
    if not (names_a_doctor or has_doctor_context(text)):
        value = out.get("doctor_name")
        if isinstance(value, str) and not _literally_in(value, norm_text):
            dropped.append(value)
            out["doctor_name"] = None
    return out, dropped
