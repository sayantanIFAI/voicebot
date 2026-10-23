"""'Who are the doctors in cardio?' -- a question about a whole department, not one doctor.

The department's short form ("cardio", "ortho", "gynae", "হার্টের ডাক্তার") is resolved by the clinic API (booking_service._DEPT_ALIASES); this module only
recognises that the caller is asking for the LIST of doctors, so the agent asks the API and answers from the real doctors table.
"""

from __future__ import annotations

import re
import unicodedata

_DOCTOR = ("ডাক্তার", "ডাক্তারেরা", "ডাক্তাররা", "डॉक्टर", "डॉक्टरों", "doctor", "doctors", "docs")
_LIST = (
    # Bengali
    "কারা", "কে কে", "কোন কোন", "কি কি", "কী কী", "কয়জন", "কতজন", "কত জন", "কারা বসেন", "কারা আছেন",
    # Hindi
    "कौन कौन", "कौन-कौन", "कौन से", "कितने", "कौन हैं", "लिस्ट", "सूची",
    # English
    "who are", "who is there", "which doctors", "what doctors", "list of doctors", "doctors in", "doctors for", "doctors are there", "doctors available",
    "how many doctors", "doctors do you have", "doctors you have",
)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    return " ".join(re.sub(r"[.,;:!?\"'()।॥]+", " ", s).split())


def _has(t: str, words: tuple[str, ...]) -> bool:
    padded = f" {t} "
    return any((f" {w} " in padded) if w.isascii() else (w in t) for w in words)


_SITS = (
    "বসছে", "বসছেন", "বসবে", "বসবেন", "বসেন", "বসে", "বসতেন", "বসেছেন", "বসছে।",
    "बैठ", "बैठते", "बैठेंगे", "बैठ रहे", "बैठती",
    "sits", "sitting", "sit ", "available", "consulting", "attending", "are there", "will be there",
)


def asks_who_sits(text: str) -> bool:
    """"কালকে কে কে বসছে কার্ডিওলজি" -- who sits (in a department) on a day. There is no doctor word in it, and "কে কে" ("who, who") is a question about people, never a
    doctor named "কে কে" (real call 2026-10-18)."""
    t = _norm(text)
    return bool(t) and len(t.split()) <= 16 and _has(t, _LIST) and _has(t, _SITS)


def asks_doctor_list(text: str) -> bool:
    t = _norm(text)
    return bool(t) and len(t.split()) <= 16 and ((_has(t, _DOCTOR) and _has(t, _LIST)) or asks_who_sits(text))


# KCD-lead-A: "There is an orthopedic doctor." / "I want to see an orthopaedic doctor." / "Do you have a dentist?" -- a request for a SPECIALTY, with no
# list word in it ("who are", "which doctors"). asks_doctor_list() above needs one, so such a sentence fell to the intent model, which took the
# specialty for a doctor's NAME ("Orthopaedics Doctor") and answered "we have no doctor called ...". This only recognises that a doctor / specialist /
# department is being asked for; WHICH department it is (any department in the clinic's database, by its own name or a short form) is decided by the
# clinic API, never here, so a department added to the database later needs no change in this file.
_DOCTOR_REQUEST_RE = re.compile(
    r"\b(doctors?|docs?|dr|specialists?|physicians?|consultants?|surgeons?|dentists?|departments?|dept|daktar|dactar|docter)\b"
    r"|ডাক্তার|ডাক্তর|বিশেষজ্ঞ|বিভাগ|डॉक्टर|डाक्टर|विशेषज्ञ|विभाग",
    re.I,
)


def mentions_doctor_request(text: str) -> bool:
    """True when a short sentence asks for / about a doctor, specialist or department at all -- the gate before the clinic API is asked which
    department (if any) it means. A named doctor is excluded by the caller (it knows the clinic's doctor catalogue); a sentence about no
    department at all ("book a doctor") simply comes back unmatched from the API and the turn goes on unchanged."""
    t = _norm(text)
    return bool(t) and len(t.split()) <= 20 and _DOCTOR_REQUEST_RE.search(t) is not None
