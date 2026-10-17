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


def asks_doctor_list(text: str) -> bool:
    t = _norm(text)
    return bool(t) and len(t.split()) <= 16 and _has(t, _DOCTOR) and _has(t, _LIST)
