"""Is this a hello? Only a hello is answered with the introduction ("Sonoscan Vaani. How can I help you?").

Real call 2026-10-17: a bare "okay" was classed as small talk and the AI introduced itself again, as if the call had just begun.
"""

from __future__ import annotations

import re
import unicodedata

_GREETINGS = (
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "namaste", "namaskar", "hallo",
    "নমস্কার", "নমস্তে", "হ্যালো", "হাই", "সুপ্রভাত", "শুভ সকাল", "শুভ সন্ধ্যা",
    "नमस्ते", "नमस्कार", "हेलो", "हैलो", "हलो", "हाय", "सुप्रभात",
)


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    return " ".join(re.sub(r"[.,;:!?\"'()।॥]+", " ", s).split())


def is_greeting(text: str) -> bool:
    t = _norm(text)
    padded = f" {t} "
    return bool(t) and any((f" {g} " in padded) if g.isascii() else (g in t) for g in _GREETINGS)
