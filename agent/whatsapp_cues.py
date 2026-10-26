"""WhatsApp, as callers and recognisers actually say and write it (owner, 2026-10-23, items 15 and 25's note).

Two jobs:

1. SPELLING. A recogniser writes the word every way it was heard -- "whatsap", "watsapp", "botsapp", "whats up", "ওয়াটসঅ্যাপ", "হোয়াটস অ্যাপ", "व्हाट्सएप",
   "वाट्सअप". `canonicalise` rewrites every such spelling to ONE form per script, so every cue elsewhere (billing_cues, owner_answers, ...) that looks for the
   word finds it. Only the matching copy of the caller's words is rewritten; nothing the agent SPEAKS comes from here.
2. THE ANSWER to "does this number have WhatsApp?" -- yes / no / "another number" in English, Hindi and Bengali, in their own scripts or in Latin letters.

Words are compared as whole tokens (not regex \\b): Bengali and Hindi vowel signs are not word characters, so a \\b would cut inside a word.
"""

from __future__ import annotations

import re

# Latin: wh|w|b|bh|v|h + a|e|o + t + (s + app|up|ep | app).  "whatsapp" "whatsap" "watsapp" "botsapp" "bhatsapp" "whats app" "what's up" "whats-app" "wotsapp"
_LATIN = re.compile(
    r"(?<![A-Za-z])(?:wh?|bh?|vh?|h)[aeo]?t+'?(?:s\s*-?\s*(?:a+p+|u+p+|e+p+)|\s*-?\s*a+p+)(?![A-Za-z])"
    r"|(?<![A-Za-z])w[ah]?sa+p+(?![A-Za-z])",
    re.IGNORECASE,
)
# Bengali script: (হোয়াট|ওয়াট|...)(স)(অ্যাপ|এপ|আপ|াপ) with an optional space before the last syllable
_BENGALI = re.compile(
    r"(?:হোয়্যাট|হোয়াট|ওয়্যাট|ওয়াট|হোআট|হোয়াত|হ্যাট|ভাট|বট|হোয়ট|ওয়ট|হাট)্?স?্?\s*(?:অ্যাপ|এ্যাপ|অ্যপ|এপ|আপ|অপ|্যাপ|াপ)"
)
# Devanagari: (व्हाट|वाट|वॉट|ह्वाट|बॉट)(्स)(एप|ऐप|अप|आप|ाप)
_DEVANAGARI = re.compile(r"(?:व्?ह?[ाॉो]?ट|ह्व[ाॉो]?ट|ब[ाॉो]?ट)्?स्?\s*(?:ए|ऐ|अ|आ|ा)?प")

CANONICAL = {"en": "WhatsApp", "bn": "হোয়াটসঅ্যাপ", "hi": "व्हाट्सएप"}


def canonicalise(text: str) -> str:
    """Every spelling of WhatsApp in `text` becomes the one form of its own script. Anything else is left exactly as it was."""
    if not text:
        return text
    text = _BENGALI.sub(CANONICAL["bn"], text)
    text = _DEVANAGARI.sub(CANONICAL["hi"], text)
    return _LATIN.sub(CANONICAL["en"], text)


def mentions_whatsapp(text: str) -> bool:
    return bool(text) and bool(_LATIN.search(text) or _BENGALI.search(text) or _DEVANAGARI.search(text))


_SPLIT = re.compile(r"[^\s,.;:!?।॥\"()\[\]{}]+")

_NO = frozenset(
    {
        "no", "nope", "nah", "not", "na", "nai", "nei", "noi", "nay", "nahi", "nahin", "nahee", "nhi", "nehi", "nahii", "dont", "don't", "doesnt", "doesn't", "never", "without",
        "keypad", "button", "নেই", "নাই", "না", "নয়", "নি", "নেইকো", "নাইকো", "নেইতো", "নাহ", "নো",
        "नहीं", "नही", "नहि", "नहीँ", "ना", "न", "नइ", "नाही", "नो", "नहिं",
    }
)  # fmt: skip
_OTHER = frozenset(
    {
        "onno", "anno", "onyo", "other", "another", "different", "alada", "alad", "alag", "dusre", "dusra", "dusri", "doosre", "doosra", "doosri", "arekta", "arek", "aar",
        "অন্য", "আলাদা", "আরেকটা", "আরেক", "অপর", "দুসরা", "दूसरे", "दूसरा", "दूसरी", "अलग", "अन्य", "और",
    }
)  # fmt: skip
_YES = frozenset(
    {
        "yes", "yeah", "yep", "yup", "ya", "yah", "ok", "okay", "sure", "absolutely", "correct", "right", "thik", "theek", "hmm", "hm", "hmmm", "ha", "haa", "haan", "haain", "han", "hanji", "hain", "hai", "hae", "hya",
        "hyan", "ji", "jee", "ache", "achhe", "ase", "acche", "achche", "aache", "achi", "ahe",
        "আছে", "আছি", "হ্যাঁ", "হ্যা", "হাঁ", "হা", "জি", "ঠিক", "ওকে", "হুম", "হ্যাঁঁ",
        "हाँ", "हां", "हा", "हाँजी", "जी", "है", "हैं", "ठीक", "हम्म", "ओके", "हूँ", "हूं", "हे",
    }
)  # fmt: skip


def _tokens(text: str) -> list[str]:
    return _SPLIT.findall(canonicalise(text or "").lower())


def answer(text: str, lang: str = "en") -> str | None:
    """The reply to "does this number have WhatsApp?": "no" (it has none, or the caller has another number), "yes", or None when it is neither.
    A no wins over a yes ("haan, nei" is a no); a sentence that only names WhatsApp ("whatsapp ache", "whatsapp e") is a yes."""
    words = _tokens(text)
    if not words:
        return None
    if any(w in _NO for w in words) or any(w in _OTHER for w in words):
        return "no"
    if any(w in _YES for w in words):
        return "yes"
    if any(w.lower() == CANONICAL["en"].lower() or w in (CANONICAL["bn"], CANONICAL["hi"]) or w.startswith(CANONICAL["bn"]) or w.startswith(CANONICAL["hi"]) for w in words):
        return "yes"
    return None


def asks_if_same_number(text: str) -> bool:
    """"this one", "ei number-ei", "yahi number", "same" -- the number already given is the WhatsApp number."""
    return any(w in {"same", "this", "ei", "eita", "eitai", "yahi", "yehi", "yeh", "ye", "এই", "এটাই", "এইটাই", "यही", "यह", "ये"} for w in _tokens(text))
