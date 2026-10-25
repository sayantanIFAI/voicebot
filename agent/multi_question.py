"""Two or three questions in one breath ("what is the CBC price AND where are you", "I want to book AND what is your address").

Owner, 2026-10-25: "if two questions are asked, give all the answers; any similar bound questions -- the voicebot must answer each, or ask a related follow-back". A sentence is cut at the
words that join questions ("আর", "এবং", "ও", "তো", "and", "also", "और", "फिर", "?" ...) into short parts; each part is then looked at on its own by the same routine-question recogniser
that answers a single question (agent/fast_path.py). This module is the pure half (the cutting); main.py's _answer_multi_questions asks the recogniser and speaks the answers.
"""

from __future__ import annotations

import re

# the words that join one question to the next (Bengali, Hindi, English, and the Latin spellings callers use)
_JOINERS = re.compile(
    r"\s+(?:আর একটা|আরেকটা|আরেকটি|এছাড়া|এছাড়াও|এবং|আর|তারপর|তারপরে|সাথে সাথে|সেই সাথে|ও|তো|आर एक|एक और|और|तथा|फिर|साथ में|"
    r"and also|and|also|plus|then|aar ekta|ar ekta|ar|aar|ebong|tarpor)\s+",
    re.IGNORECASE,
)
_QMARK = re.compile(r"[?？]+\s*")
MAX_PARTS = 4
MIN_WORDS = 2


def split_questions(text: str) -> list[str]:
    """The text cut into parts at question marks and joining words; a part of fewer than two words is glued back onto its neighbour (so "ও" or "তো" inside a sentence does not
    shred it). One part comes back when there is nothing to cut."""
    text = " ".join((text or "").split())
    if not text:
        return []
    raw: list[str] = []
    for chunk in _QMARK.split(text):
        raw.extend(p.strip(" ,।.") for p in _JOINERS.split(f" {chunk} "))
    raw = [p for p in raw if p]
    parts: list[str] = []
    for p in raw:
        if parts and (len(p.split()) < MIN_WORDS or len(parts[-1].split()) < MIN_WORDS):
            parts[-1] = f"{parts[-1]} {p}"
        else:
            parts.append(p)
    return parts[:MAX_PARTS] if len(parts) > 1 else [text]
