"""KCD-462: "the first clause is emitted as soon as it renders... a
preparation instruction of four sentences begins within the first-audio
budget."

Splits a reply into clauses at the SAME sentence-final boundaries
prosody already treats as natural pause points ("।" the Bengali danda,
and ".", "?", "!" for Hindi/English) -- not a new segmentation scheme,
just reusing the boundary a native reader already hears as one. main.py's
_speak() synthesizes and sends each clause as its own short TTS call
instead of one call for the whole reply, so the first clause reaches the
caller's ear as soon as ITS OWN (much shorter) synthesis finishes,
instead of waiting for the entire reply to render first.
"""

from __future__ import annotations

import re

from agent.speech_norm import contains_critical_figure

_RE_CLAUSE_BOUNDARY = re.compile(r"([।.!?])")

# Below this length, splitting only adds TTS round-trip overhead for no
# perceptible first-audio win -- most replies in this system already
# fit in one or two short clauses.
MIN_CHARS_TO_SPLIT = 60


def split_into_clauses(text: str) -> list[str]:
    """Sentence-final punctuation stays attached to the clause it ends
    (a caller should still hear "...লাগবে।" not "...লাগবে" then a bare
    "।"). Whitespace-only fragments are dropped. A text with no
    sentence-final punctuation at all -- or shorter than
    MIN_CHARS_TO_SPLIT -- comes back as a single-item list, so callers
    do not need to special-case "not worth splitting"."""
    text = text.strip()
    # Owner, 2026-10-31 ("আমি নম্বরটা শুনেছি 9 6 3 5 5 , 8 8 2 0 1। রোগীর বয়স কত?" was spoken slowly all the way through): a reply that carries a phone number / price / reference is slowed
    # clause by clause (main.py: contains_critical_figure), and a short reply used to be ONE clause, so the question after the number was slowed too. With a figure in it the reply is always split, so only the
    # sentence that holds the figure is slow.
    if not text or (len(text) < MIN_CHARS_TO_SPLIT and not contains_critical_figure(text)):
        return [text] if text else []

    parts = _RE_CLAUSE_BOUNDARY.split(text)
    clauses = []
    buf = ""
    for part in parts:
        buf += part
        if _RE_CLAUSE_BOUNDARY.fullmatch(part):
            stripped = buf.strip()
            if stripped:
                clauses.append(stripped)
            buf = ""
    if buf.strip():
        clauses.append(buf.strip())
    return clauses if clauses else [text]
