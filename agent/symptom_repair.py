"""A symptom word the recogniser heard as a different word, put back where the sentence can only mean the symptom (owner, 2026-10-30: "আমার খুব জাত" -- he meant "আমার খুব জ্বর", fever).

"জাত" (caste / born), "জার", "জর" sound like "জ্বর" on a phone line. In a SHORT sentence that is a complaint -- "my ... very much", "I have ...", "he has ..." -- there is no other reading, so the word is
replaced before anything else reads the sentence. Only the listed confusions, only in a short sentence with a complaint cue; a longer sentence, or one without a cue, is left exactly as heard.
"""

from __future__ import annotations

import re

# confusion -> the symptom word it stands for
_BN = {
    "জাত": "জ্বর", "জার": "জ্বর", "জর": "জ্বর", "জ্বার": "জ্বর", "জ্বোর": "জ্বর", "জাড়": "জ্বর", "জারি": "জ্বর", "জ্বরে": "জ্বর", "জ্বড়": "জ্বর", "জুর": "জ্বর",
}
_CUES = ("আমার", "আমাকে", "আমি", "আমাদের", "ওর", "ওনার", "ওনার", "ওঁর", "বাচ্চার", "মায়ের", "বাবার", "খুব", "অনেক", "বেশ", "আছে", "হয়েছে", "হচ্ছে", "হয়", "গায়ে", "সকাল থেকে", "কাল থেকে")
_MAX_WORDS = 6


def repair(text: str) -> str:
    """The sentence with a garbled symptom word put right; the sentence itself when there is nothing to repair."""
    if not text:
        return text
    words = text.split()
    if len(words) > _MAX_WORDS:
        return text
    plain = [w.strip(".,?!।") for w in words]
    if not any(c in " ".join(plain) for c in _CUES):
        return text
    hit = False
    for i, w in enumerate(plain):
        fix = _BN.get(w)
        if fix and fix != w:
            words[i] = words[i].replace(w, fix)
            hit = True
    return " ".join(words) if hit else text
