"""Respell Bengali words the Bengali voice says badly, just before they reach the voice.

Real call 2026-10-14: the caller named six words the voice speaks poorly (বলুন, দুঃখিত, সময়টা, কয়েক, নেওয়া, হয়ে).
agent/pronunciation.py cannot help: it maps ENGLISH words to Bengali script. These are ordinary Bengali words, so what
is needed is a spelling that makes the voice produce the right SOUND.

HOW THE SPELLINGS IN _RESPELL WERE CHOSEN (tools/respell_probe.py is the instrument, kept in the repo):
each candidate spelling is synthesised by the pod's Bengali voice, played through the Bengali recogniser, and kept only if
the recogniser reads the INTENDED word back. That is a measurement of intelligibility, not of naturalness: it finds words
the voice garbles badly enough that a recogniser cannot recover them, and it cannot tell "good" from "acceptable". A native
listener signs the final list off (REVIEWED below stays empty until one does), exactly as agent/pronunciation.py's does.

A word is replaced only when it is a WHOLE word in the table; sentence structure, punctuation and every other word are
untouched. The table is applied after number/date verbalisation (agent/speech_norm.verbalize) and before the cache key is
built, so a respelled clip is cached under what was actually sent to the voice.
"""

from __future__ import annotations

import os
import re
import unicodedata

# spoken word (as written everywhere else in the repo) -> spelling that makes the voice say it right.
_RESPELL: dict[str, str] = {
    # CHOSEN BY A NATIVE LISTENER, 2026-10-15, from numbered audio candidates (the round-trip recogniser was too forgiving
    # to judge: it read back words a Bengali ear still hears as Hindi-accented or wrong). The pattern in the winners: the voice
    # reads স as Hindi "s" where Bengali says "sh", so native words start with শ.
    "প্রয়োজন": "প্রোজোন",  # was candidate 3 of 5
    "সাড়া": "শাঁড়া",  # candidate 4 of 4
    "সময়টা": "শমোইটা",  # candidate 4 of 5 (the earlier machine-chosen সমোইটা is replaced)
    "নেওয়া": "নেওয়াা",  # candidate 5 of 5; listener: better, "not perfect"
}

REVIEWED: set[str] = {"প্রয়োজন", "সাড়া", "সময়টা"}  # approved by a native listener (নেওয়া: better, not perfect)

_BENGALI_WORD = re.compile(r"[ঀ-৿]+")


def _key(word: str) -> str:
    return unicodedata.normalize("NFC", word)


def respell(word: str) -> str:
    """The spelling to send to the voice for one Bengali word; the word itself when it needs no help."""
    return _RESPELL.get(_key(word), word)


def apply(text: str, lang: str) -> str:
    """Respell every whole Bengali word in `text` that the table knows. Only the Bengali voice is touched."""
    if lang != "bn" or not _RESPELL or not text or os.environ.get("BN_RESPELL", "1") == "0":
        return text
    return _BENGALI_WORD.sub(lambda m: respell(m.group(0)), text)


def table() -> dict[str, str]:
    return dict(_RESPELL)
