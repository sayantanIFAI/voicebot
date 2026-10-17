"""Letters said with a word for each: "S for September, A for Alpha, N for November" (owner, 2026-10-18).

A caller whose name or test code was not understood often spells it that way, in any of the three languages ("এস ফর সেপ্টেম্বর", "एस फॉर सितंबर", "S as in Sam").
Only the LETTER matters; the word that follows it is how the caller makes it clear, and is thrown away. Words alone (the phonetic alphabet: "alpha bravo
charlie", or months "september august") count only in a run of three or more, so an ordinary "august" or "delta" in a sentence is not read as a letter.

`collapse(text)` rewrites the phrase to plain spaced letters ("S A N"), which the rest of the turn already understands (agent/spelled_letters.py,
agent/booking_capture.py). Nothing is invented: a letter is only ever taken from what was said.
"""

from __future__ import annotations

import re
import unicodedata

from agent import spelled_letters

_FOR = frozenset({"for", "as", "se", "from", "ফর", "ফর", "ফোর", "সে", "দিয়ে", "দিয়া", "फॉर", "फार", "फर", "से", "फ़ॉर", "for.", "as in", "in", "fa", "ফ্যার"})
_NATO = {
    "alpha": "a", "alfa": "a", "bravo": "b", "charlie": "c", "delta": "d", "echo": "e", "foxtrot": "f", "golf": "g", "hotel": "h", "india": "i", "juliet": "j", "juliett": "j",
    "kilo": "k", "lima": "l", "mike": "m", "november": "n", "oscar": "o", "papa": "p", "quebec": "q", "romeo": "r", "sierra": "s", "tango": "t", "uniform": "u", "victor": "v",
    "whiskey": "w", "whisky": "w", "xray": "x", "yankee": "y", "zulu": "z",
    "january": "j", "february": "f", "march": "m", "april": "a", "may": "m", "june": "j", "july": "j", "august": "a", "september": "s", "october": "o", "december": "d",
}
_TOKEN = re.compile(r"[^\s,;:!?\"'()।॥]+")


def _norm(s: str) -> str:
    return unicodedata.normalize("NFC", s or "")


def _letter(tok: str) -> str | None:
    t = tok.strip(".-").lower()
    if len(t) == 1 and "a" <= t <= "z":
        return t
    try:
        return spelled_letters._letter_of(tok)  # noqa: SLF001 - the Bengali / Hindi letter-name table lives there
    except Exception:  # noqa: BLE001
        return None


def collapse(text: str) -> str:
    """"my name S for Sam A for Alpha N" -> "my name S A N"; "S for September A for Alpha N for November" -> "S A N". The text is returned unchanged when
    no "<letter> for <word>" run of at least two letters is present."""
    tokens = _TOKEN.findall(_norm(text))
    out: list[str] = []
    i = 0
    pairs = 0
    while i < len(tokens):
        tok = tokens[i]
        lt = _letter(tok)
        nxt = tokens[i + 1].lower().strip(".") if i + 1 < len(tokens) else ""
        if lt and nxt in _FOR and i + 2 < len(tokens):
            skip = 3
            # "as in <word>"
            if nxt == "as" and tokens[i + 2].lower() == "in" and i + 3 < len(tokens):
                skip = 4
            out.append(lt.upper())
            pairs += 1
            i += skip
            continue
        out.append(tok)
        i += 1
    if pairs < 2:
        return text
    return " ".join(out)


def nato_run(text: str) -> str | None:
    """Three or more phonetic-alphabet (or month) words in a row -> their letters: "alpha bravo charlie" -> "ABC". Else None."""
    run: list[str] = []
    best: list[str] = []
    for tok in _TOKEN.findall(_norm(text).lower()):
        if tok in _NATO:
            run.append(_NATO[tok])
            if len(run) > len(best):
                best = list(run)
        else:
            run = []
    return "".join(best).upper() if len(best) >= 3 else None


def spelled_run(text: str, min_len: int = 3) -> str | None:
    """A run of single letters ("S A N", "s a n") of at least `min_len` -> "SAN"; the longest run. Letter names in Bengali / Hindi script count."""
    best = ""
    cur = ""
    for tok in _TOKEN.findall(_norm(text)):
        lt = _letter(tok)
        if lt and (len(tok) == 1 or not tok.isascii()):
            cur += lt
            if len(cur) > len(best):
                best = cur
        else:
            cur = ""
    return best.upper() if len(best) >= min_len else None
