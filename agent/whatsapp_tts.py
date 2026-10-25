"""The word "WhatsApp" inside a Bengali or Hindi sentence is spoken with the English voice's own recording of it.

Owner, 2026-10-25: the Bengali and Hindi voices mangle "হোয়াটসঅ্যাপ" / "व्हाट्सऐप"; it is an English word and the English voice says it right. A one-second recording of the English
voice saying "WhatsApp" (assets/whatsapp_en.wav) is therefore cut into the Bengali / Hindi audio wherever the word is, in any of its spellings.

This module is the pure half (no audio, no model): it splits a sentence into the text to synthesise and the places the recording goes. tts_server.py does the rendering.
A Bengali ending glued to the word ("হোয়াটসঅ্যাপে" = on WhatsApp, "হোয়াটসঅ্যাপের" = of WhatsApp) is kept: the recording says the word and the ending is spoken after it.
"""

from __future__ import annotations

import re

# Bengali: "হোয়াটস" + "অ্যাপ" in the spellings a script or a recogniser writes (one word or two, ZWNJ inside, ওয়া / হুয়া starts, এপ for অ্যাপ), then an optional case ending.
_BN = re.compile(
    "(?:হোয়াটস|হুয়াটস|ওয়াটস|হোয়াট‌?স|হোয়াট্‌স|হোআটস|হোয়াটস্)\\s*(?:অ্যাপ|অ্যাপ্|এপ|অ্যাপ|এ্যাপ|য়াপ|অ্যাপ)"
    "(?P<suf>ের|কে|ে|র|ও)?"
    "|হোয়াটসেপ|ওয়াটসেপ|হোয়াটস্যাপ"
)
# Hindi: व्हाट्स / व्हॉट्स / वाट्स / वॉट्स + ऐप / एप / अप
_HI = re.compile("(?:व्हाट्स|व्हॉट्स|वाट्स|वॉट्स|ह्वाट्स)\\s*(?:ऐप|एप|अप|एपप)")
_LATIN = re.compile(r"whats\s?app", re.IGNORECASE)

# the ending, said as its own small word after the recording
# (the Bengali voice cannot speak a lone one-letter word: it fails on "এ", "র", "ও" -- a danda after the vowel, or the two-letter "এর", gives it enough to say)
_SUFFIX_SPOKEN = {"ে": "এ।", "ের": "এর", "কে": "কে", "র": "এর", "ও": "ও।"}


def split(text: str, lang: str) -> list[tuple[str, str]]:
    """[("text", sentence part), ("wa", "")|("wa", ending)] in speaking order. Only for Bengali and Hindi; any other language, or text without the word, comes back as one text part."""
    if lang not in ("bn", "hi") or not text:
        return [("text", text)]
    pattern = _BN if lang == "bn" else _HI
    out: list[tuple[str, str]] = []
    pos = 0
    # the Latin spelling is turned into a match too, whichever language the sentence is in
    matches = sorted([(m.start(), m.end(), (m.groupdict().get("suf") or "")) for p in (pattern, _LATIN) for m in p.finditer(text)])
    merged: list[tuple[int, int, str]] = []
    for start, end, suf in matches:
        if merged and start < merged[-1][1]:
            continue
        merged.append((start, end, suf))
    for start, end, suf in merged:
        if start > pos:
            out.append(("text", text[pos:start]))
        out.append(("wa", _SUFFIX_SPOKEN.get(suf, "")))
        pos = end
    if pos < len(text):
        out.append(("text", text[pos:]))
    if not merged:
        return [("text", text)]
    return [(k, v) for k, v in out if k == "wa" or v.strip()]
