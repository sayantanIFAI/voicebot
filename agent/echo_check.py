"""The caller checks back something the agent SAID: "বেনিয়া পুকুর কেয়া?" ("Beniapukur, is it?") after hearing the address in Hindi.

Owner, 2026-10-18: anything spoken by the AI, the patient can ask back about -- and the answer is "yes, that is right" with the sentence again, or the correct answer. The caller's
word and the agent's word are written in different scripts (Bengali letters against Devanagari, or against Latin), so they are compared by SOUND: every word is reduced to its
consonant skeleton in one common alphabet, and two words that have the same skeleton are the same word. Nothing here invents an answer: it only finds the sentence the agent actually said.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

# consonants of Bengali and Devanagari -> a common Latin letter. Vowels, vowel signs, the nukta, the virama and h / y / w are dropped, so spelling variants collapse.
_MAP = {
    **{c: "k" for c in "কখक"}, **{c: "g" for c in "গঘगघ"}, **{c: "n" for c in "ঙঞণনङञणन"}, **{c: "c" for c in "চছचछ"}, **{c: "j" for c in "জঝजझ"},
    **{c: "t" for c in "টঠতথৎटठतथ"}, **{c: "d" for c in "ডঢদধडढदध"}, **{c: "p" for c in "পফपफ"}, **{c: "b" for c in "বভबभव"}, **{c: "m" for c in "মम"},
    **{c: "r" for c in "রर"}, **{c: "l" for c in "লल"}, **{c: "s" for c in "শষসशषस"}, "ক্ষ": "ks",
}
_LATIN = {"c": "k", "q": "k", "v": "b", "w": "", "z": "j", "f": "p", "x": "ks"}
_DROP_LATIN = set("aeiouyh")
_CUES = (
    "কেয়া", "কি", "কী", "ঠিক", "সঠিক", "ঠিকই", "তাই তো", "নাকি", "ना", "क्या", "सही", "ठीक", "ही", "right", "correct", "is it", "is that", "ki", "kya", "sahi", "thik", "theek", "?",
)


def skeleton(word: str) -> str:
    w = (word or "").lower()
    for rr in ("ড়", "ঢ়", "ड़", "ढ़"):  # ড় ঢ় ड़ ढ़ are an r (flap), not a d
        w = w.replace(rr, "r")
    w = unicodedata.normalize("NFD", w)  # precomposed nukta letters (ज़ फ़ ...) fall apart into the letter and the nukta; the nukta is dropped below
    w = w.replace("ক্ষ", "ks")
    out: list[str] = []
    if w.isascii():
        w = w.replace("ch", "c").replace("sh", "s").replace("ph", "p").replace("th", "t").replace("dh", "d").replace("kh", "k").replace("gh", "g").replace("bh", "b")
        for ch in w:
            if not ch.isalpha() or ch in _DROP_LATIN:
                continue
            out.append(_LATIN.get(ch, ch))
    else:
        for ch in w:
            if ch in _MAP:
                out.append(_MAP[ch])
            elif ch.isascii() and ch.isalpha() and ch not in _DROP_LATIN:
                out.append(_LATIN.get(ch, ch))
    s = "".join(out)
    return re.sub(r"(.)\1+", r"\1", s)  # a doubled consonant is one sound


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[\s,।.?!;:\-()\"']+", unicodedata.normalize("NFC", text or "")) if t]


# Little words that answer or soften a sentence are never the thing asked back ("ঠিক আছে" is an answer, not a question about the words "ঠিক আছে" in the agent's last sentence).
_STOP = frozenset(
    "ঠিক আছে আছে হ্যাঁ হ্যা হাঁ না নাকি কি কী কেয়া তো ই জি ওকে এটা ওটা এই ওই সেই এ ও আর এবং তাই সঠিক বলুন বলবেন বললেন বলেছেন "
    "ठीक है हाँ हां जी नहीं ना क्या ही तो यह वह ये वो और सही बताइए "
    "ok okay yes no is it that the a an right correct ji haan han hai ki kya".split()
)


def _candidates(tokens: list[str]) -> list[tuple[str, str]]:
    """(the words as said, their skeleton) for every word and every run of two or three words joined: "বেনিয়া পুকুর" is the one word "বেনিয়াপুকুর" in the agent's sentence."""
    out = []
    for n in (1, 2, 3):
        for i in range(len(tokens) - n + 1):
            chunk = tokens[i : i + n]
            if all(t.lower() in _STOP for t in chunk):
                continue
            sk = skeleton("".join(chunk))
            if len(sk) >= 3:
                out.append((" ".join(chunk), sk))
    return out


def asks_back(text: str) -> bool:
    t = unicodedata.normalize("NFC", text or "").lower()
    words = set(_tokens(t))
    # a short cue ("কি", "ঠিক", "ki") counts only as a whole word: inside "বাকিটা" or "ঠিকানা" it is just letters (real call 2026-10-23: "বাকিটা হচ্ছে ফাইভ সিক্স…" was taken for a question)
    return any((c in words) if len(c) <= 3 and c != "?" else (c in t) for c in _CUES)


def find(user_text: str, last_spoken: str) -> tuple[str, str] | None:
    """(the caller's words that name something the agent said, the agent's sentence that holds it), or None. Only a short sentence that asks back is considered; the match is by
    sound, exact on the skeleton or a very close one for a longer word. The cue words themselves ("কেয়া", "ঠিক") are never the thing looked for."""
    if not user_text or not last_spoken:
        return None
    said = _tokens(user_text)
    if not 1 <= len(said) <= 8 or not asks_back(user_text):
        return None
    cue_skeletons = {skeleton(c) for c in _CUES if c != "?"} | {""}
    spoken_sentences = [s.strip() for s in re.split(r"(?<=[।.?!])\s+", last_spoken) if s.strip()] or [last_spoken]
    best = None
    for words, sk in _candidates(said):
        if sk in cue_skeletons:
            continue
        for sentence in spoken_sentences:
            for _w, sk2 in _candidates(_tokens(sentence)):
                ratio = 1.0 if sk == sk2 else difflib.SequenceMatcher(None, sk, sk2).ratio()
                need = 1.0 if min(len(sk), len(sk2)) <= 4 else 0.86
                if ratio >= need and (best is None or ratio > best[0] or (ratio == best[0] and len(sk) > len(best[3]))):
                    best = (ratio, words, sentence, sk)
    if best is None:
        return None
    # the words asked back must be most of what was said: one matching word inside a long ordinary question ("… কি বাড়ি গিয়ে নমুনা সংগ্রহ করেন") is not it
    content = [t for t in said if t.lower() not in _STOP and skeleton(t) not in cue_skeletons]
    if len(best[1].split()) * 2 < len(content):
        return None
    return best[1], best[2]
