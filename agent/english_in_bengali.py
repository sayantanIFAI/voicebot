"""English spoken, but the recogniser wrote it in Bengali letters (owner, 2026-10-31: "ইস থিয়ার পার্কিং" = "is there parking", "হু টে অর্থোপেডিক ও সিট অযাট সোনোস্কান কলকাতা" = "who are the orthopedic who sit at Sonoscan Kolkata").

A caller with a Bengali accent speaking English is labelled Bengali by language ID (0.92) and the Bengali recogniser writes the English sounds in Bengali letters, so the script says nothing. What does say it is the small closed class of English FUNCTION words -- is, there, who, the, at, what, where, can -- which Bengali has no use for. Loanwords ("পার্কিং", "ডাক্তার", "ক্লিনিক") prove nothing: Bengali speakers say them in Bengali sentences, so they are not counted.

A signal, not a verdict: a Bengali-script reading with at least two different English function words, and at least a quarter of its words, makes the turn English (agent/main.py), answered in English for this turn without locking the call.
"""

from __future__ import annotations

import re
import unicodedata

# The Bengali-letter spellings the recogniser really produces for English function words (a few variants each). Words Bengali also uses ("আর", "দি", "ইন", "অন", "টু", "ডু") are NOT here.
_FORMS: dict[str, tuple[str, ...]] = {
    "is": ("ইস", "ইজ", "ইজি"),
    "there": ("থিয়ার", "দেয়ার", "দিয়ার", "ডেয়ার", "দ্যার", "থেয়ার", "দেওয়ার"),
    "here": ("হিয়ার", "হিয়র"),
    "who": ("হু", "হুউ"),
    "what": ("হোয়াট", "ওয়াট", "হোয়াত", "হোট"),
    "where": ("হোয়ার", "ওয়ার", "হয়ার", "হোয়্যার"),
    "when": ("হোয়েন", "ওয়েন", "হুয়েন"),
    "which": ("হুইচ", "উইচ", "উইছ"),
    "how": ("হাউ", "হাও"),
    "the": ("টে", "দ্য", "দ্যা", "ডি", "দে", "ধ্য"),
    "at": ("অযাট", "অ্যাট", "এট", "আট", "এ্যাট"),
    "sit": ("সিট", "সিত", "সিটস"),
    "are": ("আর্", "অ্যার", "আরে"),
    "were": ("ওয়ার", "ওয়ের", "ওয়ে"),
    "can": ("ক্যান", "কেন", "ক্যাণ"),
    "could": ("কুড", "কুল্ড", "কুদ"),
    "will": ("উইল", "ওয়িল"),
    "would": ("উড", "ওউড"),
    "does": ("ডাজ", "ডজ", "ডাস"),
    "have": ("হ্যাভ", "হ্যাব", "হেভ"),
    "any": ("এনি", "অনি", "এনী"),
    "and": ("অ্যান্ড", "এন্ড", "অ্যান"),
    "not": ("নট", "ন্ট"),
    "for": ("ফর", "ফোর"),
    "you": ("ইউ", "য়ু", "ইয়ু"),
    "my": ("মাই", "মাঈ"),
    "your": ("ইয়োর", "ইয়র", "ইওর"),
    "this": ("দিস", "ধিস", "দিছ"),
    "that": ("দ্যাট", "ধ্যাট", "ধ্যাত", "দ্যাত"),
    "please": ("প্লিজ", "প্লীজ", "প্লিস"),
    "okay": ("ওকে", "অকে", "ওকে।"),
}
_BY_FORM: dict[str, str] = {}
for _word, _forms in _FORMS.items():
    for _f in _forms:
        _BY_FORM[unicodedata.normalize("NFC", _f)] = _word
# words that count only beside others (they are real Bengali sounds too)
_WEAK = frozenset({"the", "are", "can", "any", "at", "not", "for", "is"})


# "is parking" run together by the recogniser: "ইস্পার্কিং", "স্পারকিং", "ইসপারকিং" (the English "is" fused to the next word). A fused "is" + a clinic word counts as the word "is".
_FUSED_IS = re.compile(r"^(?:ইস|ইজ|ই)?স?্?(?:পার্?কিং|পাকিং|পারকিং|পার্কিং)$")


def _word_of(tok: str) -> str | None:
    tok = unicodedata.normalize("NFC", tok)
    w = _BY_FORM.get(tok)
    if w:
        return w
    if _FUSED_IS.match(tok) and tok not in ("পার্কিং", "পারকিং", "পাকিং"):
        return "is"
    return None


# two English words said as one ("is there")
_FUSED_TWO = {"ইসবেয়ার": {"is", "there"}, "ইসদেয়ার": {"is", "there"}, "ইসথিয়ার": {"is", "there"}, "ইজদেয়ার": {"is", "there"}, "হুআর": {"who", "are"}}


def english_words(text: str) -> set[str]:
    """The distinct English function words a Bengali-script sentence contains."""
    out = set()
    for tok in re.findall(r"[ঀ-৿]+", unicodedata.normalize("NFC", text or "")):
        w = _word_of(tok)
        if w:
            out.add(w)
        out |= _FUSED_TWO.get(unicodedata.normalize("NFC", tok), set())
    return out


def looks_english(text: str) -> bool:
    """A Bengali-script sentence made of English function words: at least two different ones, one of them not a weak one, and at least a quarter of the words."""
    toks = re.findall(r"[ঀ-৿]+", text or "")
    if not toks:
        return False
    found = english_words(text)
    hits = sum(1 for t in toks if _word_of(t) or unicodedata.normalize("NFC", t) in _FUSED_TWO)
    return len(found) >= 2 and bool(found - _WEAK) and hits * 4 >= len(toks)
