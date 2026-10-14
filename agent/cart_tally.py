"""sourav, 2026-10-14: the call's running list of priced tests, with "remove X" and "total".

WHY: after three separate price questions the session remembered only the LAST test, and the model sees only the
current sentence, so "remove CBC and tell me the total" repeated CBC's price (found by tests/test_lab_price_multilingual.py).
This module is the deterministic part of the fix, in the same spirit as agent/language_switch.py and
agent/enquiry_followup.py: it only decides WHICH remembered tests the caller means and WHAT they asked for (a total,
a removal). It never knows a price. main.py / main_pcm.py re-ask the clinic API for every test still on the list and
the existing multi_test_rate_reply sums what was actually found -- the truth-boundary rule (CLAUDE.md) is unchanged.

FLOW: _finish_enquiry_turn remembers each test a price was found for -> a later turn with a total/remove cue reaches
`plan()` before the model is asked -> main re-looks-up the remaining tests and speaks the sum. `plan()` returns None
whenever it is not sure (no list yet, a test that is not on the list is named, no cue), and the turn then goes down
the normal path exactly as before.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

_TOTAL = (
    # Bengali
    "টোটাল", "মোট", "সব মিলিয়ে", "সবমিলিয়ে", "সব মিলে", "সবগুলো মিলিয়ে", "একসাথে কত", "একসঙ্গে কত",
    # Hindi
    "टोटल", "कुल", "सब मिलाकर", "सब मिला कर", "सबको मिलाकर", "मिलाकर कितना", "कितना हुआ", "कितना हुआ सब",
    # English / romanised
    "total", "altogether", "all together", "in all", "combined", "sum", "kul", "mot koto", "sob mila",
)
_REMOVE = (
    # Bengali
    "বাদ দ", "বাদ দিন", "বাদ দিয়ে", "বাদ রেখে", "ছাড়া", "বাদ দাও", "বাদ দিয়ে", "বাদে", "সরিয়ে", "সরান", "সরাও", "কেটে দ", "ছেড়ে দ", "ডিলিট",
    # Hindi
    "हटा", "निकाल", "छोड़कर", "छोड़ के", "को छोड़", "के बिना", "बिना", "छोड़ दो", "छोड़ दीजिए", "छोड़िए", "रहने दो", "रहने दीजिए", "डिलीट", "बाद कर", "बाद दो",
    # English / romanised
    "remove", "take out", "take off", "drop", "delete", "leave out", "leave it out", "exclude", "without", "except", "excluding", "apart from", "minus", "other than",
    "bad dao", "bad din", "hata do", "hatao", "hata dijiye", "nikal do",
)
_POINTERS = (
    "ওটা", "এটা", "সেটা", "এইটা", "ওইটা", "ওটাও", "এটাও", "শেষেরটা", "শেষটা",
    "वो", "वह", "यह", "ये", "उसे", "इसे", "वही", "आखिरी",
    "that", "this", "it", "last one", "that one", "this one",
)

_ASCII_CUE = re.compile(r"^[a-z ]+$")


def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("।", " ").replace(",", " ").replace("?", " ").replace(".", " ").split())


def _has(text: str, cues: tuple[str, ...]) -> bool:
    padded = f" {text} "
    for cue in cues:
        if _ASCII_CUE.match(cue):
            if f" {cue} " in padded:  # a whole word / phrase: "sum" must not fire inside "summary"
                return True
        elif cue in text:
            return True
    return False


@dataclass
class Plan:
    remove: list[int] = field(default_factory=list)  # indexes into the list, to take off it
    total: bool = True  # always speak the new total (a removal without one would leave the caller guessing)


def wants_total_or_remove(text: str) -> tuple[bool, bool]:
    t = _norm(text)
    return _has(t, _TOTAL), _has(t, _REMOVE)


def plan(
    text: str,
    cart: list[str],
    canon: Callable[[str], str | None] | None = None,
    names_several: bool = False,
) -> Plan | None:
    """What to do with the remembered `cart` (test names as the caller said them) for this sentence, or None.

    `canon(s)` maps a phrase to the catalogue test it names (or None) so "सीबीसी" in the sentence finds the
    "cbc" saved earlier; without it only the literal saved text is looked for. `names_several` is True when the
    sentence names two or more catalogue tests -- a fresh multi-test question, left to the normal path unless it is a
    removal."""
    if not cart:
        return None
    t = _norm(text)
    total, remove = _has(t, _TOTAL), _has(t, _REMOVE)
    if not (total or remove):
        return None
    mentioned = {i for i, name in enumerate(cart) if _norm(name) and _norm(name) in t}
    key = canon(text) if canon else None
    if key:
        on_list = {i for i, name in enumerate(cart) if canon(name) == key}
        mentioned |= on_list
        if not on_list and not mentioned:
            return None  # a test that is NOT on the list was named: a normal question, not a list edit
    if remove:
        if not mentioned:
            tokens = set(t.split())
            if any((p in tokens) if _ASCII_CUE.match(p) and " " not in p else (p in t) for p in _POINTERS):
                mentioned = {len(cart) - 1}  # "remove that one" -> the test last priced
            else:
                return None
        return Plan(remove=sorted(mentioned), total=True)
    if names_several and not mentioned:
        return None
    return Plan(remove=[], total=True)


_REMOVED = {
    "bn": "{names} বাদ দিলাম।",
    "hi": "{names} हटा दिया।",
    "en": "Removed {names}.",
}
_EMPTY = {
    "bn": "আপনার তালিকায় এখন আর কোনো টেস্ট নেই। কোন টেস্টের রেট জানতে চান?",
    "hi": "आपकी लिस्ट में अब कोई टेस्ट नहीं बचा। किस टेस्ट की कीमत जाननी है?",
    "en": "There is no test left on your list now. Which test would you like the price of?",
}


def removed_text(names: list[str], lang: str) -> str:
    return _REMOVED.get(lang, _REMOVED["en"]).format(names=", ".join(names))


def empty_text(lang: str) -> str:
    return _EMPTY.get(lang, _EMPTY["en"])
