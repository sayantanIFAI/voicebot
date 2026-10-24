"""Two cheap refusals for the quick-answer shortcut (agent/fast_path.py), found in real calls of 2026-10-17.

The fast path answers a routine question from the clinic's own table with no model. It commits on character similarity to a stored
cue, which is a good test for WHICH test or doctor a caller named but a poor one for WHAT they asked: the sentence frames of different
questions look alike. Two live failures:

  * "what are your timings" was answered with the clinic's ADDRESS (0.77): the frame "what are your ..." is close to the stored
    location cue "where are you". Nothing in the sentence is about a place.
  * "tell me the CBC, uric acid and thyroid test, the price of all three together" was matched to the CBC alone (1.0): the other two
    tests were spelled by the recogniser in ways the catalogue does not hold, so only one test was "named" -- and a single price was
    spoken for a question about a total.

A wrong confident answer from this path is worse than no answer (the turn then goes to the model, which has the context). So the
shortcut is vetoed when its answer cannot be right for the sentence. Both rules only ever REMOVE an answer; they never add one.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[.,;:!?\"'()\[\]{}।॥\-–—_/]+")

# Words that make a sentence a question about WHERE the clinic is / how to reach it.
_LOCATION_WORDS = (
    # English
    "where", "address", "location", "located", "situated", "direction", "directions", "route", "reach", "get there",
    "come there", "how to come", "how do i come", "landmark", "map",
    # Bengali
    "কোথায়", "কোথায", "ঠিকানা", "লোকেশন", "অবস্থান", "কোন জায়গা", "কীভাবে আসব", "কিভাবে আসব", "কিভাবে যাব", "কীভাবে যাব", "রাস্তা", "পথ", "পৌঁছ", "ম্যাপ",
    # Bengali: "address" (the English loanword), in the spellings a recogniser actually writes -- real call 2026-10-24:
    # "আপনাদের ওখানকার অযাড্ড্রেস টা কি আছে" was not read as a location question at all ("address" in Latin script was in
    # this list, but the caller said it in Bengali script, and not even in the standard spelling).
    "অ্যাড্রেস", "অযাড্ড্রেস", "অ্যাডড্রেস", "আড্রেস", "এড্রেস", "আদ্রেস",
    # Hindi
    "कहाँ", "कहां", "पता", "ठिकाना", "ठिकाने", "लोकेशन", "कैसे आ", "कैसे जा", "कैसे पहुँच", "कैसे पहुंच", "रास्ता", "पहुँच", "पहुंच", "मैप", "जगह",
    # Hindi: "address" in Devanagari spellings
    "एड्रेस", "अड्रेस", "ऐड्रेस",
)

# Words that say the caller wants ONE figure for SEVERAL things.
_TOTAL_WORDS = (
    "total", "together", "altogether", "all three", "all of them", "all of these", "both of", "combined", "in all", "sum of",
    "টোটাল", "মোট", "সব মিলিয়ে", "সবমিলিয়ে", "একসাথে", "একসঙ্গে", "তিনটের", "তিনটার", "তিনটির", "দুটোর", "দুটির", "সবগুলোর", "সবকটার",
    "टोटल", "कुल", "एक साथ", "सबका", "सब का", "तीनों", "दोनों", "सब मिलाकर", "मिलाकर", "सभी का",
)


def _norm(text: str) -> str:
    s = unicodedata.normalize("NFC", text or "").lower().replace("़", "").replace("়", "")
    return " ".join(_PUNCT.sub(" ", s).split())


def _has_any(norm_text: str, cues: tuple[str, ...]) -> bool:
    padded = f" {norm_text} "
    for cue in cues:
        c = _norm(cue)
        if not c:
            continue
        if c.isascii():
            if f" {c} " in padded or (len(c) > 4 and c in norm_text):
                return True
        elif c in norm_text:
            return True
    return False


def location_word_present(text: str) -> bool:
    return _has_any(_norm(text), _LOCATION_WORDS)


def asks_for_total(text: str) -> bool:
    return _has_any(_norm(text), _TOTAL_WORDS)


def veto(intent: str, slots: dict | None, text: str, tests_named) -> str | None:
    """A reason to refuse the shortcut's answer, or None to let it stand. `tests_named` is a zero-argument callable returning how many
    catalogue tests the sentence names (only evaluated when a total is asked for)."""
    slots = slots or {}
    if intent == "clinic_faq" and slots.get("faq_topic") == "location" and not location_word_present(text):
        return "location answer without any word about a place"
    if intent == "test_rate" and asks_for_total(text):
        try:
            named = int(tests_named())
        except Exception:  # noqa: BLE001 - if the count cannot be made, the safe side is the model
            named = 0
        if named < 2:
            return "a total was asked for but fewer than two tests are named"
    return None
