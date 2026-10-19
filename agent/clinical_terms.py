"""Common clinical words, recognised even when the recogniser mangles them -- and recognised MORE easily where the conversation makes them likely.

Owner, 2026-10-18: "প্রেস্কৃপ্তিন" is "prescription". A medical call is full of a few words (prescription, report, sample, ...), and the agent had just said "bring the prescription": a word that
sounds nearly like it, in that moment, IS it. A plain similarity score treats every word the same and either misses the mangled term or confuses it with "পরিষ্কার" (clear). So the
bar for a match depends on context: strict by itself (0.88 on the consonant skeleton), lower (0.78) when the term appeared in what the agent said a moment ago. Sound only, no meaning
is guessed; a match is evidence for the next rule, never an answer by itself.
"""

from __future__ import annotations

import difflib

from agent.echo_check import _tokens, skeleton

TERMS: dict[str, tuple[str, ...]] = {
    "prescription": ("প্রেসক্রিপশন", "প্রেসক্রিপসন", "prescription", "প্রিস্ক্রিপশন", "पर्चा", "प्रिस्क्रिप्शन", "प्रेस्क्रिप्शन", "পর্চা"),
    "report": ("রিপোর্ট", "रिपोर्ट", "report"),
    "sample": ("স্যাম্পল", "सैंपल", "sample"),
    "appointment": ("অ্যাপয়েন্টমেন্ট", "अपॉइंटमेंट", "appointment"),
    "whatsapp": ("হোয়াটসঅ্যাপ", "व्हाट्सऐप", "whatsapp"),
    "abdomen": ("অ্যাবডোমেন", "আবডোমেন", "এবডোমেন", "অ্যাবডোমিন", "abdomen", "एब्डोमेन", "अब्डोमेन", "एब्डोमन"),
}
_STRICT = 0.88
_IN_CONTEXT = 0.78
_SKELETONS = {term: [s for s in (skeleton(f) for f in forms) if len(s) >= 3] for term, forms in TERMS.items()}


def _best(term: str, text: str) -> float:
    best = 0.0
    for sk in (skeleton(tok) for tok in _tokens(text)):  # one word at a time: joined with its neighbour, "পরিষ্কার নাই" would pass for a prescription
        if len(sk) < 4:
            continue
        for canon in _SKELETONS[term]:
            best = max(best, 1.0 if sk == canon else difflib.SequenceMatcher(None, sk, canon).ratio())
    return best


def mentions(term: str, text: str, context: str = "") -> bool:
    """Does `text` hold the clinical `term` (in any spelling the recogniser might write)? `context` is what the agent said a moment ago: if it held the term, the bar is lower."""
    if not text or term not in TERMS:
        return False
    in_context = bool(context) and _best(term, context) >= _STRICT
    return _best(term, text) >= (_IN_CONTEXT if in_context else _STRICT)
