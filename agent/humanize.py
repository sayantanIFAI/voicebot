"""Warmer, less word-for-word wording for a reply whose FACTS the code has already settled (owner, 2026-10-31).

The truth boundary stays where it is: every name, time, number and fact in a reply comes from the clinic's tables through the reply templates. Here the intent model may only REPHRASE that finished sentence
-- same language, same script, same facts -- and a checker throws the rewrite away unless it still holds:
  * every digit run of the original is in the rewrite, and the rewrite has no digit run the original lacked;
  * every clinic name (doctor / department / test words, passed in as `vocab`) in the original is in the rewrite, and the rewrite names none the original did not;
  * it is not much longer than the original, is not empty, is written in the same script, has no link, and is a question exactly when the original was one.
Any failure, a timeout or an error returns the original sentence unchanged. So the worst case is the template as it was, plus the time the attempt took.

Off by default (HUMANIZE_REPLIES=on turns it on): a rewrite costs roughly 0.3-1 s a turn on the 7B model and a new word may be pronounced worse than the template's, neither of which has been measured yet.
"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata

logger = logging.getLogger("humanize")

ENABLED = os.getenv("HUMANIZE_REPLIES", "off").strip().lower() in ("on", "1", "true", "yes")
MIN_CHARS, MAX_CHARS = 40, 420
TIMEOUT_S = float(os.getenv("HUMANIZE_TIMEOUT_S", "1.5"))
_DIGITS = re.compile(r"\d+")
_SCRIPT = {"bn": (0x0980, 0x09FF), "hi": (0x0900, 0x097F)}
_LANGUAGE_NAMES = {"bn": "Bengali", "hi": "Hindi", "en": "English"}


def worth_rewriting(text: str) -> bool:
    """Only a sentence long enough to sound stiff, and not a figure-heavy one (prices, phone numbers, ids are read exactly as they are)."""
    t = text or ""
    runs = _DIGITS.findall(t)
    return MIN_CHARS <= len(t) <= MAX_CHARS and len(runs) <= 6 and all(len(r) <= 4 for r in runs)  # times and dates are fine; a phone number, an id or a long price is read exactly as it is


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[\s,.;:!?।()\"'-]+", unicodedata.normalize("NFC", text or "")) if w]


def _script_share(text: str, lang: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    if lang == "en":
        return sum(c.isascii() for c in letters) / len(letters)
    lo, hi = _SCRIPT[lang]
    return sum(lo <= ord(c) <= hi for c in letters) / len(letters)


def acceptable(original: str, rewrite: str, lang: str, vocab: set[str]) -> bool:
    """The checker. True only when the rewrite keeps every fact of the original and adds none."""
    new = (rewrite or "").strip()
    old = (original or "").strip()
    if not new or new == old:
        return False
    if len(new) > len(old) * 1.4 + 20 or len(new) < len(old) * 0.4:
        return False
    if re.search(r"https?://|www\.|@", new):
        return False
    if sorted(_DIGITS.findall(old)) != sorted(_DIGITS.findall(new)):
        return False
    if old.rstrip().endswith(("?", "？")) != new.rstrip().endswith(("?", "？")):
        return False
    if lang in _SCRIPT or lang == "en":
        if _script_share(new, lang) < 0.85 and _script_share(old, lang) >= 0.85:
            return False
    old_w, new_w = set(_words(old)), set(_words(new))
    vocab_old = {w for w in old_w if w in vocab}
    vocab_new = {w for w in new_w if w in vocab}
    if vocab_old - new_w:  # a clinic name the original said is gone
        return False
    if vocab_new - old_w:  # a clinic name the original did not say has appeared
        return False
    return True


def _prompt(text: str, lang: str) -> str:
    language = _LANGUAGE_NAMES.get(lang, "Bengali")
    return (
        f"You rewrite ONE sentence spoken by a diagnostic clinic's phone assistant so that it sounds warmer and more natural to the caller, in {language}, in {language}'s own script.\n"
        "Rules: keep EVERY name, number, time and fact exactly as given; add NO new fact, name, number or promise; do not turn a statement into a question or the reverse; keep it about the same length; "
        "no greeting, no emoji.\n"
        'Output ONLY JSON: {"reply": "<the rewritten sentence>"}\n\n'
        f"SENTENCE:\n{text}\n\nJSON:"
    )


def rewrite(text: str, lang: str, vocab: set[str], call=None) -> str:
    """The rewritten sentence, or `text` itself when the model is off, slow, wrong or breaks a rule. `call` is the model function (agent.llm._call_ollama), injected so this module stays testable."""
    if not ENABLED or not worth_rewriting(text):
        return text
    try:
        if call is None:
            from agent.llm import _call_ollama as call  # noqa: PLC0415 - imported late: the module is optional
        raw = call(_prompt(text, lang), timeout_s=TIMEOUT_S)
        new = str(json.loads(raw).get("reply", "")).strip()
    except Exception as e:  # noqa: BLE001 - the original sentence is always a safe answer
        logger.info("rewrite skipped (%s)", e)
        return text
    if acceptable(text, new, lang, vocab):
        return new
    logger.info("rewrite rejected: %r -> %r", text[:60], new[:60])
    return text
