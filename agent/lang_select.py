"""Choosing the language of a turn when language ID alone is not enough.

Measured on the pod (docs/adr/0002 records the numbers): SpeechBrain
VoxLingua107 separates Bengali from Hindi and Hindi from Bengali well, but
Indian-accented ENGLISH is routinely labelled Hindi, at high confidence. A
confident-looking wrong answer is worse than an unsure one: the router
commits to it and the caller is answered in the wrong language, so the
decision cannot rest on LID's top label alone.

The pieces here are deliberately small and pure so they are testable
without a GPU:

  script_share   what fraction of a transcript is in a language's own script
  pick_candidate choose between two ASR results for the same audio
  speakable      can a language's TTS voice actually say this text
"""
from __future__ import annotations

from agent.speech_norm import unspeakable_spans

_SCRIPT_RANGE = {"bn": ("ঀ", "৿"), "hi": ("ऀ", "ॿ")}
_WRONG_SCRIPT = {"bn": ("ऀ", "ॿ"), "hi": ("ঀ", "৿")}
_SHARED_PUNCTUATION = frozenset(chr(0x0964) + chr(0x0965))


def script_share(text: str, lang: str) -> float:
    """Fraction of the letters in `text` written in `lang`'s own script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    if lang == "en":
        return sum(c.isascii() for c in letters) / len(letters)
    lo, hi = _SCRIPT_RANGE[lang]
    return sum(lo <= c <= hi for c in letters) / len(letters)


def speakable(text: str, lang: str) -> bool:
    """Can `lang`'s voice actually say `text`? Model-written smalltalk is the
    one place reply text is not from a template, so it is checked: a reply in
    the wrong script is dropped by the tokenizer and heard as silence."""
    if not text or unspeakable_spans(text, lang):
        return False
    wrong = _WRONG_SCRIPT.get(lang)
    # The danda (U+0964) and double danda (U+0965) sit in the Devanagari block but are the full stop of Bengali
    # too. Counting them as "wrong script" made every Bengali reply ending in "।" unspeakable, so it was
    # silently replaced by a default line.
    return not (wrong and any(wrong[0] <= c <= wrong[1] and c not in _SHARED_PUNCTUATION for c in text))


# ---------------------------------------------------------------- selection

# A second language must carry at least this much LID probability before the
# top label is treated as unsure. Measured on the pod: Indian-accented
# English is labelled Hindi at 0.90-0.91 with English at 0.09-0.36, while
# genuine Hindi and Bengali put 0.00 on English. So "any real mass on a
# second language" is the signal, not the top label's own confidence.
SECOND_LANGUAGE_FLOOR = 0.03


def languages_to_verify(language: str, scores: dict[str, float],
                        active: tuple[str, ...]) -> list[str] | None:
    """Every active language, best LID score first, when LID is not decisive;
    None when it is.

    Not decisive means: LID found none of our languages ("unknown"), or a
    second language carries real probability. Verifying with ALL active ASRs
    rather than the top two is deliberate: measured, English speech came back
    as bn 0.83 / hi 0.16 / en 0.01, so English was not even in the top two,
    and only running its ASR could have found it."""
    ranked = [lang for lang, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
              if lang in active]
    for lang in active:
        if lang not in ranked:
            ranked.append(lang)
    if language == "unknown":
        return ranked
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
    return ranked if second >= SECOND_LANGUAGE_FLOOR else None


# English gate. Measured on synthetic clips (clean, 8 kHz, and through the live
# service), decoder agreement of the ENGLISH engine: 0.67-1.00 on English
# audio (n=7 runs), 0.00-0.43 on Bengali and Hindi audio (n=8). The Bengali engine cannot be used the same way: it
# transliterates English with agreement up to 1.00, so on English audio "the
# engine that agrees with itself most" picks Bengali. REASONED from a small
# synthetic sample, not calibrated on telephony -- recalibrate on real calls.
ENGLISH_MIN_AGREEMENT = 0.6


def pick_candidate(candidates: list[tuple[str, object]]):
    """Several ASR engines ran on the same audio. Return (language, result)
    for the one that most plausibly heard its own language.

    1. English wins outright if the English engine's CTC and RNNT decoders
       agree (>= ENGLISH_MIN_AGREEMENT) and it produced Latin text -- the
       one engine whose agreement is a trustworthy "this was English" signal.
    2. Otherwise choose among the rest by transcript-in-own-script, then by
       decoder agreement (own-language 0.67-1.00 vs 0.00-0.67 for the wrong
       Indic engine, measured).
    Ties go to the earlier candidate, i.e. the LID-preferred one."""
    usable = [(lang, r) for lang, r in candidates if getattr(r, "text", "").strip()]
    if not usable:
        return candidates[0]
    for lang, r in usable:
        if (lang == "en" and r.decoder_agreement >= ENGLISH_MIN_AGREEMENT
                and script_share(r.text, "en") >= 0.9):
            return lang, r
    rest = [(lang, r) for lang, r in usable if lang != "en"] or usable
    return max(rest, key=lambda t: (round(script_share(t[1].text, t[0]), 1), t[1].decoder_agreement))
