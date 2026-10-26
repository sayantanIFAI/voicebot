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

import re
from typing import Protocol

from agent.speech_norm import unspeakable_spans

_SCRIPT_RANGE = {"bn": ("ঀ", "৿"), "hi": ("ऀ", "ॿ")}
_WRONG_SCRIPT = {"bn": ("ऀ", "ॿ"), "hi": ("ঀ", "৿")}
_SHARED_PUNCTUATION = frozenset(chr(0x0964) + chr(0x0965))


def language_of_script(text: str, fallback: str) -> str:
    """The language a transcript is WRITTEN in, judged by script: Devanagari -> hi, Bengali script -> bn, Latin -> en, and `fallback`
    when it is mixed or has no letters. The REPLY language of a call is sticky (it is locked for the whole call); the language of one
    sentence is not, and the shortcut that answers routine questions has one cue table per language. Real call 2026-10-17: a Hindi
    sentence in a call whose reply language was Bengali was matched against the BENGALI cues and answered with the clinic's address."""
    if not any(c.isalpha() for c in (text or "")):
        return fallback
    for lang in ("hi", "bn", "en"):
        if script_share(text, lang) >= 0.6:
            return lang
    return fallback


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


def languages_to_verify(language: str, scores: dict[str, float], active: tuple[str, ...]) -> list[str] | None:
    """Every active language, best LID score first, when LID is not decisive;
    None when it is.

    Not decisive means: LID found none of our languages ("unknown"), or a
    second language carries real probability. Verifying with ALL active ASRs
    rather than the top two is deliberate: measured, English speech came back
    as bn 0.83 / hi 0.16 / en 0.01, so English was not even in the top two,
    and only running its ASR could have found it."""
    ranked = [lang for lang, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if lang in active]
    for lang in active:
        if lang not in ranked:
            ranked.append(lang)
    if language == "unknown":
        return ranked
    second = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
    return ranked if second >= SECOND_LANGUAGE_FLOOR else None


# A second Indic language is only worth running a recogniser for when language ID is genuinely torn between the two.
# Below this it cannot win anyway (pick_candidate: an Indic engine other than language ID's favourite needs SWITCH_MIN_LID
# to overrule it, and needs INDIC_MIN_LID just to be considered), so its ~0.6 s of GPU time buys nothing.
#
# Lowered from 0.30. Three real calls, same caller, all genuinely Bengali but LID's top label was Hindi, never
# got a Bengali recognition pass at all: Bengali's LID score was 0.16, 0.26 and 0.29 each time, all below the old
# 0.30 floor. The symmetric case already has a pinned regression test just below here (a genuinely BENGALI call
# where Hindi scored 0.15) showing that running the OTHER engine at 0.15 bought nothing and was deliberately
# skipped -- so this floor cannot simply drop to "whatever the lowest real miss was" without re-breaking that
# measured case. 0.16 is the exact boundary that satisfies both: it is the lowest of the three new misses, and
# it is still above the 0.15 the existing test requires excluded. There is no margin either side of this value --
# both constraints are drawn from real, specific pod incidents, not a round number -- so treat 0.16 as a point
# this module's thresholds pass through today, not a value the model is confident sits in a wide "obviously
# correct" range; recalibrate properly once there is enough real-call LID data to fit a floor with actual margin,
# per this module's own docstring on measuring vs. reasoning. Trade-off: the other Indic recogniser now runs
# somewhat more often on ambiguous turns (~0.6s GPU time each time it does) -- justified here by a correctness
# bug, not assumed free; watch GPU headroom if admitted call volume grows.
OTHER_INDIC_MIN_LID = 0.16


def engines_needed(
    language: str, scores: dict[str, float], active: tuple[str, ...], prior: str | None = None
) -> list[str]:
    """The recognisers that can still change the answer, best language-ID score first (ties keep that order, so the
    LID-preferred language wins them). Measured on the pod: running all three took ~0.64 s; English took 0.21 s.

      * unknown language, or no scores: all of them, as before;
      * otherwise language ID's favourite, plus the OTHER Indic language only if LID is torn (>= OTHER_INDIC_MIN_LID);
      * English ALWAYS: it is cheap and concurrent, and language ID mislabels accented English as Bengali or Hindi
        (measured: English speech came back as bn 0.83 / hi 0.16 / en 0.01), so its probability cannot rule it out;
      * the call's OWN established language (`prior`) ALWAYS, for the identical reason, extended to Bengali/Hindi:
        a real call 2026-10-12 gave Bengali only 0.07 (confidently, wrongly, favouring Hindi at 0.93) on a call
        already well past its first turn and established as Bengali -- below EVERY floor OTHER_INDIC_MIN_LID has
        been lowered to chasing real misses (0.30 -> 0.16; this one would not have cleared 0.16 either). The
        caller's own recognised language cannot be ruled out by a LID score any more than English can. This
        costs nothing extra: `_route_and_transcribe` already starts `prior`'s recogniser in parallel with LID
        itself (the same early-start this module's own docstring describes for English) -- omitting it from
        `need` only threw away a result already being computed for free, which is what actually happened here:
        the Bengali decoder ran, and its transcript was discarded in favour of Hindi's phonetic guess at it."""
    ranked = [lang for lang, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if lang in active]
    for lang in active:
        if lang not in ranked:
            ranked.append(lang)
    if language == "unknown" or not scores:
        return ranked
    need = []
    for lang in ranked:
        if lang == "en" or lang == language or lang == prior or scores.get(lang, 0.0) >= OTHER_INDIC_MIN_LID:
            need.append(lang)
    if language in active and language not in need:
        need.insert(0, language)
    return need


# English gate. Measured on synthetic clips (clean, 8 kHz, and through the live
# service), decoder agreement of the ENGLISH engine: 0.67-1.00 on English
# audio (n=7 runs), 0.00-0.43 on Bengali and Hindi audio (n=8). The Bengali engine cannot be used the same way: it
# transliterates English with agreement up to 1.00, so on English audio "the
# engine that agrees with itself most" picks Bengali. REASONED from a small
# synthetic sample, not calibrated on telephony -- recalibrate on real calls.
ENGLISH_MIN_AGREEMENT = 0.6


# An Indic (Bengali/Hindi) recogniser may only WIN on its own confidence if language ID gave that language at
# least this much probability. Found on the live pod: LID said Bengali 0.95 / Hindi 0.05, all three recognisers ran,
# and the Hindi one (which writes Bengali speech out in Devanagari, with a confident-looking 0.50) beat the Bengali
# one (0.20), so the caller was answered in Hindi. Recogniser confidences of different models are not comparable;
# language ID is the one signal that is. English is exempt: LID mislabels accented English, which is exactly why
# the English engine has its own decoder-agreement gate.
INDIC_MIN_LID = 0.10

# English may win outright on its own decoder agreement (LID mislabels accented English), but that gate alone let a
# Bengali sentence through: the English recogniser wrote it out in Latin letters with agreement 1.00 while language ID
# had given English 0.00, and the call was answered in English. So English must ALSO be real English: language ID
# gives it some probability, or most of what it wrote is words an English speaker actually says.
ENGLISH_MIN_LID = 0.05
ENGLISH_MIN_WORD_SHARE = 0.5
STABLE_CALL_LID = 0.8  # language ID this sure of the call's language ...
STABLE_CALL_ENGLISH_SHARE = 0.75  # ... and English must be at least this plainly English (and its decoders agree this well) to take the turn
STABLE_CALL_ENGLISH_AGREEMENT = 0.7

# Owner, 2026-10-26: a one- or two-word answer is the hardest case for language ID -- the least acoustic evidence
# to go on -- so it should not be enough, on its own, to move an established bn/hi call's TRANSCRIPT to English on
# language ID barely clearing the ENGLISH_MIN_LID floor (0.05). This is reasoned, not measured against a real failing
# call (the Twilio calls read so far all happened to land the right way already); it only screens out that very low
# floor for short answers specifically, so a clearer plurality for English (SHORT_ENGLISH_MIN_LID) still goes
# through -- a short, bona fide English answer is not silently redirected. The explicit "reply in English" switch
# request above is unaffected: it is checked, and can win, before this guard ever runs.
SHORT_UTTERANCE_WORDS = 2
SHORT_ENGLISH_MIN_LID = 0.35
_ENGLISH_WORDS = frozenset(
    """
a an the and or but if so of to in on at for from with by about as is are was were be been am do does did have has had
i me my we our you your he she it they them this that these those there here what which who whom when where why how
can could will would shall should may might must not no yes ok okay please thanks thank hello hi sorry
price prices cost costs rate rates fee charge how much many test tests doctor doctors dr appointment appointments book
booking cancel reschedule time timing timings open close hours today tomorrow morning evening day week monday tuesday
wednesday thursday friday saturday sunday fasting fast empty stomach report reports result results sample blood urine
clinic address location parking insurance payment card cash number phone confirm confirmation need want tell give
know check see get take come go available sit sits sitting chamber counter staff person help
# Real call 2026-10-14: "Reply in English" was heard by the English recogniser with full agreement, but only 1 of its 3
# words was in this list, so the gate below (ENGLISH_MIN_WORD_SHARE) threw the transcript away and the caller got a
# Bengali mis-hearing. A request to change language is the one sentence an English speaker is certain to say in a
# call that is not in English yet, so its vocabulary belongs here.
speak speaking spoke talk talking reply replying respond say answer english hindi bengali bangla language
switch change continue
""".split()
)


def english_word_share(text: str) -> float:
    """Share of the tokens that are ordinary English words (a small fixed list). A Bengali or Hindi sentence written
    out in Latin letters ("sibisi test rate koto") scores low; a real English question scores high."""
    tokens = [t.strip(".,?!'\"").lower() for t in (text or "").split()]
    # A lone letter that is not a word ("c b c" -- a test code SPELLED by the caller) is neither English nor not-English: it says nothing
    # about the language, and counting it against the sentence threw away "what is the rate of c b c and uric acid together" (real call
    # 2026-10-17: share 0.58 < 0.6, so a first English question was answered in Bengali). "a" and "i" are real words and still count.
    tokens = [t for t in tokens if t and (len(t) > 1 or t in _ENGLISH_WORDS)]
    if not tokens:
        return 0.0
    return sum(t in _ENGLISH_WORDS for t in tokens) / len(tokens)


# OWNER'S INSTRUCTION, 2026-10-10: "time" was reported heard back as "tame" -- the English recogniser's own
# vowel confusion, not a language-ID or matching problem, so the fix belongs here, on the ENGLISH transcript
# text, not in any keyword list. REASONED from one reported word, not measured against a set of real calls:
# "tame" has no legitimate meaning in a diagnostics-clinic call (no test, doctor, or FAQ topic involves it),
# while "time" is one of the most common words in this domain (hours, report time, appointment slot), so
# correcting it is safe by construction even without more evidence -- there is no real caller sentence this
# substitution could break. Add further entries here only on the same evidence (a reported word, not a guess),
# and only for a wrong word that has no legitimate reading of its own.
_EN_ASR_CONFUSABLES = {
    "tame": "time",
}
_RE_EN_CONFUSABLE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _EN_ASR_CONFUSABLES) + r")\b", re.IGNORECASE)


def fix_asr_confusables(text: str) -> str:
    """Correct known English-recogniser mishearings in a transcript already decided to be English. Whole-word,
    case-preserving on the replacement's case (the table is already lower-case, matching normal ASR output)."""
    if not text:
        return text
    return _RE_EN_CONFUSABLE.sub(lambda m: _EN_ASR_CONFUSABLES[m.group(0).lower()], text)


# Leaving the language the call has been in needs language ID to believe the new one, not merely allow it.
SWITCH_MIN_LID = 0.50

# Real call 2026-10-17: "what's your location" was answered in Hindi. Language ID said Hindi at 0.99 -- and it says Hindi at 0.72-0.99
# for Indian-accented English in general (7 of 7 English questions tried on the pod) -- so the code committed to Hindi and used only
# the Hindi recogniser, which wrote the English out in Devanagari ("वाट्स योर लोकेशन"). The English recogniser had run in parallel
# and understood every one of them. The old double-check only fired when language ID was UNSURE (a second language with real
# probability); here it was confidently wrong, so a clean English sentence from the English recogniser must be allowed to overrule it.
# The bar is the same one that already lets English win elsewhere (decoder agreement, Latin script, mostly real English words),
# plus enough words that a stray syllable cannot do it: Bengali or Hindi speech was measured at agreement 0.00-0.43 on this engine.
OVERRULE_MIN_WORDS = 3
OVERRULE_MIN_WORD_SHARE = 0.6


def english_overrules_language_id(result) -> bool:
    """True when the English recogniser's own result is a clean English sentence, so it beats a confident Bengali/Hindi language ID."""
    text = (getattr(result, "text", "") or "").strip()
    words = text.split()
    agreement = float(getattr(result, "decoder_agreement", 0.0) or 0.0)
    if len(words) < 2 or agreement < ENGLISH_MIN_AGREEMENT or script_share(text, "en") < 0.9:
        return False
    share = english_word_share(text)
    if len(words) >= OVERRULE_MIN_WORDS:
        return share >= OVERRULE_MIN_WORD_SHARE
    return share == 1.0 and agreement >= 0.8  # two words ("you take"): every word real English and the decoders agree


class _Recognition(Protocol):
    """What the picker reads of one recogniser's result (agent/asr.ASRResult has both)."""

    text: str
    decoder_agreement: float


ENGLISH_SWITCH_MIN_AGREEMENT = 0.2


def _is_switch_request(text: str) -> bool:
    from agent.language_switch import detect_language_switch_request

    return detect_language_switch_request(text, "en") is not None


def pick_candidate(
    candidates: list[tuple[str, _Recognition]], lid_scores: dict[str, float] | None = None, prior: str | None = None
):
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
    # A short English COMMAND to change language wins even at low decoder agreement: pod-tested 2026-10-14, the English engine
    # heard "Reply in English" as "apply in english" (agreement 0.25), below ENGLISH_MIN_AGREEMENT, and a Bengali mis-hearing
    # won -- so the request itself was lost. Only a transcript that parses as a switch request, in Latin script, qualifies.
    for lang, r in usable:
        if (
            lang == "en"
            and r.decoder_agreement >= ENGLISH_SWITCH_MIN_AGREEMENT
            and script_share(r.text, "en") >= 0.9
            and _is_switch_request(r.text)
        ):
            return lang, r
    for lang, r in usable:
        if lang == "en" and r.decoder_agreement >= ENGLISH_MIN_AGREEMENT and script_share(r.text, "en") >= 0.9:
            # With language ID scores in hand, "English" must also be English (see ENGLISH_MIN_WORD_SHARE).
            if (
                lid_scores is None
                or lid_scores.get("en", 0.0) >= ENGLISH_MIN_LID
                or english_word_share(r.text) >= ENGLISH_MIN_WORD_SHARE
            ):
                # Real call 2026-10-17: in a call that had been Bengali for many turns, with language ID at bn 0.97 / en 0.00 and the Bengali
                # recogniser reading "তো আমার একটা বুকিং করে দিন কালকে" perfectly (agreement 1.00), the English recogniser's "the booking go to
                # then galva" (agreement 0.62, 4 of 6 words ordinary English) won, and the booking request was lost. Inside an established
                # Indic call, with language ID sure of it and not hearing English, English needs to be clearly English (>= 0.75 ordinary words, decoders agreeing >= 0.7).
                if (
                    prior in ("bn", "hi")
                    and lid_scores
                    and lid_scores.get(prior, 0.0) >= STABLE_CALL_LID
                    and lid_scores.get("en", 0.0) < ENGLISH_MIN_LID
                    and (english_word_share(r.text) < STABLE_CALL_ENGLISH_SHARE or r.decoder_agreement < STABLE_CALL_ENGLISH_AGREEMENT)
                ):
                    continue
                if (
                    prior in ("bn", "hi")
                    and lid_scores
                    and len(r.text.split()) <= SHORT_UTTERANCE_WORDS
                    and lid_scores.get("en", 0.0) < SHORT_ENGLISH_MIN_LID
                ):
                    continue
                return lang, r
    rest = [(lang, r) for lang, r in usable if lang != "en"] or usable
    if lid_scores:
        believed = [(lang, r) for lang, r in rest if lid_scores.get(lang, 0.0) >= INDIC_MIN_LID]
        rest = believed or rest
    best = max(rest, key=lambda t: (round(script_share(t[1].text, t[0]), 1), t[1].decoder_agreement))
    if lid_scores and not prior:
        # First turn: no established call language yet. Real call 2026-10-12: LID split hi 0.52 / bn 0.48 --
        # essentially a coin flip -- and the old line here ("language ID's own favourite among these is the
        # one to stay with") turned that coin flip into a protected prior, which then overrode `best` (hi
        # 0.57 vs bn 1.00 decoder agreement, on genuinely Bengali speech) via the switch-guard below, exactly
        # the class of bug this whole function exists to prevent. Anchoring to "bn" instead -- the same
        # default CallSession.lang itself starts at, never a value computed from the same shaky LID scores
        # this function is built to not fully trust -- means the switch-guard only overturns `best` when the
        # challenger's OWN score clears SWITCH_MIN_LID, the identical bar an ALREADY-established call must
        # clear to move away from Bengali (see the test just above this one). A first turn is held to the
        # same standard as every later one, not a weaker one a near-tied LID score could slip under.
        prior = "bn"
    if lid_scores and prior and best[0] != prior and lid_scores.get(best[0], 0.0) < SWITCH_MIN_LID:
        stay = [(lang, r) for lang, r in rest if lang == prior]
        if stay:
            return stay[0]  # no switch on a recogniser's say-so alone
    return best
