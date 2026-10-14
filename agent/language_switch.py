"""KCD-438: a caller explicitly asks the agent to speak a different
language ("please speak in Hindi", "bengali te bolun", "आप अंग्रेज़ी में
बोल सकते हैं?").

Deterministic keyword detection, no LLM call -- the same zero-extra-
latency, no-hallucination-risk reasoning as agent/booking_flow.py's
classify_yes_no. Scope, stated plainly: this detects the REQUEST and
main.py uses it to (a) speak the very next reply in the requested
language and (b) bias the language-ID router's ambiguous-turn prior
toward it via ASRLanguageRouter.note_response_language(), so a later
turn where LID itself is unsure leans toward what the caller asked for.
It does NOT force every later reply into the requested language
regardless of what the caller actually says next -- this project's
per-utterance language architecture (KCD-437) means a caller who goes
back to speaking their original language is still heard correctly, and
overriding that safely would mean threading a second language variable
through every reply call site in main.py's dispatch loop, a change too
large to make blind against a file with no local test coverage (it
needs the pod's ASR/TTS venv to import at all). Revisit with real call
data once that refactor can be verified live.
"""

from __future__ import annotations

import re
import unicodedata

_SWITCH_PHRASES: dict[str, dict[str, str]] = {
    # utterance language -> {phrase substring: target language code}
    "bn": {
        "হিন্দিতে বল": "hi",
        "হিন্দিতে কথা": "hi",
        "হিন্দি ভাষায়": "hi",
        "ইংরেজিতে বল": "en",
        "ইংরেজিতে কথা": "en",
        "ইংলিশে বল": "en",
        "বাংলায় বল": "bn",
        "বাংলায় কথা": "bn",
        # OWNER'S INSTRUCTION, 2026-10-10 (real call): "আহ আমি হিন্দি জানি না তো বাংলাতে কথা বলবেন" went undetected --
        # the hi/en targets just above already use the "-তে" postposition ("হিন্দিতে বল", "ইংরেজিতে বল"), but the
        # bn target only ever had the "-য়" form. A caller who phrases it the SAME way as the other two targets
        # was never covered. Adding the missing form, not inventing a new one.
        "বাংলাতে বল": "bn",
        "বাংলাতে কথা": "bn",
        # OWNER'S INSTRUCTION, real call 2026-10-13: a caller's imperative without the "-তে" postposition at
        # all ("হিন্দি বলুন"/"হিন্দি বলো", bare stem "বল") was never covered -- every target above needs the
        # postposition present. Same class of gap as the two fixes above it; adding the bare forms, all
        # three targets, not inventing a new phrasing.
        "হিন্দি বল": "hi",
        "ইংরেজি বল": "en",
        "বাংলা বল": "bn",
        # sourav, 2026-10-14 (real call): "বেঙ্গলিতে কথা বলুন" ("speak in Bengali", the English loanword spelt in Bengali)
        "বেঙ্গলিতে কথা": "bn",
        "বেঙ্গলিতে বল": "bn",
        "বেঙ্গলি বল": "bn",
        "বাংলা ভাষায়": "bn",
        "হিন্দি ভাষায় কথা": "hi",
        "ইংলিশে কথা": "en",
        "ইংরেজি ভাষায়": "en",
    },
    "hi": {
        "हिंदी में बोल": "hi",
        "हिंदी में बात": "hi",
        "अंग्रेज़ी में बोल": "en",
        "इंग्लिश में बोल": "en",
        "अंग्रेजी में बात": "en",
        "बंगाली में बोल": "bn",
        "बांग्ला में बोल": "bn",
        # OWNER'S INSTRUCTION, real call 2026-10-13: "बात कीजिए हिंदी बोलिए" (no "में" at all, bare stem
        # "बोल") went undetected -- every hi-dict target above requires "में" present. Same class of gap as
        # the bn-table fix on 2026-10-10 and the "-में बात" fix on 2026-10-12; adding the bare forms, all
        # three targets, not inventing a new phrasing.
        "हिंदी बोल": "hi",
        "अंग्रेज़ी बोल": "en",
        "इंग्लिश बोल": "en",
        "बंगाली बोल": "bn",
        "बांग्ला बोल": "bn",
        # OWNER'S INSTRUCTION, real call 2026-10-12: "बेंगोली में बात कीजिए हिंदी नहीं है" went undetected --
        # the hi/en targets just above already have a "-में बात" form alongside "-में बोल", but the bn
        # target only ever had "-में बोल". Same class of gap as the bn-table fix on 2026-10-10 (one
        # phrasing of a target was covered, its sibling was not); adding the missing "-में बात" form, and
        # the "बेंगोली"/"बेंगॉली" spellings the ASR actually produced for "bengali" (alongside the
        # dictionary "बंगाली"/"बांग्ला"), not inventing a new phrasing.
        "बंगाली में बात": "bn",
        "बांग्ला में बात": "bn",
        # sourav, 2026-10-14: the other spellings the Hindi recogniser writes for "Bangla/Bengali".
        "बांगला में बोल": "bn",
        "बांगला में बात": "bn",
        "बंगला में बोल": "bn",
        "बंगला में बात": "bn",
        "बांग्ला बोलिए": "bn",
        # sourav, 2026-10-14 (real call): "स्पीक इन बेंगोली" -- "Speak in Bengali" in Devanagari
        "बेंगोली": "bn",
        "बेंगाली": "bn",
        "बंगोली": "bn",
        "बांगला": "bn",
        "स्पीक इन हिंदी": "hi",
        "स्पीक इन हिन्दी": "hi",
        "स्पीक इन इंग्लिश": "en",
        "स्पीक इन अंग्रेजी": "en",
        "बंगाली बोलिए": "bn",
        "बेंगोली में बोल": "bn",
        "बेंगोली में बात": "bn",
        "बेंगॉली में बोल": "bn",
        "बेंगॉली में बात": "bn",
    },
    "en": {
        "speak in hindi": "hi",
        "talk in hindi": "hi",
        "hindi please": "hi",
        "speak in bengali": "bn",
        "talk in bengali": "bn",
        "bengali please": "bn",
        "speak in english": "en",
        "talk in english": "en",
        "english please": "en",
        # Same bare-imperative gap as the bn/hi fixes above (no "in"): "please speak hindi"/"speak bengali".
        "speak hindi": "hi",
        "speak bengali": "bn",
        "speak english": "en",
        # OWNER'S INSTRUCTION, real call 2026-10-13: a caller code-switches "bengali"/"hindi"/"english" as
        # loanwords into an otherwise Bengali/Hindi sentence, romanised whole by ASR/translation -- "bengali
        # te bolun hindi bujhena" ("speak in Bengali, I don't understand Hindi"), "hindi mei boliye" ("speak
        # in Hindi") -- which LID reads as "en" (Latin script) even though the content is bn/hi. Same class
        # of gap as agent/call_end.py's romanised "call kete den" fix this same day; added here so these
        # requests are read against the "en" table, which this module's own detection keys lookup by.
        "bengali te bolun": "bn",
        "bengali te bolo": "bn",
        "bangla te bolun": "bn",
        "bangla te bolo": "bn",
        "bangla bolun": "bn",
        "hindi mei boliye": "hi",
        "hindi mein boliye": "hi",
        "hindi me boliye": "hi",
        "hindi mei bolo": "hi",
        "hindi mein bolo": "hi",
        "hindi boliye": "hi",
        "english mei bolo": "en",
        "english mein bolo": "en",
        "english te bolun": "en",
        # sourav, 2026-10-14 (real call): "Speak in Bangla" went undetected -- this table knew "bengali" but not the
        # spelling "bangla" (the word callers and the ASR actually use), nor "switch/change to <language>".
        "speak in bangla": "bn",
        "talk in bangla": "bn",
        "speak bangla": "bn",
        "talk bangla": "bn",
        "bangla please": "bn",
        "in bangla please": "bn",
        "bangla language": "bn",
        "switch to bangla": "bn",
        "change to bangla": "bn",
        "switch to bengali": "bn",
        "change to bengali": "bn",
        "switch to hindi": "hi",
        "change to hindi": "hi",
        "hindi language": "hi",
        "switch to english": "en",
        "change to english": "en",
        "bangla mein bolo": "bn",
        "bangla mein boliye": "bn",
        "bangla me boliye": "bn",
        "bengali mein boliye": "bn",
        "bengali me boliye": "bn",
    },
}


def detect_language_switch_request(text: str, lang: str) -> str | None:
    """`lang` is the ASR-detected language of THIS utterance (the request
    itself is understood correctly regardless of which language it was
    made in). Returns the requested target language code, or None if this
    utterance was not a language-switch request."""
    t = text.strip().lower()
    # sourav, 2026-10-14 (real call): the call was locked to English, so the Bengali-script request "বেঙ্গলিতে কথা বলুন"
    # was looked up only in the English table and never found. A language-switch request must work whatever language
    # the call is locked to / the recogniser guessed, so after the utterance's own table the other tables are tried too
    # (the phrases are different scripts/spellings, so a Hindi phrase cannot fire on an English sentence).
    for table_lang in (lang, *(l for l in ("bn", "hi", "en") if l != lang)):
        for phrase, target in _SWITCH_PHRASES.get(table_lang, {}).items():
            # another language's table is consulted for NATIVE-SCRIPT phrases only; Latin-script phrases stay tied to the
            # utterance's own table, exactly as before (tests/test_language_switch.py pins that).
            if table_lang != lang and phrase.isascii():
                continue
            if phrase.lower() in t:
                return target
    return _generic_switch_request(text, lang)


# ---------------------------------------------------------------------------------------------------------------
# Generic detector. Real call 2026-10-14 (debug transcript): "speaking in English", "Reply in English" and "Speak in
# english" were never recognised, while the same request in Hindi and Bengali was. Not one bug but a coverage gap in
# the phrase tables above: the English table knew "speak in english" but not "reply in english" / "speaking
# in english", and the Hindi recogniser's spelling of the same words ("स्पेकिंग इंग्लिश") is in no table at all. A
# table of exact phrases can never keep up with how people ask, so the rule below is the general one: the caller
# NAMES a language (in any script or spelling) AND says a speak/reply/talk word (or "in <language>", or says just
# the language's name). It runs only after every exact phrase above has had its say, so nothing that worked changes.
# ---------------------------------------------------------------------------------------------------------------

# Written nukta-free and lower-case, matched as substrings of the same normalisation of the caller's words.
_LANGUAGE_NAMES: dict[str, tuple[str, ...]] = {
    "en": (
        "english", "inglish", "ingrezi", "angrezi", "angreji",
        "इंग्लिश", "इंग्लीश", "इंगलिश", "अंग्रेजी", "अंग्रेज", "अंग्रेजि",
        "ইংলিশ", "ইংরেজি", "ইংরেজী", "ইংরাজি", "ইংলিস",
    ),
    "hi": ("hindi", "हिंदी", "हिन्दी", "हिंदि", "হিন্দি", "হিন্দী"),
    "bn": (
        "bengali", "bangla", "bengoli", "bangali",
        "बंगाली", "बांग्ला", "बांगला", "बंगला", "बेंगोली", "बेंगाली", "बंगोली", "बेंगॉली",
        "বাংলা", "বাঙলা", "বেঙ্গলি", "বেঙ্গালি",
    ),
}
# "Bangladesh" is a place, not the language.
_NOT_A_LANGUAGE = ("bangladesh", "বাংলাদেশ", "बांग्लादेश", "बंगलादेश")

# Words that mean "speak / reply / talk / switch". Latin ones are whole words; native-script ones are stems.
_VERB_WORDS_LATIN = frozenset(
    "speak speaking spoke reply replying respond talk talking say answer switch change convert continue "
    "bolo bolun boliye boliyo bolie bolna bolen bol kotha katha baat jawab uttar karo kariye kijiye".split()
)
_VERB_STEMS_NATIVE = (
    "বল", "কথা", "উত্তর", "জবাব", "স্পিক", "রিপ্লাই", "রিপ্লে", "টক", "চেঞ্জ", "সুইচ", "পরিবর্তন", "কন্টিনিউ",
    "बोल", "बात", "जवाब", "उत्तर", "स्पीक", "स्पिक", "स्पेक", "रिप्लाई", "रिपलाई", "बदल", "चेंज", "स्विच", "कंटिन्यू",
    "टॉक", "कहिए", "कहो",
)
# A language named and then negated ("हिंदी नहीं", "hindi nahi") is the language NOT wanted.
_NEGATIONS = frozenset("nahi nahin nai not no na নয় না নেই নেই। नहीं नही ना".split())
_MAX_WORDS = 12


def _norm_words(text: str) -> str:
    s = unicodedata.normalize("NFC", text or "").lower().replace("\u093c", "")
    return " ".join(re.sub(r"[.,;:!?\"'()\[\]{}।॥\-–—_/]+", " ", s).split())


def _generic_switch_request(text: str, lang: str) -> str | None:
    t = _norm_words(text)
    words = t.split()
    if not words or len(words) > _MAX_WORDS:
        return None
    # The long-standing rule (tests/test_language_switch.py): a request that is pure Latin script is read against the
    # English recogniser's output only. A Latin sentence the Hindi or Bengali recogniser produced is not trusted.
    if t.isascii() and lang != "en":
        return None
    if any(bad in t for bad in _NOT_A_LANGUAGE):
        return None
    named: dict[str, int] = {}
    for code, names in _LANGUAGE_NAMES.items():
        positions = [t.find(n) for n in names if n in t]
        if positions:
            named[code] = min(positions)
    if not named:
        return None
    if len(named) > 1:
        # "bengali please, not hindi": drop a language that is followed (within two words) by a negation.
        for code, pos in list(named.items()):
            tail = t[pos:].split()[1:3]
            if any(w in _NEGATIONS for w in tail):
                named.pop(code)
        if len(named) != 1:
            return None
    target = next(iter(named))
    has_verb = any(w in _VERB_WORDS_LATIN for w in words) or any(stem in t for stem in _VERB_STEMS_NATIVE)
    # Owner, 2026-10-14: "Reply in English" and "apply in english" mean the same thing -- when the call is in another
    # language, the KEYWORD "English" is the request, whatever verb the recogniser made of it. So a SHORT utterance (up to
    # four words) that names exactly one language is a request to speak it; a longer one still needs a speak/reply word,
    # so "my report is in english and long" is left alone.
    short = len(words) <= 4
    return target if (has_verb or short) else None
