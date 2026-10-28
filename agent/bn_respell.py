"""Respell Bengali words the Bengali voice says badly, just before they reach the voice.

Real call 2026-10-14: the caller named six words the voice speaks poorly (বলুন, দুঃখিত, সময়টা, কয়েক, নেওয়া, হয়ে).
agent/pronunciation.py cannot help: it maps ENGLISH words to Bengali script. These are ordinary Bengali words, so what
is needed is a spelling that makes the voice produce the right SOUND.

HOW THE SPELLINGS IN _RESPELL WERE CHOSEN (tools/respell_probe.py is the instrument, kept in the repo):
each candidate spelling is synthesised by the pod's Bengali voice, played through the Bengali recogniser, and kept only if
the recogniser reads the INTENDED word back. That is a measurement of intelligibility, not of naturalness: it finds words
the voice garbles badly enough that a recogniser cannot recover them, and it cannot tell "good" from "acceptable". A native
listener signs the final list off (REVIEWED below stays empty until one does), exactly as agent/pronunciation.py's does.

A word is replaced only when it is a WHOLE word in the table; sentence structure, punctuation and every other word are
untouched. The table is applied after number/date verbalisation (agent/speech_norm.verbalize) and before the cache key is
built, so a respelled clip is cached under what was actually sent to the voice.
"""

from __future__ import annotations

import os
import re
import unicodedata

# spoken word (as written everywhere else in the repo) -> spelling that makes the voice say it right.
_RESPELL: dict[str, str] = {
    # CHOSEN BY A NATIVE LISTENER, 2026-10-15, from numbered audio candidates (the round-trip recogniser was too forgiving
    # to judge: it read back words a Bengali ear still hears as Hindi-accented or wrong). The pattern in the winners: the voice
    # reads স as Hindi "s" where Bengali says "sh", so native words start with শ.
    "প্রয়োজন": "প্রোজোন",  # was candidate 3 of 5
    "সাড়া": "শাঁড়া",  # candidate 4 of 4
    "সময়টা": "শমোইটা",  # candidate 4 of 5 (the earlier machine-chosen সমোইটা is replaced)
    "নেওয়া": "নেওআ",  # 2026-10-17: candidate 4 of 6 (the earlier নেওয়াা was "better, not perfect")
    "যাওয়া": "যাওয়াা",  # 2026-10-17: the listener chose the lengthened আ for যাওয়ার (candidate 2 of 6)
    "যাওয়ার": "যাওয়াার",
    "স্যাম্পল": "স্যাম্পেল",  # 2026-10-17: candidate 4 of 6
    # Round 3, 2026-10-17: the listener kept গিয়ে and হয় as written (candidate 1 was best) and chose নিয়ে -> নিইয়ে (candidate 4 of 6).
    "নিয়ে": "নিইয়ে",
    "নিয়েও": "নিইয়েও",
    # Round 4, 2026-10-17: ঠিকানা -> ঠিকানাহ (the listener's candidate 5 of 6); an inflected ঠিকানাটা is left as written (the added হ would sit mid-word).
    "ঠিকানা": "ঠিকানাহ",
    # নদিয়া -> নোদিয়া, chosen from the listener's own recording (My recording 50): of 15 spellings the voice said, নোদিয়া was within 0.02 of the
    # nearest by acoustic distance and is what the English recogniser heard ("nody ah"); not in REVIEWED until the listener confirms it.
    "নদিয়া": "নোদিয়া",
    "নিয়েই": "নিইয়েই",
    # কয়েক: chosen from the user's OWN recording of the word (Downloads/My recording 49.m4a): of 15 spellings the voice said, কয়এক was
    # the nearest to it acoustically (MFCC distance 3.46; the next 3.87, the current spelling 4.43). Measured against one speaker, so
    # the listener's ear still decides -- it is not in REVIEWED until they say so.
    "কয়েক": "কয়এক",
    # Same ড়া sound as সাড়া, so it gets the same treatment (nasal mark after শা). NOT yet heard by the listener.
    "সাড়ে": "শাঁড়ে",
    # Owner, 2026-10-27: "হোয়াটসঅ্যাপ" (WhatsApp) reported mispronounced on a real call -- this is the loanword's
    # standard Bengali spelling (agent/pronunciation.py's own English->Bengali table already uses it too), so the
    # word itself is not wrong, the voice's reading of its opening "হোয়া" (ho-yā) cluster is. Quick fix, NOT run
    # through tools/respell_probe.py (no pod access right now): dropping the হ, the softer-opening alternative,
    # is the one change to try first. Owner confirms by ear; swap for a different candidate if this one is not it.
    "হোয়াটসঅ্যাপ": "ওয়াটসঅ্যাপ",
    # Owner, 2026-10-27: "অ্যাপয়েন্টমেন্ট" (appointment) was reported mispronounced. CHOSEN BY A NATIVE LISTENER from six numbered audio candidates on the pod's Bengali voice
    # (the original, আপয়েন্টমেন্ট, অ্যাপোয়েন্টমেন্ট, এপয়েন্টমেন্ট, অ্যাপইন্টমেন্ট, অ্যাপয়ন্টমেন্ট): candidate 6. Its inflected forms (…টা, …র, …ের, …কে) follow the base word.
    "অ্যাপয়েন্টমেন্ট": "অ্যাপয়ন্টমেন্ট",
    # Owner, 2026-10-28: CHOSEN BY A NATIVE LISTENER from numbered audio samples of the pod's Bengali voice (kolkata_2; digit0_2, digit1_2, digit2_2, digit3_2, digit4_1, digit5_1, digit7_1, digit8_2).
    # চার, পাঁচ and সাত were chosen as written; সাত is listed so the "sh" rule below does not turn it into শাত. ছয় (6) and নয় (9) are still open: the listener gave their own recordings.
    "কলকাতা": "কোলকাতা",
    "শূন্য": "শুন্য",
    "এক": "অ্যাক",
    "দুই": "দুঈ",
    "তিন": "তীন",
    "সাত": "সাত",
    # 6 and 9 chosen from the listener's own recordings and the spelling "Chhwoy" / "Nwoy": sixw_2 and ninew_2 of the second round of samples.
    "ছয়": "ছওয়্",
    "নয়": "নওয়্",
    # Owner, 2026-10-28: the doctor's first name, chosen by the listener from numbered samples (joy_10: said as two words)
    # Owner, 2026-10-28: সরকার is said as written (sarkar_1): it is NOT in the "sh" rule's list above any more, so সরকার and সরকারের stay as they are. ডায়াবেটিস is said "ডায়া বেটিস" (diab_3).
    "ডায়াবেটিস": "ডায়া বেটিস",
    "ডায়াবিটিস": "ডায়া বেটিস",
    "জয়দীপ": "জয় দীপ",
    # the same name as the recogniser and the clinic data may spell it (owner, 2026-10-28: "cover all and similar words with similar spellings of names")
    "জয়দিপ": "জয় দীপ",
    "জয়দীপ্": "জয় দীপ",
    "জয়দিপ্": "জয় দীপ",
    "জয়দ্বীপ": "জয় দীপ",
    "জয়দ্বিপ": "জয় দীপ",
    "জোয়দীপ": "জয় দীপ",
    "জইদীপ": "জয় দীপ",
    "যয়দীপ": "জয় দীপ",
    # Owner, 2026-10-28: Dr Nirmal Roy; chosen by the listener from numbered samples (nir_3: "নির মল"). Other spellings of the name and its endings follow below (_NAME_BASES).
    "নির্মল": "নির মল",
    "নিরমল": "নির মল",
    "নির্মোল": "নির মল",
    "নীর্মল": "নির মল",
    "আট": "আট্",
}

# ---------------------------------------------------------------------------------------------------------------------
# The "sh" rule. The Bengali voice reads a plain স as Hindi "s" where Bengali says "sh" (ʃ), which is what makes সাড়া,
# সময়টা and the rest sound Hindi. Two of the four words a native listener fixed were fixed by starting them with শ.
#
# THE RULE: in a NATIVE Bengali word, every স that is not the first half of a conjunct (স্ত, স্প, স্থ ...) is written শ.
#
# WHY A LIST AND NOT EVERY WORD: Bengalis say English-origin words with a real "s" (সিবিসি, সেন্টার, সিটি, ক্যালসিয়াম,
# প্রেসক্রিপশন, সোনোস্ক্যান, অ্যাসিড ...), so a blanket rule would break exactly the lab vocabulary the bot speaks most.
# _NATIVE_SH is every Bengali word with a plain স in the bot's OWN sentences (phrases, reply templates, FAQ answers, test
# names, doctor names, number words), classified by hand, native only. A word the bot says that is not in the list is left alone.
#
# STATUS: the rule rests on two words the listener fixed this way; the rest of the list is applied on that evidence and is
# open to veto word by word (tools/respell_audit.py prints every word the rule changes). BN_RESPELL_RULES=0 turns the
# rule off and keeps only the explicit table above.
# ---------------------------------------------------------------------------------------------------------------------
_NATIVE_SH = frozenset(
    """
    সময় সময়ে সময়ের সময়গুলো সাহায্য সেটা সাথে সরাসরি সব সবচেয়ে সমস্যা সমস্যার সেন সেনগুপ্ত সঙ্গে সকাল সকালে সঠিক
    সম্ভবত সম্ভব সুবিধা অসুবিধার সহকর্মী সাধারণত সপ্তাহে আসন্ন সেখান সম্পর্ক সংস্থাগুলোর সংগ্রহ সত্যিই সন্ধ্যা
    এসে এসো আসব আসবেন আসতে আসুন বসেন বসবেন বসছেন উপবাস জিজ্ঞেস মাস মাসের সাল হাসপাতাল হাসপাতালে
    বসু দাস সাহা সিনহা সৌরভ সুব্রত মানস সুতপা সঞ্জয় সমীর সুদীপ দেবাশিস স্নেহাশিস বিশ্বাস
    সাত সাতটা সাতাশ সাতাশি সাতান্ন সাঁইত্রিশ সাতচল্লিশ সাতষট্টি সাতাত্তর সাতানব্বই সাতশো সতেরো সত্তর ঊনসত্তর সাড়ে সোয়া
    সোম সোমবার সেই সেটার সেটি সেটায় সবগুলো সবমিলিয়ে একসাথে একসঙ্গে জিনিস সর্দি সরান সরাও সরিয়ে জিজ্ঞাসা উপোস ইতিহাস
    আসল আসি অসুবিধা সম্পর্কে বয়সটা সড়ক সময়সূচি সুনীতা সত্যিকারের
    """.split()
)
_INFLECTABLE = (
    "সময় সাহায্য সেটা সব সকাল সুবিধা অসুবিধা সোম সোমবার সাত সাতটা সঠিক সমস্যা সম্পর্ক সপ্তাহ সন্ধ্যা সাল মাস "
    "সম্ভব সাধারণ সংগ্রহ সময়সূচি জিনিস সর্দি আসল উপবাস উপোস"
).split()
_ENDINGS = ("ে", "ের", "টা", "টি", "টায়", "টার", "টাকে", "গুলো", "গুলোর", "কে", "ও", "ই", "য়", "র", "েও", "ে", "তে")
_NATIVE_BASE = _NATIVE_SH  # the reviewed words themselves (what tools/respell_audit.py lists)
_NATIVE_SH = _NATIVE_SH | frozenset(unicodedata.normalize("NFC", b + e) for b in _INFLECTABLE for e in _ENDINGS)
_PLAIN_SA = re.compile("স(?!\u09cd)")  # a স that is not followed by the virama: not the start of a conjunct


def _sh(word: str) -> str:
    return _PLAIN_SA.sub("শ", word)


def rule_changes() -> dict[str, str]:
    """Every word the sh rule changes, for review (tools/respell_audit.py)."""
    return {w: _sh(w) for w in sorted(_NATIVE_BASE) if w not in _RESPELL and _sh(w) != w}


REVIEWED: set[str] = {"প্রয়োজন", "সাড়া", "সময়টা", "নেওয়া", "যাওয়ার", "স্যাম্পল", "নিয়ে", "ঠিকানা"}  # approved by a native listener

_BENGALI_WORD = re.compile(r"[ঀ-৿]+")


def _key(word: str) -> str:
    return unicodedata.normalize("NFC", word)


_RESPELL = {_key(k): v for k, v in _RESPELL.items()}  # a key typed with a precomposed য় still matches the recogniser's decomposed one


# LAYER 2 -- the same word with a common ending. A native listener chose a spelling for the BASE word ("নেওয়া", "যাওয়া", "কয়েক",
# "স্যাম্পল"); the caller hears the same sound in "নেওয়ার", "যাওয়াটা", "কয়েকটা", "স্যাম্পলের". So the chosen spelling of the base is kept and the
# ending is appended unchanged. Only these endings, only on these bases: a longer word that merely starts the same is left alone.
_FAMILY_ENDINGS = frozenset(
    ("র", "য়", "রও", "টা", "টাও", "টি", "টায়", "টার", "টাকে", "তে", "তেও", "ও", "ই", "কে", "ে", "ের", "েও", "গুলো", "গুলোর", "জন", "জনের", "দিন", "দিনের")
)


_FAMILY_BASES = ("নেওয়া", "যাওয়া", "কয়েক", "স্যাম্পল", "নদিয়া", "অ্যাপয়েন্টমেন্ট", "ডায়াবেটিস", "ডায়াবিটিস")

# A person's name with the endings it takes in speech: "জয়দীপের" (of), "জয়দীপকে" (to), "জয়দীপবাবু" (Mr), "জয়দীপদা" (elder brother) ... The chosen spelling of the name is kept and the ending
# appended, the same way _FAMILY_BASES works for ordinary words.
_NAME_BASES = tuple(unicodedata.normalize("NFC", b) for b in ("জয়দীপ", "জয়দিপ", "জয়দীপ্", "জয়দিপ্", "জয়দ্বীপ", "জয়দ্বিপ", "জোয়দীপ", "জইদীপ", "যয়দীপ", "নির্মল", "নিরমল", "নির্মোল", "নীর্মল"))
_NAME_ENDINGS = frozenset(
    ("র", "ের", "কে", "ও", "ই", "ে", "েরও", "কেও", "বাবু", "বাবুর", "বাবুকে", "বাবুও", "দা", "দার", "দাকে", "দাও", "টা", "টির", "স্যার", "স্যারের", "স্যারকে", "ডাক্তার")
)


# Owner, 2026-10-28: "only words with the জয়দেব structure -- where the second letter is য় -- should be said as two words: জয় দেব". The rule is on the SHAPE: জয় + a rest that starts with a consonant (not
# a vowel sign, not a conjunct such as ন্ত) and has at least two consonant letters (জয়দেব, জয়দীপ, জয়কৃষ্ণ, জয়গোপাল, জয়সেন, and with any ending: জয়দেবের). It is limited to the জয় prefix on purpose: other
# words with য় second (বয়স, বয়সের, শয়তান, নয়ন, পয়সা, ময়দা, দয়াল, হয়েছে) are ordinary words and must not be cut, and জয়ন্ত, জয়শ্রী, জয়িতা, জয়ী are not this shape.
_JOY_PREFIX = unicodedata.normalize("NFC", "জয়")
_VIRAMA = "\u09cd"
_CONSONANTS = frozenset(chr(c) for c in list(range(0x0995, 0x09BA)) + [0x09DC, 0x09DD, 0x09DF, 0x09CE])


def _joy_compound(w: str) -> str | None:
    if not w.startswith(_JOY_PREFIX):
        return None
    rest = w[len(_JOY_PREFIX):]
    if len(rest) < 3 or rest[0] not in _CONSONANTS or rest[1] == _VIRAMA:
        return None
    if sum(1 for ch in rest if ch in _CONSONANTS and ch not in ("\u0982", "\u0983")) < 2:
        return None
    return f"{_JOY_PREFIX} {respell(rest)}"


def _name_family(w: str) -> str | None:
    for base in _NAME_BASES:
        if w.startswith(base) and w[len(base):] in _NAME_ENDINGS:
            return _RESPELL[base] + w[len(base):]
    return None


def _family(w: str) -> str | None:
    for base in _FAMILY_BASES:
        if w.startswith(base) and w[len(base):] in _FAMILY_ENDINGS:
            return _RESPELL[base] + w[len(base):]
    return None


def respell(word: str) -> str:
    """The spelling to send to the voice for one Bengali word; the word itself when it needs no help."""
    w = _key(word)
    if w in _RESPELL:
        return _RESPELL[w]
    fam = _family(w)
    if fam is not None:
        return fam
    name = _name_family(w)
    if name is not None:
        return name
    joy = _joy_compound(w)
    if joy is not None:
        return joy
    if w in _NATIVE_SH and os.environ.get("BN_RESPELL_RULES", "1") != "0":
        return _sh(w)
    return word


def apply(text: str, lang: str) -> str:
    """Respell every whole Bengali word in `text` that the table knows. Only the Bengali voice is touched."""
    if lang != "bn" or not text or os.environ.get("BN_RESPELL", "1") == "0":
        return text
    return _BENGALI_WORD.sub(lambda m: respell(m.group(0)), text)


def table() -> dict[str, str]:
    return dict(_RESPELL)
