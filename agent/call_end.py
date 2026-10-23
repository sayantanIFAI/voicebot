"""The caller wants the call to end -- decided by words, not by the model.

A live call (2026-09-25): the agent asked "anything else? if not I will end the call", the caller said "শেষ করে দিন"
("finish it / end it"), and the agent went to the model, which heard a booking request and asked for a confirmation number.
Ending a call is a binary, low-risk, high-annoyance decision: it is decided here, deterministically, with no model.

Two strengths of cue, because the same words mean different things in different places:

  * EXPLICIT -- unmistakable at any moment: "end the call", "কল কেটে দিন", "फोन रखता हूँ", "goodbye". Mid-booking too;
    the unfinished booking is saved for the next call (main._save_unfinished_draft).
  * WHEN ASKED -- only as the answer to "is there anything else?": a bare "শেষ করে দিন", "nothing", "that's all",
    "no thanks". The same words mid-booking ("finish the booking") are not a request to hang up.

Whole-token matching for short cues (a substring hit would fire "bye" inside "maybe"), substring matching for longer
Bengali phrases (Bengali is written without spaces between a word and its endings). REASONED cue lists, not measured;
a native speaker should review them (docs/pod-verification-checklist.md).
"""

from __future__ import annotations

import re
import unicodedata

_EXPLICIT = {
    "bn": (
        "কল শেষ",
        "কল কেটে",
        "কল রেখে",
        "কল বন্ধ",
        "ফোন রাখ",
        "ফোন রেখে",
        "ফোন কেটে",
        "ফোন ছেড়ে",
        "লাইন কেটে",
        "রাখছি",
        "রেখে দিচ্ছি",
        "বাই",
        "গুডবাই",
        "টা টা",
        # "cut" on its own is unambiguous for ending a CALL (unlike "শেষ"/finish, which also means finishing a
        # booking -- that stays in _WHEN_ASKED below, gated by _TASK_WORDS). A live call (2026-09-28): the caller
        # said "কেটে দিন" mid-conversation, not right after the "anything else?" prompt, and it was not recognised.
        "কেটে দিন",
        "কেটে দাও",
        "কাটো",
        "কেটে দে",
        # The English loanword "cut" (not the native verb "কাটা" above), as a caller actually code-mixes it in:
        # "কল কাট করো", not just "কল কেটে দিন". Found missing in testing (2026-10-10), same gap in all 3 languages.
        "কল কাট",
        "কাট করো",
        "কাট দিন",
        "কাট করে দিন",
    ),
    "hi": (
        "कॉल खत्म",
        "कॉल काट",
        "कॉल बंद",
        "फोन रख",
        "फोन काट",
        "लाइन काट",
        "रखता हूं",
        "रखती हूं",
        "रखता हूँ",
        "रखती हूँ",
        "बाय",
        "अलविदा",
        "गुडबाय",
        "टाटा",
        # Same reasoning as the Bengali "কাটো" entries above.
        "काट दीजिए",
        "काट दो",
        "काटो",
        # The English loanword "कट" (not "काट" above) -- "कॉल कट कर दो" tested and missed (2026-10-10).
        "कॉल कट",
        "कट कर दो",
        "कट कर दीजिए",
        "कट करो",
    ),
    "en": (
        "end the call",
        "end this call",
        "end call",
        "hang up",
        "hanging up",
        "goodbye",
        "good bye",
        "bye",
        "that's all",
        "thats all",
        "that is all",
        "i'm done",
        "im done",
        "i am done",
        # "cut" was missing entirely for English despite being the exact example reported (2026-10-10) -- "end"/
        # "hang up" above, never "cut", even though it is one of the most natural ways to phrase this.
        "cut the call",
        "cut this call",
        "cut call",
        "please cut the call",
        "cut it off",
        # Real call 2026-10-13: a caller code-switches the English loanword "call" into an otherwise Bengali
        # sentence, and that sentence came back from ASR/translation in ROMANISED form -- "call kete den" (কল
        # কেটে দিন, "cut the call"), "call rakhun" (কল রাখুন, "hang up"), "call rakh diye"/"call rakh dicche"
        # (কল রেখে দিয়ে/দিচ্ছি, "having hung up the call") -- none of which matched the Bengali-script cues
        # above. Same verb, same three conjugation endings (-den/-dao/-de, -diye/-dicche) already covered in
        # Bengali script higher up in this tuple.
        "call kete den",
        "call kete dao",
        "call kete de",
        "call rakhun",
        "call rakho",
        "call rakh diye",
        "call rakh dicche",
        "call rekhe diye",
        "call rekhe den",
    ),
}
_WHEN_ASKED = {
    "bn": (
        "শেষ করে দিন",
        "শেষ করে দাও",
        "শেষ করুন",
        "শেষ করে দেন",
        "শেষ",
        "বন্ধ করে দিন",
        "বন্ধ করুন",
        "আর কিছু না",
        "আর কিছু নেই",
        "আর কিছু লাগবে না",
        "কিছু জানার নেই",
        "কিছু জানতে চাই না",
        "কিছু না",
        "থাক",
        "দরকার নেই",
        "লাগবে না",
        "না ধন্যবাদ",
        "না, ধন্যবাদ",
    ),
    "hi": (
        "खत्म कर दीजिए",
        "खत्म कर दो",
        "खत्म करें",
        "खत्म करिए",
        "खत्म",
        "बंद कर दीजिए",
        "बंद करें",
        "और कुछ नहीं",
        "कुछ नहीं",
        "कुछ और नहीं",
        "बस",
        "रहने दीजिए",
        "रहने दो",
        "नहीं चाहिए",
        "नहीं, धन्यवाद",
        "नहीं धन्यवाद",
    ),
    "en": (
        "end it",
        "go ahead and end",
        "finish it",
        "nothing",
        "nothing else",
        "no thanks",
        "no thank you",
        "that's it",
        "thats it",
        "nope",
        "i'm good",
        "im good",
        "all good",
        "no more",
    ),
}
# A bare "শেষ" can also mean "finish the booking": with a task word in the sentence it is not a request to hang up.
_TASK_WORDS = (
    "বুকিং",
    "অ্যাপয়েন্টমেন্ট",
    "টেস্ট",
    "সিরিয়াল",
    "बुकिंग",
    "अपॉइंटमेंट",
    "टेस्ट",
    "booking",
    "appointment",
    "test",
    "slot",
)
_PUNCT = re.compile(r"[?!.,;:।\"'()\[\]]")


def _normalise(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").replace("़", "").lower().replace("’", "'")
    t = _PUNCT.sub(lambda m: "'" if m.group(0) == "'" else " ", t)
    return " ".join(t.split())


def _has(text: str, cue: str) -> bool:
    cue = _normalise(cue)
    if " " in cue or len(cue) >= 5:
        return cue in text  # a phrase, or a long enough word to be unambiguous
    return f" {cue} " in f" {text} "  # a short word: whole tokens only ("bye" is not in "maybe")


def wants_to_end(text: str, lang: str, after_prompt: bool = False) -> bool:
    """True when the caller is asking to end the call. `after_prompt` is True right after the agent asked whether
    there was anything else (agent/phrases silence_prompt), which is when the bare cues count."""
    t = _normalise(text)
    if not t:
        return False
    languages = (lang,) + tuple(x for x in _EXPLICIT if x != lang)  # a caller may end it in any language
    if any(_has(t, c) for lg in languages for c in _EXPLICIT.get(lg, ())):
        return True
    if after_prompt and not any(w in t for w in _TASK_WORDS):
        return any(_has(t, c) for lg in languages for c in _WHEN_ASKED.get(lg, ()))
    return False

# ---------------------------------------------------------------------------------------------- a bare "okay, fine" is a goodbye
# Owner, 2026-10-23 (item 6): "achha thik ache", "ok", "thik hai" and the like, said when NOTHING is open (the agent's last words were an answer, not a question; no booking
# or other flow waits), mean "I have what I needed": the agent says its goodbye and cuts the call at once instead of waiting for another confirmation. Said while a question
# is open the very same words are a YES -- so main.py asks `_nothing_is_open` first, and this function only judges the WORDS: the sentence must be made of
# acknowledgement words alone, with at least one that is a real acknowledgement ("thik", "achha", "ok", "thanks", "bujhechi"), never a bare "ache" / "haan" / "hmm".
_ACK_PATTERNS = (
    re.compile(r"a+t?c{1,2}h{0,2}a+"),  # acha achha accha aachha atcha
    re.compile(r"o+ch+a+"),
    re.compile(r"th?(?:i+|e+)k+(?:ai|e|ei|hai|hay|ache|ase)?"),  # thik theek thiik thikai thikache thikhai thikhay
    re.compile(r"o+k+(?:a+y+|e+y+)?"),  # ok okk okay okey
    re.compile(r"th?a?n[kx]s?(?:u|you|q)?|thx|tnx|tq"),  # thanks thank thanku thankyou thanx thnx
    re.compile(r"dh?[ao]n+y?[ao]?[bv]+[ao]d|dh?[oa]nn?o[bv]+[ao]d|shu?k+u?r+i+y?a+|su?k+u?r+i+y?a+"),  # dhanyabad dhonnobad shukriya
    re.compile(r"bu?jh?(?:lam|echi|echhi|e|i|te)|samjh?(?:a|e|i)|samajh|bujhe"),  # bujhlam bujhechi samjha samajh
)
_ACK_WORDS = frozenset(
    {
        "fine", "alright", "right", "good", "great", "nice", "cool", "perfect", "understood", "noted", "sahi", "badhiya", "bariya", "bhalo", "valo", "besh", "bash", "thik",
        "gaya", "gayi", "gaye", "gelo", "holo", "chalo", "chalega", "cholbe", "cholo", "ठीक", "ठीकहै", "ठीकाछे", "अच्छा", "अच्छी", "अच्छे", "ओके", "ओके", "ओकेय", "थैंक्स", "थैंक", "थैंक्यू", "थैंकयू",
        "धन्यवाद", "शुक्रिया", "शुक्रिया", "समझ", "गया", "गयी", "गई", "बढ़िया", "बढिया", "सही", "चलो", "चलिए", "चलेगा", "बहुत",
        "আচ্ছা", "অচ্ছা", "আচ্ছা", "ঠিক", "ঠিকাছে", "ঠিকই", "ওকে", "ওকে", "ধন্যবাদ", "থ্যাংকস", "থ্যাঙ্কস", "থ্যাংক", "থ্যাঙ্ক", "বুঝেছি", "বুঝলাম", "বুঝে", "বেশ", "ভালো", "ভাল", "চলবে", "হয়েছে", "হয়ে",
    }
)  # fmt: skip
# Words that may stand BESIDE an acknowledgement but never make one: "thik ache", "ok haan", "achha, thank you", "hmm theek hai"
_ACK_FILLER = frozenset(
    {
        "ache", "achhe", "ase", "ashe", "hai", "hain", "hay", "he", "hey", "hae", "ha", "haa", "haan", "han", "ji", "jee", "hmm", "hm", "hmmm", "you", "it", "got", "thank", "bas", "all", "then", "so", "toh", "to",
        "tahole", "taile", "tobe", "aar", "ar", "ekdom", "bilkul", "ekdum", "very", "much", "a", "lot", "bhai", "dada", "didi", "madam", "sir", "bhaiya",
        "আছে", "আছি", "হ্যাঁ", "হ্যা", "হা", "হুম", "হুঁ", "জি", "তাহলে", "তবে", "তো", "একদম", "খুব", "দাদা", "দিদি", "স্যার", "ম্যাডাম",
        "है", "हैं", "हाँ", "हां", "हा", "जी", "हम्म", "तो", "फिर", "एकदम", "बिल्कुल", "बहुत", "भैया", "सर", "मैडम",
    }
)  # fmt: skip
_ACK_MAX_WORDS = 7


def _is_ack_word(word: str) -> bool:
    return word in _ACK_WORDS or (word.isascii() and any(p.fullmatch(word) for p in _ACK_PATTERNS))


def is_closing_acknowledgement(text: str, lang: str = "en") -> bool:
    """True when the whole sentence is just an acknowledgement ("achha thik ache", "ok", "theek hai, shukriya", "ঠিক আছে ধন্যবাদ", "अच्छा ठीक है") -- short, made of
    acknowledgement words and fillers only, with at least one real acknowledgement. Judges the WORDS only; whether nothing is open is the caller's to check."""
    words = _normalise(text).replace("-", " ").split()
    if not words or len(words) > _ACK_MAX_WORDS:
        return False
    if not all(_is_ack_word(w) or w in _ACK_FILLER for w in words):
        return False
    return any(_is_ack_word(w) for w in words)
# Round 7 (owner, 2026-10-23): "বেশ ধন্যবাদ আমি এটা জেনে নিলাম" ("fine, thanks, I have learned this") ended no call -- the words of "I have understood / I have what I needed" are added
# (jene, janlam, buje, jana, "হয়ে গেল", "पता चल गया", "जान लिया" ...), as acknowledgements or as fillers beside one. The sentence must still be made of these words ALONE.
_ACK_PATTERNS = _ACK_PATTERNS + (re.compile(r"jan(?:lam|ilam)|jen(?:e|echi)|buj(?:he|e)|jana\s*holo"),)
_ACK_WORDS = _ACK_WORDS | frozenset(
    {
        "জেনে", "জানলাম", "জেনেছি", "জানা", "বুঝে", "পেয়ে", "পেয়েছি", "শুনলাম",
        "जान", "जानकारी", "पता", "समझ", "सुन",
    }
)  # fmt: skip
_ACK_FILLER = _ACK_FILLER | frozenset(
    {
        "ami", "eta", "eita", "eitai", "sob", "sobkichu", "ekhon", "main", "maine", "mai", "ye", "yeh", "yah", "sab", "ab", "bhi", "liya", "chala", "ho", "hogaya", "nilam", "gelam", "gechhe", "holo", "hoye", "gelo",
        "আমি", "এটা", "এটাই", "এ", "সব", "সবকিছু", "এখন", "নিলাম", "নিয়েছি", "গেলাম", "গেছে", "গেল", "হলো", "হল", "হয়ে", "আরও", "অনেক",
        "मैं", "मैंने", "मैने", "मुझे", "यह", "ये", "सब", "अब", "भी", "लिया", "चल", "चला", "हो", "गई", "गयी", "चुका",
    }
)  # fmt: skip
