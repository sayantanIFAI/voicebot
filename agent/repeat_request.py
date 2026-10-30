"""The caller did not hear or follow the last sentence and wants it again: the agent says the LAST LINE exactly as it said it (owner, 2026-10-30).

This is COMPOSITIONAL, not a list of sentences. Every word of the caller's sentence is given a MEANING CLASS by its stem and the endings the language puts on it (Bengali script, romanised Bengali, Hindi, English):

    AGAIN   again / another time / once more       আবার, আরেক, একবার, ar ekbar, abar, फिर से, दोबारा, again, once more
    HEAR    hear / listen                          শুনতে, সুনতে, sunte, सुन, hear
    GRASP   understand / follow                    বুঝতে, bujhte, समझ, understand, catch
    ABLE    can / could / got it                   পাচ্ছি, পেলাম, পারছি, parchi, pelam, पा, can, could
    NOT     negation                               না, নি, na, nahi, नहीं, not, can't, didn't
    SAY     say / tell / speak                     বলুন, বলবেন, bolchen, bolun, बोलिए, बताइए, say, tell, repeat
    WHAT    what                                   কী, কি, ki, क्या, what
    SOFT    politeness, pronouns and fillers       প্লিজ, একটু, দাদা, please, sorry, ji, I, you

A sentence is a request to hear it again when EVERY word is one of these classes (an unknown word -- a name, a test, a doctor, a date -- means the caller is asking something else) and the classes MEAN it,
in any order and any number of times:
    * AGAIN ("আরেকবার বলবেন", "abar bolun", "one more time", just "আবার"), or
    * not being able to take it in: (HEAR or GRASP) with NOT ("শুনতে পেলাম না", "বুঝলাম না", "bujhte parchi na", "I did not catch that"), or
    * WHAT beside SAY ("কী বলছেন", "ki bolchen", "what did you say"), or WHAT alone as a one-word question ("কি", "ki?", "what?").
So "শুনতে পেলাম না আর একবার বলবেন", "বুঝতে পারছি না একটু আবার বলুন প্লিজ", "sorry ki bolchen sunte pelam na" and any other mix are understood without being written down.
"""

from __future__ import annotations

import re
import unicodedata

# (class, pattern over ONE normalised word)
_LEX: list[tuple[str, re.Pattern]] = []


def _add(cls: str, *patterns: str) -> None:
    for p in patterns:
        _LEX.append((cls, re.compile(p, re.I)))


_add(
    "AGAIN",
    r"আবার|আরেক|আরেকবার|আরও|আর|একবার|এক|বার|ফের|ফিরে|পুনরায়",
    r"a+r|aar|abar|aabar|abaar|arek|arekbar|aro|ek|ekbar|bar",
    r"फिर|फ़िर|दोबारा|दुबारा|बार|एक|और",
    r"again|once|more|another|twice|time|one|pardon",
)
_add(
    "HEAR",
    r"(?:শুন|সুন|শোন|সোন)\S*",
    r"(?:sun|shun|shon|son)\w*",
    r"(?:सुन|सुनाई)\S*",
    r"hear\w*|listen\w*|audible",
)
_add(
    "GRASP",
    r"(?:বুঝ|বুজ|বোঝ|বোজ|বুঁঝ)\S*",
    r"(?:bujh|buj|bhujh|bhuj|bojh|boj)\w*",
    r"(?:समझ)\S*",
    r"understand\w*|understood|catch|caught|follow\w*|get|got|clear",
)
_add(
    "ABLE",
    r"পা(?:চ্ছি|চ্ছ|চ্ছেন|চ্ছিস|ই|ইনি|ইনা|ব|বো|বেন|ও|য়)|পেলাম|পেলুম|পেয়েছি|পেলে|পারছি|পারছ|পারছেন|পারলাম|পারি|পারিনি|পারব|পারবো|পারেন",
    r"pa(?:chhi|cchi|chhen|i|ini|bo|ben|o)|pelam|pelum|peyechi|parchi|parchhi|parchen|parlam|pari|parini|parbo|paren",
    r"पा|पाया|पाई|पाए|पाता|पाती|पाते|दे|दिया|दी|दिए|रहा|रही|रहे|आया|आई|सका|सकी|सकता|सकती|सकते",
    r"can|could|able|manage\w*",
)
_add(
    "NOT",
    r"না|নি|নাই|নয়|নেই",
    r"na|nai|ni|noy|nei",
    r"नहीं|नही|ना|न",
    r"not|no|never|cannot|can't|couldn't|didn't|don't|doesn't|wasn't|isn't|unable|n't",
)
_add(
    "SAY",
    r"বল(?:ুন|েন|বেন|ো|ছেন|ছো|ছ|ছি|ছিস|লেন|লাম|লি|লে|ি|তে|তো|বি|বে|ব|ল)?|শোনা(?:ন|বেন|ও)?|রিপিট|পুনরাবৃত্তি",
    r"bol(?:un|en|ben|o|chen|cho|chis|li|le|lam|te|bi|be|bo)?|bolchhen|bolchho|bolchi|bollen|shona\w*|repeat",
    r"बोल\S*|बता\S*|कह\S*|सुना\S*|दोहरा\S*",
    r"say|said|saying|tell|told|speak|spoke|repeat\w*|talking|read",
)
_add(
    "WHAT",
    r"ক[িী]",
    r"ki|kii|kee|what|whats|what's|huh|hmm+|eh",
    r"क्या|kya",
)
_add(
    "SOFT",
    r"প্লিজ|একটু|দয়া|করে|তো|ও|ওই|হ্যাঁ|হ্যা|হাঁ|জি|আচ্ছা|দাদা|দিদি|ম্যাডাম|স্যার|ভাই|হ্যালো|সরি|দুঃখিত|মাফ|যদি|তাহলে|তা|আমি|আমার|আপনি|আপনার|আপনাকে|আমাকে|একটা|এটা|ওটা|সেটা|কথা|কথাটা|আরো|ধীরে|জোরে|স্পষ্ট|ঠিকমতো",
    r"please|pls|plz|sorry|hello|hi|ji|ha|haan|han|dada|didi|ekto|ektu|to|okay|ok|yes|sir|madam|bhai|ami|amar|apni|apnar|apnake|amake|ota|eta|seta|kotha|kothata|dhire|jore|ektu",
    r"जी|हाँ|हां|प्लीज़|प्लीज|ज़रा|जरा|सर|मैडम|भैया|सॉरी|माफ़|कृपया|मैं|मुझे|आप|आपकी|बात|धीरे|ज़ोर|जोर|से|को|का|की|के|में|है|हैं|तो|ये|यह|वो|वह|भी|ही|ना|कि",
    r"i|me|my|you|your|the|that|it|this|a|an|would|will|please|pardon|excuse|me|come|on|did|do|is|was|were|are|be|to|of|just|kindly|louder|slowly|slow|clearly|properly|sorry|um|uh",
)

_WORD = re.compile(r"[\wঀ-৿ऀ-ॿ']+", re.UNICODE)
# the words that, alone, are NOT an "again" (they are also "more", "one", "time" ...): they count only together
_PLAIN_AGAIN = re.compile(r"আবার|আরেক|আরেকবার|একবার|ফের|পুনরায়|abar|aabar|abaar|arek|arekbar|ekbar|again|once|twice|pardon|फिर|दोबारा|दुबारा|repeat\w*|another")
_FIRST_PERSON_SAY = re.compile(r"বলব|বলবো|বলি|বলছি|বললাম|বলতে|bolbo|boli|bolchi|bollam|bolte|बोलूं|बोलूँ|बताऊं|बताऊँ|कहूं|कहूँ")
_MORE = frozenset({"আর", "আরও", "ar", "aar", "aro", "more", "और"})
_ONE = frozenset({"এক", "ek", "एक", "one"})
_TIME = frozenset({"বার", "bar", "बार", "time"})


def _norm(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").lower()


def _classes(word: str) -> set[str]:
    return {cls for cls, rx in _LEX if rx.fullmatch(word)}


def meaning(text: str) -> set[str]:
    """The meaning classes of a sentence in which EVERY word has a class; the empty set when some word has none (a name, a test, a date, any other topic)."""
    words = _WORD.findall(_norm(text))
    if not words or len(words) > 12:
        return set()
    seen: set[str] = set()
    for w in words:
        c = _classes(w)
        if not c:
            return set()
        seen |= c
    ws = set(words)
    again = any(_PLAIN_AGAIN.fullmatch(w) for w in words) or (bool(ws & _MORE) and bool(ws & (_ONE | _TIME))) or (bool(ws & _ONE) and bool(ws & _TIME))
    if again:
        seen.add("AGAIN")
    else:
        seen.discard("AGAIN")
    return seen


def asks_repeat(text: str) -> bool:
    """True when the whole sentence, by what its words MEAN together, asks to hear the last line again (see the module docstring)."""
    m = meaning(text)
    if not m:
        return False
    words = _WORD.findall(_norm(text))
    if "AGAIN" in m:
        return True
    negated = "NOT" in m or any(re.search(r"(?:নি|নাই|ini|nai|नि)$", w) and len(w) > 3 for w in words)  # "না" apart, or fused into the verb ("বুঝিনি", "পাইনি")
    if ("HEAR" in m or "GRASP" in m) and negated:  # could not hear / understand
        return True
    say_words = [w for w in words if "SAY" in _classes(w)]
    if "WHAT" in m and say_words and not all(_FIRST_PERSON_SAY.fullmatch(w) for w in say_words):  # "কী বলছেন" / "ki bolchen" / "what did you say" -- not "কি বলব" (what shall I say)
        return True
    if "WHAT" in m and len(words) <= 2 and all(_classes(w) & {"WHAT", "SOFT"} for w in words):  # "ki?" / "কি" / "what?" alone
        return True
    return False
