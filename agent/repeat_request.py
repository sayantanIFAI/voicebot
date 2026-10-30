"""The caller did not hear or follow the last sentence and asks for it again: the agent says the LAST LINE exactly as it said it (owner, 2026-10-30).

"সুনতে পেলাম না" / "bujhte parchi na" / "ki bolchen?" / "ar ekbar bolben please" / "फिर से बोलिए" / "say that again". Pure words: the whole sentence must be such a request (a few small filler words allowed) -- a
sentence that also asks something else ("আরেকবার বলুন, ডাক্তার সেন কবে বসেন") is a question and goes on the normal path.
"""

from __future__ import annotations

import re
import unicodedata

# the request itself (Bengali script, romanised Bengali, Hindi, English)
_CORE = [
    # Bengali script
    r"(?:আ?রেক|আর\s*এক|আবার\s*এক|আবার)\s*বার\s*(?:একবার\s*)?(?:বল(?:ুন|েন|বেন|ো|ুন\s*তো|বি)?|শোনা(?:ন|বেন|ও)?)",
    r"আবার\s*(?:বল(?:ুন|েন|বেন|ো)?|শোনান|শোনাবেন)",
    r"আরেকবার\s*(?:বল(?:ুন|েন|বেন|ো)?|শোনান|শোনাবেন)",
    r"আর\s*একবার\s*(?:বল(?:ুন|েন|বেন|ো)?|শোনান|শোনাবেন)",
    r"(?:আবার|আরেকবার|আর\s*একবার)\s*এক\s*বার\s*বল(?:ুন|েন|বেন|ো)?",
    r"(?:শুনতে|সুনতে|সুন্তে|শুন্তে)\s*(?:পাচ্ছি|পারছি|পেলাম|পাইনি|পারলাম|পাই|পাচ্ছি)\s*না",
    r"শুনতে\s*পাইনি",
    r"বুঝতে\s*(?:পারছি|পেলাম|পারলাম|পাচ্ছি|পারিনি|পাইনি)\s*(?:না)?",
    r"বুঝলাম\s*না",
    r"বুঝিনি",
    r"ক[িী]\s*(?:বলছেন|বললেন|বলছিস|বললি|বলছো|বললে|বলছ)(?:\s*(?:শুনতে|সুনতে)\s*(?:পাচ্ছি|পারছি|পেলাম)\s*না)?",
    r"ক[িী]",
    r"আবার\s*বলুন\s*তো",
    # romanised Bengali
    r"(?:ar|aar|abar|arek|are|aro)\s*(?:ek\s*)?bar\s*(?:ek\s*bar\s*)?(?:bolun|bolen|bolben|bolo|bol|bolte\s*paren|bolbe)",
    r"(?:abar|arekbar|arek\s*bar|ar\s*ekbar|ar\s*ek\s*bar)\s*(?:ekbar\s*)?(?:bolun|bolen|bolben|bolo|bol)",
    r"(?:sunte|shunte|suntey)\s*(?:pelam|parchi|pachhi|pacchi|parlam|pai|parchhi)\s*na",
    r"(?:bujhte|bujte|bhujte|bujhte)\s*(?:pelam|parchi|parlam|pachhi|pacchi|parchhi)\s*na",
    r"(?:bujhlam|bujlam|bhujlam|bujhi)\s*na",
    r"ki\s*(?:bolchen|bollen|bolchis|bolli|bolle|bolchho|bolcho|bolchhen)(?:\s*(?:sunte|shunte)\s*(?:parchi|pelam|pachhi)\s*na)?",
    r"ki",
    # Hindi
    r"(?:फिर\s*से|दोबारा|एक\s*बार\s*और|फिर\s*एक\s*बार|दुबारा)\s*(?:एक\s*बार\s*)?(?:बोलिए|बोलिये|बताइए|बताइये|बोलो|बताओ|कहिए|कहिये|बोलें|बताएं|बताएँ|सुनाइए)",
    r"क्या\s*(?:बोले|बोला|कहा|बोल\s*रहे\s*हैं|कह\s*रहे\s*हैं|बोल\s*रहे\s*हो)",
    r"सुनाई\s*नहीं\s*(?:दे\s*रहा|दिया|पड़ा|पड़\s*रहा)",
    r"(?:सुन|समझ)\s*नहीं\s*(?:आया|आ\s*रहा|पाया|पा\s*रहा|पाई)",
    r"समझ\s*नहीं\s*आया",
    r"phir\s*se\s*(?:bolo|boliye|bataiye|batao)",
    r"kya\s*(?:bola|kaha|bole)",
    # English
    r"(?:say|tell|repeat)\s*(?:that|it|this)?\s*(?:once\s*)?again",
    r"(?:can|could|would|will)\s*you\s*(?:please\s*)?(?:say|repeat|tell)\s*(?:that|it|this|me)?\s*(?:once\s*)?(?:again|more)?",
    r"(?:please\s*)?repeat(?:\s*(?:that|it|please))?",
    r"(?:i\s*)?(?:did\s*not|didn't|can't|cannot|couldn't|could\s*not)\s*(?:hear|understand|catch)(?:\s*(?:you|that|it))?",
    r"pardon(?:\s*me)?",
    r"(?:what|sorry)\s*(?:did\s*you\s*say)?",
    r"once\s*more",
]
# small words that may stand before / after the request
_FILL = (
    "হ্যাঁ", "হ্যা", "হাঁ", "জি", "আচ্ছা", "ও", "ওই", "দাদা", "দিদি", "ম্যাডাম", "স্যার", "প্লিজ", "একটু", "দয়া", "করে", "ভাই", "হ্যালো", "সরি", "দুঃখিত",
    "please", "pls", "plz", "sorry", "hello", "hi", "yes", "ok", "okay", "sir", "madam", "ji", "ha", "haan", "han", "dada", "didi", "ekto", "ektu",
    "जी", "हाँ", "हां", "प्लीज़", "प्लीज", "ज़रा", "जरा", "सर", "मैडम", "भैया",
)
_ONE = "(?:" + "|".join(f"(?:{c})" for c in _CORE) + ")"
# Owner, 2026-10-30 ("শুনতে পেলাম না আর একবার বলবেন"): callers join two requests in one breath -- up to three of them in a row are still one request to hear it again
_SEP = r"(?:\s+(?:একটু|প্লিজ|দয়া|করে|তো|ও|আর|ektu|ekto|please|plz|pls|to|ज़रा|जरा|प्लीज़))*\s+"
_CORE_RE = re.compile(_ONE + "(?:" + _SEP + _ONE + "){0,2}", re.I)


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").lower()
    return " ".join(re.sub(r"[.,;:!?\"()।॥\-]+", " ", t).split())


def asks_repeat(text: str) -> bool:
    """True when the whole sentence is a request to hear the last line again (a few filler words aside)."""
    t = _norm(text)
    if not t or len(t.split()) > 9:
        return False
    # peel filler words off both ends; what is left must be exactly one request
    words = t.split()
    while words and words[0] in _FILL:
        words.pop(0)
    while words and words[-1] in _FILL:
        words.pop()
    core = " ".join(words)
    return bool(core) and _CORE_RE.fullmatch(core) is not None
