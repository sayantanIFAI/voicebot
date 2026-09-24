"""Spotting a question about the caller's OWN history, deterministically.

"When did I last do my CBC?" / "what tests have I had?" / "আমার শেষ টেস্ট কবে হয়েছিল" /
"मेरा आख़िरी टेस्ट कब हुआ था". These are questions about a patient's record, so they go
through the verification gate (agent/patient_context.history_gate) and are answered only
from retrieved fields -- and that decision must not depend on a language model deciding
what kind of question it heard. So the detection is a fixed set of cues, checked before
any model call, in the same spirit as agent/human_request.py.

Two kinds are recognised:
  last_test     "when did I last have <test>" -- the test is looked up in the catalogue
                the orchestrator already holds; if it cannot be pinned to one test the
                agent ASKS which (never guesses);
  recent_tests  "what tests have I had", "my test history".

Upcoming appointments ("when is my next appointment") are NOT here: that is the existing
booking-lookup path, which has its own phone-based authorisation (booking_service).

Cues were written by a non-native speaker and are provisional (agent/persona.py).
"""
from __future__ import annotations

import dataclasses
import re
import unicodedata


@dataclasses.dataclass
class HistoryQuestion:
    kind: str                       # "last_test" | "recent_tests"
    test_hint: str | None = None    # the words the caller used for the test, if any


_LAST = {
    "en": re.compile(r"\b(when (did|was)( i| my)?.{0,30}\b(last|previous)|(my )?(last|previous|most recent) (time )?(i )?(had|did|took|got)|"
                     r"when (did|was) i (last )?(have|do|done|take|took|get|got)|"
                     r"my (last|previous|most recent)( [\w-]+){0,2} (test|check|checkup|report))\b", re.I),
    "bn": re.compile(r"(আমার|আমি).{0,25}(শেষ(বার)?|আগে(র)?|গতবার).{0,25}(টেস্ট|পরীক্ষা|করিয়েছিলাম|করেছিলাম)|"
                     r"(শেষ(বার)?|গতবার).{0,20}(কবে|কখন).{0,25}(টেস্ট|পরীক্ষা|করিয়েছিলাম)"),
    "hi": re.compile(r"(मेरा|मैंने|मैं).{0,25}(आख़िरी|आखिरी|पिछला|पिछली बार|पहले).{0,25}(टेस्ट|जाँच|कराया|करवाया)|"
                     r"(आख़िरी|आखिरी|पिछली बार).{0,20}(कब).{0,25}(टेस्ट|जाँच|कराया|करवाया)"),
}
_RECENT = {
    "en": re.compile(r"\b(what|which) tests? (have|did) i (had|have had|done|taken|take|do)|my (test|lab|medical) (history|records?)|"
                     r"tests? i('?ve| have) (had|done|taken)|my previous tests?\b", re.I),
    "bn": re.compile(r"(আমার|আমি).{0,15}(কী কী|কোন কোন).{0,10}(টেস্ট|পরীক্ষা)|আমার (টেস্টের|পরীক্ষার) (ইতিহাস|রেকর্ড)|আমার আগের টেস্ট"),
    "hi": re.compile(r"(मेरे|मैंने).{0,15}(कौन कौन से|कौन से).{0,10}(टेस्ट|जाँच)|मेरा टेस्ट (इतिहास|रिकॉर्ड)|मेरे पिछले टेस्ट"),
}
# words that mean "test" itself and so are not a hint about WHICH one
_GENERIC = re.compile(r"\b(test|tests|check|checkup|blood|report)\b|টেস্ট|পরীক্ষা|टेस्ट|जाँच", re.I)


def _nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t or "")


def detect_history_question(text: str, lang: str = "en") -> HistoryQuestion | None:
    t = _nfc(text).strip()
    if not t:
        return None
    for lg in dict.fromkeys((lang, "en", "bn", "hi")):
        if _RECENT[lg].search(t):
            return HistoryQuestion("recent_tests")
    for lg in dict.fromkeys((lang, "en", "bn", "hi")):
        if _LAST[lg].search(t):
            return HistoryQuestion("last_test", test_hint=t)
    return None
