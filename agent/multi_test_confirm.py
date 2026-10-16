"""Several tests named in one breath, some of them misheard: read them back ONCE, then give the total.

Real call 2026-10-17: "...ক্রিকাসিড ... থাইরাইড টেস্ট এই তিনটের টোটাল কত টাকা পড়বে" (uric acid, thyroid, and a CBC the recogniser lost
entirely). The catalogue found none of the names by their written form; the model also returned junk "names" (র, ওই). The caller
heard "'ক্রিকাসিড' not found, so it is not in the total" and said it did not understand.

The clinic already knows the sound-alikes (the lookup returns did_you_mean in English, Bengali and Hindi: "ক্রিকাসিড" -> Uric Acid,
"থাইরাইড টেস্ট" -> Thyroid Profile). CLAUDE.md's rule stands -- a near match SUGGESTS, it never resolves: no price is quoted for a test
the caller has not confirmed. So when every name the catalogue could not match has a sound-alike, the agent asks one question naming
all the tests it understood and prices them only after a yes (the existing PendingEntity replay: the "yes" re-runs the same total with the
confirmed, exactly-matching names). If any name has no sound-alike at all, nothing changes: the old reply stands.
"""

from __future__ import annotations

_CONFIRM = {
    "bn": "আমি বুঝেছি: {names}। এই টেস্টগুলোর টোটাল জানতে চাইছেন তো?",
    "hi": "मैं समझी: {names}। क्या आप इन टेस्ट का टोटल जानना चाहते हैं?",
    "en": "I understood: {names}. Would you like the total for these tests?",
}
_AND = {"bn": " আর ", "hi": " और ", "en": " and "}


def _spoken(lang: str, result: dict, requested: str, suggestion_index: int | None) -> str:
    """The name to SAY, in the call's language: the suggestion's own spelling, or the matched test's spoken name."""
    key = {"bn": "did_you_mean_bn", "hi": "did_you_mean_hi"}.get(lang, "did_you_mean")
    if suggestion_index is not None:
        spoken = result.get(key) or result.get("did_you_mean") or []
        return spoken[suggestion_index] if suggestion_index < len(spoken) else (result.get("did_you_mean") or [requested])[0]
    return requested


def plan(results: list[tuple[str, dict]], lang: str):
    """`results`: [(name as heard, get_test_rate result)]. Returns None (nothing to confirm) or
    (question to speak, canonical English names to price once the caller says yes)."""
    if len(results) < 2:
        return None
    canonical: list[str] = []
    spoken: list[str] = []
    unresolved = 0
    for requested, result in results:
        if result.get("found"):
            canonical.append(result.get("test_name") or requested)
            spoken.append(result.get({"bn": "test_name_bn", "hi": "test_name_hi"}.get(lang, "test_name")) or requested)
            continue
        suggestions = result.get("did_you_mean") or []
        if not suggestions or result.get("ambiguous"):
            return None  # a name with no sound-alike (or an ambiguous one): keep the ordinary "not found" reply
        unresolved += 1
        canonical.append(suggestions[0])
        spoken.append(_spoken(lang, result, requested, 0))
    if not unresolved:
        return None  # everything matched by its written form: just price it
    # a test named twice (heard twice) is asked about once
    seen: set[str] = set()
    pairs = [(c, s) for c, s in zip(canonical, spoken) if not (c in seen or seen.add(c))]
    if len(pairs) < 2:
        return None
    names = _AND.get(lang, _AND["en"]).join(s for _, s in pairs)
    return (_CONFIRM.get(lang, _CONFIRM["bn"]).format(names=names), [c for c, _ in pairs])


# ---------------------------------------------------------------------------------------------------------------------------------
# A test the recogniser lost ENTIRELY. Real call 2026-10-17: "CBC, uric acid, thyroid -- all three together" came back with two names;
# the CBC was never heard. Reading back only the two would quietly drop a third of the question. The caller said HOW MANY tests they
# meant ("all three", "তিনটের", "तीनों"), so when fewer names were understood than that, the agent says what it caught and asks for all
# the names again. Nothing is priced from a partial list.
import re
import unicodedata

_COUNT_WORDS = {
    # English
    "two": 2, "both": 2, "three": 3, "four": 4,
    # Bengali (the forms a "total of N tests" question takes)
    "দুটো": 2, "দুটি": 2, "দুটোর": 2, "দুটির": 2, "দুইটা": 2, "দুইটার": 2,
    "তিনটে": 3, "তিনটি": 3, "তিনটা": 3, "তিনটের": 3, "তিনটার": 3, "তিনটির": 3,
    "চারটে": 4, "চারটি": 4, "চারটের": 4, "চারটার": 4,
    # Hindi
    "दोनों": 2, "दो": 2, "तीनों": 3, "तीन": 3, "चारों": 4, "चार": 4,
}
_TOTAL_RE = re.compile(
    r"total|together|altogether|combined|টোটাল|মোট|একসাথে|একসঙ্গে|সব মিলিয়ে|टोटल|कुल|एक साथ|मिलाकर|सब का|सबका"
)
_WORD_RE = re.compile(r"[\w\u0900-\u097F\u0980-\u09FF]+", re.UNICODE)


def expected_count(text: str) -> int | None:
    """How many tests the caller said they meant ("all three together"), or None. Only read when a TOTAL is asked for, so
    "three days" or "two o'clock" in an unrelated sentence never counts."""
    norm = unicodedata.normalize("NFC", text or "").lower()
    if not _TOTAL_RE.search(norm):
        return None
    counts = [_COUNT_WORDS[w] for w in _WORD_RE.findall(norm) if w in _COUNT_WORDS]
    return counts[0] if counts else None


_NUMBER = {
    "bn": {2: "দুটি", 3: "তিনটি", 4: "চারটি"},
    "hi": {2: "दो", 3: "तीन", 4: "चार"},
    "en": {2: "two", 3: "three", 4: "four"},
}
_SHORTFALL = {
    "bn": "আমি শুধু {names} বুঝতে পেরেছি, কিন্তু আপনি {n}টি টেস্ট বললেন। সবগুলো টেস্টের নাম আবার একটু ধীরে বলবেন?",
    "hi": "मैं सिर्फ {names} समझ पाई, लेकिन आपने {n} टेस्ट बताए। क्या आप सभी टेस्ट के नाम एक बार फिर धीरे से बता सकते हैं?",
    "en": "I only caught {names}, but you mentioned {n} tests. Could you say all the test names again, slowly?",
}
_SHORTFALL_NONE = {
    "bn": "আপনি {n}টি টেস্ট বললেন, কিন্তু আমি নামগুলো ঠিকমতো বুঝতে পারিনি। সবগুলো টেস্টের নাম আবার একটু ধীরে বলবেন?",
    "hi": "आपने {n} टेस्ट बताए, लेकिन मैं नाम ठीक से नहीं समझ पाई। क्या आप सभी टेस्ट के नाम एक बार फिर धीरे से बता सकते हैं?",
    "en": "You mentioned {n} tests, but I could not make out the names. Could you say all of them again, slowly?",
}


def shortfall(results: list[tuple[str, dict]], expected: int | None, lang: str):
    """`results`: [(name as heard, get_test_rate result)]. A question to speak when the caller meant `expected` tests but fewer
    were understood (a name counts as understood if it matched, or has a sound-alike to suggest); otherwise None."""
    if not expected or expected < 2:
        return None
    seen: set[str] = set()
    spoken: list[str] = []
    for requested, result in results:
        if result.get("found"):
            canon = result.get("test_name") or requested
            say = result.get({"bn": "test_name_bn", "hi": "test_name_hi"}.get(lang, "test_name")) or requested
        elif (result.get("did_you_mean") or []) and not result.get("ambiguous"):
            canon = result["did_you_mean"][0]
            say = _spoken(lang, result, requested, 0)
        else:
            continue
        if canon not in seen:
            seen.add(canon)
            spoken.append(say)
    if len(spoken) >= expected:
        return None
    number = _NUMBER.get(lang, _NUMBER["en"]).get(expected, str(expected))
    lang = lang if lang in _SHORTFALL else "en"
    if not spoken:
        return _SHORTFALL_NONE[lang].format(n=number)
    return _SHORTFALL[lang].format(names=_AND[lang].join(spoken), n=number)
