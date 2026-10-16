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
