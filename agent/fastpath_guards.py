"""Two cheap refusals for the quick-answer shortcut (agent/fast_path.py), found in real calls of 2026-10-17.

The fast path answers a routine question from the clinic's own table with no model. It commits on character similarity to a stored
cue, which is a good test for WHICH test or doctor a caller named but a poor one for WHAT they asked: the sentence frames of different
questions look alike. Two live failures:

  * "what are your timings" was answered with the clinic's ADDRESS (0.77): the frame "what are your ..." is close to the stored
    location cue "where are you". Nothing in the sentence is about a place.
  * "tell me the CBC, uric acid and thyroid test, the price of all three together" was matched to the CBC alone (1.0): the other two
    tests were spelled by the recogniser in ways the catalogue does not hold, so only one test was "named" -- and a single price was
    spoken for a question about a total.

A wrong confident answer from this path is worse than no answer (the turn then goes to the model, which has the context). So the
shortcut is vetoed when its answer cannot be right for the sentence. Both rules only ever REMOVE an answer; they never add one.
"""

from __future__ import annotations

import re
import unicodedata

_PUNCT = re.compile(r"[.,;:!?\"'()\[\]{}।॥\-–—_/]+")

# Words that make a sentence a question about WHERE the clinic is / how to reach it.
_LOCATION_WORDS = (
    # English
    "where", "address", "location", "located", "situated", "direction", "directions", "route", "reach", "get there",
    "come there", "how to come", "how do i come", "landmark", "map",
    # Bengali
    "কোথায়", "কোথায", "ঠিকানা", "লোকেশন", "অবস্থান", "কোন জায়গা", "কীভাবে আসব", "কিভাবে আসব", "কিভাবে যাব", "কীভাবে যাব", "রাস্তা", "পথ", "পৌঁছ", "ম্যাপ",
    # Bengali: "address" (the English loanword), in the spellings a recogniser actually writes -- real call 2026-10-24:
    # "আপনাদের ওখানকার অযাড্ড্রেস টা কি আছে" was not read as a location question at all ("address" in Latin script was in
    # this list, but the caller said it in Bengali script, and not even in the standard spelling).
    "অ্যাড্রেস", "অযাড্ড্রেস", "অ্যাডড্রেস", "আড্রেস", "এড্রেস", "আদ্রেস",
    # Hindi
    "कहाँ", "कहां", "पता", "ठिकाना", "ठिकाने", "लोकेशन", "कैसे आ", "कैसे जा", "कैसे पहुँच", "कैसे पहुंच", "रास्ता", "पहुँच", "पहुंच", "मैप", "जगह",
    # Hindi: "address" in Devanagari spellings
    "एड्रेस", "अड्रेस", "ऐड्रेस",
)

# Words that say the caller wants ONE figure for SEVERAL things.
_TOTAL_WORDS = (
    "total", "together", "altogether", "all three", "all of them", "all of these", "both of", "combined", "in all", "sum of",
    "টোটাল", "মোট", "সব মিলিয়ে", "সবমিলিয়ে", "একসাথে", "একসঙ্গে", "তিনটের", "তিনটার", "তিনটির", "দুটোর", "দুটির", "সবগুলোর", "সবকটার",
    "टोटल", "कुल", "एक साथ", "सबका", "सब का", "तीनों", "दोनों", "सब मिलाकर", "मिलाकर", "सभी का",
)


def _norm(text: str) -> str:
    s = unicodedata.normalize("NFC", text or "").lower().replace("़", "").replace("়", "")
    return " ".join(_PUNCT.sub(" ", s).split())


def _has_any(norm_text: str, cues: tuple[str, ...]) -> bool:
    padded = f" {norm_text} "
    for cue in cues:
        c = _norm(cue)
        if not c:
            continue
        if c.isascii():
            if f" {c} " in padded or (len(c) > 4 and c in norm_text):
                return True
        elif c in norm_text:
            return True
    return False


def location_word_present(text: str) -> bool:
    return _has_any(_norm(text), _LOCATION_WORDS)


def asks_for_total(text: str) -> bool:
    return _has_any(_norm(text), _TOTAL_WORDS)


# Owner, 2026-10-27 (real call): "যাও তো যাও আকাশে কিভাবে উঠবো আমি কিভাবে উঠবো আকাশে" ("go, go, how do I climb into the sky") was answered with the PAYMENT methods (0.75, just over the 0.72 floor): the
# stored cue "টাকা কীভাবে দেব" shares the word "how" with it. The location topic already needed a word about a place; EVERY topic now needs at least one word that belongs to it. A sentence with
# none goes to the model, which calls it small talk -- and agent/off_topic.py turns that down. Substring stems (3+ letters) and exact tokens (short words) are kept apart so "পে" does not match "পেট".
_FAQ_EVIDENCE: dict[str, tuple[tuple[str, ...], frozenset[str]]] = {
    "hours": (
        ("সময়", "খোলা", "খোলে", "খুলব", "খুলে", "খুলবে", "বন্ধ", "কখন", "কটা", "কটায়", "ঘণ্টা", "ঘন্টা", "সকাল", "রাত", "রবিবার", "ছুটি", "টাইম", "ভিজিটিং", "আওয়ার", "সন্ধ্যা", "বিকেল",
         "समय", "खुल", "बंद", "बजे", "टाइम", "घंटे", "रविवार", "छुट्टी", "सुबह", "शाम", "timing", "hours", "open", "close", "sunday", "holiday", "morning", "evening", "weekday"),
        frozenset({"কব", "কবে", "कब", "time", "when", "till", "until"}),
    ),
    "payment_methods": (
        ("টাকা", "পেমেন্ট", "কার্ড", "ইউপিআই", "নগদ", "ক্যাশ", "অনলাইন", "গুগল", "ফোনপে", "ফোন পে", "পেটিএম", "ডেবিট", "ক্রেডিট", "রসিদ", "চার্জ", "ফিস", "ভিজিট",
         "पेमेंट", "पैसे", "पैसा", "कार्ड", "यूपीआई", "नकद", "कैश", "ऑनलाइन", "गूगल", "फोन पे", "फोनपे", "पेटीएम", "रुपये", "फीस", "payment", "card", "cash", "upi", "online", "gpay", "paytm", "phonepe",
         "money", "charge", "rupee", "google pay"),
        # "কাশ" (the recogniser's spelling of ক্যাশ) is an exact word only: as a stem it sat inside "কাশি" (cough) and a patient listing "জ্বর সর্দি কাশি" was answered with the payment methods (real call 2026-10-28)
        frozenset({"পে", "পাই", "পেয়", "पे", "पाय", "pay", "fee", "fees", "gpay", "g", "কাশ"}),
    ),
    "insurance": (
        ("ইনসিওরেন্স", "ইন্সুরেন্স", "ইনস্যুরেন্স", "বিমা", "ক্যাশলেস", "মেডিক্লেম", "বীমা", "बीमा", "इंसोरेंस", "इंश्योरेंस", "इन्श्योरेंस", "कैशलेस", "मेडिक्लेम", "insur", "cashless", "mediclaim", "policy", "tpa"),
        frozenset(),
    ),
    "parking": (
        ("পার্কিং", "গাড়ি", "গাড়ী", "বাইক", "স্কুটার", "সাইকেল", "पार्किंग", "गाड़ी", "गाडी", "बाइक", "स्कूटर", "park", "vehicle", "scooter", "bike"),
        frozenset({"car", "cars"}),
    ),
    "report_collection": (
        ("রিপোর্ট", "রেজাল্ট", "রিসুল্ট", "ফলাফল", "কালেক্ট", "সংগ্রহ", "রিপোর্টে", "रिपोर्ट", "रिज़ल्ट", "रिजल्ट", "कलेक्ट", "report", "result", "collect"),
        frozenset(),
    ),
    "contact_number": (
        ("নম্বর", "নাম্বার", "নম্বার", "ফোন", "মোবাইল", "হেল্পলাইন", "যোগাযোগ", "হোয়াটসঅ্যাপ", "नंबर", "नम्बर", "फोन", "मोबाइल", "हेल्पलाइन", "संपर्क", "व्हाट्सऐप", "व्हाट्सएप", "number", "phone", "contact", "helpline",
         "whatsapp", "mobile"),
        frozenset({"call", "কল", "कॉल"}),
    ),
    "home_collection": (
        ("বাড়ি", "বাড়ী", "বাসা", "ঘরে", "হোম", "কালেকশন", "সংগ্রহ", "স্যাম্পল", "নমুনা", "ব্লাড", "घर", "होम", "कलेक्शन", "सैंपल", "सैम्पल", "ब्लड", "home", "collection", "collect", "sample", "house"),
        frozenset(),
    ),
}


def faq_topic_word_present(topic: str, text: str) -> bool:
    """True when the sentence holds a word that belongs to the FAQ topic (or the topic is not one this table knows)."""
    rule = _FAQ_EVIDENCE.get(topic)
    if rule is None:
        return True
    stems, exact = rule
    n = _norm(text)
    if any(_norm(s) in n for s in stems):
        return True
    return any(tok in exact for tok in n.split())


_DOCTOR_WORDS = ("ডাক্তার", "ডক্টর", "ডাক্তর", "डॉक्टर", "डाक्टर", "doctor", "বিভাগ", "विभाग", "জেনারেল মেডিসিন", "जनरल मेडिसिन")


def _seeks_care(text: str) -> bool:
    from agent import off_topic

    n = _norm(text)
    return any(w in n for w in _DOCTOR_WORDS) and off_topic.has_health_word(text)


def veto(intent: str, slots: dict | None, text: str, tests_named) -> str | None:
    """A reason to refuse the shortcut's answer, or None to let it stand. `tests_named` is a zero-argument callable returning how many
    catalogue tests the sentence names (only evaluated when a total is asked for)."""
    slots = slots or {}
    if intent == "clinic_faq" and _seeks_care(text):
        # a sentence that lists symptoms AND asks for a doctor / department is a patient looking for care, never a clinic FAQ whose keyword happens to appear in it
        return "a symptom with a request for a doctor is not an FAQ"
    if intent == "clinic_faq" and slots.get("faq_topic") == "location" and not location_word_present(text):
        return "location answer without any word about a place"
    if intent == "clinic_faq" and slots.get("faq_topic") not in (None, "location") and not faq_topic_word_present(str(slots.get("faq_topic")), text):
        return f"{slots.get('faq_topic')} answer without any word about that topic"
    if intent == "test_rate" and asks_for_total(text):
        try:
            named = int(tests_named())
        except Exception:  # noqa: BLE001 - if the count cannot be made, the safe side is the model
            named = 0
        if named < 2:
            return "a total was asked for but fewer than two tests are named"
    return None
