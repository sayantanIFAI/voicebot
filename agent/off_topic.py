"""Talk that has nothing to do with Sonoscan (owner, 2026-10-27).

Real examples: "আচ্ছা আমার সামনে সঙ্গে তুমি ঘুরতে যাবে" (will you go out with me?), "আমার মাথায় বাজ পড়েছে কি করবো" (an idiom: "lightning has struck my head"). The agent's job is
the clinic: FAQs, lab tests, doctor booking, and anything else about Sonoscan. Anything else is politely turned down -- "I can help with questions about Sonoscan; how can I help?" -- and
after THREE such turns in a row the call is closed with a thank-you (agent/phrases.py `offtopic_redirect` / `offtopic_final`, main.py `_handle_offtopic`).

TWO LAYERS, both conservative, because a real patient turned away is far worse than a chatterer answered once too often:

  chit_chat(text)       an explicit phrase from a short list (an invitation, romance, jokes and songs, weather / sport / politics, a few idioms). A hospital cue (doctor, test, booking,
                        clinic ... agent/unrelated_caller.has_hospital_cue) always wins. A health word (pain, fever, a body part) vetoes it too, EXCEPT for the listed idioms
                        ("বাজ পড়েছে" is not a headache).
  has_health_word(text) used by main.py for the second layer (the intent model said smalltalk / unclear and nothing in the sentence points at the clinic): a sentence that names a symptom or
                        a body part is never declined, it is a patient.

REASONED, NOT MEASURED: the Bengali / Hindi lists were written by a non-native speaker and need a native reader; add the phrasings real callers use here and in tests/test_off_topic.py.
"""

from __future__ import annotations

import re
import unicodedata

# Close the call on this many off-topic turns IN A ROW (the owner's rule).
CLOSE_AT = 3

_NORM = re.compile(r"[?!.,;:()\"“”‘’|/\\।…]+")


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").lower()
    return " ".join(_NORM.sub(" ", t).split())


# an explicit chit-chat / idiom phrase (matched as a substring of the normalised sentence)
_CHIT_CHAT_NATIVE = (
    # Bengali: invitations, romance, jokes / songs / stories, weather, sport, politics, cinema, idioms
    "ঘুরতে যাবে", "ঘুরতে যাবি", "ঘুরতে যাই", "ঘুরতে যাব", "বেড়াতে যাবে", "সিনেমা দেখতে", "সিনেমায় যাবে", "ডেটে যাবে", "ডেট করবে", "আমার সাথে বেরোবে",
    "ভালোবাসি", "ভালবাসি", "প্রেম করবে", "প্রেম করবি", "বিয়ে করবে", "বিয়ে করবি", "বন্ধু হবে", "বন্ধুত্ব করবে", "গান শোনাও", "গান গাও", "গান গেয়ে", "জোক বল", "জোকস", "মজার গল্প",
    "গল্প বল", "কবিতা শোনাও", "আবহাওয়া", "বৃষ্টি হবে", "ক্রিকেট", "ফুটবল", "আইপিএল", "নির্বাচন", "ভোট", "রাজনীতি", "মোদি", "মমতা", "বাজ পড়", "বাজ পড়েছে", "আকাশ ভেঙে পড়",
    # Hindi
    "घूमने चलोगी", "घूमने चलोगे", "घूमने चलें", "घूमने चलते", "घूमने जाओगी", "घूमने जाओगे", "मूवी देखने", "फिल्म देखने", "डेट पर", "प्यार करता", "प्यार करती", "आई लव यू", "शादी करोगी",
    "शादी करोगे", "दोस्त बनोगी", "दोस्ती करोगी", "गाना सुनाओ", "गाना गाओ", "चुटकुला", "जोक सुनाओ", "कहानी सुनाओ", "कविता सुनाओ", "मौसम कैसा", "क्रिकेट", "फुटबॉल", "चुनाव", "वोट", "राजनीति",
    "बिजली गिर", "आसमान टूट", "सिर पर बिजली",
)
_CHIT_CHAT_LATIN = re.compile(
    r"\b(?:go(?:ing)? out with me|come out with me|go for a (?:walk|drive) with me|date me|marry me|love you|i love|be my (?:friend|girlfriend|boyfriend)|tell me a joke|sing (?:me )?a song|"
    r"tell me a story|sing for me|the weather|cricket|football|ipl|election|politics|movie tonight|lightning (?:struck|hit)|(?:hit|struck) by lightning|"
    r"ghurte jabe|ghurte jabo|bhalobashi|prem korbe|biye korbe|gan shonao|joke bolo|baj poreche|baj pore geche|ghoomne chalogi|ghumne chalogi|pyaar karta|pyar karta|shaadi karogi)\b",
    re.I,
)
# idioms that contain a body part: they must not be mistaken for a symptom
_IDIOM_NATIVE = ("বাজ পড়", "আকাশ ভেঙে পড়", "बिजली गिर", "आसमान टूट", "सिर पर बिजली")

# a symptom, a body part or a health word: such a sentence is a patient, never "off topic"
_HEALTH_NATIVE = (
    "ব্যথা", "ব্যাথা", "যন্ত্রণা", "জ্বর", "কাশি", "সর্দি", "বমি", "পাতলা পায়খানা", "পেট", "মাথা", "বুক", "চোখ", "কান", "নাক", "গলা", "দাঁত", "হাত", "পা", "কোমর", "পিঠ", "ঘাড়", "চোট", "আঘাত", "অ্যাক্সিডেন্ট", "এক্সিডেন্ট", "পড়ে গিয়ে", "ভেঙে", "রক্ত", "কেটে", "পুড়ে",
    "চামড়া", "ত্বক", "চুলকানি", "ঘা ", "ফোলা", "শ্বাস", "প্রেশার", "সুগার", "থাইরয়েড", "গর্ভ", "প্রেগন্যান্সি", "ডেলিভারি", "বাচ্চা", "শিশু", "অসুস্থ", "অসুখ", "দুর্বল", "মাথা ঘোরা", "রোগী", "রোগ",
    "ওজন", "ঔষধ", "ওষুধ", "চিকিৎসা", "ডায়াবেটিস", "ক্যান্সার", "হার্ট", "কিডনি", "লিভার",
    "चोट", "एक्सीडेंट", "गिर", "खून", "पैर", "पैरों", "दर्द", "बुखार", "खांसी", "खाँसी", "जुकाम", "उल्टी", "दस्त", "पेट", "सिर", "सीने", "छाती", "आँख", "आंख", "कान", "नाक", "गला", "दांत", "हाथ", "पैर", "कमर", "पीठ", "त्वचा", "खुजली", "सूजन", "सांस",
    "प्रेशर", "शुगर", "थायराइड", "गर्भ", "प्रेग्नेंसी", "डिलीवरी", "बच्चे", "बच्चा", "बीमार", "बीमारी", "कमज़ोरी", "कमजोरी", "मरीज़", "मरीज", "दवा", "इलाज", "डायबिटीज", "कैंसर", "दिल", "किडनी", "लिवर",
)
_HEALTH_LATIN = re.compile(
    r"\b(?:pain|ache|fever|cough|cold|injur\w*|hurt|accident|fracture|bleed\w*|broken|fell|leg|legs|arm|hand|foot|back|vomit\w*|diarrh\w*|stomach|head|headache|chest|eye|eyes|ear|nose|throat|tooth|teeth|skin|rash|itch\w*|swelling|breath\w*|pressure|sugar|thyroid|pregnan\w*|"
    r"delivery|baby|child|sick|ill|illness|disease|weak\w*|patient|medicine|treatment|diabet\w*|cancer|heart|kidney|liver|byatha|betha|jor|kashi|sordi|pet|matha|dard|bukhar)\b",
    re.I,
)


def has_health_word(text: str) -> bool:
    """True when the sentence names a symptom, a body part or a health matter: a patient, never declined as off-topic."""
    t = _norm(text)
    if not t:
        return False
    padded = t + " "
    return any(w in padded for w in _HEALTH_NATIVE) or _HEALTH_LATIN.search(t) is not None


def chit_chat(text: str) -> bool:
    """True when the sentence holds an explicit chit-chat / idiom phrase from the lists above. The caller (main.py) has already checked that no hospital cue is present and that nothing
    else is open; a health word vetoes this too, except inside a listed idiom."""
    t = _norm(text)
    if not t:
        return False
    hit = any(p in t for p in _CHIT_CHAT_NATIVE) or _CHIT_CHAT_LATIN.search(t) is not None
    if not hit:
        return False
    if any(i in t for i in _IDIOM_NATIVE) or "lightning" in t or "baj pore" in t:
        return True  # "মাথায় বাজ পড়েছে" is an idiom, whatever body part it names
    return not has_health_word(t)
