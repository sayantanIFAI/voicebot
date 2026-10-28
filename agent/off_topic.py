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

# Close the call on this many off-topic turns IN A ROW (the owner's rule; 2026-10-27: two, not three).
CLOSE_AT = 2

_NORM = re.compile(r"[?!.,;:()\"“”‘’|/\\।…]+")


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").lower()
    return " ".join(_NORM.sub(" ", t).split())


# an explicit chit-chat / idiom phrase (matched as a substring of the normalised sentence)
_CHIT_CHAT_NATIVE = (
    # Bengali: invitations, romance, jokes / songs / stories, weather, sport, politics, cinema, idioms
    "ওয়েদার", "উইআথার", "উয়েদার", "উইদার", "ওয়েদার", "গরম পড়েছে", "ঠান্ডা পড়েছে", "বৃষ্টি পড়ছে", "बारिश होगी", "बारिश हो रही", "मौसम", "वेदर", "वेथर", "वेदार", "वेदर", "मौसम कैसा",
    "ঘুরতে যাবে", "ঘুরতে যাবি", "ঘুরতে যাই", "ঘুরতে যাব", "বেড়াতে যাবে", "সিনেমা দেখতে", "সিনেমায় যাবে", "ডেটে যাবে", "ডেট করবে", "আমার সাথে বেরোবে",
    "ভালোবাসি", "ভালবাসি", "প্রেম করবে", "প্রেম করবি", "বিয়ে করবে", "বিয়ে করবি", "বন্ধু হবে", "বন্ধুত্ব করবে", "গান শোনাও", "গান গাও", "গান গেয়ে", "জোক বল", "জোকস", "মজার গল্প",
    "গল্প বল", "কবিতা শোনাও", "আবহাওয়া", "বৃষ্টি হবে", "ক্রিকেট", "ফুটবল", "আইপিএল", "নির্বাচন", "ভোট", "রাজনীতি", "মোদি", "মমতা", "বাজ পড়", "বাজ পড়েছে", "আকাশ ভেঙে পড়",
    # Hindi
    "घूमने चलोगी", "घूमने चलोगे", "घूमने चलें", "घूमने चलते", "घूमने जाओगी", "घूमने जाओगे", "मूवी देखने", "फिल्म देखने", "डेट पर", "प्यार करता", "प्यार करती", "आई लव यू", "शादी करोगी",
    "शादी करोगे", "दोस्त बनोगी", "दोस्ती करोगी", "गाना सुनाओ", "गाना गाओ", "चुटकुला", "जोक सुनाओ", "कहानी सुनाओ", "कविता सुनाओ", "मौसम कैसा", "क्रिकेट", "फुटबॉल", "चुनाव", "वोट", "राजनीति",
    "बिजली गिर", "आसमान टूट", "सिर पर बिजली",
    # politics and public figures, Hindi / English / Latin Bengali (a real call: "मोदी की भावे लगा है मोदी मोदी")
    "मोदी", "ममता", "राहुल गांधी", "केजरीवाल", "भाजपा", "कांग्रेस", "सरकार बदल", "modi", "mamata", "rahul gandhi", "bjp", "congress party", "trinamool",
)
_CHIT_CHAT_LATIN = re.compile(
    r"\b(?:go(?:ing)? out with me|come out with me|go for a (?:walk|drive) with me|date me|marry me|love you|i love|be my (?:friend|girlfriend|boyfriend)|tell me a joke|sing (?:me )?a song|"
    r"tell me a story|sing for me|the weather|weather today|how(?:'s| is) the weather|cricket|football|ipl|election|politics|movie tonight|lightning (?:struck|hit)|(?:hit|struck) by lightning|"
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


# Endings a body-part / symptom word takes in speech ("নাকে", "পায়ের", "মাথায়", "পেটটা", "চোখগুলো").
_WORD_ENDINGS = frozenset(("", "ে", "ের", "য়", "য়ে", "য়ের", "তে", "েতে", "টা", "টি", "র", "ও", "কে", "গুলো", "েরও", "েই", "ই"))


def has_health_word(text: str) -> bool:
    """True when the sentence names a symptom, a body part or a health matter: a patient, never declined as off-topic.

    Owner, 2026-10-28: "আমি আপনাকে ভালোবাসি" ("I love you") was never recognised as small talk because "আপনাকে" CONTAINS "নাক" (nose) and "পা" (foot) hides inside পাব, পারি, পাচ্ছি ... Words are
    now matched as WORDS: the whole word, a long word's start, or a short word with an ending a Bengali body word really takes -- never a piece in the middle of another word."""
    t = _norm(text)
    if not t:
        return False
    tokens = t.split()
    for w in _HEALTH_NATIVE:
        w = w.strip()
        if " " in w:
            if w in t:
                return True
            continue
        for tok in tokens:
            if tok == w or (tok.startswith(w) and (len(w) >= 4 or tok[len(w):] in _WORD_ENDINGS)):
                return True
    return _HEALTH_LATIN.search(t) is not None


_HOW_ARE_YOU = re.compile(
    r"(?:আপনি|তুমি|আপনারা)?\s*(?:কেমন আছ(?:েন|ো|)|ভালো আছ(?:েন|ো|)|ভাল আছ(?:েন|ো|))|आप कैसे हैं|आप कैसी हैं|कैसे हो|कैसी हो|\bhow are you\b|\bhow do you do\b",
    re.I,
)


def how_are_you(text: str) -> bool:
    """A polite "how are you?" ("আপনি ভালো আছেন?", "কেমন আছেন", "आप कैसे हैं"): answered kindly once with a fixed line, never a strike."""
    t = _norm(text)
    return bool(t) and len(t.split()) <= 6 and _HOW_ARE_YOU.search(t) is not None


# "Is this a girls' school or a police station?" / "I called the police station": the caller dialled the wrong place. Said straight away, then the call is closed (no second chance asked).
_WRONG_PLACE_NATIVE = re.compile(
    r"(?:এটা|ইটা|এইটা|এটাই|ওটা)\s*কি\s+(?:[^\s]+\s+){0,3}?(?:স্কুল|ইস্কুল|গির্ল্স|বয়েজ|পুলিশ|থানা|স্তেতিওন|স্টেশন|স্ট্যান্ড|স্টপ|দোকান|হোটেল|ব্যাংক|ব্যাঙ্ক|অফিস|বাজার|মন্দির|মসজিদ|কলেজ)"
    r"|(?:এটা|ইটা|এইটা|এটাই|ওটা)\s*কি\s+(?:[^\s]+\s+){1,4}?(?:এর|র|-এর)?\s*(?:নম্বর|নম্বার|নাম্বার|নাম্বর)\s*(?:নাকি|না|কি)"
    r"|(?:পুলিশ\s*(?:স্টেশন|স্তেতিওন|স্টেশনে)|থানা|থানায়)\s*(?:এ|এই)?\s*ফোন\s*(?:করেছি|করেছিলাম|লাগিয়েছি|লেগেছে)"
    r"|(?:यह|ये|क्या\s+यह)\s*(?:क्या\s+)?(?:कोई\s+)?(?:[^\s]+\s+)?(?:स्कूल|पुलिस\s*स्टेशन|थाना|बैंक|दुकान|होटल|ऑफिस|बाज़ार|बाजार|बस\s*स्टैंड|कॉलेज)\s*(?:है|हैं)"
    r"|(?:यह|ये)\s+(?:क्या\s+)?(?:[^\s]+\s+){1,3}?(?:का|की)\s*नंबर\s*(?:है|हैं)?\s*(?:क्या|कि)"
    r"|(?:पुलिस\s*स्टेशन|थाने)\s*(?:में|पर)?\s*फ[ोो]न\s*(?:किया|लगाया|लग)"
)
_WRONG_PLACE_LATIN = re.compile(
    r"\b(?:is this (?:a |the )?(?:girls'? |boys'? )?(?:school|police station|bank|shop|hotel|office|market|temple)|i (?:called|dialled|dialed) (?:the )?(?:police station|school|bank|shop|hotel)|"
    r"(?:ei ta ki|eta ki|ye kya|yeh kya) (?:girls |boys )?(?:school|police station|thana)|thana e phone)\b",
    re.I,
)


_OUR_NAME = ("সোনো", "সনো", "বাণী", "सोनो", "वाणी", "sonoscan", "sono scan", "sonoscan vaani", "vaani", "bani")


def wrong_place(text: str) -> bool:
    """True when the caller is plainly asking whether this is some OTHER place (a school, a police station, a bank ...) or says they called one. A hospital cue elsewhere in the sentence
    (agent/unrelated_caller.has_hospital_cue, checked by the caller of this function) always wins; the words 'hospital', 'clinic', 'diagnostic' are not in the list on purpose."""
    t = _norm(text)
    if not t or any(b in t for b in _OUR_NAME):
        return False  # "is this Sonoscan's number?" is a question about US
    return _WRONG_PLACE_NATIVE.search(t) is not None or _WRONG_PLACE_LATIN.search(t) is not None


# Insults at the clinic or at the agent ("ফালতু আপনারা", "তুই কিচ্ছু করতে পারবি না", "बकवास", "useless"): told once, and the next non-clinic turn ends the call. Swearing itself is agent/abuse.py's; these
# are the milder insults it does not list. A REQUEST in the same sentence (book, price, report ...) means a real, if angry, caller: not counted.
_INSULT = (
    "ফালতু", "ফালদু", "ফাল দু", "ফাউল্টু", "বাজে", "জঘন্য", "বোকা", "পাগল", "রদ্দি", "নিকৃষ্ট", "পারবি না", "পারবে না তুমি", "তুই কিচ্ছু", "তুমি কিচ্ছু", "কিচ্ছু করতে পারবি", "চোর", "ঠগ",
    "फालतू", "बकवास", "बेकार", "घटिया", "बेवकूफ", "पागल", "रद्दी", "तू कुछ नहीं कर", "तुम कुछ नहीं कर",
    "useless", "stupid", "idiot", "nonsense", "rubbish", "you can't do anything", "you cannot do anything", "worst service", "bakwas", "faltu", "bekar",
)
_REQUEST = (
    "বুক", "অ্যাপয়েন্টমেন্ট", "রেট", "দাম", "কত টাকা", "টেস্ট করাত", "রিপোর্ট কবে", "বুকিং", "বুक", "अपॉइंटमेंट", "रेट", "दाम", "कितने रुपये", "बुकिंग", "book", "appointment", "price", "rate", "booking",
)


def hostile(text: str) -> bool:
    """True when the sentence insults (see _INSULT) and asks for nothing the clinic does."""
    t = _norm(text)
    return bool(t) and any(w in t for w in _INSULT) and not any(r in t for r in _REQUEST)


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
