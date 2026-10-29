"""KCD-353: "a request for a person is honoured immediately."

A caller who asks for a person must reach one, without the agent deciding they do
not really mean it, asking them to repeat it, or routing the request through a
model. So the detection is deterministic and runs BEFORE any language model call
and before intent extraction: a fixed set of phrases in each language, checked
against the transcript, with a negation guard ("I don't need a person").

It is deliberately generous. The two errors are not symmetric: a caller who says
"can I get someone to help" and is offered a person they did not quite want loses
a few seconds; a caller who asked for a person and got another round of the bot
has been refused. When in doubt this returns True. "Nurse", "doctor" and
"receptionist" as bare words do NOT count -- "book me with the doctor" is not a
request for a person -- only a request to SPEAK to someone does.

Phrase lists are provisional and were written by a non-native speaker
(agent/persona.py REVIEW_STATUS): expect a native reviewer to add more.
"""

from __future__ import annotations

import re
import unicodedata

# ---- English -------------------------------------------------------------
_EN_PERSON = r"(a |an |the |some |your |any )?(real |live |actual )?(person|human|man|woman|someone|somebody|agent|operator|representative|receptionist|staff|staff member|executive|manager|counter)"
# "I want a person" needs a narrower list than "let me talk to <X>": "I need a test at the
# counter" and "I want to know about your staff" are not requests to speak to anyone.
_EN_PERSON_ONLY = (
    r"(a |an |the |some )?(real |live |actual )?(person|human|someone|somebody|operator|representative|receptionist)"
)
_EN = [
    re.compile(
        rf"\b(talk|speak|connect|transfer|put me|get me|give me|pass me)\b.{{0,25}}\b(to|with|through to)\b\s*{_EN_PERSON}\b",
        re.I,
    ),
    re.compile(
        rf"\b(i (want|need|would like|wish)|can i (get|have)|let me (talk|speak)).{{0,25}}{_EN_PERSON_ONLY}\b", re.I
    ),
    re.compile(
        r"\b(human being|real person|live agent|customer (care|service|support)|call ?center|call centre)\b", re.I
    ),
    re.compile(r"\bnot (a )?(robot|bot|machine)\b", re.I),
]
# ---- Bengali -------------------------------------------------------------
_BN = [
    re.compile(
        r"(মানুষের|লোকের|কারও|কারো|কাউকে|কোনো একজনের|স্টাফের|কর্মীর|অপারেটরের|রিসেপশনের|কাউন্টারের|ম্যানেজারের)\s*(সাথে|সঙ্গে|সাথে|কাছে)?\s*(কথা|কথা বলতে|কথা বলব|কথা বলতে চাই|কথা বলি)"
    ),
    re.compile(r"(কাউকে|কোনো মানুষকে|স্টাফকে|অপারেটরকে|রিসেপশনকে)\s*(দিন|ধরিয়ে দিন|লাইনে দিন|দেবেন|ধরিয়ে দেবেন)"),
    re.compile(r"(আসল|সত্যিকারের|জ্যান্ত)\s*(মানুষ|লোক)"),
    re.compile(r"(অপারেটর|রিসেপশনিস্ট|কাস্টমার কেয়ার)"),
    re.compile(r"(যন্ত্র|মেশিন|রোবট)\s*(না|নয়|চাই না)"),
]
# ---- Hindi ---------------------------------------------------------------
_HI = [
    re.compile(r"(किसी|एक)?\s*(इंसान|इन्सान|आदमी|व्यक्ति|स्टाफ|ऑपरेटर|रिसेप्शन|मैनेजर|काउंटर)\s*(से|को)\s*(बात|बोल|मिला|जोड़)"),
    re.compile(r"(असली|सचमुच के)\s*(इंसान|इन्सान|आदमी|व्यक्ति)"),
    re.compile(r"(किसी को)\s*(दीजिए|दीजिये|जोड़िए|लाइन पर)"),
    re.compile(r"(ऑपरेटर|रिसेप्शनिस्ट|कस्टमर केयर)"),
    re.compile(r"(मशीन|रोबोट|बॉट)\s*(नहीं|से नहीं)"),
]

_NEGATED_EN = re.compile(
    r"\b(don'?t|do not|no need|not need|never|without)\b.{0,20}(talk|speak|person|human|agent|staff|operator)", re.I
)
_NEGATED_BN = re.compile(r"(লাগবে না|চাই না|দরকার নেই|দরকার নাই)")
_NEGATED_HI = re.compile(r"(नहीं चाहिए|नहीं चाहिये|ज़रूरत नहीं|जरूरत नहीं)")

_PATTERNS = {"en": _EN, "bn": _BN, "hi": _HI}
_NEGATED = {"en": _NEGATED_EN, "bn": _NEGATED_BN, "hi": _NEGATED_HI}


def _nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t or "")


# Owner, 2026-10-29 (live call): "অন্য কোন স্টাফ এর সাথে কথা বল", "স্টাফ এর সাথে যুক্ত করুন", "কোন মানুষের সাথে কথা বলা যাবে" were refused as "out of context". The list above wants "স্টাফের" in one piece and
# one particular verb; a caller (and the recogniser) writes "স্টাফ এর", "স্টাফকে", "ম্যানেজার", "অন্য কেউ", and says connect / join / transfer / talk / give in a dozen ways. Here a PERSON NOUN and a
# CONNECT VERB anywhere in the sentence (either order, within a few words) is a request for a person. The boundary is loose on purpose: the wrongly connected caller loses seconds, the refused one is angry.
_BN_P_NOUN = (
    r"(?:স্টাফ|ষ্টাফ|ম্যানেজার|ম্যানেজমেন্ট|কর্মী|কর্মচারী|মানুষ|লোক|অফিসার|অপারেটর|রিসেপশন\S*|রিসিপশন\S*|রিসেপশনিস্ট|কাস্টমার কেয়ার|কর্তৃপক্ষ|কাউকে|কেউ|কারো|কারও|কোনো লোক|অন্য কোন\S*|অন্য কেউ|আসল মানুষ|সিনিয়র|হেল্পডেস্ক|হেল্প ডেস্ক)"
    r"(?:\s?(?:এর|কে|র|ের|দের|গুলো|টা|টি))?"
)
_BN_P_VERB = r"(?:কথা|যুক্ত|জুড়ে|কানেক্ট|কানেকশন|ট্রান্সফার|ধরিয়ে|ধরুন|লাইনে|দিন|দেন|দাও|দিয়ে|কল|ফোন|যোগাযোগ|বলান|বলিয়ে|বলতে|মিলিয়ে|পাঠিয়ে)"
_BN_PERSON_LOOSE = [
    re.compile(_BN_P_NOUN + r".{0,28}?" + _BN_P_VERB),
    re.compile(_BN_P_VERB + r".{0,18}?" + _BN_P_NOUN),
]
_HI_P_NOUN = r"(?:स्टाफ|स्टाफ़|इंसान|इन्सान|आदमी|व्यक्ति|मैनेजर|ऑपरेटर|रिसेप्शन\S*|कर्मचारी|अधिकारी|कस्टमर केयर|हेल्पडेस्क|सीनियर|कोई और|किसी और|किसी इंसान)"
_HI_P_VERB = r"(?:बात|जोड़|जोड|कनेक्ट|ट्रांसफर|दीजिए|दीजिये|दो\b|कॉल|फोन|मिला|मिलना|बोल|मिलवा)"
_HI_PERSON_LOOSE = [
    re.compile(_HI_P_NOUN + r".{0,28}?" + _HI_P_VERB),
    re.compile(_HI_P_VERB + r".{0,18}?" + _HI_P_NOUN),
    re.compile(r"किसी\s*(?:से|को)\s*(?:बात|जोड़|कनेक्ट|मिला)"),
]


def _joined(t: str) -> str:
    """Bengali case endings written apart ("স্টাফ এর") are joined, and Devanagari / Bengali nukta forms are made plain, so one pattern reads both spellings."""
    return re.sub(r"\s+(এর|কে|র)\b", r"\1", t).replace("\u09bc", "").replace("\u093c", "")


def asks_for_a_person(text: str, lang: str = "en") -> bool:
    """True if the caller asked to speak to a person. All three languages are
    always checked (a caller may switch mid-call); `lang` only orders them."""
    t = _nfc(text).strip()
    if not t:
        return False
    tj = _joined(t)
    if not (_NEGATED_BN.search(t) or _NEGATED_HI.search(t) or _NEGATED_EN.search(t)):
        if any(p.search(tj) for p in _BN_PERSON_LOOSE + _HI_PERSON_LOOSE):
            return True
    for lg in dict.fromkeys((lang, "en", "bn", "hi")):
        if any(p.search(t) for p in _PATTERNS.get(lg, [])):
            # a bare keyword hit is dropped if the caller said they do NOT want that
            # ("customer care বলে কিছু লাগবে না"); but a NEGATED_EN/BN/HI match only
            # cancels a request that is not also a clear request form
            if _NEGATED.get(lg) and _NEGATED[lg].search(t):
                continue
            return True
    return False




# ---------------------------------------------------------------------------------------------------- KCD-lead-10
# "Book me with the doctor" / "when does Dr. Sen sit" are ordinary turns and must NOT hand off. What marks a request to
# reach the doctor HIMSELF is a TALKING verb joined to "the doctor" -- talk / speak / connect me / put me through /
# (Bengali) kotha bolte / (Hindi) baat karni -- with or without "directly / personally / myself". A bare "doctor" never
# fires: that is why asks_for_a_person's own word list leaves "doctor" out. Same provisional, non-native, deterministic,
# pre-model discipline as asks_for_a_person above.

_I = re.I
_EN_DOC = r"(?:the |a |my |our |that |this |his |her )?(?:doctor|dr\b\.?|physician|specialist)"
_EN_DOCTOR_PERSONAL = [
    re.compile(r"\b(talk|speak|connect|transfer|put me|get me|pass me|patch me|link me|chat)\b.{0,30}?\b(to|with|through to)\b\s*" + _EN_DOC, _I),
    re.compile(r"\b(i )?(personally|myself|directly) (want|need|would like|wish) to (talk|speak)\b.{0,20}\b(doctor|dr\b|physician)", _I),
    re.compile(r"\b(doctor|physician)\b.{0,12}\b(himself|herself)\b.{0,15}\b(talk|speak)", _I),
]
_BN_DOCTOR_PERSONAL = [
    re.compile(r"ডাক্তার\S*\s*(?:(?:সাথে|সঙ্গে|সংগে|সাথেই)\s*)?.{0,15}?কথা"),
    re.compile(r"(নিজে|সরাসরি)\s*ডাক্তার"),
    re.compile(r"ডাক্তার\S*\s*(?:নিজে|স্বয়ং)\s*কথা"),
]
_HI_DOCTOR_PERSONAL = [
    re.compile(r"डॉक्टर\S*\s*(?:साहब\s*)?(?:से|को)\s*.{0,15}?(?:बात|कनेक्ट|जोड़|बोल)"),
    re.compile(r"(खुद|सीधे)\s*डॉक्टर\S*\s*(?:से|को)\s*.{0,10}?(?:बात|मिल)"),
]
_RM_DOCTOR_PERSONAL = [
    re.compile(r"\bdoctor\S{0,3}\s*(?:er|ke|se)?\s*(?:sathe|shathe|songe|sange|se|ke)\b.{0,25}\b(?:kotha|baat|bat|connect|bolte|bolbo|karni|karna|karunga)\b", _I),
    re.compile(r"\b(?:kotha|baat)\b.{0,15}\bdoctor\b", _I),
]
_DOCTOR_PERSONAL_PATTERNS = {
    "en": _EN_DOCTOR_PERSONAL + _RM_DOCTOR_PERSONAL,
    "bn": _BN_DOCTOR_PERSONAL + _RM_DOCTOR_PERSONAL,
    "hi": _HI_DOCTOR_PERSONAL + _RM_DOCTOR_PERSONAL,
}


def wants_doctor_directly(text: str, lang: str = "en") -> bool:
    """True when the caller asks to TALK to the doctor himself (talk/speak/connect + doctor). An ordinary "book me with the doctor" /
    "when is the doctor available" never matches -- there is no talking verb in it. Negation handled the same way asks_for_a_person does."""
    t = _nfc(text).strip()
    if not t:
        return False
    for lg in dict.fromkeys((lang, "en", "bn", "hi")):
        if any(p.search(t) for p in _DOCTOR_PERSONAL_PATTERNS.get(lg, [])):
            if _NEGATED.get(lg) and _NEGATED[lg].search(t):
                continue
            return True
    return False


# ---------------------------------------------------------------------------------------------------- KCD-lead-12
# A caller lodging a complaint is routed to a human, same immediacy rule as asks_for_a_person. "No complaints, everything is
# fine" and a passing grumble ("the wait was long") must not fire -- hence the dedicated negation guard, and the requirement
# that the dissatisfaction be aimed at something (staff / service / treatment) or be named a complaint outright.

_EN_COMPLAINT = [
    re.compile(r"\bcomplain(?:t|ts|ing|ed)?\b", _I),
    re.compile(r"\b(unhappy|dissatisfied|not happy|not satisfied|upset|disappointed|angry|annoyed)\b.{0,15}\b(with|about|at)\b.{0,20}\b(staff|service|treatment|behaviou?r|doctor|clinic|hospital|reception\w*|nurse)\b", _I),
    re.compile(r"\b(bad|terrible|horrible|poor|awful|worst)\s+(experience|service|behaviou?r|treatment)\b", _I),
    re.compile(r"\b(staff|doctor|receptionist|nurse)\b.{0,15}\b(was|were|is|are)\s+(rude|misbehaving)\b|\b(rude|misbehaved)\b.{0,15}\b(staff|doctor|receptionist|nurse)\b", _I),
]
_BN_COMPLAINT = [
    re.compile(r"অভিযোগ|নালিশ|অসন্তুষ্ট"),
    re.compile(r"কম্?প্?ল(?:েই|াই|েন|ায়ে|েই)ন?(?:ট)?"),  # কম্প্লাইন, কমপ্লেইন, কমপ্লেন্ট ... however the recogniser spells it
    re.compile(r"(কমপ্লেইন|কমপ্লেন|কম্প্লেইন|কম্প্লেন|কমপ্লেইন্ট|কমপ্লেন্ট|কমপ্লেইনট)"),
    re.compile(r"(খারাপ|বাজে)\s*(অভিজ্ঞতা|ব্যবহার|সার্ভিস)"),
]
_HI_COMPLAINT = [
    re.compile(r"शिकायत"),
    re.compile(r"(कंप्लेंट|कम्प्लेंट|कंप्लेन|कम्प्लेन|कंप्लेन्ट)"),
    re.compile(r"(खराब|बुरा|बुरी)\s*(अनुभव|व्यवहार|सर्विस|सेवा)"),
]
_RM_COMPLAINT = [
    re.compile(r"\b(obhijog|ovijog|obhijug|shikayat|shikayet|sikayat|complen)\b", _I),
]
_COMPLAINT_PATTERNS = {"en": _EN_COMPLAINT + _RM_COMPLAINT, "bn": _BN_COMPLAINT + _RM_COMPLAINT, "hi": _HI_COMPLAINT + _RM_COMPLAINT}

_NO_COMPLAINT_EN = re.compile(r"\b(no|not|without|never|nothing|don'?t have|do not have|zero)\b.{0,15}\bcomplain|\bnothing to complain\b|\b(don'?t|do not) (want|wish|need) to complain\b", _I)
_NO_COMPLAINT_BN = re.compile(r"(অভিযোগ নেই|অভিযোগ নাই|অভিযোগ করার কিছু নেই|কোনো অভিযোগ নেই|অভিযোগ করতে চাই না)")
_NO_COMPLAINT_HI = re.compile(r"(शिकायत नहीं|शिकायत नही|शिकायत करने की जरूरत नहीं|शिकायत करना नहीं)")
_NO_COMPLAINT_RM = re.compile(r"\b(obhijog|ovijog|shikayat)\s+(nei|nai|nahi)\b", _I)
_NO_COMPLAINT = {"en": _NO_COMPLAINT_EN, "bn": _NO_COMPLAINT_BN, "hi": _NO_COMPLAINT_HI}


def wants_to_complain(text: str, lang: str = "en") -> bool:
    """True if the caller is lodging a complaint. "No complaints, everything is fine" must not match."""
    t = _nfc(text).strip()
    if not t:
        return False
    if _NO_COMPLAINT_RM.search(t):
        return False
    for lg in dict.fromkeys((lang, "en", "bn", "hi")):
        if any(p.search(t) for p in _COMPLAINT_PATTERNS.get(lg, [])):
            if _NO_COMPLAINT.get(lg) and _NO_COMPLAINT[lg].search(t):
                continue
            return True
    return False
