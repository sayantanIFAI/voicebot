"""A doctor asked about on a named WEEKDAY -- "samner robibar doctor x ke pabo?", "ei robibar", "robibar ki doc x bosbe?" (owner, 2026-10-23, item 25).

The day word is turned into the real date of the coming such day (the clinic's own calendar, never the model's guess), and the doctor's availability on THAT date is
looked up in the schedule table. Works for any weekday, in English, Hindi and Bengali -- their own scripts or Latin letters, with the spellings a recogniser writes
(robibar, rabibar, ravivar, itwar, রবিবার, रविवार, ...).

  * "samner / agami / next / अगले" -- the coming one, never today (asked on a Sunday, the next Sunday is a week away);
  * "ei / this / इस" or no marker -- the first such day from today on (asked on a Sunday, today).
Only a QUESTION about a doctor's presence matches: a weekday beside a booking, report, test, serial-number, payment or clinic-hours word is left to those flows.
"""

from __future__ import annotations

import datetime
import re

_SPLIT = re.compile(r"[^\s,.;:!?।॥\"()\[\]{}]+")

# Python weekday (Monday = 0) -> the words that name it: whole Latin words, and Bengali / Devanagari stems (a suffix -e / -এ / -को may follow).
_DAY_LATIN = {
    0: ("monday", "mondays", "mon", "somobar", "sombar", "shombar", "shomobar", "shomobaar", "somvar", "somwar", "somobaar"),
    1: ("tuesday", "tuesdays", "tues", "mongolbar", "mongalbar", "mongolbaar", "mongal", "mangalvar", "mangalwar", "mangalbar"),
    2: ("wednesday", "wednesdays", "budhbar", "budhbaar", "budhvar", "budhwar", "budbar"),
    3: (
        "thursday", "thursdays", "thurs", "brihospotibar", "brihaspatibar", "brihoshpotibar", "brishpotibar", "bispotibar", "bishyudbar", "bishudbar", "bespatibar", "bestibar",
        "brihaspativar", "guruvar", "guruwar", "veervar", "virvar", "brihospoti",
    ),
    4: ("friday", "fridays", "shukrobar", "sukrobar", "sukkurbar", "shukkurbar", "shukrobaar", "sukrobaar", "shukravar", "shukrawar", "sukravar", "sukrawar"),
    5: ("saturday", "saturdays", "shonibar", "sonibar", "shonibaar", "sonibaar", "shanivar", "shaniwar", "shanichar", "shanibar"),
    6: ("sunday", "sundays", "sun", "robibar", "rabibar", "robbar", "robibaar", "robiber", "ravivar", "raviwar", "itwar", "itwaar", "itbar", "roboar"),
}  # fmt: skip
_DAY_SCRIPT = {
    0: ("সোমবার", "সোমবারে", "सोमवार"),
    1: ("মঙ্গলবার", "मंगलवार"),
    2: ("বুধবার", "बुधवार"),
    3: ("বৃহস্পতিবার", "বেস্পতিবার", "বিষ্যুদবার", "বিষ্যুৎবার", "বৃহষ্পতিবার", "গুরুবার", "गुरुवार", "गुरूवार", "बृहस्पतिवार", "वीरवार"),
    4: ("শুক্রবার", "শুক্কুরবার", "शुक्रवार", "शुक्र्वार"),
    5: ("শনিবার", "शनिवार", "शनीचर", "शनिचर"),
    6: ("রবিবার", "রোববার", "রবিবারে", "रविवार", "इतवार", "रवीवार"),
}  # fmt: skip
_LATIN_TO_DAY = {w: d for d, words in _DAY_LATIN.items() for w in words}
_SCRIPT_STEMS = [(s, d) for d, words in _DAY_SCRIPT.items() for s in words]

# "robi bar" / "রবি বার" said as two words
_SPLIT_DAY = re.compile(r"\b(robi|rabi|shomo?|somo?|mongol|mongal|budh|shukro|sukro|shoni|soni|ravi|mangal|shukra|shani|brihospoti|guru)\s+(bar|baar|var|war)\b")
_SPLIT_DAY_BN = re.compile(r"(রবি|সোম|মঙ্গল|বুধ|বৃহস্পতি|শুক্র|শনি)\s+বার")
_SPLIT_DAY_HI = re.compile(r"(रवि|सोम|मंगल|बुध|गुरु|शुक्र|शनि)\s+वार")

_NEXT = frozenset(
    {
        "samner", "samne", "sammner", "shamner", "agami", "ashchhe", "aschhe", "ashche", "asche", "next", "coming", "upcoming", "agle", "agla", "aane", "aanewale", "aagami", "anewale",
        "আসছে", "সামনের", "আগামী", "আগামি", "আসন্ন", "अगले", "अगला", "अगली", "आने", "आगामी",
    }
)  # fmt: skip

_THIS = frozenset({"ei", "this", "এই", "इस", "ye", "yeh", "ये", "यह"})

# A verb that asks "will he be there / can I get him": pabo, bosbe, milenge, বসবেন, पाऊँगा, ...  (whole Latin words; Bengali / Devanagari stems)
_ASK_LATIN = frozenset(
    {
        "pabo", "pabe", "pabona", "pabon", "paobo", "paoa", "paowa", "pawa", "pai", "paya", "milega", "milenge", "milegi", "milte", "milta", "milti", "milunga", "available", "availability",
        "bose", "boshe", "bosen", "boshen", "bosben", "boshben", "bosbe", "boshbe", "bosbena", "boshbena", "bosche", "boschen", "boschhen", "boshchen", "bosbo", "sit", "sits", "sitting", "sitt",
        "ashben", "asben", "aashben", "asbe", "ashbe", "aschen", "asen", "ashen", "thakben", "thakbe", "thaken", "baithenge", "baithte", "baithta", "baithega", "baithegi", "baithengi",
        "aayenge", "ayenge", "aate", "aayega", "ayega", "aayegi", "ayegi", "dekhben", "dekhbe", "dekhen", "dekhte", "dekhega", "dekhenge", "come", "comes", "coming",
    }
)  # fmt: skip
_ASK_STEMS = ("পাব", "পাওয়", "পাই", "বসব", "বসে", "বসেন", "বসছ", "আসব", "আসেন", "আসছেন", "থাকব", "থাকেন", "দেখব", "দেখেন", "দেখা", "मिल", "बैठ", "आएंग", "आयेंग", "आते", "आएगा", "आएगी", "उपलब्ध", "मिलत", "देखत", "देखेंग")
# Said bare ("ache", "hai") a verb only when a doctor word is in the sentence too
_WEAK = frozenset({"ache", "achhe", "achen", "ase", "ashe", "hai", "hain", "আছে", "আছেন", "है", "हैं", "hoga", "hobe", "হবে", "होगा", "होंगे"})
_DOCTOR_WORDS = ("doctor", "doc", "dr", "daktar", "dagtar", "daktaar", "dactar", "ডাঃ", "डॉ", "डा")
_DOCTOR_STEMS = ("ডাক্তার", "ডাক্টার", "डॉक्टर", "डाक्टर")

# Topics that belong to other flows: a weekday beside one of these is not a question about a doctor being there
_OTHER_TOPICS = (
    "book", "appoint", "apoint", "apoin", "report", "test", "serial", "token", "link", "payment", "pay", "naam", "name", "khola", "khule", "bondho", "bandh", "clinic", "discount", "rate", "fees", "fee", "money", "taka",
    "বুক", "অ্যাপয়েন্ট", "অ্যাপোয়েন্ট", "রিপোর্ট", "টেস্ট", "সিরিয়াল", "টোকেন", "লিংক", "লিঙ্ক", "পেমেন্ট", "নাম", "খোলা", "খুলে", "বন্ধ", "ক্লিনিক", "ছাড়", "রেট", "দাম", "টাকা", "ফি",
    "बुक", "अपॉइंट", "अपाइंट", "रिपोर्ट", "टेस्ट", "सीरियल", "टोकन", "लिंक", "पेमेंट", "नाम", "खुल", "बंद", "क्लिनिक", "छूट", "रेट", "फीस", "रुपये", "पैसे",
)  # fmt: skip


def _norm(text: str) -> str:
    low = " ".join((text or "").lower().split())
    low = _SPLIT_DAY.sub(lambda m: m.group(1) + "bar", low)
    low = _SPLIT_DAY_BN.sub(lambda m: m.group(1) + "বার", low)
    return _SPLIT_DAY_HI.sub(lambda m: m.group(1) + "वार", low)


def _tokens(text: str) -> list[str]:
    return _SPLIT.findall(_norm(text))


def weekday_of(text: str) -> int | None:
    """The weekday (Monday = 0) the sentence names, or None. The first one named wins."""
    for tok in _tokens(text):
        if tok in _LATIN_TO_DAY:
            return _LATIN_TO_DAY[tok]
        if not tok.isascii():
            for stem, day in _SCRIPT_STEMS:
                if tok.startswith(stem):
                    return day
    return None


def _has(tokens: list[str], whole: frozenset[str], stems: tuple[str, ...] = ()) -> bool:
    return any(t in whole or (not t.isascii() and any(t.startswith(s) for s in stems)) for t in tokens)


def asks_day_availability(text: str, lang: str = "en") -> bool:
    """True for a question about a doctor being there on a named weekday: "samner robibar doctor x ke pabo", "ei robibar", "robibar ki doc x bosbe"."""
    tokens = _tokens(text)
    if len(tokens) < 2 or weekday_of(text) is None:
        return False
    if any(t.startswith(s) for t in tokens for s in _OTHER_TOPICS if s.isascii() == t.isascii()):
        return False
    if _has(tokens, _ASK_LATIN, _ASK_STEMS):
        return True
    # "ei robibar" / "samner robibar" alone: the marker and the day, nothing else, is the question
    if len(tokens) <= 5 and (_has(tokens, _NEXT, ("আসছ", "সামনের", "আগামী", "आने", "अगल")) or any(t in _THIS for t in tokens)):
        return True
    return _has(tokens, _WEAK) and any(t in _DOCTOR_WORDS or t.startswith(_DOCTOR_STEMS) for t in tokens)


def resolve_date(text: str, today: datetime.date) -> str | None:
    """The ISO date of the weekday named: the first such day from `today` on (today itself included), or the one AFTER today when the caller said "next / samner"."""
    day = weekday_of(text)
    if day is None:
        return None
    ahead = (day - today.weekday()) % 7
    if ahead == 0 and _has(_tokens(text), _NEXT, ("আসছ", "সামনের", "আগামী", "आने", "अगल")):
        ahead = 7
    return (today + datetime.timedelta(days=ahead)).isoformat()
