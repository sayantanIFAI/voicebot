"""Talking about a DEPARTMENT's doctors (owner, 2026-10-29), in Bengali, Hindi and English, for every department.

A caller says "ortho-pedic doctor sits there?" (the recogniser writes "অর্থপেটি ডাক্তার ... কেন বসে"), then "what are the doctors' names?", "are these Kolkata doctors?", "who is good, when do they sit?", or a doctor's first name
in a mangled form ("ও দিপ শুভজিৎ দিপ" for Subhadip). Each of these is answered from the clinic's own tables, in ONE fixed sentence shape, and the department stays the subject until another is named:

  * "does a <department> doctor sit here?"            -> "হ্যাঁ, বসেন।" and nothing else
  * "what are the doctors' names?"                    -> "<N> জন <department>-এর ডাক্তার বসেন: A, B এবং C।" (the count first)
  * "which one is good / when do they sit"            -> all are good, none is picked; every doctor's days and hours; then who to book
  * "are these doctors from Kolkata?"                 -> each doctor of the department, yes or no
  * a first name said in a garbled way               -> the one department doctor it sounds like: only that name

This module holds the words, the matching and the sentences (pure functions, no I/O); main.py does the lookups and the speaking.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

from agent.name_script import name_skeleton

# The words (and the ways a recogniser writes them) that name each department. Several words per department, in the three scripts; matched whole, by their beginning, or by sound (see dept_in_text).
DEPT_WORDS: dict[str, tuple[str, ...]] = {
    "Orthopaedics": (
        "অর্থোপেডিক্স", "অর্থোপেডিক", "অর্থোপেডিস্ট", "অর্থপেডিক", "অর্থপেডিক্স", "অর্থপেটি", "অর্থোপেটিক", "অরথোপেডিক", "ওর্থোপেডিক", "আর্থোপেডিক", "অর্থো", "হাড়ের", "हड्डी", "ऑर्थोपेडिक", "आर्थोपेडिक", "ऑर्थो",
        "orthopedic", "orthopaedic", "orthopetik", "orthopadic", "arthopedic", "ortho",
    ),
    "Cardiology": ("কার্ডিওলজি", "কার্ডিওলজিস্ট", "কার্ডিও", "হৃদরোগ", "কার্ডিওলোজি", "कार्डियोलॉजी", "कार्डियोलॉजिस्ट", "कार्डियो", "cardiology", "cardiologist", "cardio"),
    "Dentistry": ("ডেন্টিস্ট্রি", "ডেন্টিস্ট", "ডেনটিস্ট", "দাঁতের", "डेंटिस्ट", "डेंटिस्ट्री", "दंत", "dentist", "dentistry", "dental"),
    "Dermatology": ("ডার্মাটোলজি", "ডার্মাটোলজিস্ট", "ডারমেটোলজিস্ট", "ডারমেটোলজি", "ডার্মেটোলজি", "চর্মরোগ", "স্কিনের", "स्किन", "त्वचा", "डर्मेटोलॉजी", "डर्मेटोलॉजिस्ट", "dermatology", "dermatologist", "derma", "skin"),
    "Diabetology & Endocrinology": ("ডায়াবেটোলজি", "ডায়াবেটিস", "এন্ডোক্রিনোলজি", "এন্ডোক্রিনোলজিস্ট", "डायबिटीज़", "डायबिटीज", "एंडोक्राइन", "diabetology", "diabetologist", "diabetes", "endocrinology", "endocrinologist"),
    "ENT": ("ইএনটি", "ইএনটির", "ईएनटी", "ent"),
    "General Medicine": ("জেনারেল", "ফিজিশিয়ান", "ফিজিসিয়ান", "জেনারেল মেডিসিন", "जनरल", "फिजिशियन", "फ़िज़िशियन", "physician", "general"),
    "General Surgery": ("সার্জারি", "সার্জন", "সার্জারির", "सर्जरी", "सर्जन", "surgery", "surgeon"),
    "Gynaecology & Obstetrics": ("গাইনি", "গাইনোকোলজি", "গাইনোকোলজিস্ট", "স্ত্রীরোগ", "गायनी", "गायनेकोलॉजी", "गायनेकोलॉजिस्ट", "gynae", "gynaecology", "gynecology", "gynecologist", "gynaecologist"),
    "Neurology": ("নিউরোলজি", "নিউরোলজিস্ট", "न्यूरोलॉजी", "न्यूरोलॉजिस्ट", "neurology", "neurologist"),
    "Neurosurgery": ("নিউরোসার্জারি", "নিউরোসার্জন", "न्यूरोसर्जरी", "न्यूरोसर्जन", "neurosurgery", "neurosurgeon"),
    "Paediatrics": ("পেডিয়াট্রিক্স", "পেডিয়াট্রিশিয়ান", "শিশুরোগ", "বাচ্চাদের", "শিশুদের", "पीडियाट्रिक्स", "पीडियाट्रिशियन", "बच्चों", "paediatrics", "pediatrics", "paediatrician", "pediatrician"),
    "Urology": ("ইউরোলজি", "ইউরোলজিস্ট", "यूरोलॉजी", "यूरोलॉजिस्ट", "urology", "urologist"),
}

_TOKEN = re.compile(r"[\wঀ-৿ऀ-ॿ]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFC", text or "").lower())


def dept_in_text(text: str) -> str | None:
    """The department the sentence names (English name), else None: a department word, or a recogniser's garble of one (same consonants: "অর্থপেটি" ~ "অর্থোপেডিক্স"). The most specific wins when two are named."""
    toks = _tokens(text)
    if not toks:
        return None
    best, best_score = None, 0.0
    for dept, words in DEPT_WORDS.items():
        for w in words:
            wl = unicodedata.normalize("NFC", w).lower()
            for t in toks:
                if t == wl or (len(wl) >= 5 and len(t) >= 5 and (t.startswith(wl) or wl.startswith(t))):
                    score = 1.0 + len(wl) / 100.0
                elif len(wl) >= 5 and len(t) >= 5 and not (t.isascii() and wl.isascii()):
                    a, b = name_skeleton(t), name_skeleton(wl)
                    score = difflib.SequenceMatcher(None, a, b).ratio() if len(a) >= 4 and len(b) >= 4 else 0.0
                    if score < 0.7:
                        score = 0.0
                else:
                    score = 0.0
                if score > best_score:
                    best, best_score = dept, score
    return best


# ---------------------------------------------------------------------------------------------------------------------------------------------- cues
_SIT = re.compile(r"বসে|বসেন|বসছে|বসছেন|আছে|আছেন|থাকেন|বসবে|बैठते|बैठती|बैठता|बैठ|हैं|है(?!\S)|\bsits?\b|\bsitting\b|\bavailable\b|\bthere\b|\bhave\b", re.I)
_YESNO = re.compile(r"(?<!\S)(?:কি|কী|কেন|নাকি|না|ना)(?!\S)|\?|क्या|\bany\b|\bis there\b|\bare there\b|\bdo you\b|\bdoes\b", re.I)
_WHO_WHEN_NAME = re.compile(r"(?<!\S)কে(?!\S)|কারা|কাকে|কবে|কখন|কটা|কতক্ষণ|কোন|নাম|কত|कौन|किस|कब|कितने|नाम|\bwho\b|\bwhen\b|\bwhich\b|\bname|\bwhat time\b|\bhow many\b", re.I)
_NAMES = re.compile(r"নাম|কারা কারা|কে কে|কোন কোন|ডাক্তারদের|नाम|कौन कौन|कौन-कौन|किन-किन|\bnames?\b|\bwho are\b|\bwhich doctors\b", re.I)
_WHEN = re.compile(r"কখন|কবে|কটায়|কোন সময়|সময়|বসবে|বসেন|বসে(?!\S)|कब|समय|कितने बजे|बैठ|\bwhen\b|\bwhat time\b|\btiming|\bsit\b|\bsits\b", re.I)
_WHICH_ONE = re.compile(r"কোনটা|কোন একজন|কোনজন|কোনটি|कौन सा|कौन सी|कौन-सा|\bwhich one\b|\bwhich doctor\b", re.I)
_GOOD = re.compile(r"ভালো|ভাল|হেলো|হেল্লো|ভালু|বেস্ট|সেরা|अच्छा|अच्छे|बेहतर|बढ़िया|भला|\bgood\b|\bbest\b|\bbetter\b", re.I)
_KOLKATA = re.compile(r"কলকাতা|কোলকাতা|কলিকাতা|কোলকাতার|কলকাতার|कोलकाता|कलकत्ता|\bkolkata\b|\bcalcutta\b", re.I)
_ORIGIN_Q = re.compile(r"মালদা|বাইরের|বাইরে|অন্য জেলা|এদিকের|ওদিকের|লোক|কোথাকার|কোথা থেকে|बाहर|कहाँ के|कहां के|\boutside\b|\blocal\b|\bwhere\b", re.I)


def yes_no_existence(text: str) -> bool:
    """"Does a <department> doctor sit here?" -- a sitting / being-there word, a question particle, and none of who / when / which / name."""
    t = unicodedata.normalize("NFC", text or "")
    return len(t.split()) <= 12 and bool(_SIT.search(t)) and bool(_YESNO.search(t)) and not _WHO_WHEN_NAME.search(t)


def asks_names(text: str) -> bool:
    return bool(_NAMES.search(unicodedata.normalize("NFC", text or "")))


def asks_good(text: str) -> bool:
    t = unicodedata.normalize("NFC", text or "")
    return bool(_GOOD.search(t)) or bool(_WHICH_ONE.search(t))


def asks_when(text: str) -> bool:
    return bool(_WHEN.search(unicodedata.normalize("NFC", text or "")))


_ADDRESS_WORDS = re.compile(r"ক্লিনিক|হাসপাতাল|ঠিকানা|ল্যাব|क्लिनिक|अस्पताल|पता|\b(?:clinic|hospital|address|lab|located|location)\b", re.I)


def asks_origin(text: str) -> bool:
    """A Kolkata word in a sentence about a department's doctors ("এগুলো কলকাতা ডাক্ত না এদ মাল্লার", whatever the recogniser made of the rest) is the question "are they from Kolkata?" -- unless it is about the clinic's address."""
    t = unicodedata.normalize("NFC", text or "")
    return bool(_KOLKATA.search(t)) and not _ADDRESS_WORDS.search(t)


# ---------------------------------------------------------------------------------------------------------------------------------------------- a first name said in a garbled way
_TITLES = frozenset("dr doctor ডাঃ ডা ডাক্তার ডক্টর डॉ डॉक्टर".split())


def match_doctor(text: str, infos: list[dict], floor: float = 0.62, margin: float = 0.08) -> dict | None:
    """The ONE doctor of `infos` (catalogue rows) the sentence sounds like, else None. Windows of one to three words are written as consonant skeletons and compared with each doctor's GIVEN name
    ("সুভা জিত দীপ" -> "সভজতদপ" ~ "শুভদীপ" -> "সভদপ"); the best must clear `floor` and beat the next doctor by `margin`. Only ever used among the doctors of the department being discussed."""
    toks = [t for t in _tokens(text) if t not in _TITLES]
    if not toks or not infos:
        return None
    windows = []
    for n in (1, 2, 3):
        for i in range(len(toks) - n + 1):
            windows.append(name_skeleton("".join(toks[i : i + n])))
    scored = []
    for info in infos:
        cands = set()
        for key in ("full_name_bn", "full_name_hi", "full_name"):
            parts = [p for p in (info.get(key) or "").replace(".", " ").split() if p.lower() not in _TITLES]
            if len(parts) >= 2:
                cands.add(name_skeleton("".join(parts[:-1])))  # the given name(s)
            if parts:
                cands.add(name_skeleton("".join(parts)))
        cands = {c for c in cands if len(c) >= 3 and not c.isascii()}
        best = max((difflib.SequenceMatcher(None, w, c).ratio() for w in windows for c in cands if w and len(w) >= 2), default=0.0)
        scored.append((best, info))
    scored.sort(key=lambda x: -x[0])
    if not scored or scored[0][0] < floor:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < margin:
        return None
    return scored[0][1]


def close_doctors(text: str, infos: list[dict], floor: float = 0.55) -> list[dict]:
    """The doctors whose given name the sentence sounds LIKE, best first (at most two): the looser tier of match_doctor, used only to ask "are you asking about Dr. X?" -- never to answer about him.
    A garble such as "শুভজিৎ" sits between "শুভদীপ" and "শুভঙ্কর"; the caller is asked, with the better one first."""
    toks = [t for t in _tokens(text) if t not in _TITLES]
    if not toks or not infos:
        return []
    windows = [name_skeleton("".join(toks[i : i + n])) for n in (1, 2, 3) for i in range(len(toks) - n + 1)]
    scored = []
    for info in infos:
        cands = set()
        for key in ("full_name_bn", "full_name_hi"):
            parts = [p for p in (info.get(key) or "").replace(".", " ").split() if p.lower() not in _TITLES]
            if len(parts) >= 2:
                cands.add(name_skeleton("".join(parts[:-1])))
        cands = {c for c in cands if len(c) >= 3 and not c.isascii()}
        best = max((difflib.SequenceMatcher(None, w, c).ratio() for w in windows for c in cands if w and len(w) >= 3), default=0.0)
        if best >= floor:
            scored.append((best, info))
    scored.sort(key=lambda x: -x[0])
    return [i for _s, i in scored[:2]]


# ---------------------------------------------------------------------------------------------------------------------------------------------- a doctor that is not ours
_TITLE_WORDS = frozenset("ডাক্তার ডাক্তারের ডক্টর ডাঃ ডা डॉक्टर डॉ डॉक्टर्".split()) | {"doctor", "dr", "dr."}
_NAME_END = re.compile(r"চে[মন]্?[বদ]ার|চেম্বার|চেন্বার|বসে|বসেন|বসছ|বসবে|আসেন|আছেন|আছে|আজ|কাল|এখন|कब|आज|कल|अभी|बैठ|चेंबर|क्लिनिक|ক্লিনিক|\b(?:sits?|sitting|chamber|today|available|is|at)\b", re.I)
_NAME_STOP = frozenset("কি কী যে এক একটা ও তো না আর এই ওই সেই কোনো हैं है क्या और ये वो a an the any".split())


def unknown_doctor_name(text: str) -> str | None:
    """"বলছি কি আজ ডাক্তার ধন বাবু চেম্বারে বসছে?" -> "ধন বাবু": the name-like words right after a doctor title and before the sitting / chamber / today words (at most three), or None. Used only for a
    short sitting question whose doctor is NOT one of ours: then it is answered "no doctor by that name sits here" instead of "which doctor?"."""
    toks = _tokens(text)
    for i, t in enumerate(toks):
        if t in _TITLE_WORDS:
            name = []
            for u in toks[i + 1 : i + 5]:
                if _NAME_END.search(u) or u in _NAME_STOP or u in _TITLE_WORDS:
                    break
                name.append(u)
            if name:
                return " ".join(name[:3])
    return None


def no_such_doctor(name: str, lang: str) -> str:
    return {
        "bn": f"দুঃখিত, '{name}' নামে কোনো ডাক্তার সোনোস্ক্যানে বসেন না।",
        "hi": f"माफ़ कीजिए, '{name}' नाम के कोई डॉक्टर सोनोस्कैन में नहीं बैठते।",
        "en": f"Sorry, no doctor by the name '{name}' sits at Sonoscan.",
    }.get(lang, "")


# ---------------------------------------------------------------------------------------------------------------------------------------------- sentences
_COUNT = {
    "bn": {1: "একজন", 2: "দুইজন", 3: "তিনজন", 4: "চারজন", 5: "পাঁচজন", 6: "ছয়জন", 7: "সাতজন", 8: "আটজন", 9: "নয়জন", 10: "দশজন"},
    "hi": {1: "एक", 2: "दो", 3: "तीन", 4: "चार", 5: "पाँच", 6: "छह", 7: "सात", 8: "आठ", 9: "नौ", 10: "दस"},
    "en": {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"},
}


def spoken_name(info: dict, lang: str) -> str:
    if lang == "bn":
        return "ডাঃ " + (info.get("full_name_bn") or (info.get("aliases_bn") or [""])[0])
    if lang == "hi":
        return "डॉक्टर " + (info.get("full_name_hi") or (info.get("aliases_hi") or [""])[0])
    return info.get("full_name") or info.get("name") or "the doctor"


def _join(names: list[str], lang: str) -> str:
    if len(names) <= 1:
        return "".join(names)
    word = {"bn": " এবং ", "hi": " और ", "en": " and "}.get(lang, " and ")
    return ", ".join(names[:-1]) + word + names[-1]


def yes_sits(lang: str) -> str:
    return {"bn": "হ্যাঁ, বসেন।", "hi": "जी हाँ, बैठते हैं।", "en": "Yes, they sit here."}.get(lang, "Yes, they sit here.")


def names_reply(dept_label: str, dept_possessive: str, names: list[str], lang: str) -> str:
    n = len(names)
    word = _COUNT.get(lang, _COUNT["en"]).get(n, str(n))
    if lang == "bn":
        return f"{word} {dept_possessive} ডাক্তার বসেন: {_join(names, 'bn')}।"
    if lang == "hi":
        return f"{dept_label} विभाग में {word} डॉक्टर बैठते हैं: {_join(names, 'hi')}।"
    return f"{word.capitalize()} {dept_label} doctor{'s' if n != 1 else ''} sit here: {_join(names, 'en')}."


def origin_reply(rows: list[tuple[str, str]], lang: str) -> str:
    """rows = [(spoken name, "yes" | "no" | "")] -> "ডাঃ A কলকাতার, ডাঃ B কলকাতার নন।" ; doctors with no value are left out."""
    parts = []
    for who, value in rows:
        v = (value or "").strip().lower()
        if v == "yes":
            parts.append({"bn": f"{who} কলকাতার", "hi": f"{who} कोलकाता से हैं", "en": f"{who} is from Kolkata"}.get(lang, f"{who} is from Kolkata"))
        elif v == "no":
            parts.append({"bn": f"{who} কলকাতার নন", "hi": f"{who} कोलकाता से नहीं हैं", "en": f"{who} is not from Kolkata"}.get(lang, f"{who} is not from Kolkata"))
    if not parts:
        return {"bn": "এই তথ্য আমার কাছে নেই। কাউন্টারে জিজ্ঞেস করতে পারেন।", "hi": "यह जानकारी मेरे पास नहीं है। काउंटर पर पूछ सकते हैं।", "en": "I do not have that information. You can ask at the counter."}.get(lang, "")
    end = "।" if lang in ("bn", "hi") else "."
    return ", ".join(parts) + end


_BN_NUM = ["", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়", "দশ", "এগারো", "বারো"]
_HI_NUM = ["", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस", "ग्यारह", "बारह"]


def _clock_words(hhmm: str, lang: str) -> str:
    """"10:00" -> "সকাল দশটা" / "सुबह दस बजे" ; "17:30" -> "বিকেল সাড়ে পাঁচটা" ; the part of the day is said, and the time is words, never digits."""
    try:
        h, m = (int(x) for x in hhmm.split(":")[:2])
    except ValueError:
        return hhmm
    part_bn = "সকাল" if h < 12 else "দুপুর" if h < 15 else "বিকেল" if h < 18 else "সন্ধ্যা" if h < 20 else "রাত"
    part_hi = "सुबह" if h < 12 else "दोपहर" if h < 15 else "शाम" if h < 20 else "रात"
    h12 = h % 12 or 12
    if lang == "bn":
        if m == 30:
            return f"{part_bn} সাড়ে {_BN_NUM[h12]}টা"
        if m == 0:
            return f"{part_bn} {_BN_NUM[h12]}টা"
        return f"{part_bn} {_BN_NUM[h12]}টা {m} মিনিট"
    if lang == "hi":
        if m == 30:
            return f"{part_hi} साढ़े {_HI_NUM[h12]} बजे"
        if m == 0:
            return f"{part_hi} {_HI_NUM[h12]} बजे"
        return f"{part_hi} {_HI_NUM[h12]} बजकर {m} मिनट"
    return f"{h12}:{m:02d} {'AM' if h < 12 else 'PM'}"


def hours_words(hours: str, lang: str) -> str:
    """"10:00-12:00" -> "সকাল দশটা থেকে দুপুর বারোটা পর্যন্ত"."""
    a, _, b = (hours or "").partition("-")
    if not b:
        return hours or ""
    return {
        "bn": f"{_clock_words(a.strip(), 'bn')} থেকে {_clock_words(b.strip(), 'bn')} পর্যন্ত",
        "hi": f"{_clock_words(a.strip(), 'hi')} से {_clock_words(b.strip(), 'hi')} तक",
        "en": f"{_clock_words(a.strip(), 'en')} to {_clock_words(b.strip(), 'en')}",
    }.get(lang, hours)


def origin_all_yes(dept_possessive: str, dept_label: str, lang: str) -> str:
    return {
        "bn": f"হ্যাঁ, {dept_possessive} সব ডাক্তারই কলকাতার।",
        "hi": f"जी हाँ, {dept_label} विभाग के सभी डॉक्टर कोलकाता से हैं।",
        "en": f"Yes, all the {dept_label} doctors are from Kolkata.",
    }.get(lang, "")


def confirm_name(spoken: str, lang: str) -> str:
    return {
        "bn": f"আপনি কি {spoken}-এর কথা জিজ্ঞেস করছেন?",
        "hi": f"क्या आप {spoken} के बारे में पूछ रहे हैं?",
        "en": f"Are you asking about {spoken}?",
    }.get(lang, "")


def confirm_either(a: str, b: str, lang: str) -> str:
    return {
        "bn": f"আপনি কি {a}-এর কথা বলছেন, নাকি {b}-এর?",
        "hi": f"क्या आप {a} के बारे में पूछ रहे हैं, या {b} के?",
        "en": f"Are you asking about {a}, or {b}?",
    }.get(lang, "")


def all_are_good(lang: str) -> str:
    return {
        "bn": "আমাদের সব ডাক্তারই খুব ভালো, তাই একজনকে আলাদা করে বলতে পারব না।",
        "hi": "हमारे सभी डॉक्टर बहुत अच्छे हैं, इसलिए किसी एक को अलग से नहीं चुन सकती।",
        "en": "All our doctors are very good, so I cannot pick one over the others.",
    }.get(lang, "All our doctors are very good, so I cannot pick one over the others.")


def ask_which(lang: str, n: int) -> str:
    if n <= 1:
        return {"bn": "অ্যাপয়েন্টমেন্ট করতে চান?", "hi": "अपॉइंटमेंट लेना चाहेंगे?", "en": "Would you like to book an appointment?"}.get(lang, "")
    return {"bn": "কোন ডাক্তারের সাথে অ্যাপয়েন্টমেন্ট করতে চান?", "hi": "किस डॉक्टर के साथ अपॉइंटमेंट लेना चाहेंगे?", "en": "Which doctor would you like to book?"}.get(lang, "")
