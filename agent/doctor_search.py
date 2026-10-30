"""Finding the right doctor from how a caller asks (owner, 2026-10-18).

  * wanted_sex        -- "a lady gynaecologist", "female doctor", "মহিলা ডাক্তার", "लेडी डॉक्टर": the search is filtered to that sex, nothing else.
  * confusable        -- "neuro" / "euro" are misheard for each other and a neurologist is not a neurosurgeon: the agent asks which, in so many words,
                         before it looks anything up. Table-driven: a department added to the table is covered with no other change.
  * pick_from         -- the caller's answer to such a question, or to "which of these doctors?".
  * replies           -- the spoken sentences (bn / hi / en) for a filtered list, a nobody-sits-then answer and the "which of these?" question.

No lookup happens here: the clinic API returns the real doctors, this module only decides what to ask and how to say it.
"""

from __future__ import annotations

import re
import unicodedata


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower().replace("়", "").replace("़", "")
    return " ".join(re.sub(r"[.,;:!?\"'()।॥\-]+", " ", s).split())


def _has(t: str, words) -> bool:
    padded = f" {t} "
    for w in words:
        w = _norm(w)
        if w and ((f" {w} " in padded) if w.isascii() else (w in t)):
            return True
    return False


# ------------------------------------------------------------------------------------------------------------------------------------------ sex

_FEMALE = (
    "female doctor", "lady doctor", "woman doctor", "women doctor", "ladies doctor", "female gynecologist", "female gynaecologist", "lady gynecologist", "lady gynaecologist",
    "female gynae", "lady gynae", "female gyne", "lady gyne", "female physician", "female surgeon", "lady surgeon", "female dentist", "lady dentist", "madam doctor", "doctor madam", "female dr", "lady dr",
    "মহিলা ডাক্তার", "মহিলা গাইনো", "মহিলা গাইনি", "লেডি ডাক্তার", "লেডিস ডাক্তার", "মেয়ে ডাক্তার", "মেয়েদের ডাক্তার", "ম্যাডাম ডাক্তার", "মহিলা চিকিৎসক", "মহিলা সার্জন", "লেডি গাইনো", "লেডি গাইনি", "মহিলা",
    "महिला डॉक्टर", "लेडी डॉक्टर", "लेडीज डॉक्टर", "लड़की डॉक्टर", "मैडम डॉक्टर", "महिला चिकित्सक", "महिला गायनी", "लेडी गायनी", "लेडी",
    # owner, 2026-10-26: "लड़की लोग का डॉक्टर बैठता है क्या" ("does the doctor for girls sit?"). Each is a cue only beside a doctor word (wanted_sex).
    "लड़की लोग", "लड़कियों", "लड़कियां", "औरतों", "महिलाओं", "मेयेदेर", "মেয়েরা", "মেয়েছেলে",
)
_MALE = (
    "male doctor", "gentleman doctor", "man doctor", "gents doctor", "male gynecologist", "male gynaecologist", "male gynae", "male surgeon", "male dentist", "male dr", "sir doctor",
    "পুরুষ ডাক্তার", "পুরুষ গাইনো", "ছেলে ডাক্তার", "ছেলেদের ডাক্তার", "পুরুষ চিকিৎসক", "জেন্টস ডাক্তার",
    "पुरुष डॉक्टर", "मेल डॉक्टर", "जेंट्स डॉक्टर", "लड़का डॉक्टर", "पुरुष चिकित्सक",
)
_DOCTOR_WORD = ("doctor", "dr", "gynecologist", "gynaecologist", "gynae", "gyne", "surgeon", "physician", "dentist", "ডাক্তার", "চিকিৎসক", "গাইনো", "গাইনি", "সার্জন", "डॉक्टर", "चिकित्सक", "गायनी")


def wanted_sex(text: str) -> str | None:
    """"F" when the caller asks for a woman doctor, "M" for a man, else None. "mohila" alone is a woman-only cue only beside a doctor word."""
    t = _norm(text)
    if not t:
        return None
    if _has(t, _MALE):
        return "M"
    if _has(t, _FEMALE) and _has(t, _DOCTOR_WORD):
        return "F"
    return None


# ------------------------------------------------------------------------------------------------------------------------------ confusable families

# Each family: the stems that must be confirmed, and the departments they could mean. `bare_ok` words are spoken in full and unambiguous (no question).
FAMILIES: list[dict] = [
    {
        "stems": ("neuro", "নিউরো", "নিউরোর", "नेउरो", "न्यूरो", "नूरो", "euro", "ইউরো", "নুরো", "ऊरो", "यूरो", "nero"),
        "mishear": ("euro", "ইউরো", "যূরো", "यूरो", "ऊरो", "nuro", "নুরো"),  # Euro / Uro read for Neuro: all three are in play
        # Real call, 2026-10-29: "নিউরো মেডিসিন"/"नूरो मेडिसिन" ("neuro medicine") is a real, common way of naming the non-surgical
        # (medicine) side of neuro care -- a neurologist, not a neurosurgeon -- so it never needs the "which one?" question,
        # same as the bare specialist titles already here. Includes the real call's own dropped-vowel misspelling ("नूरो").
        "bare_ok": ("neurologist", "neurology", "neurosurgeon", "neurosurgery", "neuro surgeon", "neuro medicine", "neuro medicines", "neuromedicine", "nuero medicine", "nuro medicine", "newro medicine", "niuro medicine", "নিউরোলজিস্ট", "নিউরোলজি", "নিউরোসার্জন", "নিউরো সার্জন", "নিউরোসার্জারি", "নিউরো মেডিসিন", "নিউরোমেডিসিন", "নুরো মেডিসিন", "न्यूरोलॉजिस्ट", "न्यूरोसर्जन", "न्यूरो मेडिसिन", "नूरो मेडिसिन", "नुरो मेडिसिन", "न्यूरोमेडिसिन", "urologist", "urology"),
        "options": [
            {
                "dept": "Neurology",
                "bn": "নিউরোলজিস্ট, মানে স্নায়ু আর মস্তিষ্কের রোগের ডাক্তার",
                "hi": "न्यूरोलॉजिस्ट, यानी नसों और दिमाग़ की बीमारी के डॉक्टर",
                "en": "a neurologist, the nerve and brain physician",
                "words": ("neurolog", "নিউরোলজি", "নিউরোলোজি", "न्यूरोलॉज", "স্নায়ু", "ওষুধের", "physician", "medicine"),
            },
            {
                "dept": "Neurosurgery",
                "bn": "নিউরোসার্জন, মানে মাথা আর মেরুদণ্ডের অপারেশনের ডাক্তার",
                "hi": "न्यूरोसर्जन, यानी दिमाग़ और रीढ़ की सर्जरी के डॉक्टर",
                "en": "a neurosurgeon, who operates on the brain and spine",
                "words": ("surgeon", "surgery", "সার্জন", "সার্জারি", "সার্জেন", "অপারেশন", "सर्जन", "सर्जरी", "ऑपरेशन", "operat"),
            },
            {
                "dept": "Urology",
                "bn": "ইউরোলজিস্ট, মানে কিডনি আর প্রস্রাবের ডাক্তার",
                "hi": "यूरोलॉजिस्ट, यानी किडनी और पेशाब के डॉक्टर",
                "en": "a urologist, the kidney and urine doctor",
                "words": ("urolog", "ইউরোলজি", "ইউরোলোজি", "यूरोलॉज", "কিডনি", "প্রস্রাব", "kidney", "urine", "किडनी", "पेशाब"),
            },
        ],
    },
]

_ORDINALS = (
    (0, ("first", "1st", "one", "প্রথম", "প্রথমটা", "পहला", "पहला", "পহলা", "এক নম্বর", "১", "1")),
    (1, ("second", "2nd", "two", "দ্বিতীয়", "দ্বিতীয়টা", "দুই নম্বর", "দूसरा", "दूसरा", "২", "2")),
    (2, ("third", "3rd", "three", "তৃতীয়", "তৃতীয়টা", "তিন নম্বর", "तीसरा", "৩", "3")),
)


def confusable(text: str, existing_departments: set[str] | None = None) -> dict | None:
    """When the sentence names a family stem without saying a clear full word, the options to ask about: {"options": [...], "mishear": bool}.
    Options are limited to departments that exist (`existing_departments`, when given); a single option needs no question."""
    t = _norm(text)
    if not t:
        return None
    for fam in FAMILIES:
        if _has(t, fam["bare_ok"]) and not _has(t, fam["mishear"]):
            continue
        hit = any((w in t.split()) or (not w.isascii() and w in t) or (w.isascii() and re.search(rf"(?<![a-z]){re.escape(w)}", t)) for w in fam["stems"])
        if not hit:
            continue
        mishear = _has(t, fam["mishear"])
        options = [o for o in fam["options"] if existing_departments is None or o["dept"] in existing_departments]
        if not mishear:  # a plain "neuro": the brain and nerve pair; Urology only when "euro" was heard
            options = [o for o in options if o["dept"] != "Urology"]
        if len(options) < 2:
            continue
        return {"options": options, "mishear": mishear}
    return None


def question(options: list[dict], lang: str) -> str:
    """"Do you mean A, B or C?" -- each option with what it is, so the caller can choose by meaning."""
    lab = [o.get(lang) or o["en"] for o in options]
    if lang == "bn":
        return "আপনি কোন ডাক্তারের কথা বলছেন — " + ", নাকি ".join(lab) + "?"
    if lang == "hi":
        return "आप किस डॉक्टर की बात कर रहे हैं — " + ", या ".join(lab) + "?"
    return "Which doctor do you mean — " + ", or ".join(lab) + "?"


def pick_from(text: str, options: list[dict]) -> dict | None:
    """The option the caller's answer chooses (by its words, or "the first / second"), or None."""
    t = _norm(text)
    if not t:
        return None
    hits = [o for o in options if _has(t, o["words"])]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        for idx, words in _ORDINALS:
            if idx < len(options) and _has(t, words) and len(t.split()) <= 4:
                return options[idx]
    return None


# --------------------------------------------------------------------------------------------------------------------------------------- replies

_SEX_WORD = {"F": {"bn": "মহিলা", "hi": "महिला", "en": "lady"}, "M": {"bn": "পুরুষ", "hi": "पुरुष", "en": "male"}}


def _join(names: list[str], lang: str) -> str:
    sep = {"bn": " আর ", "hi": " और ", "en": " and "}[lang]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + sep + names[-1]


def filtered_reply(result: dict, sex: str, lang: str, dept_label: str) -> str:
    """The list for "a lady gynaecologist": only that sex; plainly says so when the department has none."""
    names = result.get("doctors") or []
    today = result.get("doctors_today") or []
    w = _SEX_WORD[sex][lang]
    if not names:
        other = "M" if sex == "F" else "F"
        return {
            "bn": f"দুঃখিত, {dept_label} বিভাগে এখন কোনো {w} ডাক্তার নেই। {_SEX_WORD[other]['bn']} ডাক্তারের কাছে দেখাতে চাইলে বলুন, আমি বিভাগের ডাক্তারদের নাম বলে দেব।",
            "hi": f"माफ़ कीजिए, {dept_label} विभाग में अभी कोई {w} डॉक्टर नहीं हैं। {_SEX_WORD[other]['hi']} डॉक्टर को दिखाना चाहें तो बताइए, मैं विभाग के डॉक्टरों के नाम बता दूँगी।",
            "en": f"Sorry, there is no {w} doctor in {dept_label} right now. If you are willing to see a {_SEX_WORD[other]['en']} doctor, tell me and I will read out the department's doctors.",
        }[lang]
    tail = {
        "bn": f" আজ বসছেন {_join(today, lang)}। কাকে চান?" if today else " কার কাছে বুক করব?",
        "hi": f" आज {_join(today, lang)} बैठ रही हैं। किसे चाहिए?" if today else " किसके पास बुक करूँ?",
        "en": f" Sitting today: {_join(today, lang)}. Whom would you like?" if today else " Whom shall I book?",
    }[lang]
    return {
        "bn": f"{dept_label} বিভাগে {w} ডাক্তার আছেন {_join(names, lang)}।" + tail,
        "hi": f"{dept_label} विभाग में {w} डॉक्टर हैं {_join(names, lang)}।" + tail,
        "en": f"In {dept_label}, the {w} doctors are {_join(names, lang)}." + tail,
    }[lang]


# Owner, 2026-10-30: a list of the doctors who sit is told by the part of the day they START in, with no hours ("সকালে বসছেন A, B আর C, দুপুরে বসছেন D, বিকালে বসছেন E, রাতে বসছেন F"); the time of
# the doctor the caller chooses is told in the booking. Starts before 12:00 are morning, 12:00-15:00 noon, 15:00 until NIGHT_FROM afternoon ("বিকাল": the owner's example puts a doctor who starts at 18:00 under
# বিকালে and one who starts at 22:00 under রাতে), NIGHT_FROM onward night. (The caller's own "বিকালের দিকে" filter, part_of_day_of, is unchanged.)
NIGHT_FROM = 20
_SPOKEN_PARTS = ("morning", "noon", "afternoon", "night")
_SPOKEN_PART_WORD = {
    "morning": {"bn": "সকালে", "hi": "सुबह", "en": "in the morning"},
    "noon": {"bn": "দুপুরে", "hi": "दोपहर में", "en": "around noon"},
    "afternoon": {"bn": "বিকালে", "hi": "शाम को", "en": "in the afternoon"},
    "night": {"bn": "রাতে", "hi": "रात में", "en": "at night"},
}


def spoken_part_of(start: str | None) -> str | None:
    try:
        h = int((start or "").split(":")[0])
    except ValueError:
        return None
    return "morning" if h < 12 else "noon" if h < 15 else "afternoon" if h < NIGHT_FROM else "night"


def grouped_by_part(items: list[dict], lang: str) -> str | None:
    """"সকালে বসছেন A, B আর C, দুপুরে বসছেন D, রাতে বসছেন F" -- None when a doctor has no start time (the caller then names them plainly)."""
    if not items or any(spoken_part_of(i.get("start")) is None for i in items):
        return None
    lg = lang if lang in ("bn", "hi") else "en"
    groups = []
    for part in _SPOKEN_PARTS:
        names = [i["name"] for i in items if spoken_part_of(i.get("start")) == part]
        if not names:
            continue
        w = _SPOKEN_PART_WORD[part][lg]
        groups.append({"bn": f"{w} বসছেন {_join(names, lg)}", "hi": f"{w} {_join(names, lg)} बैठ रहे हैं", "en": f"{w}: {_join(names, lg)}"}[lg])
    return ", ".join(groups)


def sitting_pick_question(names: list[str], when_phrase: str, lang: str, items: list[dict] | None = None) -> str:
    """Several doctors sit at the asked time: name them and ask which, with a confirmation, never choosing one for the caller. With `items` (the doctors' rows, with their start times) the names
    are told by part of the day."""
    grouped = grouped_by_part(items or [], lang)
    if grouped:
        return {
            "bn": f"{when_phrase} {grouped}। কার কাছে অ্যাপয়েন্টমেন্ট করব?",
            "hi": f"{when_phrase} {grouped}। किसके पास अपॉइंटमेंट करूँ?",
            "en": f"{when_phrase} {grouped}. Whom shall I book?",
        }[lang]
    return {
        "bn": f"{when_phrase} এই ডাক্তাররা বসেন: {_join(names, lang)}। কার কাছে অ্যাপয়েন্টমেন্ট করব?",
        "hi": f"{when_phrase} ये डॉक्टर बैठते हैं: {_join(names, lang)}। किसके पास अपॉइंटमेंट करूँ?",
        "en": f"{when_phrase} these doctors sit: {_join(names, lang)}. Whom shall I book?",
    }[lang]


def sitting_list_reply(sitting: list[dict], when_phrase: str, dept_label: str, lang: str) -> str:
    """"Who sits in cardiology tomorrow": the doctors by part of the day (no hours -- the time is told in the booking), from the real schedule. Never today's list for a question about tomorrow."""
    names = [s["name"] for s in sitting]
    if not names:
        return nobody_sits(when_phrase, [], lang)
    grouped = grouped_by_part(sitting, lang)
    if len(names) == 1:
        # Owner, 2026-10-27: with ONE doctor there is nobody to choose between -- "whom?" is the wrong question; ask whether to book him.
        g = grouped or names[0]
        return {
            "bn": f"{when_phrase} {dept_label} বিভাগে {g}। অ্যাপয়েন্টমেন্ট করতে চান?",
            "hi": f"{when_phrase} {dept_label} विभाग में {g}। अपॉइंटमेंट करना चाहेंगे?",
            "en": f"{when_phrase} in {dept_label}, {g}. Would you like an appointment?",
        }[lang]
    g = grouped or _join(names, lang)
    return {
        "bn": f"{when_phrase} {dept_label} বিভাগে {g}। কার কাছে অ্যাপয়েন্টমেন্ট করতে চান?",
        "hi": f"{when_phrase} {dept_label} विभाग में {g}। किसके पास अपॉइंटमेंट चाहिए?",
        "en": f"{when_phrase} in {dept_label}, {g}. Whom would you like an appointment with?",
    }[lang]


def nobody_sits(when_phrase: str, others: list[str], lang: str) -> str:
    if others:
        return {
            "bn": f"{when_phrase} ওই বিভাগের কেউ বসেন না। ওই দিন বসছেন {_join(others, lang)}, অন্য সময়ে। চাইলে ওনাদের কারও কথা বলুন।",
            "hi": f"{when_phrase} उस विभाग का कोई डॉक्टर नहीं बैठता। उसी दिन {_join(others, lang)} दूसरे समय बैठते हैं। चाहें तो किसी का नाम बताइए।",
            "en": f"Nobody from that department sits {when_phrase}. On that day {_join(others, lang)} sit at other times. Tell me if you would like one of them.",
        }[lang]
    return {
        "bn": f"{when_phrase} ওই বিভাগের কেউ বসেন না। অন্য দিনের কথা বলবেন?",
        "hi": f"{when_phrase} उस विभाग का कोई डॉक्टर नहीं बैठता। किसी और दिन के लिए बताएँ?",
        "en": f"Nobody from that department sits {when_phrase}. Would another day do?",
    }[lang]


# ------------------------------------------------------------------------------------------------------------------------------ the bare word "doctor"

# owner, 2026-10-26: "डॉक्टर दिखाना था" ("I wanted to see a doctor") was searched as a doctor NAMED "डॉक्टर" and answered "no doctor of that name -- did you mean Chakraborty?".
# The word "doctor" in any script, with the little words around it, is not a name: the caller is asked which doctor (or what the problem is).
_DOCTOR_TOKENS = {"doctor", "doctors", "dr", "doc", "daktar", "doktor", "dactar", "ডাঃ", "ডক্টর", "डॉ", "डा"}
_DOCTOR_STEMS = ("ডাক্তার", "ডাক্টার", "ডক্টর", "डॉक्टर", "डाक्टर", "डॉक्टर्", "doctor")
_FILLER_TOKENS = {
    "ei", "oi", "ek", "ekta", "ekjon", "kon", "kono", "ki", "ke", "the", "a", "an", "my", "some", "any", "which",
    "এই", "ওই", "এক", "একটা", "একজন", "কোন", "কোনো", "কি", "কে", "টা", "আছে", "আছেন", "আমার", "একটি",
    "एक", "कोई", "कौन", "कोन", "किस", "ये", "यह", "वो", "वह", "है", "हैं", "मेरे", "मेरा", "का", "की", "के",
}


def is_generic_doctor_phrase(name: str) -> bool:
    """True when `name` holds only the word "doctor" (any script, any ending) and little filler words: it names nobody."""
    tokens = _norm(name).split()
    if not tokens:
        return False
    saw_doctor = False
    for t in tokens:
        if t in _DOCTOR_TOKENS or t.startswith(_DOCTOR_STEMS):
            saw_doctor = True
        elif t not in _FILLER_TOKENS:
            return False
    return saw_doctor


# ------------------------------------------------------------------------------------------------------------------------------ "which doctors do you have?"

# Owner, 2026-10-27: "কোন কোন ডাক্তার বসে আপনাদের ওখানে", "क्या क्या डॉक्टर है आप लोगों के पास", "কে কে বসে" were answered "which doctor do you mean?" / "sorry, say again". They are a request for the
# range of doctors: answered with a few real examples from the catalogue and the question of whom to book. Needs a doctor word AND a "which / who / how many" shape.
_DOCTOR_RANGE = re.compile(
    r"(?:কোন\s*কোন|কে\s*কে|কারা\s*কারা|কি\s*কি|কী\s*কী|কয়জন|কত\s*জন|ডাক্তার\s*কারা|কোন\s*কোন\s*ডাক্তার)"
    r"|(?:कौन\s*[- ]?\s*कौन|क्या\s*क्या|कितने|कौन\s*से\s*कौन\s*से|डॉक्टर\s*कौन)"
    r"|\b(?:which doctors|what doctors|who are the doctors|how many doctors|list of doctors|doctors do you have|doctors are there|which all doctors)\b",
    re.I,
)
_DOCTOR_WORDS_RANGE = ("ডাক্তার", "ডক্টর", "डॉक्टर", "डाक्टर", "doctor", "doctors")


def asks_which_doctors(text: str) -> bool:
    t = _norm(text)
    return bool(t) and any(w in t for w in _DOCTOR_WORDS_RANGE) and _DOCTOR_RANGE.search(t) is not None


def overview_reply(examples: list[tuple[str, str]], lang: str) -> str:
    """`examples` = [(department label, a real doctor of it), ...]. Says every department has doctors, gives the examples, asks whom to book."""
    listed = {
        "bn": "; ".join(f"{d} বিভাগে {n}" for d, n in examples),
        "hi": "; ".join(f"{d} विभाग में {n}" for d, n in examples),
        "en": "; ".join(f"{n} in {d}" for d, n in examples),
    }[lang if lang in ("bn", "hi") else "en"]
    if not examples:
        return {
            "bn": "আমাদের সব বিভাগেই ডাক্তার আছেন। কোন ডাক্তারের কাছে অ্যাপয়েন্টমেন্ট করতে চান?",
            "hi": "हमारे सभी विभागों में डॉक्टर हैं। किस डॉक्टर के साथ अपॉइंटमेंट लेना चाहेंगे?",
            "en": "We have doctors in every department. Which doctor would you like to book?",
        }[lang if lang in ("bn", "hi") else "en"]
    return {
        "bn": f"আমাদের সব বিভাগেই ভালো ডাক্তার আছেন, যেমন {listed}। কোন ডাক্তারের কাছে অ্যাপয়েন্টমেন্ট করতে চান?",
        "hi": f"हमारे सभी विभागों में अच्छे डॉक्टर हैं, जैसे {listed}। किस डॉक्टर के साथ अपॉइंटमेंट लेना चाहेंगे?",
        "en": f"We have good doctors in every department, for example {listed}. Which doctor would you like to book?",
    }[lang if lang in ("bn", "hi") else "en"]


# ------------------------------------------------------------------------------------------------------------------------------ questions about the doctors JUST offered

# Owner, 2026-10-27: after "আগামীকাল এই ডাক্তাররা বসেন: দেবাশিস দাস আর অমিত কর। কার কাছে অ্যাপয়েন্টমেন্ট করব?" the caller asked "কে ভালো হবে এর মধ্যে সকালে কে বসে" (who is better among these, who sits in the
# morning): the answer was "sorry, say again" and then "I cannot say who is better; tell me your problem", which lost the whole conversation. The doctors offered stay the subject.
_GOOD_AMONG = re.compile(r"(?:ভালো|ভাল|ভাল|অভিজ্ঞ|সেরা)|(?:अच्छा|अच्छे|बेहतर|सबसे अच्छ)|\b(?:good|better|best)\b", re.I)
_PARTS = {
    "morning": ("সকাল", "ভোর", "सुबह", "सवेरे", "morning"),
    "afternoon": ("দুপুর", "दोपहर", "afternoon"),
    "evening": ("বিকেল", "বিকাল", "সন্ধ্যা", "সন্ধ্যে", "রাত", "शाम", "रात", "evening", "night"),
}
_PART_WORD = {
    "morning": {"bn": "সকালে", "hi": "सुबह", "en": "in the morning"},
    "afternoon": {"bn": "দুপুরে", "hi": "दोपहर में", "en": "in the afternoon"},
    "evening": {"bn": "সন্ধ্যায়", "hi": "शाम को", "en": "in the evening"},
}


def asked_part_of_day(text: str) -> str | None:
    t = _norm(text)
    # Owner, 2026-10-28: "সকালে হবে বা ... আজকে বিকালে" names morning first and evening last: the LAST part of the day said is the one meant
    best, best_pos = None, -1
    for part, words in _PARTS.items():
        for w in words:
            pos = t.rfind(w)
            if pos > best_pos:
                best, best_pos = part, pos
    return best


def asks_who_is_good(text: str) -> bool:
    return _GOOD_AMONG.search(_norm(text)) is not None


def part_of_day_of(start: str | None) -> str | None:
    try:
        h = int((start or "").split(":")[0])
    except ValueError:
        return None
    return "morning" if h < 12 else "afternoon" if h < 16 else "evening"


def _with_hours(item: dict) -> str:
    hours = f" ({item['start']}-{item['end']})" if item.get("start") and item.get("end") else ""
    return f"{item['name']}{hours}"


def offered_followup_reply(items: list[dict], part: str | None, good: bool, lang: str) -> tuple[str, dict | None]:
    """The answer about the doctors just offered and, when exactly one of them fits, that doctor (the question then is whether to book him). `part` = morning / afternoon / evening or None;
    `good` = the caller asked who is better. No doctor is ever called better than another: "all of our doctors are good" is the approved answer."""
    lg = lang if lang in ("bn", "hi") else "en"
    names = "; ".join(_with_hours(i) for i in items)
    lead = {"bn": "আমাদের সব ডাক্তারই ভালো। ", "hi": "हमारे सभी डॉक्टर अच्छे हैं। ", "en": "All our doctors are good. "}[lg] if good else ""
    if part:
        fits = [i for i in items if part_of_day_of(i.get("start")) == part]
        word = _PART_WORD[part][lg]
        if len(fits) == 1:
            one = fits[0]
            say = {
                "bn": f"{lead}{one['name']} {word} বসেন ({one.get('start')}-{one.get('end')})। ওনার কাছেই অ্যাপয়েন্টমেন্ট করব?",
                "hi": f"{lead}{one['name']} {word} बैठते हैं ({one.get('start')}-{one.get('end')})। क्या उन्हीं के साथ अपॉइंटमेंट कर दूँ?",
                "en": f"{lead}{one['name']} sits {word} ({one.get('start')}-{one.get('end')}). Shall I book him?",
            }[lg]
            return say, one
        if not fits:
            return {
                "bn": f"{lead}{word} এদের কেউ বসেন না। এঁরা বসেন: {names}। এদের মধ্যে কার কাছে অ্যাপয়েন্টমেন্ট করব?",
                "hi": f"{lead}{word} इनमें से कोई नहीं बैठता। ये बैठते हैं: {names}। इनमें से किसके पास अपॉइंटमेंट करूँ?",
                "en": f"{lead}None of them sits {word}. They sit: {names}. Whom shall I book?",
            }[lg], None
        shown = "; ".join(_with_hours(i) for i in fits)
        return {
            "bn": f"{lead}{word} বসেন: {shown}। এদের মধ্যে কার কাছে অ্যাপয়েন্টমেন্ট করব?",
            "hi": f"{lead}{word} बैठते हैं: {shown}। इनमें से किसके पास अपॉइंटमेंट करूँ?",
            "en": f"{lead}Sitting {word}: {shown}. Whom shall I book?",
        }[lg], None
    return {
        "bn": f"{lead}এদের মধ্যে আমি কার কাছে অ্যাপয়েন্টমেন্ট করব? {names}।",
        "hi": f"{lead}इनमें से मैं किसके पास अपॉइंटमेंट करूँ? {names}।",
        "en": f"{lead}Whom among them shall I book? {names}.",
    }[lg], None


# ------------------------------------------------------------------------------------------------------------------------------ choosing one of the doctors just named

# owner, 2026-10-26: "তাহলে ওই সঞ্জয় গুহার অ্যাপয়েন্টমেন্ট টা দেন কালকে" -- the doctors "রঞ্জন পাল আর সঞ্জয় গুহ" had just been offered, and the caller chose one the way people do: with the
# Bengali "of" ending on the surname ("গুহার"). The word did not equal the name, so the choice was lost and the agent asked "which doctor?" all over again.
_NAME_TITLES = {"dr", "ডাঃ", "ডাক্তার", "ডক্টর", "डॉ", "डॉक्टर", "doctor", "mr", "mrs", "sri", "shri"}
_NAME_TAILS = ("", "র", "ার", "ের", "এর", "কে", "েকে", "কেও", "দের", "রা", "ও", "টা", "টি")


def _name_tokens(s: str) -> list[str]:
    return [t for t in _norm(s.replace(".", " ")).split() if t not in _NAME_TITLES]


def _token_is(word: str, name_token: str) -> bool:
    return word == name_token or (word.startswith(name_token) and word[len(name_token):] in _NAME_TAILS)


def choose_by_name(text: str, doctors: list[list[str]]) -> int | None:
    """The index of the ONE doctor the caller's words name, or None. `doctors[i]` are the spellings of doctor i (short name, whole name, any script). A name word counts when the caller
    said it with or without a case ending ("গুহ" / "গুহার"). Two doctors named equally well -- or none -- is None: never a guess. A two-letter word ("কর") alone is not a match, it is
    too common a word ("বুক করে দিন")."""
    words = _norm(text).split()
    scores: list[int] = []
    for spellings in doctors:
        best = 0
        for spelling in spellings:
            parts = _name_tokens(spelling or "")
            hit = [p for p in parts if len(p) >= 2 and any(_token_is(w, p) for w in words)]
            n = len(hit)
            if n == 1 and len(hit[0]) < 3 and len(parts) > 1:
                n = 0
            best = max(best, n)
        scores.append(best)
    top = max(scores, default=0)
    if top == 0:
        return _choose_by_sound(words, doctors)
    if scores.count(top) != 1:
        return None
    return scores.index(top)


def _choose_by_sound(words: list[str], doctors: list[list[str]]) -> int | None:
    """Owner, 2026-10-29 (offered Dr. সুতপা চ্যাটার্জী, the caller's "सुतापा चटर्जी" was written by the Hindi recogniser in Devanagari): a name word that matches no spelling letter for letter is still
    the offered doctor when its consonant skeleton is the same (Bengali / Devanagari / spelling variants). One doctor must be named best; a skeleton shorter than three letters never counts."""
    from agent.name_script import name_skeleton, same_given_name

    said = [(w, name_skeleton(w)) for w in words if not w.isascii()]
    said = [(w, k) for w, k in said if len(k) >= 3]
    if not said:
        return None
    scores = []
    for spellings in doctors:
        hit: set[str] = set()
        for spelling in spellings:
            for part in _name_tokens(spelling or ""):
                key = name_skeleton(part)
                if len(key) >= 3 and any(same_given_name(w, part) for w, _ in said):
                    hit.add(key)
        scores.append(len(hit))
    top = max(scores, default=0)
    if top == 0 or scores.count(top) != 1:
        return None
    return scores.index(top)
