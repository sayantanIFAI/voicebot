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
        "stems": ("neuro", "নিউরো", "নিউরোর", "नेउरो", "न्यूरो", "euro", "ইউরো", "নুরো", "ऊरो", "यूरो", "nero"),
        "mishear": ("euro", "ইউরো", "যূরো", "यूरो", "ऊरो", "nuro", "নুরো"),  # Euro / Uro read for Neuro: all three are in play
        "bare_ok": ("neurologist", "neurology", "neurosurgeon", "neurosurgery", "neuro surgeon", "নিউরোলজিস্ট", "নিউরোলজি", "নিউরোসার্জন", "নিউরো সার্জন", "নিউরোসার্জারি", "न्यूरोलॉजिस्ट", "न्यूरोसर्जन", "urologist", "urology"),
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


def sitting_pick_question(names: list[str], when_phrase: str, lang: str) -> str:
    """Several doctors sit at the asked time: name them and ask which, with a confirmation, never choosing one for the caller."""
    return {
        "bn": f"{when_phrase} এই ডাক্তাররা বসেন: {_join(names, lang)}। কার কাছে অ্যাপয়েন্টমেন্ট করব?",
        "hi": f"{when_phrase} ये डॉक्टर बैठते हैं: {_join(names, lang)}। किसके पास अपॉइंटमेंट करूँ?",
        "en": f"{when_phrase} these doctors sit: {_join(names, lang)}. Whom shall I book?",
    }[lang]


def sitting_list_reply(sitting: list[dict], when_phrase: str, dept_label: str, lang: str) -> str:
    """"Who sits in cardiology tomorrow": each doctor with the hours, from the real schedule. Never today's list for a question about tomorrow."""
    def one(s: dict) -> str:
        hours = f" ({s['start']}-{s['end']})" if s.get("start") and s.get("end") else ""
        return f"{s['name']}{hours}"

    names = [one(s) for s in sitting]
    if not names:
        return nobody_sits(when_phrase, [], lang)
    return {
        "bn": f"{when_phrase} {dept_label} বিভাগে বসছেন {_join(names, lang)}। কার কাছে অ্যাপয়েন্টমেন্ট করতে চান?",
        "hi": f"{when_phrase} {dept_label} विभाग में बैठ रहे हैं {_join(names, lang)}। किसके पास अपॉइंटमेंट चाहिए?",
        "en": f"{when_phrase} in {dept_label}: {_join(names, lang)}. Whom would you like an appointment with?",
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
