"""'Is the CT scan done on Thursday?' / 'which days is it done?' -- answered from the lab table's days column, never from the model.

sourav, 2026-10-24: the owner added a days column to the lab-test table (clinic-api models.LabTest.available_days: Monday = 0 ... Sunday = 6). A test done "every day
except Thursday" must be told that way: the caller who asks about Thursday hears no, and the days it IS done. The truth boundary (CLAUDE.md) is untouched: the days come from
the clinic API's answer for the test the catalogue found; this module only decides that a sentence is such a question and words the reply.

Only a QUESTION about days matches. A weekday beside a booking, price, preparation or report word is left to those flows (same exclusions as test_availability), and a
sentence about a doctor is the doctor-weekday flow's (main.py only claims it when a lab test is actually named in the sentence).
"""

from __future__ import annotations

from agent import test_availability as _ta
from agent import weekday_cues

# a phrase that asks WHICH days, with no day named
_WHICH_DAYS = (
    "which day", "which days", "what day", "what days", "on which day", "days is it", "days are", "any day", "every day", "everyday", "all days", "all day", "daily",
    "kon din", "kon kon din", "kon kon bar", "kon bar", "kon din hoy", "kono din", "roj", "roz", "rojh", "protidin", "pratidin", "pratidin",
    "কোন দিন", "কোন কোন দিন", "কোন কোন বার", "কোন বার", "কোন দিনে", "কোন কোন দিনে", "প্রতিদিন", "রোজ", "সব দিন", "কোনো দিন", "কত দিন",
    "kaun se din", "kaun kaun se din", "kis din", "kis kis din", "kaun sa din", "roz", "roz hota", "har din", "sabhi din", "sab din",
    "कौन से दिन", "कौन कौन से दिन", "किस दिन", "किस किस दिन", "कौन सा दिन", "रोज़", "रोज", "हर दिन", "सभी दिन", "सब दिन", "रोजाना",
)  # fmt: skip
# the sentence asks whether the test is DONE / possible (the tails test_availability already knows, plus the ones that go with a day word)
_DONE_WORDS = (
    "done", "available", "possible", "open", "happen", "happens", "do you do", "is it", "can i", "can we", "will it", "does it", "get it", "get done",
    "hoy", "hoi", "hobe", "hoye", "hoyna", "jay", "jabe", "kora", "kore", "paoa", "pawa", "pabo", "ache", "achhe", "bondho", "bandh", "khola", "khule",
    "hota", "hote", "hoti", "hoga", "hogi", "hota hai", "milta", "milega", "ho jata", "karte", "kar sakte", "uplabdh", "hai",
    "হয়", "হয", "হবে", "করা", "যায়", "যাবে", "আছে", "পাওয়া", "বন্ধ", "খোলা", "করেন", "করানো",
    "होता", "होती", "होते", "होगा", "होगी", "मिलता", "मिलेगा", "उपलब्ध", "करते", "सकते", "है", "हो",
)  # fmt: skip

_DAY_NAMES = {
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
    "bn": ("সোমবার", "মঙ্গলবার", "বুধবার", "বৃহস্পতিবার", "শুক্রবার", "শনিবার", "রবিবার"),
    "hi": ("सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"),
}


def asks_test_days(text: str) -> bool:
    """sourav, 2026-10-24: True for a question about the DAYS a test is done ("is it done on Thursday?", "kon kon din hoy?", "किस दिन होता है?"). Why: a sentence with a weekday or
    "which days" in it must reach the days answer instead of the generic availability/price paths. Flow: the sentence must not be a booking / price / preparation / report
    request (test_availability's exclusions); then either it names a weekday AND asks whether something is done, or it asks "which days" outright."""
    t = _ta._norm(text)
    if not t or len(t.split()) > 22 or _ta._contains_cue(t, _ta._NOT_AVAILABILITY):
        return False
    if _ta._contains_cue(t, tuple(w for w in weekday_cues._DOCTOR_WORDS)) or any(tok.startswith(weekday_cues._DOCTOR_STEMS) for tok in t.split()):
        return False  # a doctor's day is weekday_cues' / doctor_enquiry_cues' question
    if _ta._contains_cue(t, _WHICH_DAYS):
        return True
    return weekday_cues.weekday_of(text) is not None and _ta._contains_cue(t, _DONE_WORDS)


def named_weekday(text: str) -> int | None:
    """sourav, 2026-10-24: the weekday (Monday = 0) the caller named, or None. Why: the answer says yes/no for THAT day. Flow: weekday_cues' own reader, so every spelling a
    recogniser writes for a weekday (robibar, ravivar, রবিবার, रविवार ...) works here too."""
    return weekday_cues.weekday_of(text)


def _join(items: list[str], lang: str) -> str:
    if len(items) <= 1:
        return "".join(items)
    sep = {"bn": " আর ", "hi": " और "}.get(lang, " and ")
    return ", ".join(items[:-1]) + sep + items[-1]


def day_names(days: list[int], lang: str) -> str:
    """sourav, 2026-10-24: weekday numbers -> the spoken list in the call's language ("Monday, Wednesday and Friday")."""
    names = _DAY_NAMES.get(lang, _DAY_NAMES["en"])
    return _join([names[d] for d in days if 0 <= d <= 6], lang)


def _when_phrase(open_days: list[int], closed: list[int], lang: str) -> str:
    """sourav, 2026-10-24: the days a test IS done, said the shorter way: "every day except Thursday" when at most three days are closed, else "on Monday, Wednesday and Friday"."""
    if len(closed) <= 3:
        c = day_names(closed, lang)
        return {"bn": f"{c} ছাড়া প্রতিদিন", "hi": f"{c} को छोड़कर रोज़"}.get(lang, f"every day except {c}")
    o = day_names(open_days, lang)
    return {"bn": f"{o}-এ", "hi": f"{o} को"}.get(lang, f"on {o}")


def days_reply(spoken_name: str, open_days: list[int], lang: str, asked_weekday: int | None = None) -> str:
    """sourav, 2026-10-24: the spoken answer, built only from the table's days. Why: this is a fact about the clinic, so it is a template over the API's answer (CLAUDE.md's one
    rule). Flow: every day -> "done every day" (a named day is then a plain yes); a named day -> yes/no for that day, and on a no the days it IS done; no day named -> the days."""
    lang = lang if lang in ("bn", "hi") else "en"
    names = _DAY_NAMES[lang]
    open_days = sorted(set(open_days)) or [0, 1, 2, 3, 4, 5, 6]
    closed = [d for d in range(7) if d not in open_days]
    every = not closed
    if asked_weekday is not None:
        day = names[asked_weekday]
        if asked_weekday in open_days:
            yes = {"bn": f"হ্যাঁ, {spoken_name} {day}ে হয়।", "hi": f"हाँ, {spoken_name} {day} को होता है।", "en": f"Yes, {spoken_name} is done on {day}."}[lang]
            return yes if every else yes + " " + {"bn": f"এটা {_when_phrase(open_days, closed, lang)} হয়।", "hi": f"यह {_when_phrase(open_days, closed, lang)} होता है।", "en": f"It is done {_when_phrase(open_days, closed, lang)}."}[lang]
        no = {"bn": f"না, {spoken_name} {day}ে হয় না।", "hi": f"नहीं, {spoken_name} {day} को नहीं होता।", "en": f"No, {spoken_name} is not done on {day}."}[lang]
        return no + " " + {"bn": f"এটা {_when_phrase(open_days, closed, lang)} হয়।", "hi": f"यह {_when_phrase(open_days, closed, lang)} होता है।", "en": f"It is done {_when_phrase(open_days, closed, lang)}."}[lang]
    if every:
        return {"bn": f"{spoken_name} প্রতিদিনই হয়।", "hi": f"{spoken_name} रोज़ होता है।", "en": f"{spoken_name} is done every day."}[lang]
    return {"bn": f"{spoken_name} {_when_phrase(open_days, closed, lang)} হয়।", "hi": f"{spoken_name} {_when_phrase(open_days, closed, lang)} होता है।", "en": f"{spoken_name} is done {_when_phrase(open_days, closed, lang)}."}[lang]
