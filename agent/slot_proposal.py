"""The machine picks the time; the caller is not asked for one.

Real call 2026-10-17: after "yes, Dr Sen sits today 10:00-12:00" the AI asked "which time do you want the appointment?" four times. The caller cannot
know which slots are taken, so the question has no good answer ("any time you can", "between ten and twelve", "from where should I say a time").
The AI knows the doctor's sitting hours, the slots already booked and the time now, so it tells the caller directly, e.g. "at 10:45 we can
book a slot for you -- please come 15 minutes early". The slot is the first FREE one that leaves time to get to the clinic; the caller can
still ask for a different time and the usual hold check then judges it.

Also the day is said the way people say it: "today" for today, "tomorrow, 18 October" for tomorrow, the date otherwise.
"""

from __future__ import annotations

import datetime

try:  # the clinic's own clock: a Kolkata clinic, whatever time zone the server runs in
    from zoneinfo import ZoneInfo

    CLINIC_TZ = ZoneInfo("Asia/Kolkata")
except Exception:  # noqa: BLE001 - no tz database: fall back to the machine's local time
    CLINIC_TZ = None

LEAD_MINUTES = 30  # a same-day slot must start at least this long from now (time to get there; the caller is also asked to come 15 min early)
COME_EARLY_MINUTES = 15


def clinic_now() -> datetime.datetime:
    return datetime.datetime.now(CLINIC_TZ) if CLINIC_TZ else datetime.datetime.now()


def _minutes(slot: str) -> int:
    h, m = slot.split(":")
    return int(h) * 60 + int(m)


def pick(free_slots: list[str], date_iso: str, now: datetime.datetime | None = None, lead_minutes: int = LEAD_MINUTES, skip=()) -> str | None:
    """The first free slot, and on today's date the first one that starts at least `lead_minutes` from now. None when there is none."""
    now = now or clinic_now()
    slots = sorted(set(free_slots or []) - set(skip or ()), key=_minutes)
    if not slots:
        return None
    if date_iso == now.date().isoformat():
        earliest = now.hour * 60 + now.minute + lead_minutes
        slots = [s for s in slots if _minutes(s) >= earliest]
    return slots[0] if slots else None


def day_kind(date_iso: str, now: datetime.datetime | None = None) -> str:
    """'today' | 'tomorrow' | 'other'."""
    today = (now or clinic_now()).date()
    try:
        d = datetime.date.fromisoformat(date_iso)
    except ValueError:
        return "other"
    if d == today:
        return "today"
    if d == today + datetime.timedelta(days=1):
        return "tomorrow"
    if d == today + datetime.timedelta(days=2):
        return "day_after"
    return "other"


_DAY = {
    "bn": {
        "today": "ডাক্তার {doc} আজই বসবেন।",
        "tomorrow": "ডাক্তার {doc} আগামীকাল, {date} তারিখে বসবেন।",
        "other": "ডাক্তার {doc} {date} তারিখে বসবেন।",
    },
    "hi": {
        "today": "डॉक्टर {doc} आज ही बैठेंगे।",
        "tomorrow": "डॉक्टर {doc} कल, {date} तारीख को बैठेंगे।",
        "other": "डॉक्टर {doc} {date} तारीख को बैठेंगे।",
    },
    "en": {
        "today": "Doctor {doc} sits today.",
        "tomorrow": "Doctor {doc} sits tomorrow, {date}.",
        "other": "Doctor {doc} sits on {date}.",
    },
}
_SLOT = {
    # a statement, not a question: the very next thing asked is the caller's name and number (owner 2026-10-17)
    # a QUESTION: the caller says yes (or another time) BEFORE the name and number are asked (owner 2026-10-17)
    "bn": "{time}-এ আপনার জন্য স্লট রাখছি, ঠিক আছে? দয়া করে {early} মিনিট আগে চলে আসবেন।",
    "hi": "{time} बजे आपके लिए स्लॉट रख रही हूँ, ठीक है? कृपया {early} मिनट पहले आ जाइएगा।",
    "en": "I am holding the {time} slot for you -- is that all right? Please come {early} minutes early.",
}


_NONE_LEFT = {
    "bn": "দুঃখিত, এই দিনের জন্য আর কোনো স্লট নেই।",
    "hi": "माफ़ कीजिए, इस दिन के लिए कोई स्लॉट नहीं बचा है।",
    "en": "Sorry, there is no slot left for that day.",
}


_NOT_SITTING = {
    "bn": "ওই দিন ডাক্তার বসেন না।",
    "hi": "उस दिन डॉक्टर नहीं बैठते।",
    "en": "The doctor does not sit on that day.",
}


def not_sitting(lang: str) -> str:
    return _NOT_SITTING.get(lang, _NOT_SITTING["en"])


def none_left(lang: str) -> str:
    return _NONE_LEFT.get(lang, _NONE_LEFT["en"])


_WHEN = {
    # the day after tomorrow: the DATE first, then the word ("পরশু" / "परसों"), owner 2026-10-17
    "bn": {"today": "আজই", "tomorrow": "আগামীকাল, {date} তারিখে", "day_after": "{date} তারিখে, পরশু", "other": "{date} তারিখে"},
    "hi": {"today": "आज ही", "tomorrow": "कल, {date} तारीख को", "day_after": "{date} को, परसों", "other": "{date} को"},
    "en": {"today": "today", "tomorrow": "tomorrow, {date}", "day_after": "on {date}, the day after tomorrow", "other": "on {date}"},
}


def when_phrase(date_iso: str | None, lang: str, now: datetime.datetime | None = None) -> str:
    """The day as a caller says it: "আজই" / "আগামীকাল, <date> তারিখে" / "<date> তারিখে" (Hindi "आज ही" / "कल, <date> तारीख को", English "today" / "tomorrow, <date>")."""
    lang = lang if lang in _WHEN else "en"
    if not date_iso:
        return _WHEN[lang]["today"]
    return _WHEN[lang][day_kind(date_iso, now)].format(date=date_iso)


_RECONFIRM = {
    "bn": "ঠিক আছে, {time}। এই সময়টা ঠিক আছে তো?",
    "hi": "ठीक है, {time} बजे। यह समय ठीक है न?",
    "en": "All right, {time}. Is that time fine?",
}


def reconfirm(time_slot: str, lang: str) -> str:
    return _RECONFIRM.get(lang, _RECONFIRM["en"]).format(time=time_slot)


def day_line(doctor: str, date_iso: str, lang: str, now: datetime.datetime | None = None) -> str:
    lang = lang if lang in _DAY else "en"
    if day_kind(date_iso, now) == "day_after":
        when = when_phrase(date_iso, lang, now)
        return {"bn": f"ডাক্তার {doctor} {when} বসবেন।", "hi": f"डॉक्टर {doctor} {when} बैठेंगे।", "en": f"Doctor {doctor} sits {when}."}[lang]
    return _DAY[lang][day_kind(date_iso, now)].format(doc=doctor, date=date_iso)


# Owner, 2026-10-18: the caller is NOT given a ten-minute slot. They are told when the doctor SITS and to come then, with one to two hours in hand
# (tokens are given in the order of payment, so nobody can be told an exact minute). The slot is still held internally.
_SITS = {
    "bn": "ডাক্তার {doc} {when} {time}-এ বসবেন। আপনি ওই সময়ে চলে আসুন, হাতে এক থেকে দুই ঘণ্টা সময় নিয়ে আসবেন। ঠিক আছে?",
    "hi": "डॉक्टर {doc} {when} {time} बजे बैठेंगे। आप उसी समय आ जाइए, हाथ में एक से दो घंटे का समय लेकर आइए। ठीक है?",
    "en": "Doctor {doc} sits {when} at {time}. Please come at that time, with one to two hours in hand. Is that all right?",
}


def proposal(doctor: str, date_iso: str, slot: str, lang: str, now: datetime.datetime | None = None, sits_at: str | None = None) -> str:
    """What the caller hears: the day and the hour the doctor SITS (`sits_at`, the start of the chamber hours; the held `slot` when that is not known),
    and to come then with one to two hours in hand."""
    lang = lang if lang in _DAY else "en"
    when = when_phrase(date_iso, lang, now)
    return _SITS[lang].format(doc=doctor, when=when, time=sits_at or slot)


_CONCLUDE = {
    "bn": "ডাক্তার {doc} {when} {time}-এ বসবেন। আপনি ওই সময়ে চলে আসুন, হাতে এক থেকে দুই ঘণ্টা সময় নিয়ে আসবেন।",
    "hi": "डॉक्टर {doc} {when} {time} बजे बैठेंगे। आप उसी समय आ जाइए, हाथ में एक से दो घंटे का समय लेकर आइए।",
    "en": "Doctor {doc} sits {when} at {time}. Please come at that time, with one to two hours in hand.",
}


def conclusion(doctor: str, date_iso: str, time: str, lang: str, now: datetime.datetime | None = None) -> str:
    """The same sentence as the proposal, without the question: said at the END of the booking to conclude it (owner, 2026-10-18)."""
    lang = lang if lang in _CONCLUDE else "en"
    doctor = doctor.replace("Dr. ", "", 1).replace("Dr ", "", 1).replace("ডাঃ ", "", 1).replace("ডাক্তার ", "", 1).replace("डॉ. ", "", 1)  # the sentence already says "Doctor"
    return _CONCLUDE[lang].format(doc=doctor, when=when_phrase(date_iso, lang, now), time=time)


_PAY_NOTICE = {
    "bn": "আপনার হোয়াটসঅ্যাপে একটা পেমেন্ট লিংক পাঠানো হবে। পনেরো মিনিটের মধ্যে চল্লিশ টাকা পেমেন্ট করবেন, না হলে লিংকটা নিজে থেকেই বাতিল হয়ে যাবে। ",
    "hi": "आपके व्हाट्सऐप पर एक पेमेंट लिंक भेजा जाएगा। पंद्रह मिनट के अंदर चालीस रुपये भुगतान कीजिए, वरना लिंक अपने आप बंद हो जाएगा। ",
    "en": "A payment link will be sent to your WhatsApp. Please pay forty rupees within fifteen minutes, otherwise the link will expire automatically. ",
}
_ARRIVAL = {
    "bn": "{when} {time}-এ আপনি ওই সময়ে চলে আসুন, হাতে এক থেকে দুই ঘণ্টা সময় নিয়ে আসবেন। ঠিক আছে?",
    "hi": "{when} {time} बजे आप उसी समय आ जाइए, हाथ में एक से दो घंटे का समय लेकर आइए। ठीक है?",
    "en": "{when} at {time}, please come at that time, with one to two hours in hand. Is that all right?",
}


def arrival_question(date_iso: str, time: str, lang: str, now: datetime.datetime | None = None) -> str:
    """The LAST thing asked before a doctor booking is written (owner, 2026-10-18), after the name, the number, the age and the address have each been taken and
    confirmed one at a time: the day, the hour, and to come then with one to two hours in hand. The caller's "ঠিক আছে" books it."""
    lang = lang if lang in _ARRIVAL else "en"
    # Owner, 2026-10-18: first what to pay and by when (the link goes to WhatsApp), THEN the day, the hour and "come with one to two hours in hand".
    return _PAY_NOTICE[lang] + _ARRIVAL[lang].format(when=when_phrase(date_iso, lang, now), time=time)


_CALL_AGAIN = {
    "bn": "দুঃখিত, এই ডাক্তারের জন্য এখন কোনো ফাঁকা দিন বা সময় নেই। দয়া করে পরের তারিখের কথা জানতে আবার ফোন করবেন।",
    "hi": "माफ़ कीजिए, इस डॉक्टर के लिए अभी कोई खाली दिन या समय नहीं है। कृपया अगली तारीख के बारे में पूछने के लिए दोबारा फ़ोन कीजिए।",
    "en": "Sorry, there is no free day or time for this doctor right now. Please call again to ask about the next date.",
}


def call_again(lang: str) -> str:
    """Owner, 2026-10-18: when no date and time is available for the doctor, the patient is asked to call again to ask about the next date."""
    return _CALL_AGAIN.get(lang, _CALL_AGAIN["en"])
