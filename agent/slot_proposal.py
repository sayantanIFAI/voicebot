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


def pick(free_slots: list[str], date_iso: str, now: datetime.datetime | None = None, lead_minutes: int = LEAD_MINUTES) -> str | None:
    """The first free slot, and on today's date the first one that starts at least `lead_minutes` from now. None when there is none."""
    now = now or clinic_now()
    slots = sorted(set(free_slots or []), key=_minutes)
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
    "bn": "{time}-এ আপনার জন্য স্লট বুক করতে পারি। দয়া করে {early} মিনিট আগে চলে আসবেন।",
    "hi": "{time} बजे आपके लिए स्लॉट बुक कर सकते हैं। कृपया {early} मिनट पहले आ जाइएगा।",
    "en": "I can book a slot for you at {time}. Please come {early} minutes early.",
}


def day_line(doctor: str, date_iso: str, lang: str, now: datetime.datetime | None = None) -> str:
    lang = lang if lang in _DAY else "en"
    return _DAY[lang][day_kind(date_iso, now)].format(doc=doctor, date=date_iso)


def proposal(doctor: str, date_iso: str, slot: str, lang: str, now: datetime.datetime | None = None) -> str:
    lang = lang if lang in _DAY else "en"
    return f"{day_line(doctor, date_iso, lang, now)} " + _SLOT[lang].format(time=slot, early=COME_EARLY_MINUTES)
