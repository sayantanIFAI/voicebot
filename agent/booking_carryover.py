"""sourav, 2026-10-14: "book HIM / her / that doctor, for THAT day" straight after a doctor-availability answer.

WHY: a caller asks "is Dr. Sen sitting tomorrow?", hears "yes, 10 to 12", and says "unar appointment korun, aye din" / "उनका अपॉइंटमेंट
उसी दिन करो" / "book an appointment with him for that day". The model sees only the CURRENT sentence, so it extracts a booking with NO doctor and NO date;
the flow then asked "which doctor?" again, or worse silently defaulted to General Medicine. This module is the deterministic fix, in the
same spirit as agent/enquiry_followup.py: it decides only WHICH doctor / date the caller is pointing at from what was just answered. The
doctor is still validated against the clinic API by the normal booking flow, and the reply always names the doctor and date, so a wrong
assumption is heard and corrected.
FLOW: _finish_enquiry_turn remembers {doctor_name, date, turn} after a found doctor_availability answer -> a book_appointment turn within
FOLLOWUP turns that names no doctor gets doctor_name from it, and the date too when the caller said "that day / aye din / उसी दिन".
"""

from __future__ import annotations

MAX_GAP_TURNS = 3  # same distance agent/enquiry_followup.FOLLOWUP_MAX_GAP treats as "just now"

_SAME_DAY = (
    # Bengali
    "এই দিন", "ওই দিন", "সেই দিন", "ওইদিন", "এইদিন", "একই দিন", "এই তারিখ", "ওই তারিখ", "সেই তারিখ", "একই তারিখ", "ঐ দিন", "ঐ তারিখ",
    # Hindi
    "उसी दिन", "उस दिन", "इसी दिन", "इस दिन", "वही दिन", "उसी तारीख", "इसी तारीख", "उस तारीख", "इस तारीख", "वही तारीख", "उसी डेट", "उस डेट",
    # English
    "that day", "same day", "this day", "that date", "same date", "this date", "that same day", "for the same day",
    # romanised (Banglish / Hinglish)
    "aye din", "ayi din", "oi din", "ei din", "shei din", "oi tarik", "ei tarik", "aye tarik", "ayi tarik", "shei tarik", "ekoi din",
    "usi din", "us din", "isi din", "is din", "wahi din", "usi tareekh", "usi tarikh", "us tareekh", "same din", "usi date", "us date",
)
_HIM_HER = (
    "তাঁর", "উনার", "ওনার", "উনি", "ওঁর", "তার", "ওনাকে", "উনাকে", "ওই ডাক্তার", "এই ডাক্তার", "সেই ডাক্তার",
    "उनका", "उनकी", "उनके", "उन्हें", "इनका", "उन डॉक्टर", "उस डॉक्टर", "इस डॉक्टर", "वही डॉक्टर",
    "him", "her", "his", "that doctor", "this doctor", "the same doctor", "same doctor", "them",
    "unar", "onar", "unake", "onake", "oi doctor", "ei doctor", "oi daktar", "ei daktar", "unka", "unki", "unke", "unhe", "usi doctor", "wahi doctor",
)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("।", " ").replace(",", " ").replace("?", " ").replace(".", " ").split())


def _has(text: str, cues: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any((f" {c} " in padded) if c.isascii() else (c in text) for c in cues)


def says_same_day(text: str) -> bool:
    return _has(_norm(text), _SAME_DAY)


def points_at_doctor(text: str) -> bool:
    return _has(_norm(text), _HIM_HER)


def carry_over(ctx: dict | None, slots: dict, text: str, booking_slots: dict | None, turn_count: int) -> dict:
    """The booking `slots` with doctor_name (and date) filled from the doctor just asked about, or `slots` unchanged.

    Rules: nothing remembered, too long ago, a doctor already named this turn or already in the booking, or the booking already
    has a doctor -> unchanged. The date is added only when the caller said "that day" (or similar) and no date was given."""
    if not ctx or not ctx.get("doctor_name"):
        return slots
    if turn_count - ctx.get("turn", -99) > MAX_GAP_TURNS:
        return slots
    if slots.get("doctor_name") or (booking_slots or {}).get("doctor_name"):
        return slots
    out = dict(slots)
    out["doctor_name"] = ctx["doctor_name"]
    if not out.get("date") and not (booking_slots or {}).get("date") and ctx.get("date") and says_same_day(text):
        out["date"] = ctx["date"]
    return out
