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


_SAME_TIME = (
    # Bengali
    "ওই সময়", "এই সময়", "সেই সময়", "ওই সময়ে", "এই সময়ে", "সেই সময়ে", "ঐ সময়", "একই সময়", "ওই টাইম", "ওই টাইমে", "সেই টাইমে", "ওই সময়েই",
    # Hindi
    "उसी समय", "उस समय", "इसी समय", "वही समय", "उसी टाइम", "उस टाइम", "इस टाइम", "वही टाइम",
    # English
    "that time", "same time", "this time", "at that time", "that slot", "same slot", "that timing",
    # romanised
    "oi somoy", "ei somoy", "shei somoy", "oi time", "usi time", "us time", "isi time", "usi samay", "us samay",
)


def says_same_time(text: str) -> bool:
    return _has(_norm(text), _SAME_TIME)


def window_start(window) -> str | None:
    """"10:00-12:00" / "10:00 AM - 12:00 PM" -> "10:00" (the first time in the sitting hours), or None."""
    import re

    m = re.search(r"(\d{1,2})[:.](\d{2})", str(window or ""))
    return f"{int(m.group(1)):02d}:{m.group(2)}" if m else None


def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("।", " ").replace(",", " ").replace("?", " ").replace(".", " ").split())


def _has(text: str, cues: tuple[str, ...]) -> bool:
    padded = f" {text} "
    return any((f" {c} " in padded) if c.isascii() else (c in text) for c in cues)


_BOOK_REQUEST = (
    "বুকিং দিন", "বুকিং দিয়ে", "বুকিং করে দিন", "বুকিং করুন", "বুকিং নিন", "বুকিং নেন", "বুকিং করে দেন", "বুক করে দিন", "বুক করুন", "বুক করে দেন",
    "অ্যাপয়েন্টমেন্ট দিন", "অ্যাপয়েন্টমেন্ট দিয়ে", "অ্যাপয়েন্টমেন্ট করে দিন", "অ্যাপয়েন্টমেন্ট করুন", "বুকিং চাই", "অ্যাপয়েন্টমেন্ট চাই", "একটা বুকিং",
    "बुकिंग कर दो", "बुकिंग कर दीजिए", "बुक कर दो", "बुक कर दीजिए", "अपॉइंटमेंट दे दो", "अपॉइंटमेंट दीजिए", "अपॉइंटमेंट चाहिए", "बुकिंग चाहिए", "बुकिंग कर दें",
    "book it", "book me", "make a booking", "make an appointment", "book an appointment", "book that", "give me a booking", "give me an appointment",
    "i want a booking", "i want an appointment",
)
_LOOKUP = (
    "আমার বুকিং", "বুকিং কোথায়", "বুকিং চেক", "বুকিং স্ট্যাটাস", "কনফার্মেশন", "বুকিংয়ের তথ্য", "মেরী बुकिंग", "मेरी बुकिंग", "बुकिंग की स्थिति",
    "my booking", "my appointment", "booking status", "check my booking", "confirmation id", "confirmation number",
)


def asks_to_book(text: str) -> bool:
    """The caller asks for a booking (and is not asking about one they already have)."""
    t = _norm(text)
    return _has(t, _BOOK_REQUEST) and not _has(t, _LOOKUP)


def recent(ctx: dict | None, turn_count: int) -> bool:
    """A doctor was just asked about (within MAX_GAP_TURNS turns)."""
    return bool(ctx and ctx.get("doctor_name") and turn_count - ctx.get("turn", -99) <= MAX_GAP_TURNS)


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
    booked = booking_slots or {}
    out = dict(slots)
    # the doctor only when none is named this turn or already in the booking
    if not slots.get("doctor_name") and not booked.get("doctor_name"):
        out["doctor_name"] = ctx["doctor_name"]
    # Real call 2026-10-17: after "yes, Dr Sen sits on 2026-10-17, 10:00-12:00", "ওই সময় একটা বুকিং দিয়ে দেন" ("book me at that time") was asked
    # "which day?", and a second "ওই দিনেরই ওই সময়েই বুকিং নেন" ("that same day, that same time") changed nothing because the doctor was by then
    # already in the booking. The caller is pointing at what the agent JUST said: "that day" is its date, "that time" is its date AND its sitting
    # hours (the start of them), whether or not the doctor was already carried over.
    wants_day, wants_time = says_same_day(text), says_same_time(text)
    if (wants_day or wants_time) and ctx.get("date") and not out.get("date") and not booked.get("date"):
        out["date"] = ctx["date"]
    if wants_time and not out.get("time_slot") and not booked.get("time_slot"):
        start = window_start(ctx.get("window"))
        if start:
            out["time_slot"] = start
    return out
