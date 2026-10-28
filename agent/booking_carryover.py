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

import re

MAX_GAP_TURNS = 3  # same distance agent/enquiry_followup.FOLLOWUP_MAX_GAP treats as "just now"

_SAME_DAY = (
    # Bengali
    "সেদিন", "ওদিন", "এদিন", "এই দিন", "ওই দিন", "সেই দিন", "ওইদিন", "এইদিন", "একই দিন", "এই তারিখ", "ওই তারিখ", "সেই তারিখ", "একই তারিখ", "ঐ দিন", "ঐ তারিখ",
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
    "আমার বুকিং", "আমার অ্যাপয়েন্টমেন্ট", "আমার appointment", "মেরা অ্যাপয়েন্টমেন্ট", "मेरा अपॉइंटमेंट", "मेरी अपॉइंटमेंट", "বুকিং কোথায়", "বুকিং চেক", "বুকিং স্ট্যাটাস", "কনফার্মেশন", "বুকিংয়ের তথ্য", "মেরী बुकिंग", "मेरी बुकिंग", "बुकिंग की स्थिति",
    "my booking", "my appointment", "booking status", "check my booking", "confirmation id", "confirmation number",
)


# Any booking word at all, once a doctor has just been asked about: "okay at the book again", "বুক তো করিনি", "অযাপ্ফোন্টমেন্ট বুক করতে বলছি".
_BOOK_WORDS = ("বুক", "অ্যাপয়েন্টমেন্ট", "অযাপ্ফোন্টমেন্ট", "ন্টমেন্ট", "বুকিং", "बुक", "बुकिंग", "अपॉइंटमेंट", "appointment", "booking")


_CHANGE_CUES = (
    "cancel", "reschedule", "postpone", "change", "shift", "move", "বাতিল", "ক্যান্সেল", "পরিবর্তন", "বদল", "পিছিয়ে", "সরিয়ে", "রিশিডিউল", "তারিখ পাল্টা",
    "रद्द", "कैंसल", "बदल", "आगे बढ़ा", "रीशेड्यूल", "पोस्टपोन",
)


def mentions_change_or_cancel(text: str) -> bool:
    return _has(_norm(text), _CHANGE_CUES)


# A bare "do it" / "take it" said right after a doctor was asked about ("এই সেদিনেরই করুন", "ওই দিনই করে দিন", "मैं ले लूँगा"): the booking goes on.
_DO_IT = (
    "করুন", "করে দিন", "করে দেন", "করে দাও", "করব", "করবো", "করতে চাই", "নেব", "নেবো", "নিতে চাই", "নিয়ে নিন", "দেখাব", "দেখাবো", "দেখাতে চাই",
    "कर दो", "कर दीजिए", "कर दें", "करना है", "करूँगा", "करूंगा", "करूँगी", "लेना है", "ले लूँगा", "ले लूंगा", "दिखाना है",
    "do it", "go ahead", "take it", "i will take", "i'll take", "i want to see",
)


# Owner, 2026-10-29 ("তো একটা নাম লিখে দিন" right after "Dr X sits tomorrow, the 30th" was answered "which doctor?"): RIGHT AFTER a doctor was talked about, every way of saying "do it" is a
# booking of THAT doctor on the day just told. People never say "book": they say write my name, put a serial, hold it, fix it, I want to see him. None of these is a booking request in a cold
# sentence (so they are used only where a doctor was just discussed, see continues_booking / carry_over); each is a phrase, never a single common word.
_COMMIT = (
    # Bengali
    "নাম লিখে", "নাম লিখুন", "নাম লেখ", "নামটা লিখে", "নামটা লেখ", "নাম লিখিয়ে", "নাম তুলে", "নাম নথিভুক্ত", "নাম ঢুকিয়ে", "নাম্বার লাগিয়ে", "নম্বর লাগিয়ে", "নাম্বার লাগান", "নম্বর লাগান",
    "সিরিয়াল দিন", "সিরিয়াল দেন", "সিরিয়াল করে", "সিরিয়াল লাগিয়ে", "সিরিয়াল লাগান", "সিরিয়াল নিয়ে", "সিরিয়াল চাই", "সিরিয়াল ধরে", "টোকেন দিন", "টোকেন দেন", "টোকেন করে", "টোকেন চাই",
    "ধরে দিন", "ধরে দেন", "ধরে রাখুন", "রেখে দিন", "রেখে দেন", "ঠিক করে দিন", "ঠিক করে দেন", "ঠিক করে রাখুন", "ফিক্স করে", "ফিক্স করুন", "রিজার্ভ", "কনফার্ম করে", "কনফার্ম করুন", "নিশ্চিত করে",
    "করে দিন", "করে দেন", "করে দাও", "করে রাখুন", "তাহলে করুন", "তাহলে দিন", "তাহলে নিন", "তাহলে রাখুন", "দেখাতে চাই", "দেখাতে যাব", "দেখাতে যাবো", "দেখাতে আসব", "দেখাতে আসবো",
    "দেখাব", "দেখাবো", "দেখাতাম", "দেখাতে এসে", "দেখা করতে চাই", "দেখা করব", "দেখা করবো", "সিরিয়ালটা", "অ্যাপয়েন্টমেন্টটা", "অ্যাপয়েন্টমেন্ট নিতে", "অ্যাপয়েন্টমেন্ট নেব", "অ্যাপয়েন্টমেন্ট নেবো",
    "নিতে চাই", "নেব", "নেবো", "নিয়ে নিন", "নিয়ে নেন", "ওই দিনেই", "ওই দিন করুন", "ওই দিন করে", "ওই দিন দিন", "এই দিনেই করুন", "ওই তারিখেই", "ওই সময়ই", "ঠিক আছে করুন", "আচ্ছা করুন", "হ্যাঁ করুন",
    # Hindi
    "नाम लिख", "नाम लिखवा", "नाम दर्ज", "नाम डाल", "नाम चढ़ा", "नाम जोड़", "सीरियल लगा", "सीरियल दे", "सीरियल कर", "सीरियल चाहिए", "टोकन दे", "टोकन चाहिए", "नंबर लगा", "नम्बर लगा",
    "रख दीजिए", "रख दो", "रख दें", "रख लीजिए", "रख लो", "फिक्स कर", "रिज़र्व", "रिजर्व", "कन्फर्म कर", "कन्फ़र्म कर", "पक्का कर", "कर दीजिए", "कर दो", "कर दें", "कर लीजिए", "कर लो", "तो कर", "तब कर",
    "दिखाना है", "दिखाना चाहता", "दिखाना चाहती", "दिखाऊंगा", "दिखाऊँगा", "दिखाऊंगी", "दिखाएंगे", "दिखाने आऊं", "दिखाने आ", "मिलना है", "मिलना चाहता", "मिलना चाहती", "अपॉइंटमेंट ले", "अपॉइंटमेंट चाहिए", "ले लीजिए", "ले लो", "ले लूं", "ले लूँ", "उसी दिन कर", "उसी दिन के लिए",
    # English
    "put my name", "write my name", "write down my name", "note my name", "add my name", "enter my name", "take my name", "register me", "sign me up", "put me down", "write me down", "note me down",
    "get me a serial", "get me a token", "get me a number", "give me a serial", "give me a token", "give me a number", "a serial for", "a token for", "reserve", "hold it", "hold a slot", "hold the slot", "fix it",
    "fix an appointment", "fix a appointment", "fix the appointment", "confirm it", "confirm that", "lock it", "go ahead", "go for it", "i want to see", "i'd like to see", "i would like to see", "i want to consult",
    "like to consult", "want to meet", "want to visit", "i will come", "i'll come", "i will take", "i'll take", "take me", "do that", "that works", "that day then", "that one then", "ok then book", "okay then book",
)


def commits_to_booking(text: str) -> bool:
    """A way of saying "do it / take my name / put me down / I want to see him" -- meaningful as a booking only after a doctor was just talked about (see the list above)."""
    t = _norm(text)
    return bool(t) and not _has(t, _LOOKUP) and not _has(t, _CHANGE_CUES) and (_has(t, _COMMIT) or _write_name_cue(text))


def _write_name_cue(text: str) -> bool:
    try:
        from agent import doctor_enquiry_cues

        return doctor_enquiry_cues.asks_to_write_name(text)
    except Exception:  # noqa: BLE001 - a cue module hiccup must not stop the booking
        return False


def continues_booking(text: str) -> bool:
    """Right after a doctor was asked about: a booking word, a pointer to the day/time just given, a bare \"do it\", or any other way of committing (commits_to_booking)."""
    t = _norm(text)
    if _has(t, _LOOKUP) or _has(t, _CHANGE_CUES):
        return False
    return asks_to_book(text) or says_same_day(text) or says_same_time(text) or _has(t, _DO_IT) or commits_to_booking(text)


# Review 2026-10-28: "don't book an appointment", "বুক করবেন না", "बुक मत करो" asked to book. A clause that names a booking AND refuses it is not a request; another clause ("but book Dr Sen")
# still is. (NOT "বুক তো করিনি" -- "I did not book" -- which is a complaint that the booking is missing, and stays a booking request.)
_NOT_BOOKING = (
    "করবেন না", "করবো না", "করব না", "করো না", "করবে না", "করতে চাই না", "চাই না", "চাইনা", "দিবেন না", "দেবেন না", "নেবেন না", "দিতে হবে না", "লাগবে না", "দরকার নেই", "প্রয়োজন নেই", "করতে হবে না", "করার দরকার নেই",
    "मत ", "नहीं करना", "नहीं करनी", "नहीं चाहिए", "नहीं चाहता", "नहीं चाहती", "ना करें", "ना करो", "जरूरत नहीं", "ज़रूरत नहीं",
    "don't", "dont", "do not", "no need", "not to book", "not book", "not want", "never", "no booking", "no appointment",
)
_CLAUSE_SPLIT = re.compile(r"\s+(?:but|however|কিন্তু|তবে|বরং|लेकिन|बल्कि)\s+|[,;।.?!]")


def _refuses(clause: str) -> bool:
    return _has(clause + " ", _NOT_BOOKING)


def asks_to_book(text: str) -> bool:
    """The caller asks for a booking (and is not asking about one they already have, nor refusing one)."""
    t = _norm(text)
    if _has(t, _LOOKUP):
        return False
    for raw in _CLAUSE_SPLIT.split((text or "").lower()):
        c = _norm(raw)
        if c and (_has(c, _BOOK_REQUEST) or any(w in c for w in _BOOK_WORDS) or _has(c, ("book",))) and not _refuses(c):
            return True
    return False


def recent(ctx: dict | None, turn_count: int) -> bool:
    """A doctor was just asked about (within MAX_GAP_TURNS turns)."""
    return bool(ctx and ctx.get("doctor_name") and turn_count - ctx.get("turn", -99) <= MAX_GAP_TURNS)


# "ওই উনিশ তারিখেই" / "उसी उन्नीस तारीख" / "that 19th": a pointing word, one word (the day number), then 'date' -- the date the agent just gave
_POINTED_DATE = re.compile(r"(?:ওই|এই|সেই|ঐ|उसी|उस|इसी|इस|वही|that|same|this)\s+\S+\s+(?:তারিখ|तारीख|तारिख|date)")


def says_same_day(text: str) -> bool:
    t = _norm(text)
    return _has(t, _SAME_DAY) or bool(_POINTED_DATE.search(t))


def points_at_doctor(text: str) -> bool:
    return _has(_norm(text), _HIM_HER)


def carry_over(ctx: dict | None, slots: dict, text: str, booking_slots: dict | None, turn_count: int, time_said: bool = False) -> dict:
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
    # A commitment ("নাম লিখে দিন", "put my name", "ঠিক করে দিন") with no day of its own is for the day the agent JUST told -- the doctor's sitting day, with the start of his hours.
    if commits_to_booking(text) and ctx.get("date") and not out.get("date") and not booked.get("date") and not slots.get("time_slot"):
        out["date"] = ctx["date"]
        start = window_start(ctx.get("window"))
        if start and not out.get("time_slot") and not booked.get("time_slot"):
            out["time_slot"] = start
    if (wants_day or wants_time) and ctx.get("date") and not out.get("date") and not booked.get("date"):
        out["date"] = ctx["date"]
    elif wants_day and ctx.get("date"):
        # the caller POINTED at the date just stated; a different date the model read out of the same sentence (2026-10-17 for "that 19th") is its error
        out["date"] = ctx["date"]
    # Real call 2026-10-17: the agent said "Dr Sen sits on the 19th, 10-12. Which day?" and the caller answered with a TIME only ("আরে হৈদিনি করে দিন
    # এগারোটার সময়", the day-pointer garbled): the only day the caller can mean is the one just given.
    if (time_said or out.get("time_slot")) and ctx.get("date") and not out.get("date") and not booked.get("date"):
        out["date"] = ctx["date"]
    if wants_time and not out.get("time_slot") and not booked.get("time_slot"):
        start = window_start(ctx.get("window"))
        if start:
            out["time_slot"] = start
    return out
