"""How a date is SAID to the caller (owner, 2026-10-24; credit: sourav).

Why: the clinic's data carries dates as 2026-10-18, and a template that drops that string in front of the caller makes the voice read digits ("two zero two six ...") or
a date format. A person says it by what is nearest in time, with no format and no month when the month is understood:
    same month        18 October  ->  "আঠারো তারিখ"          / "अठारह तारीख"             / "the 18th"
    the month after   2 September ->  "সামনের মাসের দুই তারিখ"  / "अगले महीने की दो तारीख"      / "the 2nd of next month"
    the month before  25 September ->  "গত মাসের পঁচিশ তারিখ"   / "पिछले महीने की पच्चीस तारीख"   / "the 25th of last month"
    further away      3 January   ->  "জানুয়ারির তিন তারিখ"   / "जनवरी की तीन तारीख"        / "the 3rd of January"
(the year is added only when it is not the current or the next one.)  Today / tomorrow / the day after stay words in slot_proposal; they only change what follows them.

`speak_dates(text, lang)` is the one chokepoint: every ISO date inside a reply is rewritten just before it is spoken (main._speak), so no template has to remember to, and a
reply with no ISO date is returned untouched. The number is the clinic's own date -- only its wording changes. Pure functions, no I/O.
"""

from __future__ import annotations

import datetime
import re

_BN_NUM = ["", "এক", "দুই", "তিন", "চার", "পাঁচ", "ছয়", "সাত", "আট", "নয়", "দশ", "এগারো", "বারো", "তেরো", "চোদ্দ", "পনেরো", "ষোলো", "সতেরো", "আঠারো", "উনিশ",
           "বিশ", "একুশ", "বাইশ", "তেইশ", "চব্বিশ", "পঁচিশ", "ছাব্বিশ", "সাতাশ", "আটাশ", "ঊনত্রিশ", "ত্রিশ", "একত্রিশ"]
_HI_NUM = ["", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ", "दस", "ग्यारह", "बारह", "तेरह", "चौदह", "पंद्रह", "सोलह", "सत्रह", "अठारह", "उन्नीस",
           "बीस", "इक्कीस", "बाईस", "तेईस", "चौबीस", "पच्चीस", "छब्बीस", "सत्ताईस", "अट्ठाईस", "उनतीस", "तीस", "इकतीस"]
_BN_MONTH_OF = ["", "জানুয়ারির", "ফেব্রুয়ারির", "মার্চের", "এপ্রিলের", "মে মাসের", "জুনের", "জুলাইয়ের", "আগস্টের", "সেপ্টেম্বরের", "অক্টোবরের", "নভেম্বরের", "ডিসেম্বরের"]
_HI_MONTH = ["", "जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]
_EN_MONTH = ["", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _month_gap(d: datetime.date, today: datetime.date) -> int:
    return (d.year - today.year) * 12 + (d.month - today.month)


def core(date_iso: str, lang: str, today: datetime.date | None = None) -> str | None:
    """The date as said WITHOUT the word for 'date' (bn/hi): "আঠারো" / "সামনের মাসের দুই" / "अठারह" ...; English is the whole phrase ("the 18th", "the 2nd of next month").
    None when `date_iso` is not a real date."""
    try:
        d = datetime.date.fromisoformat(date_iso)
    except (TypeError, ValueError):
        return None
    today = today or datetime.date.today()
    gap = _month_gap(d, today)
    far = abs(gap) > 1
    year_note = d.year not in (today.year, today.year + 1) if far else False
    if lang == "bn":
        n = _BN_NUM[d.day]
        if gap == 0:
            return n
        if gap == 1:
            return f"সামনের মাসের {n}"
        if gap == -1:
            return f"গত মাসের {n}"
        return f"{_BN_MONTH_OF[d.month]} {n}" if not year_note else f"{d.year} সালের {_BN_MONTH_OF[d.month]} {n}"
    if lang == "hi":
        n = _HI_NUM[d.day]
        if gap == 0:
            return n
        if gap == 1:
            return f"अगले महीने की {n}"
        if gap == -1:
            return f"पिछले महीने की {n}"
        return f"{_HI_MONTH[d.month]} की {n}" if not year_note else f"{d.year} {_HI_MONTH[d.month]} की {n}"
    o = _ordinal(d.day)
    if gap == 0:
        return f"the {o}"
    if gap == 1:
        return f"the {o} of next month"
    if gap == -1:
        return f"the {o} of last month"
    return f"the {o} of {_EN_MONTH[d.month]}" + (f" {d.year}" if year_note else "")


def say(date_iso: str, lang: str, today: datetime.date | None = None) -> str:
    """The whole spoken date with the word for 'date': "আঠারো তারিখ" / "अठारह तारीख" / "the 18th". An unreadable date is returned as it came."""
    c = core(date_iso, lang, today)
    if c is None:
        return date_iso
    return f"{c} তারিখ" if lang == "bn" else f"{c} तारीख" if lang == "hi" else c


_ISO = re.compile(r"(?<![\w-])(\d{4})-(\d{2})-(\d{2})(?![\w-])")
_BN_WORD = re.compile(r"\s*(তারিখ)")
_HI_WORD = re.compile(r"\s*(तारीख|तारिख)")


def speak_dates(text: str, lang: str, today: datetime.date | None = None) -> str:
    """Every ISO date in `text` rewritten the way it is said. If the template already follows the date with 'তারিখ' / 'तारीख' (any case ending after it) that word is kept, so
    "2026-10-19 তারিখে" -> "উনিশ তারিখে" and "2026-10-19 को" -> "उन्नीस तारीख को"; a date with no such word gets it. A text with no ISO date is returned unchanged."""
    if not text or "-" not in text or _ISO.search(text) is None:
        return text
    lang = lang if lang in ("bn", "hi", "en") else "en"

    def _one(m: re.Match[str]) -> str:
        c = core(m.group(0), lang, today)
        if c is None:
            return m.group(0)
        rest = text[m.end():]
        if lang == "bn":
            return c if _BN_WORD.match(rest) else f"{c} তারিখ"
        if lang == "hi":
            return c if _HI_WORD.match(rest) else f"{c} तारीख"
        return c

    return _ISO.sub(_one, text)
