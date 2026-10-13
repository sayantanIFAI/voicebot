"""How long a report takes, as a sentence a caller can follow -- not a bare hour count.

OWNER'S INSTRUCTION, 2026-10-12: the price reply used to say "রিপোর্ট {hours} ঘণ্টার মধ্যে পাবেন" / "within {hours}
hours" literally, for every test -- "7 ঘণ্টার মধ্যে" and "72 ঘণ্টার মধ্যে" read out the same way. A caller does not
think in raw hour counts: a short turnaround is "a few hours", a day-plus one is "1 din" / "1 day", longer than
that is "N din" / "N days". This module buckets `report_time_hours` into that everyday phrasing, in all three
languages, the same way agent/sample_wording.py turns `sample_type` into a sentence instead of a bare label.

Buckets (REASONED from the owner's own two worked examples -- 7-8 hours said as "a few hours", 24 hours said as
"1 din" -- not measured against real calls; revisit if a native reviewer disagrees):
  <= 12 hours   -> "a few hours"
  13-24 hours   -> "1 day"
  > 24 hours    -> ceil(hours / 24) days

The SAFETY-CHECK sentence (report_time_safety_note) is appended after every one of these on purpose: a bucketed
answer is deliberately less precise than the raw hour count was, so the caller is told, every time, that staff
can confirm the exact time once the test itself is actually done -- never left thinking the bucket is a promise.
"""

from __future__ import annotations

import math


def _bucket_days(hours: int) -> tuple[str, int]:
    """("few_hours" | "days", N) -- N is meaningless for "few_hours"."""
    if hours <= 12:
        return "few_hours", 0
    if hours <= 24:
        return "days", 1
    return "days", math.ceil(hours / 24)


def report_time_phrase(hours: int | None, lang: str) -> str:
    """One sentence saying when the report will be ready, bucketed -- or "" if `hours` is not a usable number."""
    if not hours or hours <= 0:
        return ""
    kind, days = _bucket_days(int(hours))
    if lang == "hi":
        if kind == "few_hours":
            return "रिपोर्ट कुछ घंटों में मिल जाएगी।"
        return f"रिपोर्ट {days} दिन में मिल जाएगी।" if days == 1 else f"रिपोर्ट {days} दिनों में मिल जाएगी।"
    if lang == "en":
        if kind == "few_hours":
            return "You will get the report within a few hours."
        return (
            f"You will get the report within {days} day."
            if days == 1
            else (f"You will get the report within {days} days.")
        )
    if kind == "few_hours":
        return "রিপোর্ট কয়েক ঘণ্টার মধ্যে পাবেন।"
    return f"রিপোর্ট {days} দিনের মধ্যে পাবেন।"


def report_time_safety_note(lang: str) -> str:
    """Always follows report_time_phrase(): the bucket is an estimate, not a confirmed slot -- staff confirm the
    exact time once the test itself is done, same discipline CLAUDE.md's truth boundary asks of every other
    fact this agent states (a number the model could get wrong is never left sounding like a guarantee)."""
    if lang == "hi":
        return "टेस्ट होने के बाद हमारा स्टाफ सही समय और सटीक तरीके से बता पाएगा।"
    if lang == "en":
        return "Once the test is done, our staff will be able to tell you the exact time more accurately."
    return "টেস্ট হয়ে যাওয়ার পর আমাদের স্টাফ সময়টা আরও ঠিকভাবে বলতে পারবেন।"
