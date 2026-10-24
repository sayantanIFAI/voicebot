"""The weekdays a lab test is done on (models.LabTest.available_days).

sourav, 2026-10-24: the owner asked for a "days" column on the lab-test table -- a CT scan of the brain, say, is done every day except Thursday -- so the agent can
answer "is it done on Thursday?" and "which days is it done?" from the table, never from the model. The column holds Python weekday numbers, Monday = 0 ... Sunday = 6
(the SAME convention as DoctorSchedule.weekday), comma-joined: "0,1,2,4,5,6" = every day except Thursday.

Rules that keep an existing test behaving exactly as before:
  * an EMPTY or unreadable value means EVERY day -- a test is never silently made unavailable by a typo in the column;
  * numbers outside 0-6 and repeats are ignored.

Pure functions, no database and no web framework, so they are unit-tested on their own (tests/test_lab_test_days.py).
"""

from __future__ import annotations

import datetime

ALL_DAYS: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
DEFAULT_DAYS = "0,1,2,3,4,5,6"


def parse_days(raw: object) -> list[int]:
    """sourav, 2026-10-24: the column text -> sorted weekday numbers. Why: one tolerant reader for every caller of the column. Flow: split on commas, keep the integers
    0-6, drop repeats; nothing usable left = every day (see the module docstring)."""
    out: set[int] = set()
    for part in str(raw if raw is not None else "").replace(";", ",").replace(" ", ",").split(","):
        part = part.strip()
        if part.isdigit() and 0 <= int(part) <= 6:
            out.add(int(part))
    return sorted(out) if out else list(ALL_DAYS)


def format_days(days: object) -> str:
    """sourav, 2026-10-24: weekday numbers -> the text stored in the column (sorted, no repeats, "0,1,..."). Why: a writer (seed, an admin script) goes through this so the
    column is always in one shape."""
    return ",".join(str(d) for d in parse_days(",".join(str(d) for d in days) if not isinstance(days, str) else days))


def closed_days(days: list[int]) -> list[int]:
    """sourav, 2026-10-24: the weekdays the test is NOT done (the complement), so the agent can say "every day except Thursday"."""
    return [d for d in ALL_DAYS if d not in days]


def is_open_on(days: list[int], day: datetime.date) -> bool:
    """sourav, 2026-10-24: is the test done on this calendar date? Flow: the date's weekday (Monday = 0) is looked up in the list."""
    return day.weekday() in days


def next_open_date(days: list[int], from_date: datetime.date, include_from: bool = True) -> datetime.date:
    """sourav, 2026-10-24: the first date on or after `from_date` (after it when include_from is False) the test is done. Why: "not on Thursday -- the next day is Friday".
    `days` is never empty (parse_days), so within seven days there is always one."""
    start = 0 if include_from else 1
    for offset in range(start, start + 7):
        d = from_date + datetime.timedelta(days=offset)
        if d.weekday() in days:
            return d
    return from_date  # unreachable: days is never empty
