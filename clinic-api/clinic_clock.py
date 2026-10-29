"""The clinic's own clock (owner, 2026-10-29): a Kolkata clinic, whatever time zone the server runs in.

The pod runs on UTC, 5.5 hours behind Kolkata, so "has today's chamber closed?" and "is this date past?" were answered on the wrong clock: at 17:00 in Kolkata a 10:00-13:00 chamber still counted as open
(it was 11:30 UTC) and was announced as "sitting today". The agent already used Kolkata time (agent/slot_proposal.clinic_now); the clinic API now does too. CLINIC_TZ=local turns it off.
"""

from __future__ import annotations

import datetime
import os

try:
    from zoneinfo import ZoneInfo

    _name = os.environ.get("CLINIC_TZ", "Asia/Kolkata")
    _TZ = None if _name == "local" else ZoneInfo(_name)
except Exception:  # noqa: BLE001 - no tz database: the machine's own time
    _TZ = None


def now() -> datetime.datetime:
    """The clinic's current time as a naive datetime (the database keeps naive local times)."""
    return datetime.datetime.now(_TZ).replace(tzinfo=None) if _TZ else datetime.datetime.now()


def today() -> datetime.date:
    return now().date()
