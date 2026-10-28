"""Limits and validation for what a client may send over the call socket (review 2026-10-28).

Before this, a control frame that was valid JSON but not an object (`null`, `[]`, `7`) raised in the handler, a "sampleRate" of 0, 10**9 or a string raised or was believed,
a client could stream audio for ever, and one message could be of any size. A bad client only ever ended its own call (the handler catches), but through an unhandled error and with the
buffer growing until then. Here every one of those is a controlled rejection with a close code, and the limits are named, in one place, and tunable by environment.
"""

from __future__ import annotations

import json
import os

MAX_CALL_SECONDS = float(os.environ.get("MAX_CALL_SECONDS", str(30 * 60)))  # a call longer than this is closed (a clinic call is minutes)
MAX_CALL_BYTES = int(os.environ.get("MAX_CALL_BYTES", str(128 * 1024 * 1024)))  # audio bytes per call: 30 minutes of 16 kHz PCM is 58 MB, of 48 kHz 173 MB
MAX_CHUNK_BYTES = int(os.environ.get("MAX_CHUNK_BYTES", str(1024 * 1024)))  # one audio message
MAX_CONTROL_BYTES = int(os.environ.get("MAX_CONTROL_BYTES", "8192"))  # one control (text) message
ALLOWED_SAMPLE_RATES = frozenset({8000, 16000, 22050, 24000, 32000, 44100, 48000})
CONTROL_TYPES = frozenset({"hello", "playback_done", "interrupt", "ping", "stop_playback"})

# WebSocket close codes: 1003 unsupported data, 1008 policy violation, 1009 message too big
CLOSE_BAD_DATA, CLOSE_POLICY, CLOSE_TOO_BIG = 1003, 1008, 1009


class Rejected(Exception):
    """The client sent something the call socket does not accept: the call is closed with `code`."""

    def __init__(self, code: int, reason: str):
        super().__init__(reason)
        self.code, self.reason = code, reason


def parse_control(raw) -> dict | None:
    """A control frame as a dict, or None when it is not one (logged by the caller, the call goes on). Raises Rejected for a frame that is oversized."""
    if not isinstance(raw, str):
        return None
    if len(raw.encode("utf-8", "ignore")) > MAX_CONTROL_BYTES:
        raise Rejected(CLOSE_TOO_BIG, "control message too large")
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(msg, dict) or not isinstance(msg.get("type"), str):
        return None
    if msg["type"] not in CONTROL_TYPES:
        return None
    return msg


def valid_sample_rate(value) -> int | None:
    """The declared rate if it is one of the supported rates (an int or a whole-number float, never a bool or a string), else None."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and value in ALLOWED_SAMPLE_RATES:
        return value
    return None


class CallLimits:
    """Tracks one call's audio bytes and wall time; `check_*` raise Rejected when a limit is passed."""

    def __init__(self, now: float):
        self.started = now
        self.bytes = 0

    def check_audio(self, chunk_len: int, now: float) -> None:
        if chunk_len > MAX_CHUNK_BYTES:
            raise Rejected(CLOSE_TOO_BIG, "audio message too large")
        self.bytes += chunk_len
        if self.bytes > MAX_CALL_BYTES:
            raise Rejected(CLOSE_POLICY, "too much audio for one call")
        self.check_time(now)

    def check_time(self, now: float) -> None:
        if now - self.started > MAX_CALL_SECONDS:
            raise Rejected(CLOSE_POLICY, "call too long")
