"""WHERE a real SMS / WhatsApp provider plugs in.

Today nothing leaves the building: `send()` is a PLACEHOLDER that records what would be sent (the caller, models.SmsOutbox, already keeps the row)
and returns status "queued". Owner's decision, 2026-10-17: the pilot is run with this placeholder and the agent tells the caller, aloud, that the
payment link and booking number have been sent by SMS and WhatsApp.

TO MAKE IT REAL, either
  * set NOTIFY_PROVIDER_URL (and optionally NOTIFY_PROVIDER_TOKEN) to an HTTP endpoint that accepts
        POST {"channel": "sms" | "whatsapp", "to": "<10-digit number>", "message": "...", "reference": "<booking id>"}
    and answers 2xx when it has accepted the message -- then `send()` returns status "sent" (any other answer: "failed"); or
  * replace the body of `_deliver()` with the provider's own SDK call.
Nothing else in the clinic API or the voice agent needs to change.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger("clinic.notifications")

CHANNELS = ("sms", "whatsapp")


def _deliver(channel: str, to_phone: str, message: str, reference: str | None) -> dict:
    url = os.environ.get("NOTIFY_PROVIDER_URL", "")
    if not url:
        return {"status": "queued", "provider": "placeholder"}  # nothing is connected yet
    headers = {"Authorization": f"Bearer {os.environ['NOTIFY_PROVIDER_TOKEN']}"} if os.environ.get("NOTIFY_PROVIDER_TOKEN") else {}
    try:
        r = httpx.post(url, json={"channel": channel, "to": to_phone, "message": message, "reference": reference}, headers=headers, timeout=5.0)
        return {"status": "sent" if r.is_success else "failed", "provider": url, "http_status": r.status_code}
    except httpx.HTTPError as e:
        logger.warning("notification provider unreachable (%s)", e)
        return {"status": "failed", "provider": url, "error": str(e)}


def send(channel: str, to_phone: str, message: str, reference: str | None = None) -> dict:
    """Hand one message to the provider. Never raises: a failure is a status, not an exception that would undo a booking."""
    if channel not in CHANNELS:
        return {"status": "failed", "error": f"unknown channel {channel!r}"}
    result = _deliver(channel, to_phone, message, reference)
    logger.info("notification %s -> ...%s [%s] %s", channel, to_phone[-4:], reference, result["status"])
    return result
