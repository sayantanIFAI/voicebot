"""WHERE a real payment gateway plugs in.

`create_payment_link()` returns the link the caller is sent to pay a doctor's appointment fee. Today it is a PLACEHOLDER on a reserved, never-resolvable
domain (RFC 2606 `.invalid`), so nobody can pay on it by accident. Owner's decision, 2026-10-17: the pilot runs with the placeholder.

TO MAKE IT REAL, set PAYMENT_LINK_URL to a gateway endpoint that accepts
    POST {"reference": "<booking id>", "amount_inr": <fee>}
and answers {"url": "https://..."}; or replace the body below with the gateway's SDK call. Nothing else needs to change.
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger("clinic.payments")

PLACEHOLDER_BASE = "https://pay.kolkata-care-voice-agent.invalid"


# Owner, 2026-10-18: the booking is held by a Rs 40 payment made within 15 minutes; the link then expires on its own.
BOOKING_FEE_INR = int(os.environ.get("BOOKING_FEE_INR", "40"))
LINK_EXPIRY_MINUTES = int(os.environ.get("PAYMENT_LINK_EXPIRY_MINUTES", "15"))


def create_payment_link(confirmation_id: str, amount_inr: int, expires_in_minutes: int = LINK_EXPIRY_MINUTES) -> dict:
    url = os.environ.get("PAYMENT_LINK_URL", "")
    if not url:
        return {"url": f"{PLACEHOLDER_BASE}/{confirmation_id}", "provider": "placeholder"}
    try:
        r = httpx.post(url, json={"reference": confirmation_id, "amount_inr": amount_inr, "expires_in_minutes": expires_in_minutes}, timeout=8.0)
        r.raise_for_status()
        return {"url": r.json()["url"], "provider": url}
    except (httpx.HTTPError, ValueError, KeyError) as e:
        logger.warning("payment gateway failed (%s) -- falling back to the placeholder link", e)
        return {"url": f"{PLACEHOLDER_BASE}/{confirmation_id}", "provider": "placeholder", "error": str(e)}
