"""KCD-lead-1: inter-center referrals -- staff at another Sonoscan center calling to pass along a patient's
number so an outbound call can be made later to arrange a doctor appointment. No outbound-calling system exists
in this repo (same gap models.CallbackRequest's own docstring already documents for an ordinary callback) -- this
is a worklist for staff to work from, not an auto-dialer, so it is logged to a plain CSV file rather than a new
database table: easy for a non-technical colleague to open directly and work through, which is the actual job
here, and no new schema migration is needed for something this simple.

Same "log exactly what would be needed, never claim an action that didn't happen" discipline as
clinic-api/notifications.py and models.CallbackRequest: every call here appends one row and returns success, it
never fails silently, and nothing anywhere claims the outbound call itself has happened.
"""

from __future__ import annotations

import csv
import datetime
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CSV_PATH = os.environ.get("REFERRAL_CSV_PATH", os.path.join(DATA_DIR, "referral_callbacks.csv"))

FIELDS = ["logged_at", "patient_name", "phone", "doctor_name", "symptom_description", "notes"]


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def log_referral(
    patient_name: str = "",
    phone: str = "",
    doctor_name: str = "",
    symptom_description: str = "",
    notes: str = "",
) -> dict:
    """Appends one row -- whatever was actually captured on the call, blank fields included, never invented.
    Creates the CSV (with header) on first write if it does not exist yet."""
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    is_new = not os.path.exists(CSV_PATH)
    row = {
        "logged_at": _now().isoformat(),
        "patient_name": patient_name or "",
        "phone": phone or "",
        "doctor_name": doctor_name or "",
        "symptom_description": symptom_description or "",
        "notes": notes or "",
    }
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return {"success": True, "logged_at": row["logged_at"]}
