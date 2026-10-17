"""Test-suite environment.

SLOT_STEP_MIN: production books one appointment per 10-minute slot (owner, 2026-10-17). The older end-to-end booking tests were written against a
15-minute grid ("10:15", "11:15", "quarter past eleven") and pin those times, so the suite runs on the 15-minute grid unless a test says otherwise
(tests/test_booking_slots_and_sen.py checks the 10-minute default explicitly). Set before any clinic-api module is imported."""
import os

os.environ.setdefault("SLOT_STEP_MIN", "15")
# SEED_EXTRA_DOCTORS: production seeds a SECOND Dr. Sen (Dr. Biswajit Sen, owner 2026-10-17). The older suite was written for exactly one "Dr Sen" and pins
# that; it runs without the extra doctor. tests/test_two_sens.py seeds WITH it and checks the new behaviour.
os.environ.setdefault("SEED_EXTRA_DOCTORS", "0")
