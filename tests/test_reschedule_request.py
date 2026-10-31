"""Owner, 2026-10-31: a caller who already HAS a booking and asks to move it (earlier, later, another day, "I cannot come") is changing that booking, never making a new one.

python -m pytest tests/test_reschedule_request.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import booking_carryover, reschedule_request as rr  # noqa: E402


@pytest.mark.parametrize(
    "text",
    [
        "আমার একটা অযাপ্ফোন্টমেন্ট আছে দু তারিখে তো সেদিনকে আসতে পারবো না ওটা একদিন আগানো যাবে",  # the real call
        "অযাপ্ফোন্টমেন্ট না একটা আমার অযাপ্ফোন্টমেন্ট আছে দু তারিখে সেদিন কি আসতে পারবো না তো ওটা কি একদিন আগানো যাবে",
        "আমার অ্যাপয়েন্টমেন্টটা এক সপ্তাহ পিছিয়ে দিন",
        "আমার অ্যাপয়েন্টমেন্ট বদলাতে চাই",
        "my appointment on the 2nd, can i prepone it by one day",
        "I cannot come for my appointment tomorrow",
        "मेरी अपॉइंटमेंट एक दिन पहले हो सकती है क्या",
        "मैं अपॉइंटमेंट के दिन नहीं आ सकता",
        "appointment ta ektu pichiye din",
        "amar appointment ta ek din aage kore den",
    ],
)
def test_moving_an_existing_booking_is_recognised(text):
    assert rr.asks_reschedule(text)
    assert booking_carryover.mentions_change_or_cancel(text)


@pytest.mark.parametrize(
    "text",
    [
        "পরের সোমবার ডাক্তার সেনের অ্যাপয়েন্টমেন্ট করতে চাই",
        "ডাক্তার সেন কি দু তারিখে বসছে",
        "আমি একটা অ্যাপয়েন্টমেন্ট করতে চাই",
        "আমি নাম লেখাতে চাই",
        "my age is 45 and I want an appointment",
        "appointment pore korbo",
    ],
)
def test_a_new_booking_is_not_a_move(text):
    assert not rr.asks_reschedule(text)


@pytest.mark.parametrize(
    "text,days",
    [("একদিন আগে করা যাবে", -1), ("এক সপ্তাহ পরে", 7), ("দুদিন পিছিয়ে", 2), ("ওটা একদিন আগানো যাবে", -1), ("move it two days later", 2)],
)
def test_how_far_it_moves(text, days):
    assert rr.shift_days(text) == days
