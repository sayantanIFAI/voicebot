"""Owner, 2026-10-26: real Bengali and Hindi calls whose opening-hours / payment-method question matched no stored phrase ("sorry, say again", "जी, बताइए").
Real clinic catalogue -> real FastPath -> the sentence must come out as clinic_faq with the right topic, whatever language the call is locked to."""

import datetime
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "clinic-api"))

from agent.fast_path import Catalogue, FastPath  # noqa: E402

TODAY = datetime.date(2026, 9, 25)

HOURS = [
    "আপনাদের এই ক্লিনিক টা কটা থেকে কটা খুলবে",
    "ক্লিনিক টা কটা থেকে খুলবে",
    "কখন আসবো কটা থেকে খুলে এটা",
    "बोल रहे थे कि कितने बजे से खुलता है आप लोग का ये",
    "बोल रहे थे कितने बजे से खुलता है आप लोग का ये क्लिनिक",
    "बोल रहे थे कितने बजे से खुलता है आप लोग का क्लीनिक",
]
PAYMENT = [
    "वो गूगल पाय हो जायेगा तो गूगल पाय फोन पाय सब",
    "बोल रहे थे की वो गूगल पे फोन पे सब हो जायेगा",
    "बोल रहे थे की गूगल पाय या फोन पाय से पैसा देंगे तो चलेगा ना",
    "बोल रहे थे कि वो गूगल पाय से पाय्मेंट करेंगे या फोन पाय से ग पाय तो पैसा ले लीजियेगा तो",
    "वो कैश आनी नहीं तो ऑनलाइन गूगल पाय सब चुल गए",
]


@pytest.fixture(scope="module")
def payload():
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        return client.get("/api/v1/catalogue").json()


@pytest.mark.parametrize("topic,sentences", [("hours", HOURS), ("payment_methods", PAYMENT)])
def test_phrasing_is_answered_as_the_right_faq(payload, topic, sentences):
    """The agent reads each sentence in the language of its SCRIPT (main.language_of_script), so Bengali script is "bn" and Devanagari is "hi"."""
    fp = FastPath(Catalogue(payload), today=TODAY)
    missed = []
    for s in sentences:
        lang = "hi" if any("ऀ" <= c <= "ॿ" for c in s) else "bn"
        r = fp.resolve(s, lang)
        got = None if r is None else (r.intent, r.slots.get("faq_topic"))
        if got != ("clinic_faq", topic):
            missed.append((s, got))
    assert not missed, f"{topic}: {missed}"
