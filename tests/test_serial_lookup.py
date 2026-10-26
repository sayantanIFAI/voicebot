"""'I booked and paid but the serial number never came on WhatsApp' -- and 'is the CT scan done on Thursday?' through the real orchestrator (sourav, 2026-10-24).

Three layers: the pure cue + number state machine (agent/serial_lookup.py), the clinic-api behaviour (payment simulated by booking_service.mark_paid), and the real
`_dispatch_turn` with a fake clinic API. The serial is always the DATABASE's: no test lets the model compose one.

    python -m pytest tests/test_serial_lookup.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sourav: clinic-api is deliberately NOT added to sys.path here -- clinic_app() does that for the duration of a test; a permanent entry makes other test files import the wrong `main`
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app  # noqa: E402

from agent import serial_lookup  # noqa: E402

DEBASHIS = "9830023456"  # the sample patient whose booking is marked paid by clinic-api/patient_seed.py

# ------------------------------------------------------------------------------------------------------------------------------ the cue


@pytest.mark.parametrize(
    "text",
    [
        "I booked and paid but I did not get the serial number on WhatsApp",
        "payment kore diyechi kintu serial number paini",
        "আমি টাকা দিয়েছি কিন্তু সিরিয়াল নম্বর পাইনি",
        "मैंने पेमेंट किया लेकिन सीरियल नंबर नहीं आया",
        "serial number whatsapp e asheni",
    ],
)
def test_the_complaint_is_recognised_in_three_languages(text):
    assert serial_lookup.reports_serial_not_received(text)


@pytest.mark.parametrize("text", ["what is my serial number", "how much is the CBC", "I did not get the report", "after paying will I get a serial number", ""])
def test_other_sentences_keep_their_old_answers(text):
    assert not serial_lookup.reports_serial_not_received(text)


# ------------------------------------------------------------------------------------------------------------------- the number state machine
def test_ask_then_confirm_then_lookup():
    f = serial_lookup.new_flow(1)
    assert serial_lookup.advance(f, "nine eight three zero zero two three four five six", None) == ("confirm", DEBASHIS)
    assert serial_lookup.advance(f, "yes", "yes") == ("lookup", DEBASHIS)


def test_a_no_goes_back_to_asking_and_a_corrected_number_replaces_it():
    f = serial_lookup.new_flow(1)
    serial_lookup.advance(f, DEBASHIS, None)
    assert serial_lookup.advance(f, "no it is 9831111111", "no") == ("confirm", "9831111111")
    assert serial_lookup.advance(f, "no", "no")[0] == "ask_again"


def test_digits_arriving_over_several_turns_are_joined():
    f = serial_lookup.new_flow(1)
    assert serial_lookup.advance(f, "98300", None)[0] == "ask_again"
    assert serial_lookup.advance(f, "23456", None) == ("confirm", DEBASHIS)


def test_the_caller_is_not_asked_forever():
    f = serial_lookup.new_flow(1)
    acts = [serial_lookup.advance(f, "umm", None)[0] for _ in range(serial_lookup.MAX_TRIES)]
    assert acts[-1] == "give_up"


# ------------------------------------------------------------------------------------------------------------------------ the clinic backend
@pytest.fixture(scope="module")
def client():
    with clinic_app() as (_app, c):
        yield c


def test_a_paid_booking_returns_its_serial(client):
    r = client.get("/api/v1/bookings/serial", params={"phone": DEBASHIS}).json()
    assert r["found"] and r["bookings"][0]["paid"] is True and isinstance(r["bookings"][0]["serial_number"], int)


def test_an_unknown_number_finds_nothing(client):
    r = client.get("/api/v1/bookings/serial", params={"phone": "9000000000"}).json()
    assert r == {"found": False, "bookings": []}


def test_resend_queues_the_message_and_never_claims_delivery(client):
    b = client.get("/api/v1/bookings/serial", params={"phone": DEBASHIS}).json()["bookings"][0]
    r = client.post("/api/v1/bookings/serial/resend", json={"phone": DEBASHIS, "confirmation_id": b["confirmation_id"]}, headers={"Idempotency-Key": "t1"}).json()
    assert r["success"] and r["sent_to_last4"] == DEBASHIS[-4:]


def test_resend_is_refused_for_a_number_that_is_not_the_bookings(client):
    b = client.get("/api/v1/bookings/serial", params={"phone": DEBASHIS}).json()["bookings"][0]
    r = client.post("/api/v1/bookings/serial/resend", json={"phone": "9000000000", "confirmation_id": b["confirmation_id"]}, headers={"Idempotency-Key": "t2"}).json()
    assert r["success"] is False


# ---------------------------------------------------------------------------------------------------------------------- the real orchestrator
from _pod_stubs import pod_stubs  # noqa: E402
from test_orchestrator_booking_flow import FakeTools, env, m  # noqa: E402,F401


class _SerialTools(FakeTools):
    def __init__(self, bookings, resend_ok=True):
        super().__init__()
        self.bookings, self.resend_ok, self.looked_up, self.resent, self.callbacks = bookings, resend_ok, [], [], []

    async def lookup_serial(self, phone):
        self.looked_up.append(phone)
        return {"found": bool(self.bookings), "bookings": self.bookings}

    async def resend_serial(self, phone, confirmation_id):
        self.resent.append((phone, confirmation_id))
        return {"success": self.resend_ok, "sent_to_last4": phone[-4:]}

    async def request_callback(self, phone, call_id, window, reason="", summary=""):
        self.callbacks.append((phone, reason))
        return {"success": True}

    async def get_test_days(self, name):
        return {"found": True, "test_name": "CT Scan Brain", "available_days": [0, 1, 2, 4, 5, 6], "closed_days": [3]}


PAID = {"confirmation_id": "KCD-1", "doctor_name": "Dr. A. Sen", "date": "2026-11-02", "time_slot": "10:30", "patient_name": "Debashis", "paid": True, "serial_number": 7}
UNPAID = {**PAID, "paid": False, "serial_number": None}


async def _complain(env):
    return await env.say("I booked and paid but did not get the serial number on WhatsApp")


@pytest.mark.asyncio
async def test_paid_booking_serial_comes_from_the_database_after_the_number_is_confirmed(env, m):
    env.tools.__class__ = _SerialTools
    _SerialTools.__init__(env.tools, [PAID])
    out = await _complain(env)
    assert "mobile number" in " ".join(out) and not env.tools.looked_up
    out = await env.say("nine eight three zero zero two three four five six")
    assert "9 8 3 0 0 , 2 3 4 5 6" in " ".join(out) and not env.tools.looked_up, "the number is read back first"
    out = await env.say("yes")
    text = " ".join(out)
    assert env.tools.looked_up == [DEBASHIS]
    assert "serial number is 7" in text and "sending the serial number to WhatsApp again" in text
    assert env.tools.resent == [(DEBASHIS, "KCD-1")]
    assert env.session.serial_flow is None and not env.session.ws.closed, "a yes to the read-back must not end the call"


@pytest.mark.asyncio
async def test_unpaid_booking_is_explained_and_a_callback_is_queued(env, m):
    env.tools.__class__ = _SerialTools
    _SerialTools.__init__(env.tools, [UNPAID])
    await _complain(env)
    await env.say(DEBASHIS)
    out = await env.say("yes")
    text = " ".join(out)
    assert "do not see the payment" in text and "serial number is" not in text
    assert env.tools.callbacks == [(DEBASHIS, "paid_but_no_serial")] and not env.tools.resent


@pytest.mark.asyncio
async def test_no_booking_asks_for_another_number(env, m):
    env.tools.__class__ = _SerialTools
    _SerialTools.__init__(env.tools, [])
    await _complain(env)
    await env.say(DEBASHIS)
    out = await env.say("yes")
    assert "could not find" in " ".join(out) and env.session.serial_flow is not None
    out = await env.say("9831111111")
    assert "9 8 3 1 1 , 1 1 1 1 1" in " ".join(out)


@pytest.mark.asyncio
async def test_a_failing_lookup_is_said_plainly_and_the_flow_ends(env, m):
    env.tools.__class__ = _SerialTools
    _SerialTools.__init__(env.tools, [PAID])

    async def boom(phone):
        raise m.ToolCallError("down")

    env.tools.lookup_serial = boom
    await _complain(env)
    await env.say(DEBASHIS)
    out = await env.say("yes")
    assert "cannot check the serial number" in " ".join(out) and env.session.serial_flow is None


@pytest.mark.asyncio
async def test_the_old_generic_serial_answer_is_unchanged_for_a_plain_question(env, m):
    env.tools.__class__ = _SerialTools
    _SerialTools.__init__(env.tools, [PAID])
    await env.say("what is my serial number")
    assert env.session.serial_flow is None


@pytest.mark.asyncio
async def test_ct_scan_thursday_is_answered_from_the_days_column(env, m, monkeypatch):
    env.tools.__class__ = _SerialTools
    _SerialTools.__init__(env.tools, [])

    class FP:
        def named_tests(self, text, lang):
            return ["CT Scan Brain"]

    monkeypatch.setattr(m, "_fast_path", FP())
    out = await env.say("is the CT scan of the brain done on Thursday")
    text = " ".join(out)
    assert "not done on Thursday" in text and "every day except Thursday" in text
    out = await env.say("is the CT scan of the brain done on Friday")
    assert "Yes, CT Scan Brain is done on Friday" in " ".join(out)
