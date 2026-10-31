"""Owner, 2026-10-31: the serial-number simulator.

  * every booking takes the next token number of its doctor and day (appointments.token_number), one above the last booked;
  * the moment the payment link is sent the fee counts as paid (no gateway is wired), so the serial can be told;
  * the same caller phones again, gives the mobile number, and is told the token number (bookings/serial);
  * "which serial is going on?" is answered with the serial last booked for the doctor that day (next-serial: one above it).

    python -m pytest tests/test_serial_simulation.py -q
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from test_booking_service import _doctor, _next_weekday, clinic_modules  # noqa: E402,F401

from agent import serial_lookup, visit_cues  # noqa: E402


def _book(bs, db, doc, date, slot, name, phone):
    hold = bs.hold_slot(db, doc.id, date, slot)
    assert hold["success"], hold
    result = bs.confirm_booking(db, hold["hold_token"], doc.id, date, slot, name, phone, phone)
    assert result["success"], result
    return result["confirmation_id"]


def test_each_booking_takes_the_next_token_and_the_link_marks_it_paid(clinic_modules):
    bs, db_mod, m = clinic_modules
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        slots = bs.available_slots(db, doc.id, date)
        first = _book(bs, db, doc, date, slots[0], "Ravi Das", "9800000011")
        second = _book(bs, db, doc, date, slots[1], "Anita Sen", "9800000012")
        rows = {a.confirmation_id: a for a in db.query(m.Appointment).filter(m.Appointment.doctor_id == doc.id, m.Appointment.date == date).all()}
        assert (rows[first].token_number, rows[second].token_number) == (1, 2)  # the token number is in the column, one more after each booking
        assert rows[first].paid_at is None  # not paid until the payment link goes out

        sent = bs.send_payment_link(db, first)
        assert sent["success"] and sent["payment_simulated"] is True and sent["serial_number"] == 1
        status = bs.serial_status(db, "9800000011")
        assert status["found"] and status["bookings"][0]["paid"] is True and status["bookings"][0]["serial_number"] == 1  # "the same user calls again, gives the mobile number"

        assert bs.send_payment_link(db, second)["serial_number"] == 2
        assert bs.next_serial(db, doc.id, date) == 3  # the serial last booked is 2: "2 is going on", the next booking gets 3
    finally:
        db.close()


def test_the_simulation_can_be_switched_off(clinic_modules, monkeypatch):
    bs, db_mod, m = clinic_modules
    monkeypatch.setenv("SIMULATE_PAYMENT_ON_LINK", "off")
    db = db_mod.SessionLocal()
    try:
        doc = _doctor(db, m)
        date = _next_weekday(doc.schedule[0].weekday if doc.schedule else 0)
        cid = _book(bs, db, doc, date, bs.available_slots(db, doc.id, date)[0], "Ravi Das", "9800000013")
        sent = bs.send_payment_link(db, cid)
        assert sent["success"] and "serial_number" not in sent
        assert bs.serial_status(db, "9800000013")["bookings"][0]["paid"] is False
    finally:
        db.close()


@pytest.mark.parametrize(
    "text",
    ["কোন সিরিয়াল চলছে", "এখন কত নম্বর চলছে", "kon serial cholche", "अभी कौन सा नंबर चल रहा है", "which serial is going on now"],
)
def test_which_serial_is_running_is_recognised(text):
    assert visit_cues.asks_running_serial(text)
    assert not serial_lookup.asks_own_serial(text)


@pytest.mark.parametrize(
    "text",
    ["আমার সিরিয়াল নম্বর কত", "আমি গতকাল বুক করেছিলাম আমার সিরিয়াল কত বলুন", "what is my token number", "mera serial kitna hai"],
)
def test_my_own_serial_starts_the_mobile_number_lookup(text):
    assert serial_lookup.asks_own_serial(text)
    assert not visit_cues.asks_running_serial(text)


@pytest.mark.parametrize("text", ["এখন নাম লিখালে আমার সেরিয়াল কত হবে", "আজ বুক করলে সিরিয়াল কত হবে"])
def test_what_would_mine_be_if_i_book_now_is_not_a_lookup(text):
    assert visit_cues.asks_next_serial(text)
    assert not serial_lookup.asks_own_serial(text)
