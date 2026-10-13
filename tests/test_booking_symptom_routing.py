"""KCD-dentistry (owner's instruction, 2026-10-12): a caller who wants an appointment but does not know which
doctor to see, and describes a symptom instead ("mujhe ek doctor ka appointment chahiye... kaan mein bahut dard
hai"), must be routed to a real department and a real doctor -- or, if the symptom is unclear/unmatched, to a
General Medicine checkup, so the doctor can refer them onward. See main_pcm.py's _resolve_doctor_from_symptom.

    python -m pytest tests/test_booking_symptom_routing.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_orchestrator_booking_flow import ASK_DAY_TIME, env, m, text_of  # noqa: F401 -- reused fixtures


@pytest.mark.asyncio
async def test_a_clearly_matched_symptom_names_the_department_and_asks_for_the_date_next(env):
    """The owner's own example: ear pain, no doctor name -- ENT's real doctors are named and the flow moves
    straight to the date, never re-asking for a doctor's name the caller already said they do not have."""
    env.tools.route_department_result = {
        "matched": True,
        "department_name": "ENT",
        "doctors": ["Dr. R. Pal", "Dr. K. Halder"],
    }
    said = await env.say(
        "mujhe doctor dekhana tha, kaan mein bahut dard hai",
        "book_appointment",
        {"symptom_description": "kaan mein bahut dard hai"},
    )
    text = text_of(said)
    assert "ENT" in text
    assert "Dr. R. Pal" in text and "Dr. K. Halder" in text
    assert ASK_DAY_TIME in text  # doctor_name is now filled -- the next question is the date, not the name
    # Stored WITHOUT "Dr. " -- booking_confirmation_readback's English template prepends it itself, the same
    # convention a caller-given name follows (clinic-api's fuzzy matching resolves the bare name either way).
    assert env.session.booking.slots["doctor_name"] == "R. Pal"
    assert "symptom_description" not in env.session.booking.slots  # consumed, never leaks into the readback


@pytest.mark.asyncio
async def test_an_unmatched_symptom_defaults_to_a_general_medicine_checkup(env):
    """Owner's instruction: when the symptom (or lack of one) does not clearly point anywhere, default to a
    general checkup rather than asking the caller to choose a specialist department they cannot judge."""
    env.tools.route_department_result = {"matched": False}
    env.tools.department_doctors_result = {"matched": True, "doctors": ["Dr. S. Mukherjee"]}
    said = await env.say(
        "mujhe doctor dekhana tha, bas thoda unwell feel ho raha hai",
        "book_appointment",
        {"symptom_description": "thoda unwell feel ho raha hai"},
    )
    text = text_of(said)
    assert "Dr. S. Mukherjee" in text
    assert ASK_DAY_TIME in text
    assert env.session.booking.slots["doctor_name"] == "S. Mukherjee"


@pytest.mark.asyncio
async def test_an_ambiguous_symptom_also_defaults_to_a_general_medicine_checkup(env):
    """route_department's own "ambiguous" shape (two departments matched) -- same owner's instruction applies:
    General Medicine decides and refers onward, rather than asking the caller to pick between two departments."""
    env.tools.route_department_result = {
        "matched": False,
        "ambiguous": True,
        "candidates": ["Orthopaedics", "General Medicine"],
        "doctors_by_department": {"Orthopaedics": ["Dr. D. Das"], "General Medicine": ["Dr. S. Mukherjee"]},
    }
    env.tools.department_doctors_result = {"matched": True, "doctors": ["Dr. S. Mukherjee"]}
    said = await env.say(
        "mujhe doctor dekhana tha, haath mein dard hai",
        "book_appointment",
        {"symptom_description": "haath mein dard hai"},
    )
    text = text_of(said)
    assert "Dr. S. Mukherjee" in text
    assert env.session.booking.slots["doctor_name"] == "S. Mukherjee"


@pytest.mark.asyncio
async def test_a_direct_doctor_name_never_triggers_symptom_routing_at_all(env):
    """The existing shortcut -- caller names a doctor directly -- must be completely unaffected."""
    said = await env.say("mujhe Dr Sen se appointment chahiye", "book_appointment", {"doctor_name": "Sen"})
    assert env.tools.routed_symptoms == []  # route_department was never even called
    assert ASK_DAY_TIME in text_of(said)


@pytest.mark.asyncio
async def test_the_full_flow_completes_after_symptom_routing(env):
    """End to end, the owner's whole scenario: symptom -> department/doctor -> date -> patient name and
    phone -> a real confirmation, exactly like a booking that started with a doctor's name already known."""
    env.tools.route_department_result = {
        "matched": True,
        "department_name": "ENT",
        "doctors": ["Dr. R. Pal"],
    }
    await env.say(
        "mujhe doctor dekhana tha, kaan mein dard hai",
        "book_appointment",
        {"symptom_description": "kaan mein dard hai"},
    )
    await env.say("kal subah", "book_appointment", {"date": "2026-10-13", "time_slot": "10:00"})
    said = await env.say(
        "mera naam Ravi hai, number 9876543210",
        "book_appointment",
        {"patient_name": "Ravi", "phone": "9876543210"},
    )
    assert "shall i confirm" in text_of(said).lower()
    said = await env.say("yes", "unclear", {})  # the readback's own yes/no, same as every other booking test
    text = text_of(said)
    assert "KCD-1001" in text  # FakeTools.confirm_booking's confirmation id
    assert env.tools.holds and env.tools.holds[0][0] == "R. Pal"
