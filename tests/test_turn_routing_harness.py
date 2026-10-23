"""The real turn logic of BOTH orchestrator entry points -- main.py and main_pcm.py (the one the Twilio phone bridge runs) -- driven with typed sentences.

This is what the unit tests of the cue functions cannot show: that the detector is actually CALLED, that its result is RESPECTED by the routing branch, and that
exactly ONE appropriate reply comes back. Each scenario is a failure seen in testing:

  A. "There is an orthopedic doctor."   -> was "we have no doctor called 'Orthopaedics Doctor'"
  B. "Do you have any health packages?" -> was the insurance / cashless FAQ
  C. "Do I have any money left to pay?"  -> was "I didn't understand, say that again"
  D. a request to talk to the doctor / to complain -> never reached a human

The intent model is SCRIPTED to make the mistakes it made on those calls (so the repair is shown, not assumed); the clinic API is the real clinic-api business logic
over a seeded throw-away database; the voice stack is stood in for (subprocess, tests/_turn_harness_runner.py). What this cannot show -- the live recogniser, the
live Ollama model, real audio -- is listed in the final report as pod verification.

    python -m pytest tests/test_turn_routing_harness.py -v
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO_ROOT, "tests", "_turn_harness_runner.py")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import owner_answers  # noqa: E402

ENTRY_POINTS = ("main", "main_pcm")

CASES = [
    # --- C: money still to pay
    ("Do I have any money left to pay?", "en"),
    ("क्या मेरा कोई पैसा बाकी है?", "hi"),
    ("আমার কি কোনো টাকা বাকি আছে?", "bn"),
    ("Amar kono taka baki ache?", "bn"),
    # --- B: health package (the model, scripted, would have said clinic_faq / insurance)
    ("Do you have any health packages?", "en"),
    ("फुल बॉडी चेकअप पैकेज है क्या?", "hi"),
    ("আপনাদের এখানে হেলথ প্যাকেজ আছে?", "bn"),
    ("Health package-er dam koto?", "bn"),
    # --- genuine insurance must keep using the insurance flow
    ("Do you accept cashless insurance?", "en"),
    # --- D: the doctor himself / a complaint / a person
    ("I want to speak directly to the doctor.", "en"),
    ("मुझे डॉक्टर से सीधे बात करनी है।", "hi"),
    ("আমি সরাসরি ডাক্তারের সঙ্গে কথা বলতে চাই।", "bn"),
    ("Ami doctor-er sathe directly kotha bolte chai.", "bn"),
    ("I want to make a complaint.", "en"),
    ("मुझे शिकायत करनी है।", "hi"),
    ("আমি অভিযোগ জানাতে চাই।", "bn"),
    ("Ami complaint korte chai.", "bn"),
    ("I want to talk to a person", "en"),
    # --- A: a specialty is not a doctor's name
    ("There is an orthopedic doctor.", "en"),
    ("Orthopaedics Doctor", "en"),
    ("Orthopaedics please.", "en"),
    ("Dentistry please.", "en"),
    # --- symptoms the model could not place
    ("My leg hurts.", "en"),
    ("আমার পায়ে ব্যথা।", "bn"),
    ("मेरे कान में दर्द है।", "hi"),
    ("Amar paye fracture hoyeche.", "bn"),
    # --- a department the clinic does not have is never invented
    ("My eyes hurt.", "en"),
    ("Ami jhapsa dekhchi.", "bn"),
]


@pytest.fixture(scope="module")
def turns():
    payload = json.dumps([{"text": t, "lang": lg} for t, lg in CASES], ensure_ascii=False)
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, RUNNER], input=payload.encode("utf-8"), capture_output=True, env=env, timeout=600
    )
    out = proc.stdout.decode("utf-8", "replace")
    assert "@@RESULT@@" in out, "harness did not finish:\n" + proc.stderr.decode("utf-8", "replace")[-3000:]
    raw = json.loads(out.split("@@RESULT@@", 1)[1].strip().splitlines()[0])
    return {entry: {r["text"]: r for r in rows} for entry, rows in raw.items()}


def _run(turns, entry, text):
    r = turns[entry][text]
    assert r["error"] is None, (entry, text, r["error"])
    return r


def _one_reply(r):
    assert len(r["said"]) == 1, ("exactly one response per caller turn", r["said"])
    return r["said"][0]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", CASES[0:4])
def test_C_money_still_to_pay_gets_the_billing_redirect_and_never_a_balance(turns, entry, text, lang):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert said == owner_answers.billing_dues_reply(lang)
    assert not r["model_called"] and not r["handoffs"]
    assert "didn't understand" not in said.lower() and not any(ch.isdigit() for ch in said)  # no invented figure


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", CASES[4:8])
def test_B_a_health_package_question_is_answered_no_packages_never_with_the_insurance_faq(turns, entry, text, lang):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert said == owner_answers.health_package_reply(lang)
    assert "get_faq" not in r["clinic_calls"] and not r["model_called"]
    for word in ("cashless", "insurance", "ক্যাশলেস", "बीमा", "कैशलेस"):
        assert word not in said.lower()


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_B_a_genuine_insurance_question_still_uses_the_insurance_answer(turns, entry):
    r = _run(turns, entry, "Do you accept cashless insurance?")
    assert "Cashless facility is available" in _one_reply(r)
    assert not r["handoffs"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", CASES[9:13])
def test_D_a_request_to_talk_to_the_doctor_reaches_the_human_handoff(turns, entry, text, lang):
    r = _run(turns, entry, text)
    assert r["handoffs"] == ["wants_doctor_personally"], r
    assert not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", CASES[13:17])
def test_D_a_complaint_reaches_the_human_handoff(turns, entry, text, lang):
    r = _run(turns, entry, text)
    assert r["handoffs"] == ["complaint"], r
    assert not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_D_a_request_for_a_person_still_reaches_the_human_handoff(turns, entry):
    assert _run(turns, entry, "I want to talk to a person")["handoffs"] == ["caller_requested"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text", ["There is an orthopedic doctor.", "Orthopaedics Doctor"])
def test_A_a_specialty_in_a_sentence_lists_that_departments_real_doctors(turns, entry, text):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert "Orthopaedics" in said and "Dr." in said, said
    assert "don't have a doctor" not in said and "no doctor" not in said.lower()
    assert "get_doctor_availability" not in r["clinic_calls"], "a specialty is never looked up as a doctor's name"
    assert not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,dept", [("Orthopaedics please.", "Orthopaedics"), ("Dentistry please.", "Dentistry")])
def test_A_a_department_the_model_put_in_the_doctor_slot_is_repaired_into_the_department_list(turns, entry, text, dept):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert r["model_called"] and dept in said and "Dr." in said, (
        said
    )  # the (scripted) model made the mistake; the fallback repaired it
    assert "don't have a doctor" not in said


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize(
    "text,needle",
    [
        ("My leg hurts.", "Orthopaedics"),
        ("আমার পায়ে ব্যথা।", None),
        ("मेरे कान में दर्द है।", None),
        ("Amar paye fracture hoyeche.", None),
    ],
)
def test_a_symptom_the_model_could_not_place_is_routed_to_a_department_with_real_doctors(turns, entry, text, needle):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert "clinic_calls" in r and "route_department" in r["clinic_calls"], r
    assert "Dr." in said or "ডা" in said or "डॉ" in said, said
    if needle:
        assert needle in said


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text", ["My eyes hurt.", "Ami jhapsa dekhchi."])
def test_a_department_the_clinic_does_not_have_is_never_invented(turns, entry, text):
    said = _one_reply(_run(turns, entry, text))
    assert "Dr." not in said and "ডা" not in said, said
