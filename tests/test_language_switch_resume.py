"""Context-preservation regression (2026-10-12, real-call bug): a caller who stays in another language for
two turns in a row gets the "shall I switch?" offer instead of an answer to what they just asked -- the
question is meant to be answered on the turn after, once the offer resolves (see main_pcm.py's
_continue_language_switch_offer). Live testing found it was answered only on a clean "yes": a "no", or any
reply that is not a clean yes/no (a caller who simply repeats or rephrases their original question instead
of literally saying "yes"), silently dropped the question and left the caller unanswered twice over.

    python -m pytest tests/test_language_switch_resume.py -v
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _synth_voices import FATHER, utterance
from test_orchestrator_booking_flow import env, m  # noqa: F401 -- reused fixtures (m, env)
from test_orchestrator_persona import _wav

# Real script text in each turn's own language, matching how the live bug was reproduced -- transcript_problem's
# wrong_script check would otherwise flag a romanised ASR text as a mismatch with its claimed detected_lang.
BN_LOCATION = "এটা কোথায়"
HI_PAYMENT = "कैश नहीं है तो ऑनलाइन पेमेंट हो जाएगा"
HI_PARKING_HI = "पार्किंग है क्या"
HI_YES = "हाँ ठीक है"
HI_NO = "नहीं"
HI_REPHRASE = "वो पूछ रहे थे कैश नहीं है तो ऑनलाइन पेमेंट हो जाएगा"


async def _turn(m, session, state, text, tmp_path, name):
    """One caller turn, WITHOUT touching state['data'] -- unlike env.say(), so a turn that resolves via the
    language-switch-offer continuation (which answers a STASHED turn's intent/slots, not this one's) is not
    accidentally overwritten by this call's own intent/slots."""
    state["text"] = text
    voice = utterance(FATHER, dur=3.0, seed=hash(name) % 1000, amp=0.3)
    path = _wav(tmp_path, voice, f"{name}.wav")
    session.turn_epoch = session.speak_epoch
    before = len(session.ws.spoken())
    await m._dispatch_turn(session, path)
    return session.ws.spoken()[before:]


async def _set_up_the_offer(env, state, monkeypatch, m):
    """Bengali first (locks session.lang to "bn"), then two Hindi turns in a row -- the second one triggers
    the offer instead of being answered, exactly the two-in-a-row condition next to session.lang_history."""
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi"))
    state["lang"] = "bn"
    await env.say(BN_LOCATION, "clinic_faq", {"faq_topic": "location"})

    state["lang"] = "hi"
    await env.say(HI_PARKING_HI, "clinic_faq", {"faq_topic": "parking"})

    state["reply"] = "PAYMENT_ANSWER"
    said = await env.say(HI_PAYMENT, "clinic_faq", {"faq_topic": "payment_methods"})
    assert env.session.awaiting_language_switch_offer is True
    assert "PAYMENT_ANSWER" not in " ".join(said)  # set aside this turn, not answered yet


@pytest.mark.asyncio
async def test_yes_still_answers_the_pending_question_in_the_new_language(m, env, tmp_path, monkeypatch):
    """The already-working case, pinned down so a future change cannot quietly break it."""
    session, state = env.session, env.state
    await _set_up_the_offer(env, state, monkeypatch, m)

    said = await _turn(m, session, state, HI_YES, tmp_path, "yes-reply")
    assert session.awaiting_language_switch_offer is False
    assert "PAYMENT_ANSWER" in " ".join(said)


@pytest.mark.asyncio
async def test_a_rephrase_that_is_not_yes_or_no_still_gets_the_original_question_answered(
    m, env, tmp_path, monkeypatch
):
    """The real-call bug: the caller does not say a bare "yes" -- they repeat/rephrase the question instead,
    exactly as a real caller does. The question must still be answered, not dropped a second time."""
    session, state = env.session, env.state
    await _set_up_the_offer(env, state, monkeypatch, m)

    # Not yes/no -- reported speech restating the same question, the live transcript's actual wording.
    said = await _turn(m, session, state, HI_REPHRASE, tmp_path, "rephrase")
    assert session.awaiting_language_switch_offer is False  # the offer is resolved either way, not left dangling
    assert "PAYMENT_ANSWER" in " ".join(said)  # and the original question is finally answered


@pytest.mark.asyncio
async def test_a_clear_no_still_gets_the_original_question_answered(m, env, tmp_path, monkeypatch):
    """Declining the switch used to silently drop the question too -- only the LANGUAGE should depend on
    yes/no, never whether the question gets answered at all."""
    session, state = env.session, env.state
    await _set_up_the_offer(env, state, monkeypatch, m)

    said = await _turn(m, session, state, HI_NO, tmp_path, "no-reply")
    assert session.awaiting_language_switch_offer is False
    assert "PAYMENT_ANSWER" in " ".join(said)
