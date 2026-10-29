"""A caller who says nothing at all is called "hello" three times, then the call is cut (owner, 2026-10-23).

sourav, 2026-10-24: the hello logic itself (main.py `_handle_silence`) was already there, but it was never reached on a real call: the poll loop only
told it "the line is quiet" when the speech detector found NO speech at all, and a real line (a phone line, a room, a breath, a click) is never that clean.
Any blip made the detector answer "transient" (speech shorter than a word), the loop decided "not quiet", reset the hello clock, and the caller who said
nothing was never called "hello". These tests drive the REAL `_turn_poll_loop` with a fake turn detector, so the whole path is covered, not only the
`_handle_silence` function the older tests call directly with `quiet=True`.

    python -m pytest tests/test_silence_hellos.py -v
"""

import asyncio
import os
import sys
import time
import types

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _pod_stubs import pod_stubs
from test_orchestrator_persona import FakeWS

from agent import silence_policy
from agent.endpointing import TurnDecision
from agent.phrases import phrase


class _Tail:
    """What `session.audio.tail_tensor()` returns: enough audio to be judged."""

    def numel(self):
        return 16000


class _FakeAudio:
    """A call whose audio clock runs in real time (the silence timer is measured on the audio clock since 2026-10-29)."""

    sample_rate = 16000

    def __init__(self):
        self._t0 = time.monotonic()

    @property
    def duration_s(self):
        return 30.0 + (time.monotonic() - self._t0)

    def tail_tensor(self, _from_s):
        return _Tail()


@pytest.fixture(scope="module")
def m():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main_pcm")


async def _run_loop(m, monkeypatch, reasons, *, hellos=3, turn_count=0, max_s=5.0):
    """Run the real poll loop until it returns (call cut) or `max_s` passes. `reasons` is the detector's verdict on each poll, the last one repeating."""
    monkeypatch.setattr(m, "SILENCE_PROMPT_S", 0.05)
    monkeypatch.setattr(m, "SILENCE_CLOSE_S", 0.05)
    monkeypatch.setattr(m, "SILENCE_HELLOS", hellos)
    monkeypatch.setattr(m, "POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(m, "ACTIVE_POLL_INTERVAL_S", 0.01)
    monkeypatch.setattr(m, "MIN_WAKE_S", 0.005)

    class FakeTTS:
        async def synthesize(self, lang, text, speed=1.0, prosody=None):
            from test_orchestrator_persona import FakeTTS as T

            return await T().synthesize(lang, text, speed)

    monkeypatch.setattr(m, "_tts_router", FakeTTS())
    session = m.CallSession(FakeWS())
    session.audio = _FakeAudio()
    session.lang = "en"
    session.turn_count = turn_count
    session.next_wake_s = 0.01
    calls = {"n": 0}

    async def fake_decide(sess, tail, sr):
        r = reasons[min(calls["n"], len(reasons) - 1)]
        calls["n"] += 1
        return TurnDecision(None, r != "no_speech", r)

    monkeypatch.setattr(m, "_decide_turn", fake_decide)

    seen = {"n": 0}

    async def client_playback_done():  # the client's "playback_done": sent once, after each clip the agent spoke has played
        while True:
            if len(session.ws.spoken()) > seen["n"]:
                seen["n"] = len(session.ws.spoken())
                await asyncio.sleep(0.01)
                session.release_gate()
            await asyncio.sleep(0.005)

    opener = asyncio.create_task(client_playback_done())
    try:
        await asyncio.wait_for(m._turn_poll_loop(session), timeout=max_s)
        finished = True
    except asyncio.TimeoutError:
        finished = False
    finally:
        opener.cancel()
    return session, finished


@pytest.mark.asyncio
async def test_pure_silence_gets_three_hellos_and_then_the_call_is_cut(m, monkeypatch):
    session, finished = await _run_loop(m, monkeypatch, ["no_speech"])
    assert finished, "the poll loop never ended the call"
    assert session.ws.spoken().count(phrase("hello_check", "en")) == 3
    assert session.ws.closed and session.outcome == "silence_timeout"


@pytest.mark.asyncio
@pytest.mark.parametrize("noise", ["transient", "too_little_audio"])
async def test_line_noise_that_is_not_a_word_does_not_stop_the_hellos(m, monkeypatch, noise):
    """The real-call failure: a click, a breath or hiss makes the detector answer 'transient' (speech shorter than a word). That is not the caller speaking."""
    session, finished = await _run_loop(m, monkeypatch, [noise])
    assert finished, f"'{noise}' kept the hello clock from ever running"
    assert session.ws.spoken().count(phrase("hello_check", "en")) == 3
    assert session.ws.closed and session.outcome == "silence_timeout"


@pytest.mark.asyncio
async def test_noise_between_the_hellos_does_not_reset_the_count(m, monkeypatch):
    session, finished = await _run_loop(m, monkeypatch, ["no_speech", "transient", "no_speech", "transient"])
    assert finished
    assert session.ws.spoken().count(phrase("hello_check", "en")) == 3


@pytest.mark.asyncio
async def test_a_caller_who_is_really_speaking_is_never_called_hello(m, monkeypatch):
    """'waiting_silence' = real speech that has not finished yet. The caller is talking: no hello, no cut."""
    session, finished = await _run_loop(m, monkeypatch, ["waiting_silence"], max_s=0.6)
    assert not finished
    assert session.ws.spoken().count(phrase("hello_check", "en")) == 0 and not session.ws.closed


@pytest.mark.asyncio
async def test_a_caller_who_has_already_spoken_keeps_the_old_anything_else_flow(m, monkeypatch):
    session, finished = await _run_loop(m, monkeypatch, ["no_speech"], turn_count=1)
    assert finished
    spoken = session.ws.spoken()
    assert phrase("hello_check", "en") not in spoken
    assert spoken[0] == phrase("silence_prompt", "en")
    assert spoken[-1] == phrase("idle_close", "en")


def test_which_detector_verdicts_count_as_a_quiet_line():
    for r in ("no_speech", "too_little_audio", "transient"):
        assert silence_policy.line_is_quiet(r)
    for r in ("waiting_silence", "silence_confirmed", "max_utterance", "", None):
        assert not silence_policy.line_is_quiet(r)
