"""Real call 2026-10-15: a caller on headphones talked over the agent and was ignored.

Replaying it showed the detector firing (9 events) and every one being discarded: the trust rule needed the echo
canceller's ERLE to reach 5 dB, and a canceller cannot improve on an echo that is not there, so on an echo-free line ERLE
stays at 0 dB for the whole call. agent/full_duplex.py now also trusts events once the agent has spoken for NO_ECHO_AFTER_S
and its voice has never appeared in the microphone. Real echo keeps the old rule (tests/test_barge_in.py pins that).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

for _p in (os.path.dirname(__file__),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _synth_echo import SR, caller, echo_scene, far_end

from agent.full_duplex import NO_ECHO_AFTER_S, FullDuplexProcessor


def _echo_free_scene(seed=0, agent_s=6.0):
    """The agent speaks; the microphone hears only a quiet room (headphones, or a clean phone line)."""
    far = far_end(seed, dur_each=agent_s, voices=1, same_voice=True)
    rng = np.random.default_rng(500 + seed)
    return far, (0.001 * rng.standard_normal(far.size)).astype(np.float32)


def _run(far, mic, chunk=1600):
    fd = FullDuplexProcessor()
    fd.place_reference(far)
    acted = []
    for i in range(0, mic.size, chunk):
        r = fd.process(mic[i : i + chunk])
        if r.barge_in is not None:
            acted.append(r.barge_in)
    return fd, acted


@pytest.mark.parametrize("seed", range(4))
def test_a_caller_over_an_echo_free_line_is_heard(seed):
    far, mic = _echo_free_scene(seed)
    c = caller(seed, dur=1.5, f0=110 + 12 * seed)
    on = int(3.0 * SR)
    mic[on : on + c.size] += c
    fd, acted = _run(far, mic)
    assert fd.echo_free()
    assert any(on <= ev.detected_at_sample < on + c.size for ev in acted), "the caller was ignored"


def test_an_echo_free_line_is_not_trusted_before_enough_agent_speech_has_been_heard():
    far, mic = _echo_free_scene(1)
    c = caller(1, dur=1.0)
    mic[int(0.3 * SR) : int(0.3 * SR) + c.size] += c  # talks almost at once
    _, acted = _run(far, mic)
    assert not any(ev.detected_at_sample < NO_ECHO_AFTER_S * SR for ev in acted)


def test_no_caller_on_an_echo_free_line_means_no_barge_in():
    far, mic = _echo_free_scene(2)
    fd, acted = _run(far, mic)
    assert acted == [] and fd.barge_ins == 0


@pytest.mark.parametrize("seed", range(3))
def test_real_echo_is_never_mistaken_for_an_echo_free_line(seed):
    far, mic, _ = echo_scene(seed, dur_each=5.0, same_voice=True)
    fd, acted = _run(far, mic)  # the agent's own voice comes back through the room; nobody is talking
    assert not fd.echo_free()
    assert acted == [], "the agent's own echo triggered a barge-in"
