"""When is the line "quiet" -- the one question the 'hello, hello, hello, cut the call' behaviour depends on.

sourav, 2026-10-24: the owner's rule (2026-10-23) is that a caller who says nothing at all after the greeting is called "hello" up to SILENCE_HELLOS times and
then the call is cut. The hello logic lives in main.py `_handle_silence`; it only runs while the poll loop calls it with `quiet=True`. That flag used to be
true ONLY when the speech detector found no speech at all ("no_speech") or too little audio. On a real call (a phone line, a room, a click, a breath, line hiss)
the detector almost always finds SOME blip, answers "transient" (speech shorter than a word, endpointing.decide Guard 1), the loop took that for "the caller is
talking", reset the hello clock every poll, and the hellos never came. This module is the single, pure place that says which verdicts mean "nobody is talking".

Kept free of any model or transport import so it is tested off-pod (tests/test_silence_hellos.py).

Not quiet, on purpose:
  * "waiting_silence"   real speech that has not been followed by enough silence yet -- the caller is mid-sentence.
  * "silence_confirmed" a finished utterance -- a turn, handled by dispatch.
  * "max_utterance"     a very long utterance cut by the runaway guard -- also a turn.
"""

from __future__ import annotations

# endpointing.TurnDecision.reason values that mean "no word is being spoken on the line right now"
QUIET_REASONS = frozenset({"no_speech", "too_little_audio", "transient"})


def line_is_quiet(reason: str | None) -> bool:
    """sourav, 2026-10-24: True when the detector's verdict means nobody is speaking. Flow: the poll loop passes `result.reason` of every undecided poll here and
    hands the answer to `_handle_silence`; True keeps the hello / anything-else clock running, False (a real, unfinished utterance) resets it."""
    return reason in QUIET_REASONS
