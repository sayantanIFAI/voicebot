"""KCD-461: agent/filler.py's race logic. Real (short) asyncio.sleep
calls, no TTS/network.

    python -m pytest tests/test_filler.py -v
"""
import asyncio
import os
import sys
import time

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent.filler import await_with_filler


async def _fast_lookup() -> str:
    await asyncio.sleep(0.02)
    return "the real answer"


async def _slow_lookup() -> str:
    await asyncio.sleep(0.15)
    return "the real answer, eventually"


@pytest.mark.asyncio
async def test_filler_is_suppressed_when_the_result_arrives_first():
    filler_calls = []

    async def on_timeout():
        filler_calls.append(1)

    result = await await_with_filler(_fast_lookup(), threshold_s=0.1, on_timeout=on_timeout)
    assert result == "the real answer"
    assert filler_calls == []   # never spoken


@pytest.mark.asyncio
async def test_filler_fires_exactly_once_when_the_stage_exceeds_the_threshold():
    filler_calls = []

    async def on_timeout():
        filler_calls.append(1)

    result = await await_with_filler(_slow_lookup(), threshold_s=0.05, on_timeout=on_timeout)
    assert result == "the real answer, eventually"
    assert filler_calls == [1]   # exactly once, not repeated while still waiting


@pytest.mark.asyncio
async def test_the_real_result_is_still_returned_after_the_filler():
    async def on_timeout():
        pass

    result = await await_with_filler(_slow_lookup(), threshold_s=0.05, on_timeout=on_timeout)
    assert result == "the real answer, eventually"


@pytest.mark.asyncio
async def test_an_exception_from_the_awaitable_still_propagates():
    async def failing():
        await asyncio.sleep(0.02)
        raise ValueError("boom")

    async def on_timeout():
        pass

    with pytest.raises(ValueError, match="boom"):
        await await_with_filler(failing(), threshold_s=0.1, on_timeout=on_timeout)


@pytest.mark.asyncio
async def test_the_filler_does_not_delay_the_real_result_past_its_own_completion():
    # The filler callback itself takes real time (as speaking one would);
    # the total elapsed time should still be dominated by the slow
    # lookup, not doubled by waiting for the filler AND then restarting
    # the wait for the lookup from zero.
    async def slow_filler():
        await asyncio.sleep(0.02)

    t0 = time.monotonic()
    await await_with_filler(_slow_lookup(), threshold_s=0.05, on_timeout=slow_filler)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.15 + 0.05   # slow_lookup's own 0.15s plus slack, not summed twice
