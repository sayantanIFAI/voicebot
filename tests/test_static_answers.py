"""Owner, 2026-10-29: test prices, preparation and FAQ answers are loaded once and answered from memory until the next load."""

import asyncio

import httpx

from agent.tools_client import ClinicToolsClient


def _tools(handler):
    t = ClinicToolsClient("http://clinic.test")
    t._client = type(t._client)(base_url="http://clinic.test", transport=httpx.MockTransport(handler))
    return t


def test_a_price_is_asked_once_and_then_answered_from_memory_until_cleared():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"found": True, "test_name": "CBC", "rate_inr": 300, "sample_type": "Blood", "report_time_hours": 24})

    t = _tools(handler)

    async def run():
        a = await t.get_test_rate("CBC")
        b = await t.get_test_rate("CBC")
        t.clear_static()
        c = await t.get_test_rate("CBC")
        return a, b, c

    a, b, c = asyncio.run(run())
    assert a == b == c and a["rate_inr"] == 300
    assert len(calls) == 2  # the second question came from memory; the clear made the third ask again


def test_a_not_found_answer_is_never_kept():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(200, json={"found": False, "query": "zzz", "did_you_mean": []})

    t = _tools(handler)

    async def run():
        await t.get_test_rate("zzz")
        await t.get_test_rate("zzz")

    asyncio.run(run())
    assert len(calls) == 2
