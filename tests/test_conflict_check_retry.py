import asyncio

import httpx
import pytest

from agent.tools_client import ClinicToolsClient, ToolCallError


def _tools(handler):
    t = ClinicToolsClient("http://clinic.test")
    t._client = type(t._client)(base_url="http://clinic.test", transport=httpx.MockTransport(handler))
    return t


def test_one_read_timeout_is_asked_again():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("", request=request)
        return httpx.Response(200, json={"conflict": False, "existing": None})

    t = _tools(handler)
    assert asyncio.run(t.booking_conflict("9635500201", "2026-10-30", "17:10")) == {"conflict": False, "existing": None}
    assert len(calls) == 2


def test_two_timeouts_are_a_tool_error():
    def handler(request):
        raise httpx.ReadTimeout("", request=request)

    t = _tools(handler)
    with pytest.raises(ToolCallError):
        asyncio.run(t.booking_conflict("9635500201", "2026-10-30", "17:10"))
