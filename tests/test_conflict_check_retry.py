import asyncio

import httpx
import pytest

from agent.tools_client import ClinicToolsClient, ToolCallError


class _Resp:
    status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"conflict": False, "existing": None}


def _tools(effects):
    t = ClinicToolsClient.__new__(ClinicToolsClient)

    class C:
        calls = 0

        async def get(self, *a, **k):
            C.calls += 1
            e = effects.pop(0)
            if isinstance(e, Exception):
                raise e
            return e

    t._client = C()
    return t, C


def test_one_read_timeout_is_asked_again():
    t, c = _tools([httpx.ReadTimeout(""), _Resp()])
    assert asyncio.run(t.booking_conflict("9635500201", "2026-10-30", "17:10")) == {"conflict": False, "existing": None}
    assert c.calls == 2


def test_two_timeouts_are_a_tool_error():
    t, _ = _tools([httpx.ReadTimeout(""), httpx.ReadTimeout("")])
    with pytest.raises(ToolCallError):
        asyncio.run(t.booking_conflict("9635500201", "2026-10-30", "17:10"))
