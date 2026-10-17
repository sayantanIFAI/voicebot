"""A garbled reply must not erase the question that was just asked, and "these ones" refers to the tests just named (real call 2026-10-17)."""
import asyncio
import types

import pytest

main = pytest.importorskip("main")  # needs the full voice-agent environment: runs on the pod


class _Tools:
    async def get_test_rate(self, name):
        return {"found": True, "test_name": name, "test_name_bn": name, "test_name_hi": name, "rate_inr": {"Thyroid Profile": 700, "Blood Sugar Fasting": 120}.get(name, 100)}


def _session(turn=5):
    s = types.SimpleNamespace(turn_count=turn, last_multi=(["Thyroid Profile", "Blood Sugar Fasting"], turn - 1), lapsed_pending=None, pending_question=None,
                              pending_entity=None, booking=None)
    return s


def _run(session, text, monkeypatch):
    said = []

    async def fake_speak(sess, t, lang_, **kw):
        said.append(t)

    monkeypatch.setattr(main, "_tools", _Tools())
    monkeypatch.setattr(main, "_speak", fake_speak)
    done = asyncio.run(main._resume_after_unclear(session, text, "en"))
    return done, said


def test_these_ones_after_a_total_gives_the_total(monkeypatch):
    done, said = _run(_session(), "yes these ones", monkeypatch)
    assert done and "820" in said[0].replace(",", "") or done and "820" in said[0]


def test_a_garbled_reply_asks_the_same_question_again_and_keeps_it_open(monkeypatch):
    s = _session()
    s.last_multi = None
    pending = object()
    s.lapsed_pending = (pending, s.turn_count)
    s.pending_question = ("I understood: Thyroid and Sugar. Would you like the total?", s.turn_count - 1)
    done, said = _run(s, "eh eh you look", monkeypatch)
    assert done and said == ["I understood: Thyroid and Sugar. Would you like the total?"]
    assert s.pending_entity is pending  # the yes is still awaited


def test_nothing_to_resume_returns_false(monkeypatch):
    s = _session()
    s.last_multi = None
    done, said = _run(s, "eh eh you look", monkeypatch)
    assert not done and said == []
