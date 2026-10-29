"""Owner, 2026-10-29: with several API worker processes, a retry that reaches another worker while the first request is still running must NOT run the operation again; a claim left by a dead process is taken over."""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app  # noqa: E402


def _claim_with(response_json):
    import idempotency
    from db import SessionLocal
    from fastapi import HTTPException
    from models import IdempotencyRecord

    db = SessionLocal()
    try:
        db.add(IdempotencyRecord(key="k-x", scope="s-x", request_hash="h", response_json=response_json))
        db.commit()
        try:
            got = idempotency._claim(db, "k-x", "s-x", "h")
            return ("owns" if got is None else "replay", None)
        except HTTPException as e:
            return ("http", e.status_code)
    finally:
        db.close()


def test_a_claim_held_by_a_live_other_process_is_busy():
    with clinic_app(sample_patients=False):
        import idempotency

        other_live_pid = os.getppid() if os.getppid() not in (0, os.getpid()) else 1
        assert idempotency._process_alive(str(other_live_pid))
        assert _claim_with(f"{idempotency._RUNNING}{other_live_pid}") == ("http", 409)


def test_a_claim_from_a_dead_process_is_taken_over():
    with clinic_app(sample_patients=False):
        import idempotency

        assert not idempotency._process_alive("999999999")
        assert _claim_with(f"{idempotency._RUNNING}999999999") == ("owns", None)


def test_a_finished_answer_is_replayed():
    with clinic_app(sample_patients=False):
        assert _claim_with('{"success": true}')[0] == "replay"


def test_garbage_in_the_marker_counts_as_dead():
    with clinic_app(sample_patients=False):
        import idempotency

        assert not idempotency._process_alive("not-a-pid")
        assert not idempotency._process_alive(str(os.getpid()))  # our own claim that is not in _live died inside this process
