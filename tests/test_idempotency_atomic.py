"""clinic-api/idempotency.py claims the key BEFORE the operation runs (external review 2026-10-28)."""

import threading
import time

from fastapi import HTTPException
from tests._clinic_app import clinic_app


def _runner(op, results, key="k1"):
    import db as dbmod
    import idempotency

    session = dbmod.SessionLocal()
    idempotency.set_key_from_header(key)
    try:
        results.append(("ok", op(1, db=session)))
    except HTTPException as e:
        results.append(("http", e.status_code))
    finally:
        session.close()


def test_two_concurrent_requests_with_one_key_run_the_operation_once():
    with clinic_app():
        import idempotency

        calls, gate, results = [], threading.Event(), []

        @idempotency.idempotent("test.once")
        def op(x, db=None):
            calls.append(x)
            gate.wait(3)
            return {"done": True}

        first = threading.Thread(target=_runner, args=(op, results))
        first.start()
        time.sleep(0.3)  # the first now holds the claim and is running
        second = threading.Thread(target=_runner, args=(op, results))
        second.start()
        second.join(3)
        gate.set()
        first.join(3)
        assert len(calls) == 1
        assert ("http", 409) in results and ("ok", {"done": True}) in results


def test_a_finished_key_replays_the_stored_answer_and_a_failed_one_can_be_retried():
    with clinic_app():
        import db as dbmod
        import idempotency

        calls = []

        @idempotency.idempotent("test.replay")
        def op(x, db=None):
            calls.append(x)
            return {"n": len(calls)}

        s = dbmod.SessionLocal()
        idempotency.set_key_from_header("k2")
        assert op(1, db=s) == {"n": 1}
        assert op(1, db=s) == {"n": 1}  # replayed, not run again
        assert len(calls) == 1

        fails = []

        @idempotency.idempotent("test.fail")
        def bad(x, db=None):
            fails.append(x)
            if len(fails) == 1:
                raise RuntimeError("boom")
            return {"ok": True}

        idempotency.set_key_from_header("k3")
        try:
            bad(1, db=s)
        except RuntimeError:
            pass
        assert bad(1, db=s) == {"ok": True}  # the failed attempt released its claim
        s.close()


def test_a_stale_claim_left_by_a_dead_process_is_taken_over():
    with clinic_app():
        import db as dbmod
        import idempotency
        from models import IdempotencyRecord

        s = dbmod.SessionLocal()
        idempotency.set_key_from_header("k4")

        @idempotency.idempotent("test.stale")
        def op(x, db=None):
            return {"ran": True}

        digest = idempotency._fingerprint({"x": 1})
        s.add(IdempotencyRecord(key="k4", scope="test.stale", request_hash=digest, response_json=""))
        s.commit()
        assert op(x=1, db=s) == {"ran": True}  # not in the live set: stale, so it runs
        s.close()
