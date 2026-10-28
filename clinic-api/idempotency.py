"""Idempotent writes: a retried POST does what the first one did, once.

A voice agent on a bad line retries: the connection drops after the server has committed, the client never hears the
answer and sends the same request again. Without a guard that is a second booking, a second SMS, a second callback.

The contract (the standard `Idempotency-Key` header, as used by payment APIs):

  * the client sends `Idempotency-Key: <opaque string>` with a write, and REUSES it when it retries the same operation;
  * the key is CLAIMED FIRST, atomically, before the operation runs (an INSERT that the unique (key, scope) constraint
    makes succeed for exactly one request); only the claimant runs the operation and then stores its JSON answer;
  * a repeat with the same key AND the same body after the answer is stored returns that answer and runs nothing;
  * a repeat while the first is STILL RUNNING is refused with 409 + Retry-After (the client waits and asks again with the
    same key), never executed a second time -- this closes the check-then-act race of the earlier version, where two
    concurrent requests could both run (external review 2026-10-28);
  * the same key with a DIFFERENT body is a client bug and is refused (422);
  * an operation that FAILS releases its claim, so a retry may run it again (a failure is not a stored answer);
  * no key: the write runs as before, and the endpoints whose operation has a natural identity (an SMS to the same
    number with the same text, a callback for the same call and window, ...) de-duplicate on that instead.

A claim row with an empty answer means "running". The in-memory set below says which of those claims belong to a live request
IN THIS PROCESS; an empty claim that is not in it (the service crashed or restarted mid-request) is stale and is taken over.
This assumes ONE clinic-api process (as deployed: uvicorn without --workers). With several workers the "live" set would have to
move into the database (a lease timestamp column).

What this does NOT do: it cannot undo an external side effect (an SMS already handed to a provider) -- that needs a transactional
outbox with its own unique send key, to be built when a real provider is connected (SmsOutbox is the placeholder for it).
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import threading

from fastapi import HTTPException
from models import IdempotencyRecord
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

KEY_HEADER = "idempotency-key"
MAX_KEY_CHARS = 200

_key: contextvars.ContextVar[str | None] = contextvars.ContextVar("idempotency_key", default=None)
_live: set[tuple[str, str]] = set()  # claims held by a request that is running right now in this process
_live_lock = threading.Lock()


def set_key_from_header(value: str | None) -> None:
    """Called by the request middleware; a blank or oversized key is ignored (the write then runs unguarded)."""
    value = (value or "").strip()
    _key.set(value if 0 < len(value) <= MAX_KEY_CHARS else None)


def current_key() -> str | None:
    return _key.get()


def _fingerprint(kwargs: dict) -> str:
    """A stable hash of the request body(s): every pydantic argument, every path/query value; never the DB session."""
    parts = {}
    for name, value in sorted(kwargs.items()):
        if isinstance(value, Session):
            continue
        parts[name] = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _claim(db: Session, key: str, scope: str, digest: str) -> IdempotencyRecord | None:
    """None: this request now OWNS the key and must run the operation. A record: the answer is already stored (replay it)."""
    for _ in range(3):
        try:
            db.add(IdempotencyRecord(key=key, scope=scope, request_hash=digest, response_json=""))
            db.commit()
            with _live_lock:
                _live.add((key, scope))
            return None
        except IntegrityError:
            db.rollback()
        rec = db.query(IdempotencyRecord).filter_by(key=key, scope=scope).first()
        if rec is None:
            continue  # released between the two statements: claim again
        if rec.request_hash != digest:
            raise HTTPException(status_code=422, detail="Idempotency-Key was already used with a different request")
        if rec.response_json:
            return rec
        with _live_lock:
            running = (key, scope) in _live
        if running:
            raise HTTPException(status_code=409, detail="A request with this Idempotency-Key is still running; retry with the same key shortly", headers={"Retry-After": "1"})
        db.delete(rec)  # a stale claim from a request that died with the process: take it over
        db.commit()
    raise HTTPException(status_code=409, detail="Idempotency-Key is busy; retry with the same key shortly", headers={"Retry-After": "1"})


def _release(db: Session, key: str, scope: str) -> None:
    """The operation failed or returned something that cannot be stored: drop the claim so a retry can run."""
    with _live_lock:
        _live.discard((key, scope))
    try:
        db.rollback()
        db.query(IdempotencyRecord).filter_by(key=key, scope=scope, response_json="").delete()
        db.commit()
    except Exception:  # noqa: BLE001 - never mask the original outcome
        db.rollback()


def idempotent(scope: str):
    """Decorator for a synchronous endpoint that takes `db: Session`. The endpoint's signature is unchanged (FastAPI
    still sees the original parameters through functools.wraps)."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            key, db = current_key(), kwargs.get("db")
            if not key or db is None:
                return fn(*args, **kwargs)
            digest = _fingerprint(kwargs)
            stored = _claim(db, key, scope, digest)
            if stored is not None:
                return json.loads(stored.response_json)
            try:
                result = fn(*args, **kwargs)
            except BaseException:
                _release(db, key, scope)
                raise
            try:
                payload = json.dumps(result, default=str)
            except (TypeError, ValueError):
                _release(db, key, scope)
                return result  # not JSON-shaped (a raw Response): nothing to store
            try:
                db.query(IdempotencyRecord).filter_by(key=key, scope=scope).update({"response_json": payload})
                db.commit()
            except Exception:  # noqa: BLE001 - the operation already happened; a failure to record it must not turn it into an error
                db.rollback()
            finally:
                with _live_lock:
                    _live.discard((key, scope))
            return result

        return wrapper

    return deco
