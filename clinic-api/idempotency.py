"""Idempotent writes: a retried POST does what the first one did, once.

A voice agent on a bad line retries: the connection drops after the server has committed, the client never hears the
answer and sends the same request again. Without a guard that is a second booking, a second SMS, a second callback.

The contract (the standard `Idempotency-Key` header, as used by payment APIs):

  * the client sends `Idempotency-Key: <opaque string>` with a write, and REUSES it when it retries the same operation;
  * the first request with a key runs and its JSON answer is stored under (key, scope);
  * a repeat with the same key AND the same body returns the stored answer, and runs nothing;
  * the same key with a DIFFERENT body is a client bug and is refused (422), never silently treated as either;
  * no key: the write runs as before, and the endpoints whose operation has a natural identity (an SMS to the same
    number with the same text, a callback for the same call and window, ...) de-duplicate on that instead.

Race: two concurrent requests with one key may both run before either stores its answer; the unique index makes the
second store fail, and that request then returns the first one's stored answer. It narrows a retry storm to at most one
extra execution of an operation that is itself made safe to repeat by its own guards (holds, confirms, unique slots).
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json

from fastapi import HTTPException
from models import IdempotencyRecord
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

KEY_HEADER = "idempotency-key"
MAX_KEY_CHARS = 200

_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "idempotency_key", default=None
)


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
        parts[name] = (
            value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        )
    return hashlib.sha256(
        json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


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
            rec = db.query(IdempotencyRecord).filter_by(key=key, scope=scope).first()
            if rec is not None:
                return _replay(rec, digest)
            result = fn(*args, **kwargs)
            try:
                payload = json.dumps(result, default=str)
            except (TypeError, ValueError):
                return result  # not JSON-shaped (a raw Response): nothing to store
            try:
                db.add(
                    IdempotencyRecord(
                        key=key, scope=scope, request_hash=digest, response_json=payload
                    )
                )
                db.commit()
            except IntegrityError:  # a concurrent twin stored first: answer as it did
                db.rollback()
                rec = (
                    db.query(IdempotencyRecord).filter_by(key=key, scope=scope).first()
                )
                if rec is not None:
                    return _replay(rec, digest)
            return result

        return wrapper

    return deco


def _replay(rec: IdempotencyRecord, digest: str):
    if rec.request_hash != digest:
        raise HTTPException(
            status_code=422,
            detail="Idempotency-Key was already used with a different request",
        )
    return json.loads(rec.response_json)
