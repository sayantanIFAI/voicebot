"""Integration check for the test_prep and clinic_faq endpoints added
alongside agent/fast_path.py's test_prep/clinic_faq intents.

Runs clinic-api directly against a throwaway SQLite file via FastAPI's
TestClient -- no live pod needed, unlike tests/test_smoke.py. This is
exactly the kind of check CLAUDE.md asks for: verified against a live
system (even if a local, temporary one), not asserted from reading the
code.

    python -m pytest tests/test_clinic_api_new_endpoints.py -v
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


@pytest.fixture()
def clinic_client():
    """A fresh clinic-api app + freshly seeded SQLite file per test."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)

    if CLINIC_API_DIR not in sys.path:
        sys.path.insert(0, CLINIC_API_DIR)

    # Fresh import each time so module-level `engine`/`SessionLocal` in
    # db.py pick up THIS test's CLINIC_DB_PATH rather than a previous
    # test's cached module.
    for mod in ("main", "db", "models", "seed"):
        sys.modules.pop(mod, None)

    import main as clinic_main  # noqa: PLC0415
    from fastapi.testclient import TestClient

    with TestClient(clinic_main.app) as client:
        yield client

    try:
        os.remove(db_path)
    except OSError:
        pass


def test_catalogue_includes_faq_topics(clinic_client):
    resp = clinic_client.get("/api/v1/catalogue")
    assert resp.status_code == 200
    payload = resp.json()
    assert "faq_topics" in payload
    topics = {row["topic"] for row in payload["faq_topics"]}
    assert "hours" in topics
    assert "location" in topics
    hours_row = next(r for r in payload["faq_topics"] if r["topic"] == "hours")
    assert any("সময়" in k for k in hours_row["keywords_bn"])


def test_faq_answer_found(clinic_client):
    resp = clinic_client.get("/api/v1/faq", params={"topic": "hours"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["topic"] == "hours"
    assert "খোলা" in body["answer"] or "সময়" in body["answer"]


def test_faq_answer_not_found(clinic_client):
    resp = clinic_client.get("/api/v1/faq", params={"topic": "does_not_exist"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False


def test_prep_found_for_fasting_test(clinic_client):
    resp = clinic_client.get("/api/v1/tests/prep", params={"name": "লিপিড প্রোফাইল"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["fasting_required"] is True
    assert "উপবাস" in body["prep_instructions"]


def test_prep_found_for_non_fasting_test(clinic_client):
    resp = clinic_client.get("/api/v1/tests/prep", params={"name": "সিবিসি"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["fasting_required"] is False
    assert body["prep_instructions"]  # the generic default, not empty


def test_prep_not_found_returns_suggestions_shape(clinic_client):
    resp = clinic_client.get("/api/v1/tests/prep", params={"name": "নাথিং টেস্ট"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert "did_you_mean" in body


def test_existing_test_search_endpoint_still_works(clinic_client):
    """Regression guard: the _find_test refactor in clinic-api/main.py must
    not change search_test's existing, already-relied-upon behaviour."""
    resp = clinic_client.get("/api/v1/tests/search", params={"name": "সিবিসি"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["rate_inr"] == 400
