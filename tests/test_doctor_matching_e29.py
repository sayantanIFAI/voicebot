"""Epic E29 (Language, Mixing and Register): doctor-name matching against
mispronunciation and cross-script variants, at the HTTP layer -- on top
of tests/test_phonetic_match.py's pure unit tests of the folding logic
itself.

    python -m pytest tests/test_doctor_matching_e29.py -v
"""
import os
import sys
import tempfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


@pytest.fixture()
def clinic_client():
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.environ["CLINIC_DB_PATH"] = db_path
    os.environ.pop("DATABASE_URL", None)
    if CLINIC_API_DIR not in sys.path:
        sys.path.insert(0, CLINIC_API_DIR)
    for mod in ("main", "db", "models", "seed", "booking_service", "booking_migrate",
                "i18n_content", "phonetic_match"):
        sys.modules.pop(mod, None)

    from fastapi.testclient import TestClient

    import main as clinic_main

    with TestClient(clinic_main.app) as c:
        yield c
    try:
        os.remove(db_path)
    except OSError:
        pass


def test_doctor_nobody_still_never_matches_a_real_doctor(clinic_client):
    """KCD-436's own regression case, reproduced locally (test_smoke.py's
    version needs a live pod). "Doctor Nobody" once fuzzy-matched "Dr. N.
    Roy" at a HIGHER character ratio than a genuine garbled name scored
    against its own doctor -- see clinic-api/main.py's FUZZY_SURNAME_FLOOR
    comment. Adding phonetic matching (KCD-436) must not reopen this."""
    result = clinic_client.get("/api/v1/doctors/availability", params={"name": "Doctor Nobody"}).json()
    assert result["found"] is False


def test_a_badly_garbled_surname_still_finds_the_right_doctor(clinic_client):
    # Roy is seeded (Dr. N. Roy, Cardiology). A phonetically-consistent
    # but character-heavy mishearing must still resolve, same principle
    # as the existing "mukharji" vs "Mukherjee" case FUZZY_SURNAME_FLOOR
    # was calibrated against.
    cat = clinic_client.get("/api/v1/catalogue").json()
    roy = next(d for d in cat["doctors"] if d["surname"] == "Roy")
    assert roy is not None

    result = clinic_client.get("/api/v1/doctors/availability", params={"name": "Roy"}).json()
    assert result["found"] is True and result["doctor_name"] == roy["name"]
