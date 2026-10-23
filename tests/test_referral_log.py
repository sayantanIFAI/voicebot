"""Unit tests for clinic-api/referral_log.py (KCD-lead-1): the CSV worklist an inter-center referral call is
logged to. Pure file I/O, no database -- no clinic-api app import needed, so this runs anywhere.

    python -m pytest tests/test_referral_log.py -v
"""

import csv
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLINIC_API_DIR = os.path.join(REPO_ROOT, "clinic-api")


@pytest.fixture()
def referral_log(monkeypatch, tmp_path):
    if CLINIC_API_DIR not in sys.path:
        sys.path.insert(0, CLINIC_API_DIR)
    sys.modules.pop("referral_log", None)
    import referral_log as rl

    csv_path = tmp_path / "referral_callbacks.csv"
    monkeypatch.setattr(rl, "CSV_PATH", str(csv_path))
    return rl, str(csv_path)


def _rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_log_referral_creates_the_file_with_a_header_on_first_write(referral_log):
    rl, csv_path = referral_log
    assert not os.path.exists(csv_path)
    result = rl.log_referral(patient_name="Ravi Das", phone="9800000001", doctor_name="Sen", notes="from Malda branch")
    assert result["success"] is True and result["logged_at"]
    assert os.path.exists(csv_path)
    rows = _rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["patient_name"] == "Ravi Das"
    assert rows[0]["phone"] == "9800000001"
    assert rows[0]["doctor_name"] == "Sen"
    assert rows[0]["notes"] == "from Malda branch"
    assert rows[0]["symptom_description"] == ""  # blank fields are kept blank, never invented
    assert rows[0]["logged_at"]


def test_log_referral_appends_rather_than_overwriting(referral_log):
    rl, csv_path = referral_log
    rl.log_referral(patient_name="Ravi Das", phone="9800000001")
    rl.log_referral(patient_name="Mina Das", phone="9800000002")
    rows = _rows(csv_path)
    assert len(rows) == 2
    assert [r["patient_name"] for r in rows] == ["Ravi Das", "Mina Das"]


def test_log_referral_never_requires_any_field(referral_log):
    """Nothing usable was said on the call -- the row is still written (never silently dropped), just blank."""
    rl, csv_path = referral_log
    result = rl.log_referral()
    assert result["success"] is True
    rows = _rows(csv_path)
    assert rows[0] == {
        "logged_at": rows[0]["logged_at"],
        "patient_name": "",
        "phone": "",
        "doctor_name": "",
        "symptom_description": "",
        "notes": "",
    }
