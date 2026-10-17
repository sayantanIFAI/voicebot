"""One booking per 10-minute slot; Dr. A. Sen sits every Thursday and Sunday 10:00-12:00 (owner, 2026-10-17)."""
import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "clinic-api"))


def test_slots_are_ten_minutes_apart_by_default(monkeypatch):
    monkeypatch.delenv("SLOT_STEP_MIN", raising=False)  # production default
    bs = importlib.reload(importlib.import_module("booking_service"))
    try:
        slots = bs._generate_slots("10:00", "12:00")
        assert slots[:4] == ["10:00", "10:10", "10:20", "10:30"] and slots[-1] == "11:50" and len(slots) == 12
    finally:
        monkeypatch.setenv("SLOT_STEP_MIN", "15")
        importlib.reload(bs)


def test_the_api_and_the_service_read_the_same_setting():
    src = (ROOT / "clinic-api" / "main.py").read_text(encoding="utf-8")
    bs_src = (ROOT / "clinic-api" / "booking_service.py").read_text(encoding="utf-8")
    line = 'SLOT_STEP_MIN = int(os.environ.get("SLOT_STEP_MIN", "10"))'
    assert line in src and line in bs_src


def test_dr_sen_sits_thursday_and_sunday_ten_to_twelve():
    seed = importlib.import_module("seed")
    o = seed.SCHEDULE_OVERRIDES["Dr. A. Sen"]
    assert sorted(o["days"]) == [3, 6] and (o["start"], o["end"]) == ("10:00", "12:00")
