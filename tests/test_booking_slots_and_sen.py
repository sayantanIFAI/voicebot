"""One booking per 10-minute slot; Dr. A. Sen sits every Thursday and Sunday 10:00-12:00 (owner, 2026-10-17)."""
import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "clinic-api"))


def test_slots_are_ten_minutes_apart():
    bs = importlib.import_module("booking_service")
    slots = bs._generate_slots("10:00", "12:00")
    assert slots[:4] == ["10:00", "10:10", "10:20", "10:30"] and slots[-1] == "11:50" and len(slots) == 12


def test_the_api_and_the_service_use_the_same_step():
    src = (ROOT / "clinic-api" / "main.py").read_text(encoding="utf-8")
    bs = importlib.import_module("booking_service")
    assert f"SLOT_STEP_MIN = {bs.SLOT_STEP_MIN}" in src


def test_dr_sen_sits_thursday_and_sunday_ten_to_twelve():
    seed = importlib.import_module("seed")
    o = seed.SCHEDULE_OVERRIDES["Dr. A. Sen"]
    assert sorted(o["days"]) == [3, 6] and (o["start"], o["end"]) == ("10:00", "12:00")
