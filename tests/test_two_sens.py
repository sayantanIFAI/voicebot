"""Owner, 2026-10-17: two doctors named Sen with different first names; a doctor is ALWAYS named by the whole name."""
import importlib
import pathlib
import sys
from types import SimpleNamespace

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "clinic-api"))


def test_the_second_dr_sen_is_defined_with_a_different_first_name_in_three_scripts():
    seed = importlib.import_module("seed")
    names = seed.doctor_full_names()
    a, b = names["Dr. A. Sen"], names["Dr. B. Sen"]
    assert a[0] == "Dr. Arindam Sen" and b[0] == "Dr. Biswajit Sen"
    assert a[1] != b[1] and a[2] != b[2] and all(a) and all(b)  # Bengali and Devanagari full names exist and differ
    dept, short, _, template = seed.EXTRA_DOCTORS[0]
    assert (dept, short) == ("General Medicine", "Dr. B. Sen") and template["start"] == "10:00" and template["end"] == "12:00"


def test_a_doctor_is_named_by_the_whole_name_even_when_he_is_the_only_one_with_that_surname():
    main = importlib.import_module("main")
    roy = SimpleNamespace(id=5, name="Dr. N. Roy", full_name="Dr. Nirmal Roy")

    class _Q:
        def filter(self, *a, **k):
            return self

        def count(self):
            return 0  # no twin

    db = SimpleNamespace(query=lambda *_: _Q())
    assert main._doctor_ref(db, roy) == "Dr. Nirmal Roy"
    bare = SimpleNamespace(id=6, name="Dr. X. Y", full_name="")
    assert main._doctor_ref(db, bare) == "Dr. X. Y"  # no whole name recorded: the short one, as before


def test_spoken_department_names_use_the_whole_name_in_every_language():
    bs = importlib.import_module("booking_service")
    d = SimpleNamespace(name="Dr. N. Roy", full_name="Dr. Nirmal Roy", full_name_bn="নির্মল রায়", full_name_hi="निर्मल रॉय")
    assert bs._doctor_spoken_name(d, "en") == "Dr. Nirmal Roy"
    assert bs._doctor_spoken_name(d, "bn") == "নির্মল রায়"
    assert bs._doctor_spoken_name(d, "hi") == "निर्मल रॉय"
