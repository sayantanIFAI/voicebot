"""Live replay 2026-10-28: "डॉक्टर् अरुण कुमार" (a title with a halant, said by the recogniser) was refused as 'no such doctor' by the given-name check; the new Dr. Arun Kumar Saha exists."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app  # noqa: E402


def _found(client, name):
    r = client.get("/api/v1/doctors/availability", params={"name": name}).json()
    return r


def test_a_title_with_a_halant_does_not_hide_the_doctor():
    with clinic_app(sample_patients=False) as (_app, client):
        for said in ("डॉक्टर् अरुण कुमार", "डॉक्टर अरुण कुमार", "अरुण कुमार", "ডাক্তার অরুণ কুমার", "Dr Arun Kumar Saha"):
            r = _found(client, said)
            assert r.get("found") and "Arun" in (r.get("doctor_name") or r.get("name") or str(r)), (said, r)


def test_a_given_name_that_is_nobody_still_is_not_the_surname_doctor():
    with clinic_app(sample_patients=False) as (_app, client):
        r = _found(client, "শুভদীপ মন্ডল")
        assert not r.get("found"), r


def test_sir_before_after_or_around_a_name_is_ignored():
    with clinic_app(sample_patients=False) as (_app, client):
        for said in (
            "Dr Arun Kumar Sir", "Arun Kumar Sir", "Arun Kumar Saha Sir", "Sir Arun Kumar", "Sir Arun Kumar Saha", "Dr Arun Kumar Saha Sir",
            "অরুণ কুমার স্যার", "ডাক্তার অরুণ কুমার সির", "স্যার অরুণ কুমার", "अरुण कुमार सर", "डॉक्टर् अरुण कुमार साहब", "Arun Kumar sar",
        ):
            r = _found(client, said)
            assert r.get("found") and "Arun" in str(r), (said, r)
        # a name that is nobody stays nobody, with or without Sir
        for said in ("Rahul Verma Sir", "Sir Rahul Verma", "রাহুল ভার্মা স্যার"):
            assert not _found(client, said).get("found"), said
