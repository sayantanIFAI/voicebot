"""sourav, 2026-10-14: a caller naming a doctor in ANY accent must reach THAT doctor, in every call language.

WHY: the doctor appointment flow starts with "is Dr X sitting today?"; if the name is not recognised the whole booking never starts.
FLOW: real clinic API catalogue -> every doctor x every spelling/script (agent/doctor_cues.py) x sentence frames x the three call
languages -> real FastPath.resolve. Two outcomes are allowed per sentence: the RIGHT doctor is answered (doctor_availability), or
the fast path abstains (the model / "which one?" takes over). A DIFFERENT doctor being answered is always a failure -- that is a
wrong fact spoken to a patient. The `caught` share is asserted high so a regression in recognition is visible.
Set DOCTOR_CUE_REPORT=<file> to also write the human-readable list (doctors, spellings, sentences, results)."""

import datetime
import os
import sys
from collections import Counter, defaultdict

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "clinic-api"))

from agent import doctor_cues as dc  # noqa: E402
from agent.fast_path import Catalogue, FastPath  # noqa: E402
from agent.fast_path import _normalize as norm  # noqa: E402

TODAY = datetime.date(2026, 9, 25)


@pytest.fixture(scope="module")
def payload():
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        return client.get("/api/v1/catalogue").json()


def _script(s: str) -> str:
    if any("ঀ" <= ch <= "৿" for ch in s):
        return "bn"
    if any("ऀ" <= ch <= "ॿ" for ch in s):
        return "hi"
    return "lat"


def _sentences(d: dict):
    """(spoken name with/without title, script, sentence) for one doctor."""
    forms = dc.doctor_extra_forms(d)
    out = []
    for f in forms:
        sc = _script(f)
        if sc == "bn":
            names, frames = [f, f"{dc.TITLES['bn'][0]} {f}", f"{dc.TITLES['bn'][1]} {f}"], dc.FRAMES["bn"][:3] + dc.FRAMES["mixed"][:3]
        elif sc == "hi":
            names, frames = [f, f"{dc.TITLES['hi'][0]} {f}", f"{dc.TITLES['hi'][2]} {f}"], dc.FRAMES["hi"][:3] + dc.FRAMES["mixed"][:3]
        else:
            names = [f, f"dr {f}", f"daktar {f}"]
            frames = dc.FRAMES["en"][:3] + dc.FRAMES["mixed"][:4]
        for n in names:
            for fr in frames:
                out.append((n, sc, fr.format(d=n)))
    return out


def test_every_doctor_in_every_spelling_reaches_the_right_doctor_or_nobody(payload):
    cat = Catalogue(payload)
    fp = FastPath(cat, today=TODAY)
    stats = Counter()
    wrong, per_doc = [], defaultdict(Counter)
    for i, d in enumerate(payload["doctors"]):
        for name, sc, sentence in _sentences(d):
            for lang in ("bn", "hi", "en"):
                r = fp.resolve(sentence, lang)
                if r is None:
                    stats["abstain"] += 1
                    per_doc[d["name"]]["abstain"] += 1
                    continue
                if r.intent != "doctor_availability":
                    stats["other_intent"] += 1
                    per_doc[d["name"]]["other"] += 1
                    continue
                owners = cat._doctor_owners.get((lang, norm(r.slots["doctor_name"])), set())
                if i in owners:
                    stats["caught"] += 1
                    per_doc[d["name"]]["caught"] += 1
                else:
                    stats["WRONG"] += 1
                    wrong.append((sentence, lang, r.slots["doctor_name"], d["name"]))
    total = sum(stats.values())
    path = os.environ.get("DOCTOR_CUE_REPORT")
    if path:
        _write_report(path, payload, stats, per_doc, wrong, total)
    assert not wrong, f"a DIFFERENT doctor was answered: {wrong[:8]}"
    assert stats["caught"] / total > 0.5, dict(stats)


def _write_report(path, payload, stats, per_doc, wrong, total):
    lines = ["DOCTOR CUE LIST (generated from the clinic catalogue)", f"Doctors in the catalogue: {len(payload['doctors'])}",
             f"Sentences tried: {total} (each doctor x spelling x title x sentence shape x 3 call languages)",
             f"Right doctor answered: {stats['caught']}   fast path stepped aside (model decides): {stats['abstain'] + stats['other_intent']}   WRONG doctor: {stats['WRONG']}", ""]
    lines.append("SENTENCE SHAPES  ({d} = doctor name, with or without title)")
    for k, v in dc.FRAMES.items():
        lines += [f"  [{k}]"] + [f"    {x}" for x in v]
    lines += ["", "TITLES"] + [f"  [{k}] {', '.join(v)}" for k, v in dc.TITLES.items()] + ["", "PER DOCTOR: every spoken form recognised (all scripts)"]
    for d in payload["doctors"]:
        c = per_doc[d["name"]]
        lines.append(f"- {d['name']}  ({d.get('full_name','')} / {d.get('full_name_bn','')} / {d.get('full_name_hi','')})  caught={c['caught']} stepped-aside={c['abstain'] + c['other']}")
        lines.append("    " + " | ".join(dc.doctor_extra_forms(d)))
    open(path, "w", encoding="utf-8").write("\n".join(lines))
