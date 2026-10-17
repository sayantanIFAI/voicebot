"""Run on the pod:  python tools/spelled_gate_live.py   (env: CLINIC_API_TOKEN, CLINIC_API_BASE)
The live gate: for EVERY letters-only test code in the running catalogue, its spelled form in Bengali, Hindi and English must (1) be read back as
the code by agent/spelled_letters, and (2) find the test in the clinic's own name search. Prints one line per code and exits 1 on any miss."""
import os
import sys

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "clinic-api"))
import spelled_forms as sf  # noqa: E402

from agent import spelled_letters as sl  # noqa: E402
from agent.catalogue_forms import lab_test_code  # noqa: E402

base = os.environ.get("CLINIC_API_BASE", "http://localhost:8080")
headers = {"Authorization": f"Bearer {os.environ.get('CLINIC_API_TOKEN', '')}"}
cat = httpx.get(f"{base}/api/v1/catalogue", headers=headers, timeout=15).json()
codes = {}
for t in cat["tests"]:
    c = lab_test_code(t["name"])
    if c.isalpha() and c.isascii() and c.upper() == c and len(c) >= 2:
        codes.setdefault(c, set()).add(t["name"])
codes = {c: next(iter(n)) for c, n in codes.items() if len(n) == 1}
bad = 0
for code, name in sorted(codes.items()):
    for label, said in (("bn", " ".join(sf.BN_LETTER[c] for c in code)), ("hi", " ".join(sf.HI_LETTER[c] for c in code)), ("en", " ".join(code.lower()))):
        sentence = f"{said} test"
        out, found = sl.normalise_spelled_codes(sentence, codes)
        res = httpx.get(f"{base}/api/v1/tests/search", params={"name": code}, headers=headers, timeout=10).json()
        ok = found == [code] and (res.get("found") or res.get("did_you_mean") or res.get("ambiguous"))
        bad += 0 if ok else 1
        print(f"{'ok  ' if ok else 'MISS'} {code:5} {label}  {sentence!r:32} -> {out!r}  lookup: found={res.get('found')} name={res.get('test_name')} suggest={res.get('did_you_mean')}")
print("codes in catalogue:", sorted(codes))
sys.exit(1 if bad else 0)
