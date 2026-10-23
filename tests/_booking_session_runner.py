"""Subprocess runner for tests/test_booking_sessions.py: a MULTI-turn booking through the real _dispatch_turn of main.py and main_pcm.py (one session, many caller turns).

Same stand-ins as tests/_turn_harness_runner.py, which it extends: the recogniser's text, the voice, the websocket, the intent model (scripted) and the clinic API (the real
booking_service / geo_service over a throw-away database, plus the booking calls faked). Reads {scenario name: [[text, lang], ...]} from stdin, prints one JSON document.

    python tests/_booking_session_runner.py < scenarios.json
"""

import asyncio
import importlib.util
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("hr", os.path.join(REPO, "tests", "_turn_harness_runner.py"))
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
sys.path.insert(0, os.path.join(REPO, "clinic-api"))
import geo_service  # noqa: E402

geo_service.seed_geo(r._DBSESSION)


async def booking_conflict(self, phone, date, time):
    self.calls.append(("booking_conflict", phone))
    return {"conflict": False}


async def geo_match(self, text):
    self.calls.append(("geo_match", text))
    return geo_service.match_geo(r._DBSESSION, text)


async def doctor_weekdays(self, name):
    return {"found": True, "doctor_name": name, "weekdays": [0, 2, 4], "hours": {}}


async def hold_slot(self, doctor_name, date, time_slot):
    self.calls.append(("hold_slot", time_slot))
    return {"success": True, "hold_token": "T1", "doctor_id": 1, "date": date, "time_slot": time_slot}


async def get_patient_senior(self, phone, caller_phone=None):
    return False


async def confirm_booking(self, *a, **k):
    self.calls.append(("confirm_booking", k.get("phone") or (a[5] if len(a) > 5 else None)))
    return {
        "success": True,
        "confirmation_id": "KCD-1",
        "doctor_name": "Dr. A. Sen",
        "date": "2026-10-30",
        "time_slot": "10:00",
    }


async def send_payment_link(self, *a, **k):
    self.calls.append(("send_payment_link",))
    return {"success": True}


async def send_booking_confirmation_whatsapp(self, *a, **k):
    self.calls.append(("send_booking_confirmation_whatsapp",))
    return {"success": True}


for fn in (
    booking_conflict,
    geo_match,
    doctor_weekdays,
    hold_slot,
    get_patient_senior,
    confirm_booking,
    send_payment_link,
    send_booking_confirmation_whatsapp,
):
    setattr(r.FakeTools, fn.__name__, fn)

BOOK = {"intent": "book_appointment", "slots": {"doctor_name": "Sen", "date": "2026-10-30", "time_slot": "10:00"}}
r.SCRIPTED["I want to book Dr Sen"] = BOOK
r.SCRIPTED["আমি ডাক্তার সেনকে দেখাতে চাই"] = BOOK
r.SCRIPTED["मुझे डॉक्टर सेन को दिखाना है"] = BOOK
for nm in ("Rahul Dey", "রাহুল দে", "राहुल डे"):
    r.SCRIPTED[nm] = {"intent": "book_appointment", "slots": {"patient_name": nm}}

NOBOOK = {"intent": "book_appointment", "slots": {}}
for _t in (
    "আমি অ্যাপয়েন্টমেন্ট বুক করতে চাই",
    "আমার কোমরে ব্যথা",
    "मुझे अपॉइंटमेंट बुक करना है",
    "मेरी कमर में दर्द है",
    "I want to book an appointment",
    "I have back pain",
    "কানে ব্যথা",
    "I have ear pain",
):
    r.SCRIPTED[_t] = NOBOOK
SCEN = json.loads(sys.stdin.read())  # {name: [[text, lang], ...]}


def run():
    out = {}
    for name, filename in (("main", "main.py"), ("main_pcm", "main_pcm.py")):
        mod = r._load("voicebot_" + name, filename)
        orig = mod.CallSession
        out[name] = {}
        for sc_name, turns in SCEN.items():
            cache = [None]

            def make(ws, orig=orig, cache=cache):
                if cache[0] is None:
                    cache[0] = orig(ws)
                else:
                    cache[0].ws = ws
                return cache[0]

            mod.CallSession = make
            out[name][sc_name] = asyncio.run(r.run_entry(mod, [tuple(t) for t in turns], True))
    return out


res = run()
print("@@RESULT@@" + json.dumps(res, ensure_ascii=False))
