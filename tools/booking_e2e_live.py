"""Run on the pod:  python tools/booking_e2e_live.py   (env: CLINIC_API_TOKEN, CLINIC_API_BASE)
Books, finds by phone number, moves, confirms and cancels one appointment through the clinic API, checking after every step that the per-slot
store (slot_locks / appointments) says what it should. Uses a fictional phone number; the cancelled test row stays as history (nothing is deleted)."""
import datetime
import os
import sys

import httpx

BASE = os.environ.get("CLINIC_API_BASE", "http://localhost:8080")
H = {"Authorization": f"Bearer {os.environ.get('CLINIC_API_TOKEN', '')}"}
PHONE = "9000000017"
c = httpx.Client(base_url=BASE, headers=H, timeout=20)
bad = 0


def check(label, ok, extra=""):
    global bad
    bad += 0 if ok else 1
    print(("ok   " if ok else "FAIL ") + label, extra)


def next_weekday(wd, after=0):
    d = datetime.date.today() + datetime.timedelta(days=after)
    while d.weekday() != wd:
        d += datetime.timedelta(days=1)
    return d.isoformat()


sunday, thursday = next_weekday(6, 1), next_weekday(3, 1)
fs = c.get("/api/v1/doctors/free-slots", params={"name": "Sen", "date": sunday}).json()
check(f"Dr Sen sits on Sunday {sunday} 10:00-12:00 in 10-minute slots", fs.get("sits") and fs["chamber_hours"] == "10:00-12:00" and len(fs["valid_slots"]) == 12, fs.get("valid_slots"))
fs_t = c.get("/api/v1/doctors/free-slots", params={"name": "Sen", "date": thursday}).json()
check(f"Dr Sen sits on Thursday {thursday}", fs_t.get("sits") and len(fs_t["valid_slots"]) == 12)
tue = c.get("/api/v1/doctors/free-slots", params={"name": "Sen", "date": next_weekday(1, 1)}).json()
check("Dr Sen does not sit on a Tuesday", not tue.get("sits"))

slot = fs["free_slots"][0]
hold = c.post("/api/v1/bookings/hold", json={"doctor_name": "Sen", "date": sunday, "time_slot": slot}).json()
check(f"hold {sunday} {slot}", hold.get("success"), {k: hold.get(k) for k in ("success", "reason")})
after_hold = c.get("/api/v1/doctors/free-slots", params={"name": "Sen", "date": sunday}).json()["free_slots"]
check("a held slot is no longer free", slot not in after_hold)

conf = c.post("/api/v1/bookings/confirm", json={"hold_token": hold["hold_token"], "doctor_id": hold["doctor_id"], "date": sunday, "time_slot": slot,
                                                 "patient_name": "Test Patient", "phone": PHONE, "caller_phone": PHONE}).json()
cid = conf.get("confirmation_id")
check("confirm returns a booking id", bool(cid), cid)

found = c.get("/api/v1/bookings/lookup", params={"phone": PHONE}).json()
mine = [b for b in found.get("bookings", []) if b.get("confirmation_id") == cid]
check("the booking is found by phone number alone", bool(mine), mine[:1])

moved = c.post("/api/v1/bookings/reschedule", json={"confirmation_id": cid, "new_date": thursday, "new_time_slot": fs_t["free_slots"][1], "caller_phone": PHONE}).json()
check(f"postpone to {thursday} {fs_t['free_slots'][1]}", moved.get("success"), {k: moved.get(k) for k in ("success", "reason")})
back = c.get("/api/v1/doctors/free-slots", params={"name": "Sen", "date": sunday}).json()["free_slots"]
check("the old slot is free again after the move", slot in back)
now = c.get("/api/v1/bookings/lookup", params={"phone": PHONE}).json()["bookings"]
check("lookup by phone shows the new date", any(b.get("date") == thursday for b in now), [(b.get("date"), b.get("time_slot"), b.get("status")) for b in now])

new_cid = next((b["confirmation_id"] for b in now if b.get("date") == thursday and b.get("status", "confirmed") == "confirmed"), cid)
res = c.post("/api/v1/bookings/resend", json={"confirmation_id": new_cid}).json()
check("confirmation can be sent again for the booking", bool(res) and res.get("success", True) is not False, res)

canc = c.post("/api/v1/bookings/cancel", json={"confirmation_id": new_cid, "confirm_charge": True, "caller_phone": PHONE}).json()
check("cancel", canc.get("success", False) or canc.get("status") == "cancelled", {k: canc.get(k) for k in ("success", "status", "reason")})
free_after = c.get("/api/v1/doctors/free-slots", params={"name": "Sen", "date": thursday}).json()["free_slots"]
check("the cancelled slot is free again", fs_t["free_slots"][1] in free_after)
sys.exit(1 if bad else 0)
