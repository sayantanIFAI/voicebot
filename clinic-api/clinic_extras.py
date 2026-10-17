"""Owner's 2026-10-18 additions to the clinic data, applied idempotently on every boot (nothing here overwrites a value a clinician has edited).

  * doctors.sex, so "a lady gynaecologist" filters the search;
  * departments the clinic needs for routing (General Surgery for a hernia, Neurology and Neurosurgery apart, Urology) with FICTIONAL placeholder
    doctors like every other seeded row, and the symptom routes that point at them;
  * relationship_rules, so "for my mother" tells the system the patient is a woman;
  * a doctor going on leave: the leave row, every confirmed appointment of that doctor in the range cancelled WITH the reason, and an apology queued
    for each patient (a message is queued, never claimed as sent -- see booking_service.queue_sms).
"""

from __future__ import annotations

import datetime
import logging

from models import (
    Appointment,
    Department,
    DepartmentRoute,
    Doctor,
    DoctorLeave,
    DoctorSchedule,
    RelationshipRule,
    SlotLock,
)
from sqlalchemy.orm import Session

log = logging.getLogger("clinic-api")

# short name -> "F" | "M". The given names are the fictional placeholders in seed.DOCTOR_FULL_NAMES.
DOCTOR_SEX: dict[str, str] = {
    "Dr. S. Mukherjee": "M", "Dr. A. Sen": "M", "Dr. B. Sen": "M", "Dr. P. Ghosh": "M", "Dr. R. Chowdhury": "M",
    "Dr. K. Bhattacharya": "M", "Dr. N. Roy": "M", "Dr. S. Banerjee": "M", "Dr. M. Dutta": "M",
    "Dr. S. Chatterjee": "F", "Dr. A. Basu": "F", "Dr. R. Mitra": "F", "Dr. P. Sengupta": "F",
    "Dr. D. Das": "M", "Dr. T. Bose": "M", "Dr. A. Kar": "M", "Dr. S. Nandi": "M",
    "Dr. R. Pal": "M", "Dr. K. Halder": "M", "Dr. S. Guha": "M", "Dr. B. Chanda": "M",
    "Dr. M. Saha": "F", "Dr. A. Dey": "M", "Dr. P. Adhikari": "F", "Dr. S. Bagchi": "M",
    "Dr. N. Biswas": "F", "Dr. R. Majumder": "F", "Dr. A. Mondal": "M", "Dr. S. Ganguly": "F",
    "Dr. K. Sinha": "M", "Dr. P. Ray": "M", "Dr. A. Sarkar": "M", "Dr. S. Chakraborty": "M",
    "Dr. S. Kundu": "M", "Dr. M. Mallick": "F", "Dr. R. Pramanik": "M", "Dr. T. Bhowmik": "F",
}

NEW_DEPARTMENTS = ["General Surgery", "Neurology", "Neurosurgery", "Urology"]

# (department, short name, whole name, bn, hi, qualifications, sex, fee, [bn surname aliases], [hi surname aliases], schedule)
_MWF_NOON = {"days": [0, 2, 4], "start": "11:00", "end": "13:00"}
_TTS_NOON = {"days": [1, 3, 5], "start": "11:00", "end": "13:00"}
_MWF_EVE = {"days": [0, 2, 4], "start": "17:00", "end": "19:00"}
_TTS_EVE = {"days": [1, 3, 5], "start": "17:30", "end": "19:30"}
NEW_DOCTORS = [
    ("General Surgery", "Dr. A. Lahiri", "Dr. Ashis Lahiri", "আশিস লাহিড়ী", "आशीष लाहिड़ी", "MBBS, MS (General Surgery)", "M", 700, ["লাহিড়ী", "লাহিড়ি"], ["लाहिड़ी"], _TTS_NOON),
    ("General Surgery", "Dr. S. Dasgupta", "Dr. Sharmila Dasgupta", "শর্মিলা দাশগুপ্ত", "शर्मिला दासगुप्ता", "MBBS, MS (General Surgery)", "F", 700, ["দাশগুপ্ত", "দাসগুপ্ত"], ["दासगुप्ता"], _MWF_NOON),
    ("Neurology", "Dr. J. Sanyal", "Dr. Joydeep Sanyal", "জয়দীপ সান্যাল", "जयदीप सान्याल", "MBBS, MD, DM (Neurology)", "M", 900, ["সান্যাল"], ["सान्याल"], _MWF_EVE),
    ("Neurosurgery", "Dr. A. Maitra", "Dr. Arup Maitra", "অরূপ মৈত্র", "अरूप मैत्रा", "MBBS, MS, MCh (Neurosurgery)", "M", 1000, ["মৈত্র", "মৈত্রা"], ["मैत्रा"], _TTS_EVE),
    ("Urology", "Dr. T. Ghoshal", "Dr. Tanmoy Ghoshal", "তন্ময় ঘোষাল", "तन्मय घोषाल", "MBBS, MS, MCh (Urology)", "M", 800, ["ঘোষাল"], ["घोषाल"], _TTS_NOON),
    # a male gynaecologist, so "a lady gynaecologist" is a real filter and not the whole department
    ("Gynaecology & Obstetrics", "Dr. A. Pathak", "Dr. Alok Pathak", "আলোক পাঠক", "आलोक पाठक", "MBBS, MS (Obs & Gynae)", "M", 700, ["পাঠক"], ["पाठक"], _MWF_NOON),
]

# (department, keywords_bn, keywords_hi, keywords_en) -- symptoms the clinic's clinicians route to that department. Specific on purpose: a bare
# "pain" must never route to a surgeon.
NEW_ROUTES = [
    (
        "General Surgery",
        ["হার্নিয়া", "হারনিয়া", "অ্যাপেনডিক্স", "অ্যাপেন্ডিক্স", "পাইলস", "অর্শ", "ফিস্টুলা", "ফিসার", "পিত্তথলির পাথর", "গলব্লাডারের পাথর", "পেটে চাকা", "ফোঁড়া কাটা"],
        ["हर्निया", "हार्निया", "अपेंडिक्स", "बवासीर", "पाइल्स", "फिस्टुला", "पित्त की पथरी", "गॉलब्लैडर में पथरी"],
        ["hernia", "appendix", "appendicitis", "piles", "haemorrhoids", "hemorrhoids", "fistula", "fissure", "gallstone", "gall bladder stone", "lump in abdomen"],
    ),
    (
        "Neurology",
        ["খিঁচুনি", "খিচুনি", "মৃগী", "মাইগ্রেন", "স্ট্রোক", "প্যারালাইসিস", "হাত কাঁপা", "স্মৃতিভ্রংশ", "ভুলে যাওয়া রোগ"],
        ["मिर्गी", "दौरे", "माइग्रेन", "लकवा", "स्ट्रोक", "हाथ काँपना"],
        ["epilepsy", "seizure", "fits", "migraine", "stroke", "paralysis", "tremor", "dementia"],
    ),
    (
        "Neurosurgery",
        ["ব্রেন টিউমার", "মস্তিষ্কের টিউমার", "মেরুদণ্ডের অপারেশন", "স্লিপ ডিস্কের অপারেশন", "মাথার অপারেশন"],
        ["ब्रेन ट्यूमर", "रीढ़ की सर्जरी", "स्लिप डिस्क की सर्जरी"],
        ["brain tumour", "brain tumor", "spine surgery", "slip disc surgery", "head surgery"],
    ),
    (
        "Urology",
        ["কিডনিতে পাথর", "কিডনির পাথর", "প্রোস্টেট", "প্রস্রাবে জ্বালা", "প্রস্রাবে রক্ত", "মূত্রথলির পাথর"],
        ["किडनी में पथरी", "किडनी की पथरी", "प्रोस्टेट", "पेशाब में जलन", "पेशाब में खून"],
        ["kidney stone", "prostate", "burning urine", "blood in urine", "bladder stone"],
    ),
]

# (key, sex, [bn], [hi], [en])
RELATIONSHIPS = [
    ("mother", "F", ["মা", "মাকে", "মায়ের", "আম্মা", "আম্মু", "মম"], ["माँ", "मां", "मम्मी", "अम्मी", "माता"], ["mother", "mom", "mum", "mummy", "mother's"]),
    ("father", "M", ["বাবা", "বাবাকে", "বাবার", "আব্বা", "আব্বু"], ["पिता", "पापा", "पिताजी", "अब्बू", "बाप"], ["father", "dad", "daddy", "papa", "father's"]),
    ("wife", "F", ["স্ত্রী", "বউ", "বৌ", "ঘরণী", "বিবি"], ["पत्नी", "बीवी", "वाइफ"], ["wife", "spouse (wife)"]),
    ("husband", "M", ["স্বামী", "বর"], ["पति", "हसबैंड"], ["husband"]),
    ("son", "M", ["ছেলে", "ছেলেকে", "পুত্র"], ["बेटा", "बेटे", "लड़का"], ["son", "boy"]),
    ("daughter", "F", ["মেয়ে", "মেয়েকে", "কন্যা"], ["बेटी", "लड़की"], ["daughter", "girl"]),
    ("sister", "F", ["বোন", "দিদি", "দিদিকে", "ননদ", "বৌদি"], ["बहन", "दीदी", "भाभी"], ["sister"]),
    ("brother", "M", ["ভাই", "দাদা", "দাদাকে", "ভাইকে"], ["भाई", "भैया"], ["brother"]),
    ("grandmother", "F", ["ঠাকুমা", "দিদা", "দিদিমা", "ঠাম্মা", "দিমা"], ["दादी", "नानी"], ["grandmother", "grandma", "granny"]),
    ("grandfather", "M", ["ঠাকুরদা", "দাদু", "দাদুকে", "দাদামশাই"], ["दादा", "नाना"], ["grandfather", "grandpa"]),
    ("aunt", "F", ["মাসি", "মাসিমা", "পিসি", "পিসিমা", "কাকিমা", "মামি", "জেঠিমা"], ["मौसी", "बुआ", "चाची", "मामी"], ["aunt", "auntie"]),
    ("uncle", "M", ["কাকা", "মামা", "জেঠু", "পিসেমশাই", "মেসোমশাই"], ["चाचा", "मामा", "ताऊ", "फूफा"], ["uncle"]),
    ("mother_in_law", "F", ["শাশুড়ি", "শাশুড়ী"], ["सास"], ["mother-in-law", "mother in law"]),
    ("father_in_law", "M", ["শ্বশুর", "শ্বশুরমশাই"], ["ससुर"], ["father-in-law", "father in law"]),
    ("child", "", ["বাচ্চা", "বাচ্চাকে", "শিশু"], ["बच्चा", "बच्चे", "बच्ची"], ["child", "baby", "kid"]),
    ("friend", "", ["বন্ধু", "বন্ধুকে"], ["दोस्त", "मित्र"], ["friend"]),
    ("relative", "", ["আত্মীয়", "পরিচিত"], ["रिश्तेदार"], ["relative", "relation"]),
]


def apply_extras(db: Session) -> dict:
    """Idempotent. Returns what it changed."""
    out = {"doctor_sex_set": 0, "departments": 0, "doctors": 0, "routes": 0, "relationships": 0}
    for d in db.query(Doctor).filter(Doctor.sex == "").all():
        s = DOCTOR_SEX.get(d.name)
        if s:
            d.sex = s
            out["doctor_sex_set"] += 1
    by_name = {d.name: d for d in db.query(Department).all()}
    for name in NEW_DEPARTMENTS:
        if name not in by_name:
            dept = Department(name=name)
            db.add(dept)
            db.flush()
            by_name[name] = dept
            out["departments"] += 1
    have = {d.name for d in db.query(Doctor).all()}
    for dept, short, full, bn, hi, quals, sex, fee, al_bn, al_hi, sched in NEW_DOCTORS:
        if short in have:
            continue
        doc = Doctor(
            name=short, full_name=full, full_name_bn=bn, full_name_hi=hi, qualifications=quals, aliases_bn="|".join(al_bn), aliases_hi="|".join(al_hi),
            department_id=by_name[dept].id, consultation_fee_inr=fee, sex=sex,
        )
        db.add(doc)
        db.flush()
        for wd in sched["days"]:
            db.add(DoctorSchedule(doctor_id=doc.id, weekday=wd, start_time=sched["start"], end_time=sched["end"]))
        out["doctors"] += 1
    routed = {r.department_id: r for r in db.query(DepartmentRoute).all()}
    for dept, kw_bn, kw_hi, kw_en in NEW_ROUTES:
        d = by_name.get(dept)
        if not d:
            continue
        row = routed.get(d.id)
        if row is None:
            db.add(DepartmentRoute(department_id=d.id, keywords_bn="|".join(kw_bn), keywords_hi="|".join(kw_hi), keywords_en="|".join(kw_en)))
            out["routes"] += 1
            continue
        for attr, new in (("keywords_bn", kw_bn), ("keywords_hi", kw_hi), ("keywords_en", kw_en)):
            cur = [k for k in (getattr(row, attr) or "").split("|") if k]
            add = [k for k in new if k not in cur]
            if add:
                setattr(row, attr, "|".join(cur + add))
                out["routes"] += 1
    known = {r.relationship_key for r in db.query(RelationshipRule).all()}
    for key, sex, bn, hi, en in RELATIONSHIPS:
        if key not in known:
            db.add(RelationshipRule(relationship_key=key, inferred_sex=sex, keywords_bn="|".join(bn), keywords_hi="|".join(hi), keywords_en="|".join(en)))
            out["relationships"] += 1
    db.commit()
    return out


def relationship_rules(db: Session) -> list[dict]:
    return [
        {
            "relationship": r.relationship_key,
            "sex": r.inferred_sex,
            "keywords_bn": [k for k in r.keywords_bn.split("|") if k],
            "keywords_hi": [k for k in r.keywords_hi.split("|") if k],
            "keywords_en": [k for k in r.keywords_en.split("|") if k],
        }
        for r in db.query(RelationshipRule).order_by(RelationshipRule.id).all()
    ]


# --------------------------------------------------------------------------------------------------------------------------- doctor on leave

APOLOGY = {
    "bn": "দুঃখিত, ডাক্তার {doctor} {start} থেকে {end} পর্যন্ত বসবেন না, তাই আপনার অ্যাপয়েন্টমেন্ট ({cid}, {date}) বাতিল করতে বাধ্য হয়েছি। অসুবিধার জন্য আন্তরিকভাবে দুঃখিত। নতুন তারিখে বুক করতে আবার ফোন করুন।",
    "en": "We are very sorry: Dr {doctor} will not be sitting from {start} to {end}, so your appointment ({cid}, {date}) has been cancelled. Our sincere apologies for the trouble. Please call again to book a new date.",
}


def register_leave(
    db: Session, doctor: Doctor, start_date: str, end_date: str, return_date: str | None = None, queue_sms=None
) -> dict:
    """The doctor will not consult from start_date to end_date (ISO, inclusive). Records the leave, cancels every CONFIRMED appointment in the
    range with cancel_reason="doctor_leave", frees its slot, and queues an apology per patient. `queue_sms(db, phone, key, message, cid)` is
    booking_service.queue_sms -- passed in so this module stays free of that import. Returns the cancelled bookings."""
    datetime.date.fromisoformat(start_date)
    datetime.date.fromisoformat(end_date)
    leave = DoctorLeave(doctor_id=doctor.id, start_date=start_date, end_date=end_date, return_date=return_date)
    db.add(leave)
    cancelled = []
    rows = (
        db.query(Appointment)
        .filter(Appointment.doctor_id == doctor.id, Appointment.status == "confirmed", Appointment.date >= start_date, Appointment.date <= end_date)
        .all()
    )
    now = datetime.datetime.now()
    for a in rows:
        a.status = "cancelled"
        a.cancelled_at = now
        a.cancel_reason = "doctor_leave"
        a.cancellation_charge_inr = 0  # the clinic cancelled: the patient is never charged
        db.query(SlotLock).filter_by(doctor_id=a.doctor_id, date=a.date, time_slot=a.time_slot).delete()
        cancelled.append({"confirmation_id": a.confirmation_id, "date": a.date, "time_slot": a.time_slot, "patient_name": a.patient_name, "phone": a.phone})
        if queue_sms is not None:
            for lang in ("bn", "en"):
                msg = APOLOGY[lang].format(doctor=doctor.full_name or doctor.name, start=start_date, end=end_date, cid=a.confirmation_id, date=a.date)
                queue_sms(db, a.phone, f"doctor_leave_cancel_{lang}", msg, a.confirmation_id)
    db.commit()
    log.info("doctor %s on leave %s..%s: %d appointment(s) cancelled", doctor.name, start_date, end_date, len(cancelled))
    return {"success": True, "doctor": doctor.full_name or doctor.name, "cancelled": cancelled, "leave_id": leave.id}
