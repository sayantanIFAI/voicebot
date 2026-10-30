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
import random

from models import (
    Appointment,
    AuditLog,
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
    "Dr. K. Bhattacharya": "M", "Dr. N. Roy": "M", "Dr. S. Banerjee": "M", "Dr. M. Dutta": "M", "Dr. R. Dutta": "M",
    "Dr. S. Chatterjee": "F", "Dr. A. Basu": "F", "Dr. R. Mitra": "F", "Dr. P. Sengupta": "F",
    "Dr. D. Das": "M", "Dr. T. Bose": "M", "Dr. A. Kar": "M", "Dr. S. Nandi": "M",
    "Dr. R. Pal": "M", "Dr. K. Halder": "M", "Dr. S. Guha": "M", "Dr. B. Chanda": "M",
    "Dr. M. Saha": "F", "Dr. A. Dey": "M", "Dr. P. Adhikari": "F", "Dr. S. Bagchi": "M",
    "Dr. N. Biswas": "F", "Dr. R. Majumder": "F", "Dr. A. Mondal": "M", "Dr. S. Ganguly": "F",
    "Dr. K. Sinha": "M", "Dr. P. Ray": "M", "Dr. A. Sarkar": "M", "Dr. S. Chakraborty": "M",
    "Dr. S. Kundu": "M", "Dr. M. Mallick": "F", "Dr. R. Pramanik": "M", "Dr. T. Bhowmik": "F",
}

NEW_DEPARTMENTS = ["General Surgery", "Neurology", "Neurosurgery", "Urology", "Gastroenterology", "Psychiatry"]

# (department, short name, whole name, bn, hi, qualifications, sex, fee, [bn surname aliases], [hi surname aliases], schedule)
_MWF_NOON = {"days": [0, 2, 4], "start": "11:00", "end": "13:00"}
_TTS_NOON = {"days": [1, 3, 5], "start": "11:00", "end": "13:00"}
_MWF_EVE = {"days": [0, 2, 4], "start": "17:00", "end": "19:00"}
_TTS_EVE = {"days": [1, 3, 5], "start": "17:30", "end": "19:30"}
# Owner's own words on the call: sits from 1pm.
_MWF_1PM = {"days": [0, 2, 4], "start": "13:00", "end": "15:00"}
NEW_DOCTORS = [
    ("General Surgery", "Dr. A. Lahiri", "Dr. Ashis Lahiri", "আশিস লাহিড়ী", "आशीष लाहिड़ी", "MBBS, MS (General Surgery)", "M", 700, ["লাহিড়ী", "লাহিড়ি"], ["लाहिड़ी"], _TTS_NOON),
    ("General Surgery", "Dr. S. Dasgupta", "Dr. Sharmila Dasgupta", "শর্মিলা দাশগুপ্ত", "शर्मिला दासगुप्ता", "MBBS, MS (General Surgery)", "F", 700, ["দাশগুপ্ত", "দাসগুপ্ত"], ["दासगुप्ता"], _MWF_NOON),
    ("Neurology", "Dr. J. Sanyal", "Dr. Joydeep Sanyal", "জয়দীপ সান্যাল", "जयदीप सान्याल", "MBBS, MD, DM (Neurology)", "M", 900, ["সান্যাল"], ["सान्याल"], _MWF_EVE),
    ("Neurosurgery", "Dr. A. Maitra", "Dr. Arup Maitra", "অরূপ মৈত্র", "अरूप मैत्रा", "MBBS, MS, MCh (Neurosurgery)", "M", 1000, ["মৈত্র", "মৈত্রা"], ["मैत्रा"], _TTS_EVE),
    ("Urology", "Dr. T. Ghoshal", "Dr. Tanmoy Ghoshal", "তন্ময় ঘোষাল", "तन्मय घोषाल", "MBBS, MS, MCh (Urology)", "M", 800, ["ঘোষাল"], ["घोषाल"], _TTS_NOON),
    # a male gynaecologist, so "a lady gynaecologist" is a real filter and not the whole department
    ("Gynaecology & Obstetrics", "Dr. A. Pathak", "Dr. Alok Pathak", "আলোক পাঠক", "आलोक पाठक", "MBBS, MS (Obs & Gynae)", "M", 700, ["পাঠক"], ["पाठक"], _MWF_NOON),
    # Real call, 2026-10-28: a caller asked for "Dr. Sanjay Saha" and the booking never flowed -- the doctor was never in the catalogue at all, not a mishearing of an
    # existing one. Added here (never from seed.seed(), which would wipe every booking already made). "সাহা"/"साहा" (Saha) is close enough to "শাহ"/"शाह" (Shah) that
    # the recogniser wrote the wrong but real surname rather than a typo on the real call ("संजो शाह", "डॉ संजोय शाह" for "Dr. Sanjay Saha") -- those exact heard forms
    # are kept as aliases alongside the correct spelling, same discipline as every other alias in this file (a confirmed real form for a real doctor, never a guess).
    (
        "General Medicine", "Dr. S. Saha", "Dr. Sanjay Saha", "সঞ্জয় সাহা", "संजय साहा", "MBBS, MD (Gen. Med.)", "M", 600,
        ["সাহা", "শাহ", "সঞ্জয় সাহা", "সঞ্জয় শাহ", "সঞ্জো শাহ"],
        ["साहा", "शाह", "संजय साहा", "संजय शाह", "संजो शाह", "संजोय शाह"],
        _MWF_1PM,
    ),
    # Owner, 2026-10-28: sample data -- Dr. Arun Kumar Saha, General Physician, Monday / Wednesday / Friday, 10 pm to 11:45 pm (as given). Bengali "অরুণ", Hindi "अरुण"; the surname
    # "সাহা" is shared with the other Sahas, which is right: "Dr Saha" is then asked about, never guessed.
    (
        "General Medicine", "Dr. A. K. Saha", "Dr. Arun Kumar Saha", "অরুণ কুমার সাহা", "अरुण कुमार साहा", "MBBS, MD (Gen. Med.)", "M", 600,
        ["অরুণ কুমার সাহা", "অরুণ সাহা", "অরুণ কুমার", "অরুণ কুমার শাহ", "অরুণ শাহ"],
        ["अरुण कुमार साहा", "अरुण साहा", "अरुण कुमार", "अरुण कुमार शाह", "अरुण शाह"],
        {"days": [0, 2, 4], "start": "22:00", "end": "23:45"},
    ),
    # Owner, 2026-10-29: sample data -- Dr. Ziaul Haq, paediatrician, sitting tomorrow (Friday 2026-10-30). The table is by weekday, so he sits on Fridays; hours were not given, so a morning
    # chamber (10:00-13:00) is used until a clinician sets the real ones.
    (
        "Paediatrics", "Dr. Z. Haq", "Dr. Ziaul Haq", "জিয়াউল হক", "ज़ियाउल हक़", "MBBS, MD (Paediatrics)", "M", 600,
        ["জিয়াউল হক", "জিয়াউল", "হক"],
        ["ज़ियाउल हक़", "जियाउल हक", "जियाउल", "हक", "हक़", "ज़ियाउल"],
        {"days": [0, 1, 2, 3, 4], "start": "10:00", "end": "13:00"},  # owner, 2026-10-29: from next week, weekdays only (Monday to Friday)
    ),
    # Owner, 2026-10-29: Dr. Chandan Kumar, gastroenterologist, every weekday (Monday to Friday). Hours were not given: 10:00-13:00 until a clinician sets the real ones. His slots for today and tomorrow are
    # filled at the first start (see apply_extras), so the first free day is next Monday.
    (
        "Gastroenterology", "Dr. C. Kumar", "Dr. Chandan Kumar", "চন্দন কুমার", "चंदन कुमार", "MBBS, MD, DM (Gastroenterology)", "M", 800,
        ["চন্দন কুমার", "চন্দন", "চণ্ডন কুমার", "চন্দন কুমারের"],
        ["चंदन कुमार", "चन्दन कुमार", "चंदन"],
        {"days": [0, 1, 2, 3, 4], "start": "10:00", "end": "13:00"},
    ),
    # Owner, 2026-10-29: Dr. Subhadip Mondal, Orthopaedics. Hours were not given: Tuesday / Thursday / Saturday 12:00-14:00 until a clinician sets the real ones. "মন্ডল" is shared with Dr. Anirban Mondal, so the
    # surname alone still asks "which one?"; the first name picks him (Bengali spellings: শুভদীপ / সুভদীপ, মন্ডল / মণ্ডল).
    (
        "Orthopaedics", "Dr. S. Mondal", "Dr. Subhadip Mondal", "শুভদীপ মন্ডল", "शुभदीप मंडल", "MBBS, MS (Orthopaedics)", "M", 700,
        ["শুভদীপ মন্ডল", "শুভদীপ মণ্ডল", "শুভদীপ", "সুভদীপ মন্ডল", "সুভদীপ", "শুভদীপা মণ্ডল", "শুভদীপ মোণ্ডল", "মন্ডল", "মণ্ডল"],
        ["शुभदीप मंडल", "शुभदीप मण्डल", "शुभदीप", "सुभदीप मंडल", "सुभदीप", "मंडल", "मण्डल"],
        {"days": [1, 3, 5], "start": "12:00", "end": "14:00"},
    ),
    # Owner, 2026-10-29: Dr. Rajarshi Dutta, General Medicine. Hours were not given: Monday to Friday 10:00-13:00 until a clinician sets the real ones. "দত্ত" is shared with Dr. Manas Dutta (Cardiology), so the
    # surname alone asks "which one?"; the first name picks him (Bengali spellings: রাজর্ষি / রাজর্শি, Hindi: राजर्षि / राजऋषि).
    (
        "General Medicine", "Dr. R. Dutta", "Dr. Rajarshi Dutta", "রাজর্ষি দত্ত", "राजर्षि दत्ता", "MBBS, MD (Gen. Med.)", "M", 600,
        ["রাজর্ষি দত্ত", "রাজর্ষি", "রাজর্শি দত্ত", "রাজর্শি", "রাজর্ষী দত্ত", "রাজর্ষী"],
        ["राजर्षि दत्ता", "राजर्षि दत्त", "राजर्षि", "राजऋषि दत्ता", "राजऋषि", "राजर्शि"],
        {"days": [0, 1, 2, 3, 4], "start": "10:00", "end": "13:00"},
    ),
    # Real call, 2026-10-29: a caller asked about a psychiatrist and the department does not exist in the catalogue at
    # all -- added with a real department + doctor, same pattern as General Surgery/Neurology/Urology above.
    (
        "Psychiatry", "Dr. A. Bhadra", "Dr. Amitava Bhadra", "অমিতাভ ভদ্র", "अमिताव भद्र", "MBBS, MD (Psychiatry), DPM", "M", 700,
        ["ভদ্র"],
        ["भद्र"],
        _TTS_NOON,
    ),
]

# (department, keywords_bn, keywords_hi, keywords_en) -- symptoms the clinic's clinicians route to that department. Specific on purpose: a bare
# "pain" must never route to a surgeon.
NEW_ROUTES = [
    (
        "Gastroenterology",
        ["গ্যাস্ট্রো", "গ্যাস্ট্রোএন্টারোলজি", "গ্যাস্ট্রোএন্টারোলজিস্ট", "অম্বল", "বুক জ্বালা", "বদহজম", "পেটে আলসার", "লিভারের সমস্যা", "জন্ডিস", "পিত্তথলি", "গ্যাসের সমস্যা", "হজমের সমস্যা", "অ্যাসিডিটি"],
        ["गैस्ट्रो", "गैस्ट्रोएंटेरोलॉजी", "एसिडिटी", "अपच", "सीने में जलन", "पीलिया", "लिवर की समस्या", "पेट में अल्सर"],
        ["gastro", "gastroenterology", "gastroenterologist", "acidity", "heartburn", "indigestion", "jaundice", "liver problem", "stomach ulcer"],
    ),
    # Owner, 2026-10-29: "বডি ফুলে গেছে তো ডাক্তার দেখাবো" (the body has swelled) was not recognised as a symptom and was asked "which doctor?". Swelling of the body, face, feet or hands goes to General Medicine
    # (the ordinary doctor decides if a specialist is needed), in all three languages and the way the recogniser writes it.
    (
        "General Medicine",
        ["ফুলে গেছে", "ফুলে যাচ্ছে", "ফুলে আছে", "ফুলে উঠেছে", "ফুলছে", "ফোলা", "শরীর ফুলে", "বডি ফুলে", "গা ফুলে", "পা ফুলে", "মুখ ফুলে", "হাত ফুলে", "শরীর ফোলা"],
        ["सूजन", "सूज गया", "सूज गई", "सूजा", "फूल गया", "फूल गई", "फूले", "फूलेगे", "बॉडी फूल", "शरीर फूल", "बदन फूल"],
        ["swelling", "swollen", "body swelling", "body is swollen", "puffy", "swelled up"],
    ),
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
        # Owner, 2026-10-30: "আমার ডেঙ্গু হয়েছে" (I have dengue) was answered "tell me what the problem is": a fever illness named as the illness is General Medicine's, with today's doctors.
        "General Medicine",
        ["ডেঙ্গু", "ডেঙ্গি", "ডেংগু", "ম্যালেরিয়া", "ম্যালেরিয়ায়", "টাইফয়েড", "চিকুনগুনিয়া", "ভাইরাল জ্বর", "ভাইরাল ফিভার"],
        ["डेंगू", "डेंगी", "मलेरिया", "टाइफाइड", "चिकनगुनिया", "वायरल बुखार"],
        ["dengue", "malaria", "typhoid", "chikungunya", "viral fever"],
    ),
    (
        "Urology",
        ["কিডনিতে পাথর", "কিডনির পাথর", "প্রোস্টেট", "প্রস্রাবে জ্বালা", "প্রস্রাবে রক্ত", "মূত্রথলির পাথর", "ইউটিআই", "ইউ টি আই", "ইউটি", "ইউরিন ইনফেকশন", "প্রস্রাবের ইনফেকশন"],  # owner, 2026-10-30: UTI
        ["किडनी में पथरी", "किडनी की पथरी", "प्रोस्टेट", "पेशाब में जलन", "पेशाब में खून", "यूटीआई", "यू टी आई", "यूरिन इन्फेक्शन", "पेशाब का इन्फेक्शन"],
        ["kidney stone", "prostate", "burning urine", "blood in urine", "bladder stone", "u t i", "uti problem", "uti infection", "urine infection", "urinary infection", "urinary tract"]  # (a bare "uti" would match inside "utility": keywords are substrings),
    ),
    (
        "Psychiatry",
        ["ডিপ্রেশন", "ডিপ্রেসন", "বিষণ্নতা", "দুশ্চিন্তা", "অ্যাংজাইটি", "অ্যানজাইটি", "প্যানিক অ্যাটাক", "ঘুম হয় না", "ঘুমের সমস্যা", "মানসিক চাপ", "মানসিক সমস্যা", "মনের অসুখ", "মন খারাপ থাকে", "ওসিডি"],
        ["डिप्रेशन", "अवसाद", "चिंता", "घबराहट", "पैनिक अटैक", "नींद नहीं आती", "नींद की समस्या", "मानसिक तनाव", "मानसिक समस्या", "मन अशांत रहता है", "ओसीडी"],
        ["depression", "anxiety", "panic attack", "insomnia", "sleep problem", "sleeplessness", "mental stress", "mood swings", "ocd", "obsessive compulsive"],
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


# Owner, 2026-10-28: sample data -- Dr. Subrata Banerjee (Dr. S. Banerjee) sits on EVERY day of the week, so every date this month and next month is one he sits. The table is by weekday, so
# seven rows cover every date; the hours stay the ones he already has. Added only where a weekday is missing (never changes an hour a clinician edited).
SITS_EVERY_DAY = ("Dr. S. Banerjee",)
# Owner, 2026-10-29: Dr. Ziaul Haq, weekdays only (Monday to Friday), from next week. Tomorrow (Friday 2026-10-30) was already promised; the only other weekday before next Monday is today (Thursday), whose
# 10:00-13:00 chamber has already closed when this is read, so a plain weekday schedule is exactly "from next week". Rows are only ADDED for a missing weekday.
SITS_WEEKDAYS = {"Dr. Z. Haq": ((0, 1, 2, 3, 4), "10:00", "13:00")}
# Spellings the recognisers write for a doctor that already exists: "জেড হাক" / "ज़ेड हाक" for Dr. Z. Haq (live replay, 2026-10-29). Added when missing; nothing is ever removed.
ALL_FROM_KOLKATA = ("Orthopaedics",)
# Owner, 2026-10-29: doctors booked only on the day they sit (set once, when this runs for the first time; changed afterwards in the table, it stays as changed)
ON_THE_DAY_DOCTORS = ("Dr. Z. Haq",)
EXTRA_ALIASES = {
    "Dr. Z. Haq": (["হাক", "জেড হাক", "জেড হক", "হাক সাহেব"], ["हाक", "जेड हाक", "जेड हक", "ज़ेड हक", "हॉक"]),
    # Owner, 2026-10-29: "Majumder" and "Majumdar" are the same name, said the same
    "Dr. R. Majumder": (["majumdar", "mazumdar", "mozumdar", "majumder"], ["majumdar", "mazumdar", "मजूमदार", "मजुमदार"]),
    # Owner, 2026-10-29: the surname "দত্ত" belongs to Dr. Rajarshi Dutta as well as Dr. Manas Dutta -- the bare surname must name BOTH ("which one?"), never answer for one of them
    "Dr. R. Dutta": (["দত্ত", "দত্ত সাহেব", "dutta", "datta"], ["दत्ता", "दत्त", "दत्ता साहब"]),
}


def apply_extras(db: Session) -> dict:
    """Idempotent. Returns what it changed."""
    out = {"doctor_sex_set": 0, "departments": 0, "doctors": 0, "routes": 0, "relationships": 0, "weekdays_added": 0, "from_kolkata_set": 0}
    for short, (al_bn, al_hi) in EXTRA_ALIASES.items():
        doc = db.query(Doctor).filter_by(name=short).first()
        if doc is None:
            continue
        cur_bn = [a for a in (doc.aliases_bn or "").split("|") if a and a not in ("হোক", "জিয়াউল হোক")]  # "হোক" is the ordinary word "let it be": it must never name a doctor
        cur_hi = [a for a in (doc.aliases_hi or "").split("|") if a]
        add_bn, add_hi = [a for a in al_bn if a not in cur_bn], [a for a in al_hi if a not in cur_hi]
        if add_bn or add_hi or ("|".join(cur_bn) != (doc.aliases_bn or "")):
            doc.aliases_bn = "|".join(cur_bn + add_bn)
            doc.aliases_hi = "|".join(cur_hi + add_hi)
            out["aliases_added"] = out.get("aliases_added", 0) + len(add_bn) + len(add_hi)
    for short, (days, start, end) in SITS_WEEKDAYS.items():
        doc = db.query(Doctor).filter_by(name=short).first()
        if doc is None:
            continue
        have = {r.weekday for r in db.query(DoctorSchedule).filter_by(doctor_id=doc.id).all()}
        for wd in days:
            if wd not in have:
                db.add(DoctorSchedule(doctor_id=doc.id, weekday=wd, start_time=start, end_time=end))
                out["weekdays_added"] += 1
    for short in SITS_EVERY_DAY:
        doc = db.query(Doctor).filter_by(name=short).first()
        if doc is None:
            continue
        rows = db.query(DoctorSchedule).filter_by(doctor_id=doc.id).all()
        have = {r.weekday for r in rows}
        start, end = (rows[0].start_time, rows[0].end_time) if rows else ("18:00", "20:00")
        for wd in range(7):
            if wd not in have:
                db.add(DoctorSchedule(doctor_id=doc.id, weekday=wd, start_time=start, end_time=end))
                out["weekdays_added"] += 1
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
    # Owner, 2026-10-29: the "From Kolkata?" column is filled AT RANDOM (about three in four "yes") for every doctor that has no value yet -- sample data, once; a value anyone has set is never changed.
    _rng = random.Random(20261029)
    for d in db.query(Doctor).order_by(Doctor.id).all():
        pick = "yes" if _rng.random() < 0.75 else "no"  # drawn for every doctor, so a doctor's value does not depend on who else has one
        if not (getattr(d, "from_kolkata", "") or "").strip():
            d.from_kolkata = pick
            out["from_kolkata_set"] += 1
    if not db.query(AuditLog).filter_by(action="seed.appointment_on_the_day").first():
        for d in db.query(Doctor).filter(Doctor.name.in_(ON_THE_DAY_DOCTORS)).all():
            d.appointment_on_the_day = True
        db.add(AuditLog(at=datetime.datetime.now(), actor="system", action="seed.appointment_on_the_day", outcome="ok", detail_json="{}"))
        out["on_the_day_seeded"] = len(ON_THE_DAY_DOCTORS)
    # Owner, 2026-10-29: every doctor of these departments is from Kolkata (set on every start; the other departments keep their own values)
    for d in db.query(Doctor).all():
        if d.department is not None and d.department.name in ALL_FROM_KOLKATA and (d.from_kolkata or "") != "yes":
            d.from_kolkata = "yes"
            out["from_kolkata_set"] += 1
    # Owner, 2026-10-29: "per doctor slot for each day should be 5 only" -- every sitting day of every doctor takes at most five patients (the table's max_bookings_per_day, which available_slots honours). Set ONCE
    # (marker in the audit log); a number a clinician has changed afterwards stays as changed.
    if not db.query(AuditLog).filter_by(action="seed.max_bookings_5").first():
        db.flush()
        for row in db.query(DoctorSchedule).all():
            row.max_bookings_per_day = 5
        db.add(AuditLog(at=datetime.datetime.now(), actor="system", action="seed.max_bookings_5", outcome="ok", detail_json="{}"))
        out["max_bookings_set"] = True
    # Owner, 2026-10-29: Dr. Chandan Kumar's five patients for today and for tomorrow are already booked (sample data, once): the first free day is next Monday.
    if not db.query(AuditLog).filter_by(action="seed.chandan_kumar_full").first():
        chandan = db.query(Doctor).filter_by(name="Dr. C. Kumar").first()
        if chandan is not None:
            import booking_service as bs
            import clinic_clock

            db.flush()
            sched_hours = db.query(DoctorSchedule).filter_by(doctor_id=chandan.id).first()
            slots = bs._generate_slots(sched_hours.start_time, sched_hours.end_time)[:5]  # noqa: SLF001
            for offset in (0, 1):
                day = (clinic_clock.today() + datetime.timedelta(days=offset)).isoformat()
                for n, slot in enumerate(slots, start=1):
                    held = bs.hold_slot(db, chandan.id, day, slot)
                    if held.get("success"):
                        phone = f"90000{offset}{n:04d}"
                        bs.confirm_booking(db, held["hold_token"], chandan.id, day, slot, f"Sample Patient {offset}{n}", phone, phone, 40)
            out["chandan_filled"] = True
        db.add(AuditLog(at=datetime.datetime.now(), actor="system", action="seed.chandan_kumar_full", outcome="ok", detail_json="{}"))
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
