"""sourav, 2026-10-14: every way a caller may NAME a doctor, built from the clinic's own doctor list (never a fixed list of names).

WHY: callers say a doctor's name in Bengali, Hindi or English words, in Latin letters with any spelling ("Mukherjee / Mukharji /
Mukherji"), with a title ("doctor", "ডাক্তার", "डॉक्टर", "daktar babu", "sir", "madam") or without one, and the recogniser may write the
same spoken name in ANY script whatever the call's language is (Bengali speech written in Devanagari, as seen on the pod). The
fast path matches a doctor only against the forms the clinic API sends for the call's own language, so a spelling or script it does
not hold is not matched and the question goes to the model (slow, and it may guess). This module adds the missing FORMS for every
doctor in the catalogue payload, in all three scripts, to all three languages' doctor tables (agent/fast_path.py
Catalogue._doctor_forms_of). It only widens what is recognised: which doctor a form belongs to is still decided by the owner count
(a form that two doctors share, like "রায়" or "বসু", is still never answered on -- the caller is asked "which one?").

FLOW: Catalogue builds the doctor tables -> doctor_extra_forms(doctor_row) is appended to each language's forms -> the existing fast
path/gazetteer matching runs unchanged. FRAMES are the sentence shapes (availability, booking, "book him/her", this day) used by
tests/test_doctor_cues.py and by Doctor_Cue_List.txt to prove each form reaches the right doctor.
"""

from __future__ import annotations

# Latin spellings callers/recognisers actually use, by the surname as it appears in the doctor list.
LATIN_VARIANTS: dict[str, list[str]] = {
    "Mukherjee": ["mukherji", "mukharji", "mukerjee", "mukherjee", "mukhopadhyay", "mukhopadhyaya"],
    "Sen": ["shen", "sen"],
    "Ghosh": ["ghose", "ghosh", "gosh"],
    "Chowdhury": ["choudhury", "chaudhury", "chowdhuri", "chaudhuri", "choudhuri", "chowdhury"],
    "Bhattacharya": ["bhattacharjee", "bhattacharyya", "bhattacharia", "bhattacharaya", "bhattacharya"],
    "Roy": ["roi", "roy"],
    "Banerjee": ["bannerjee", "banerji", "bandyopadhyay", "bandopadhyay", "banarjee", "banerjee"],
    "Dutta": ["dutt", "datta", "dutta"],
    "Chatterjee": ["chatterji", "chaterjee", "chattopadhyay", "chatarjee", "chatterjee"],
    "Basu": ["bosu", "basu", "boshu"],
    "Mitra": ["mitro", "mitter", "mitra"],
    "Sengupta": ["sengupto", "sengupta"],
    "Das": ["dash", "dass", "das"],
    "Bose": ["bos", "bose"],
    "Kar": ["kor", "kaar", "kar"],
    "Nandi": ["nondi", "nandy", "nandi"],
    "Pal": ["paul", "paal", "pal"],
    "Halder": ["haldar", "holdar", "halder"],
    "Guha": ["guho", "guha"],
    "Chanda": ["chondo", "chand", "chanda"],
    "Saha": ["shaha", "saha", "shaah"],
    "Dey": ["dey", "de"],
    "Adhikari": ["adhikary", "odhikari", "adhikari"],
    "Bagchi": ["bagchee", "bagchi"],
    "Biswas": ["bishwas", "viswas", "bisshas", "biswas"],
    "Majumder": ["majumdar", "mazumdar", "mojumdar", "majumder"],
    "Mondal": ["mandal", "mondol", "mandol", "mondal"],
    "Ganguly": ["ganguli", "gangopadhyay", "gangooly", "ganguly"],
    "Sinha": ["singha", "sinhaa", "sinha"],
    "Ray": ["rey", "ray"],
    "Sarkar": ["sorkar", "sarker", "sircar", "sarkar"],
    "Chakraborty": ["chakrabarti", "chakraborti", "chokroborty", "chakravarty", "chakraborty"],
    "Kundu": ["kundoo", "kundu"],
    "Mallick": ["mallik", "mullick", "malik", "mallick"],
    "Pramanik": ["pramanick", "promanik", "pramanic", "pramanik"],
    "Bhowmik": ["bhoumik", "bhaumik", "bhowmick", "bhowmik"],
}

# Titles/forms of address, per script. Used to build sentences (FRAMES) and spoken "title + name" forms.
TITLES = {
    "bn": ["ডাক্তার", "ডক্টর", "ডাঃ", "ডাক্তারবাবু", "স্যার", "ম্যাডাম"],
    "hi": ["डॉक्टर", "डाक्टर", "डॉ", "डॉक्टर साहब", "सर", "मैडम"],
    "en": ["doctor", "dr", "dr.", "doc", "sir", "madam"],
    "rom": ["daktar", "daktar babu", "doktor", "dactar", "doctor babu", "dr"],
}

# Sentence shapes ({d} = the doctor, with or without a title). Availability first, then booking (including "book HIM/HER",
# "that day"): the carrying-over of doctor and date from the previous answer is done by the booking flow, these just prove the
# question is recognised in every accent.
FRAMES: dict[str, list[str]] = {
    "bn": [
        "{d} আজ বসবেন?", "{d} কাল বসবেন কি?", "{d} কবে বসেন", "{d} কখন বসবেন", "{d} কি আজ আছেন?", "{d} এর চেম্বার কখন",
        "{d} কখন পাওয়া যাবে", "{d} কি আজ চেম্বারে আসবেন", "আজ {d} বসবেন কি", "{d} এর সময় কখন",
    ],
    "hi": [
        "{d} आज बैठेंगे?", "{d} कब बैठते हैं", "{d} कितने बजे बैठेंगे", "{d} आज मिलेंगे क्या", "{d} का चैंबर कब है",
        "{d} आज उपलब्ध हैं", "आज {d} बैठेंगे क्या", "{d} का टाइम क्या है", "{d} कब मिलेंगे",
    ],
    "en": [
        "is {d} available today", "when does {d} sit", "what time does {d} sit", "will {d} be there today",
        "{d} timings", "{d} availability", "is {d} in today", "when can I see {d}", "{d} schedule please",
    ],
    "mixed": [
        "{d} aj bosben?", "{d} ki aj bosbe", "{d} kobe boshen", "{d} aaj baithenge kya", "{d} kab milenge",
        "{d} today bosben", "{d} aj available achhen?", "{d} kal baithenge?", "x doctor {d} ki aj bosbe",
        "{d} ka timing kya hai", "{d} er chamber kokhon",
    ],
}

_SHORT_VOWEL = {"ী": "ি", "ূ": "ু"}  # long vowel signs are routinely written/recognised short (Bengali and Devanagari alike)


def _short_vowel_variant(s: str) -> str:
    out = s
    for a, b in _SHORT_VOWEL.items():
        out = out.replace(a, b)
    return out


def _clean(x: str) -> str:
    return " ".join((x or "").lower().replace(".", " ").split())


def doctor_extra_forms(d: dict) -> list[str]:
    """Extra spoken forms for ONE doctor row of the catalogue payload, all scripts (they are added to every language's table).

    Includes: the Latin spelling variants of the surname, the whole Latin / Bengali / Devanagari name, the given name alone and
    "given + surname", and the short-vowel spelling of every Bengali/Devanagari form. A form is returned only for this doctor; the
    catalogue's owner counting still treats a form two doctors share as ambiguous."""
    surname = d.get("surname") or (d.get("name") or "").split()[-1]
    forms: list[str] = []
    forms += LATIN_VARIANTS.get(surname, [surname])
    forms += [*(d.get("aliases_bn") or []), *(d.get("aliases_hi") or []), *(d.get("aliases_en") or [])]
    for key in ("full_name", "full_name_bn", "full_name_hi"):
        whole = " ".join(w for w in (d.get(key) or "").split() if w.strip(".").lower() not in ("dr", "doctor"))
        parts = whole.split()
        if whole:
            forms.append(whole)
        if len(parts) > 1:
            if len(parts[0]) >= 4:
                forms.append(parts[0])  # the given name alone ("arindam doctor")
    forms += [_short_vowel_variant(f) for f in list(forms)]
    return list(dict.fromkeys(c for c in (_clean(f) for f in forms) if len(c) >= 2))
