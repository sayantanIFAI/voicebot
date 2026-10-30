"""Lay doctor phrases -> department (owner, 2026-10-24; credit: sourav).

Why: callers do not say "orthopaedic" or "urologist". They say "haarer doctor" (the bone doctor), "booker doctor" (chest = heart), "peter doctor",
"mathar doctor" (head = neurologist), "pecchaper doctor" (urine = urologist), "operation er doctor" / "kata chherar doctor" (surgeon), "meyeder doctor"
(women = gynaecologist), "pain er doctor" / "plain doctor" / "emni physian" (an ordinary physician), "bachhader doctor" (children = paediatrician).
The owner gave the mapping below and asked for the same in Bengali script, Hindi script and romanised / mixed speech.

How: ONE ordered table of (department, strength, pattern) built from stems, so a new lay word is one line. A pattern is  STEM [up to two small filler words] DOCTOR-WORD
("haarer ekta doctor", "heart er doctor", "হাড়ের ডাক্তার", "दिल का डॉक्टर"). Pure text -> department NAME; the ids / doctors come from the database in
booking_service. No model, no guess.

Rules (owner):
  * A CHILD word with another lay word ("bachhader haarer doctor") -> Paediatrics wins ... EXCEPT
  * a patient of 13 YEARS OR MORE is never sent to the paediatrician: the normal (General Medicine) doctor, or -- when another specialty was named too -- that
    specialty (a 15-year-old's "bachhader haarer doctor" = the bone doctor). `PAEDIATRIC_MAX_AGE` = 12.
  * "weak" phrases (pain er doctor, plain doctor, emni physian: an ordinary physician) never beat a specific one in the same sentence.
Neurology, Urology and General Surgery come from clinic_extras.py (already in the catalogue); a database without one simply gets no lay answer for it.
This module imports nothing from the agent (clinic-api is a separate service).
"""

from __future__ import annotations

import re

PAEDIATRIC_MAX_AGE = 12  # 13 and above go to the normal doctor (owner, 2026-10-24)
PAEDIATRICS = "Paediatrics"
GENERAL_MEDICINE = "General Medicine"

_I = re.I

# the word "doctor" in the ways callers and recognisers write it
_DOC_EN = r"(?:doctors?|docters?|doktors?|daktars?|dagtars?|dactars?|daktar\w*|dakter|doc|dr|specialists?|physicians?|physians?|fijishian|fisician)"
_DOC_BN = r"(?:ডাক্তার\S*|ডাক্তর\S*|ডক্টর\S*|ডাক্টার\S*|স্পেশালিস্ট|বিশেষজ্ঞ|ফিজিশিয়ান|ফিজিসিয়ান)"
_DOC_HI = r"(?:डॉक्टर\S*|डाक्टर\S*|डॉ\.?|डाॅक्टर\S*|वैद्य|स्पेशलिस्ट|विशेषज्ञ|फिजिशियन|फिज़िशियन)"
_GAP = r"(?:\s+[^\s]{1,9}){0,4}"  # "haarer ekta doctor", "meyeder problem er doctor"; owner, 2026-10-26: up to four small words ("लड़की लोगों के लिए वो डॉक्टर")


def _en(stem: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![a-z])(?:{stem}){_GAP}\s+{_DOC_EN}(?![a-z])", _I)


def _script(stem: str, doc: str) -> re.Pattern[str]:
    return re.compile(rf"(?:{stem})\S*{_GAP}\s+{doc}")


# (department, weak, romanised/English stems, Bengali stems, Hindi stems)
_RULES: list[tuple[str, bool, str, str, str]] = [
    (PAEDIATRICS, False,
     r"bac+h+c?h*[aeou]\w*|baccha\w*|bacha\w*|shishu\w*|chh?o+to+der|choto+der|chotoder|kids?|child|children|baby|babies|infant|toddler",
     r"বাচ্চা|বাচ্ছা|বাচ্চা|শিশু|ছোটদের|খোকা|খুকি|ছেলেমেয়ে",
     r"बच्चों|बच्चे|बच्चा|बच्चो|शिशु|छोटे\s*बच्चों|नवजात"),
    ("Orthopaedics", False,
     r"ha+r(?:er|r|ke|ki|ka|e|ey)?|ha+d(?:er|dir|der)?|hadd?i(?:r|er|ka|ke|ki)?|haddiyon\w*|bones?|joints?",
     # owner, 2026-10-24: "হার" (missing the ড়) -- a recogniser's real miss of "হাড়" ("bone") -- was already covered
     # for the ROMANISED stem above ("harer") but never added here for Bengali SCRIPT; "হারের ডাক্তার" fell through
     # to a plain doctor-name lookup and fuzzy-matched the surname "হালদার" instead. Safe to add bare despite "হার"
     # also meaning "necklace"/"defeat" in isolation: this stem only ever fires immediately beside a doctor word
     # (the _GAP + DOC_BN match below), and nobody asks for "the necklace's doctor".
     r"হাড়|হাড|হাঁড়|হাড্ডি|হার(?!্)|জয়েন্ট|গাঁট|গাট",
     r"हड्डी|हड्डियों|हड्डि|जोड़ों|जोड़|बोन"),
    ("Cardiology", False,
     r"bu+k(?:er|ar|r|ke|ki|e|ey)?|boo?k(?:er|ar)|heart(?:er|r)?|hart(?:er)?|hridoy\w*|hriday\w*|hridpindo\w*|dil(?:ka|ke|ki)?|chest|sine(?:ke|ka|ki)?",
     r"বুক|হার্ট|হৃদ|হৃৎ|হৃদয়|হৃদপিণ্ড",
     r"दिल|हार्ट|सीने|छाती|हृदय|ह्रदय"),
    ("Diabetology & Endocrinology", False,
     r"pe+t(?:er|r|e|ey|ke|ki|ka)?|poter|pait|stomach|tummy|belly|sugar|shugar|diabet\w*|madhumeh|thyroid|gland|hormone",
     r"পেট|সুগার|ডায়াবেটিস|থাইরয়েড|হরমোন",
     r"पेट|शुगर|डायबिटीज|डायबिटीस|मधुमेह|थायराइड|हार्मोन"),
    ("Neurology", False,
     r"matha(?:r|y|er|ke|ki)?|mathar|mather|mathaa?r|nerves?|nerver|nerv|snayu\w*|brain|dimag\w*|dimaag\w*|head|sir(?=\s+(?:ka|ke|ki|ko|er|r)\s)|fits?|migraine",
     # owner, 2026-10-27: real call -- "মাফার ডাক্তার" (the recogniser's "মাথার", missing the থ) fell through
     # to a plain doctor-name lookup and fuzzy-matched the surnames "হালদার"/"মজুমদার" instead, the exact same
     # bug class as "হার" for "হাড়়" above (2026-10-24): the romanised stem already had "mather" for this same
     # miss, but the Bengali SCRIPT stem never did. Safe to add bare for the same reason "হার" was: this stem
     # only ever fires immediately beside a doctor word (the _GAP + DOC_BN match below).
     r"মাথা|মাফা|নার্ভ|স্নায়ু|ব্রেন|মস্তিষ্ক|মগজ",
     # Real call, 2026-10-29: bare "नस" (nerve) with no word-boundary check matched INSIDE "मानसिक" (mental, मा-नस-िक)
     # -- "मानसिक रोग के डॉक्टर" (psychiatrist) was sent to Neurology instead of the new Psychiatry department. The
     # negative lookahead blocks that one collision; "नस"/"नसों" (nerve/nerves) on their own still match everywhere else.
     r"सिर|दिमाग|नस(?!िक)|नसों|नर्व|नर्वस|ब्रेन|मस्तिष्क|माइग्रेन"),
    ("Urology", False,
     r"pe?c+h*a?[bp](?:er|ar|r|e|ey)?|pechh?a[bp]\w*|pecchap\w*|pechap\w*|pesha[bp]\w*|pesa[bp]\w*|prosra[bp]\w*|prasra[bp]\w*|prosa[bp]\w*|urine|urinary|kidneys?|kidni\w*|kidnir",
     r"পেচ্ছাপ|পেছাপ|পেটছাপ|পেট\s*ছাপ|পেট\s*চাপ|পেটচ্ছাপ|পেশাব|প্রস্রাব|প্রসাব|কিডনি|কিডনী|ইউরিন|মূত্র",  # owner, 2026-10-30: "পেট ছাপ করতে গেলে ব্যথা হয়" is the recogniser's pechhap (urine)
     r"पेशाब|पेसाब|मूत्र|किडनी|यूरिन|पेशाब"),
    ("General Surgery", False,
     r"op[ea]r[ae]?sh[ao]n(?:er|r|ar|ke|ki|e)?|operation(?:er|r)?|kata[\s-]*ch+h?e+n?r(?:a|ar|er|r)?|katachh?e+r\w*|cut[\s-]*(?:and|n|o)?[\s-]*tear\w*|hernia|appendix|piles|surgery|surgeon",
     r"অপারেশন|কাটাছেঁড়া|কাটা\s*ছেঁড়া|কাটা-ছেঁড়া|কাটাছেড়া|কাটা\s*ছেড়া|সার্জারি|সার্জন|সার্জেন",
     r"ऑपरेशन|आपरेशन|ओपरेशन|कटने\s*फटने|कटा\s*फटा|काट\s*छांट|चीर\s*फाड़|सर्जरी|सर्जन"),
    ("Gynaecology & Obstetrics", False,
     r"meye?(?:der|ader|r\s*der)|meyeder|mohila\w*|mahila\w*|womens?|women's|woman|ladies|lady|ladki\s*log\w*|ladkiyon\w*|ladkiyo\w*|aurat\w*|aurton|auraton|stree\w*|prosuti|pregnan\w*",
     # owner, 2026-10-26: "लड़की लोग का डॉक्टर" (the doctor for girls) is "meyeder doctor" said in Hindi. A bare "মেয়ের / लड़की" (my daughter) is NOT a cue: that is the patient, not the doctor's kind.
     r"মেয়েদের|মহিলাদের|মহিলা|নারী|স্ত্রী|প্রসূতি|মেয়েরা|মেয়েছেলে|মেয়ে\s*লোক|লেডিজ|লেডি",
     r"औरतों|औरत|महिलाओं|महिला|लड़कियों|लड़कियां|लड़कियाँ|लडकियों|लड़की\s*लोग|लडकी\s*लोग|लेडीज़?|लेडी|लेडिज|स्त्री|गर्भवती"),
    # owner, 2026-10-24: "naak kaan golar daktar" (nose ear throat) -- ENT had no lay rule at all until now.
    ("ENT", False,
     r"naa?k(?:er|e)?|kaan?(?:er|e)?|gol[ae]y?(?:r|er)?|throats?|noses?|ears?|\bent\b|\bint\b",
     # owner, 2026-10-26: "ইন্ট ডাক্তার আছে" -- the recogniser's spelling of "ENT doctor"
     r"নাক|কান|গলা|গলার|ইন্ট|ইনটি|ইএনটি|ই\s*এন\s*টি",
     r"नाक|कान|गला|गले|गले\s*का|इंट|इन्ट|इनटी|ईएनटी|ई\s*एन\s*टी"),
    (GENERAL_MEDICINE, True,
     r"pain(?:er|r|e)?|plain|plane|byath[ae]?r?|byatha\w*|dard(?:ka|ke|ki)?|normal|emni|emnei|erokom|ordinary|common|simple|sadha?ron\w*|sadharan|samany\w*|general|jwor\w*|jor(?:er)?|fever",
     r"ব্যথার|ব্যাথার|প্লেন|প্লেইন|এমনি|এমনই|সাধারণ|সাধারন|নরমাল|সাধারণ|জ্বরের",
     r"दर्द|साधारण|नॉर्मल|नार्मल|ऐसे\s*ही|यूँ\s*ही|यूं\s*ही|जनरल|बुखार"),
]

_COMPILED: list[tuple[str, bool, list[re.Pattern[str]]]] = []
for _dept, _weak, _en_stem, _bn_stem, _hi_stem in _RULES:
    _COMPILED.append((_dept, _weak, [_en(_en_stem), _script(_bn_stem, _DOC_BN), _script(_hi_stem, _DOC_HI)]))

# "emni physian" / "physician" alone: an ordinary physician -- no stem word needed
_PHYSICIAN_ALONE = re.compile(r"(?<![a-z])(?:physians?|physicians?|fijishian|fisician|fizisian)(?![a-z])|ফিজিশিয়ান|ফিজিসিয়ান|फिजिशियन|फिज़िशियन", _I)

# a CHILD word at all (the precedence rule: "bachhader haarer doctor" is the paediatrician's)
_CHILD = re.compile(
    r"(?<![a-z])(?:bac+h+c?h*[aeou]\w*|baccha\w*|bacha\w*|shishu\w*|chh?o+to+der|choto+der|chotoder|kids?|child|children|baby|babies|infant)(?![a-z])"
    r"|বাচ্চা|বাচ্ছা|শিশু|ছোটদের|बच्चों|बच्चे|बच्चा|बच्चो|शिशु",
    _I,
)


def has_child_word(text: str) -> bool:
    return _CHILD.search((text or "").lower()) is not None


def lay_hits(text: str) -> dict[str, bool]:
    """{department name: weak?} for every lay doctor phrase in `text` (a weak hit is an ordinary-physician phrase)."""
    q = (text or "").lower()
    out: dict[str, bool] = {}
    if not q.strip():
        return out
    for dept, weak, pats in _COMPILED:
        if any(p.search(q) for p in pats):
            out[dept] = weak and out.get(dept, True)
    if _PHYSICIAN_ALONE.search(q):
        out[GENERAL_MEDICINE] = out.get(GENERAL_MEDICINE, True)
    return out


def resolve(text: str, age: int | None = None) -> list[str]:
    """The department NAMES the lay phrases in `text` stand for, best first; [] when the sentence has none.
    Rules: a specific (strong) phrase beats an ordinary-physician (weak) one; a child word beats the others (paediatrician) -- unless the patient is 13 or older:
    then the paediatrician is dropped (the other specialty if one was named, else General Medicine)."""
    hits = lay_hits(text)
    if not hits:
        return []
    strong = [d for d, weak in hits.items() if not weak]
    names = strong or [d for d in hits]
    # a child word in the same sentence as a specific lay phrase ("বাচ্চার হাড়ের ডাক্তার", "बच्चों की हड्डी का डॉक्टर") makes it the paediatrician's, even when
    # the child word sits too far from "doctor" to form a phrase of its own
    if strong and PAEDIATRICS not in names and has_child_word(text):
        names = [PAEDIATRICS, *names]
    if PAEDIATRICS in names:
        teen = age is not None and age > PAEDIATRIC_MAX_AGE
        if teen:
            names = [d for d in names if d != PAEDIATRICS] or [GENERAL_MEDICINE]
        elif len(names) > 1 and has_child_word(text):
            names = [PAEDIATRICS]
    return names


def apply_age_rule(names: list[str], age: int | None) -> list[str]:
    """Any route (a symptom keyword, a lay phrase) that ended on the paediatrician for a patient of 13 or more goes to General Medicine instead."""
    if age is None or age <= PAEDIATRIC_MAX_AGE or PAEDIATRICS not in names:
        return names
    out = [GENERAL_MEDICINE if n == PAEDIATRICS else n for n in names]
    return list(dict.fromkeys(out))
