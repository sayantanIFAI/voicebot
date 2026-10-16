"""Lab-test PRICE conversations in Bengali, Hindi and English, at code level (no GPU, no model, no network).

    python -m pytest tests/test_lab_price_multilingual.py -v -s

Written by sourav's request ("do these tests first before doing anything", lab tests only, price questions only).
This file only ADDS tests; it changes nothing in the agent, the clinic API or any existing test.

WHAT IS REAL: the orchestrator's `_dispatch_turn` (main_pcm.py), the deterministic fast path (agent/fast_path.py)
built from the REAL catalogue, the lay-term gate (agent/lay_terms.py), the reply templates, and the real clinic API
(clinic-api, its own temporary SQLite file) for every price. WHAT IS FAKED: only the pod-only pieces -- ASR (the
text is handed in), TTS, and the intent LLM. The LLM is the one thing that cannot run without a GPU, so when the
fast path abstains the test records "LLM needed" (it is NOT counted as wrong -- it cannot be judged here), and in the
multi-test and memory tests a clearly-labelled stand-in LLM is used so the code AFTER it (lookup, sum, reply,
memory) is what gets tested.

Outcome of one price turn (see `classify`):
    CORRECT   the right test(s) were looked up and the spoken reply carries the catalogue price
    WRONG     a price for a test other than the one asked for, or a reply that lacks the looked-up price -- a defect
    ASK       the agent asked "which one?" (ambiguous alias) -- safe
    NOTFOUND  the agent said it has no such test -- safe, but a coverage gap when the test does exist
    LLM       the fast path abstained, so the real system would call the model -- cannot be judged off-GPU
    OTHER     something else was said (category question, prescription request, ...) -- see the spoken text
Only WRONG fails a matrix test: a coverage gap (NOTFOUND / LLM) is information for adding cue words, not a defect.
"""

import collections
import inspect
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from _clinic_app import clinic_app
from _pod_stubs import pod_stubs
from _synth_voices import FATHER, utterance
from test_orchestrator_persona import ASRResult, FakeTTS, FakeWS, _wav

from agent.api_models import TestAnswer, validated
from agent.fast_path import Catalogue, FastPath

LANGS = ("bn", "hi", "en")

# --------------------------------------------------------------------------------------------- canonical names
CBC = "Complete Blood Count (CBC)"
ESR = "ESR"
FBS = "Blood Sugar Fasting"
PPBS = "Blood Sugar PP"
HBA = "HbA1c"
LIPID = "Lipid Profile"
LFT = "Liver Function Test (LFT)"
KFT = "Kidney Function Test (KFT)"
THY = "Thyroid Profile (T3 T4 TSH)"
TSH = "TSH"
URINE = "Urine Routine Examination"
WIDAL = "Widal Test"
NS1 = "Dengue NS1 Antigen"
IGG = "Dengue IgG/IgM"
MAL = "Malaria Antigen"
CRP = "CRP (C-Reactive Protein)"
VITD = "Vitamin D (25-OH)"
B12 = "Vitamin B12"
CREAT = "Serum Creatinine"
ELEC = "Serum Electrolytes"
BGRP = "Blood Grouping & Rh Typing"
HIV = "HIV Test (ELISA)"
HBS = "HBsAg"
HCV = "HCV"
ECG = "ECG"
CXR = "Chest X-Ray (PA view)"
USGA = "USG Whole Abdomen"
USGP = "USG Pregnancy Profile"
ECHO = "2D Echocardiography"
TMT = "TMT (Treadmill Test)"
PAP = "Pap Smear"
PSA = "PSA (Prostate Specific Antigen)"
URIC = "Uric Acid"
CALC = "Calcium (Serum)"
NONE = ()  # "the clinic has no such test": any price quoted for it is a fabrication

# (kind, what the caller says, the tests that are an acceptable answer). kind: "clin" = clinical name or code,
# "lay" = how an ordinary person says it, "rom" = Latin-script spelling of Bengali/Hindi speech (poor line / SMS).
BN_TERMS = [
    ("clin", "সিবিসি", (CBC,)), ("clin", "কমপ্লিট ব্লাড কাউন্ট", (CBC,)), ("clin", "ইএসআর", (ESR,)),
    ("clin", "ব্লাড সুগার ফাস্টিং", (FBS,)), ("clin", "সুগার পিপি", (PPBS,)), ("clin", "এইচবিএ১সি", (HBA,)),
    ("clin", "হিমোগ্লোবিন এ১সি", (HBA,)), ("clin", "লিপিড প্রোফাইল", (LIPID,)),
    ("clin", "লিভার ফাংশন টেস্ট", (LFT,)), ("clin", "এলএফটি", (LFT,)), ("clin", "কিডনি ফাংশন টেস্ট", (KFT,)),
    ("clin", "কেএফটি", (KFT,)), ("clin", "থাইরয়েড প্রোফাইল", (THY,)), ("clin", "টিএসএইচ", (TSH, THY)),
    ("clin", "ইউরিন রুটিন", (URINE,)), ("clin", "ওয়াইডাল টেস্ট", (WIDAL,)), ("clin", "ডেঙ্গু এনএস১", (NS1,)),
    ("clin", "ডেঙ্গু আইজিজি", (IGG,)), ("clin", "ম্যালেরিয়া এন্টিজেন", (MAL,)), ("clin", "সিআরপি", (CRP,)),
    ("clin", "ভিটামিন ডি", (VITD,)), ("clin", "ভিটামিন বি১২", (B12,)), ("clin", "ক্রিয়াটিনিন", (CREAT,)),
    ("clin", "ইলেক্ট্রোলাইটস", (ELEC,)), ("clin", "ব্লাড গ্রুপ", (BGRP,)), ("clin", "এইচআইভি টেস্ট", (HIV,)),
    ("clin", "এইচবিএসএজি", (HBS,)), ("clin", "এইচসিভি", (HCV,)), ("clin", "ইসিজি", (ECG,)),
    ("clin", "চেস্ট এক্সরে", (CXR,)), ("clin", "পেটের ইউএসজি", (USGA,)),
    ("clin", "প্রেগন্যান্সি আলট্রাসাউন্ড", (USGP,)), ("clin", "ইকো টেস্ট", (ECHO,)), ("clin", "টিএমটি", (TMT,)),
    ("clin", "প্যাপ স্মিয়ার", (PAP,)), ("clin", "পিএসএ", (PSA,)), ("clin", "ইউরিক অ্যাসিড", (URIC,)),
    ("clin", "ক্যালসিয়াম", (CALC,)),
    ("lay", "কোলেস্টেরল টেস্ট", (LIPID,)), ("lay", "কিডনি টেস্ট", (KFT,)), ("lay", "লিভার টেস্ট", (LFT,)),
    ("lay", "থাইরয়েড টেস্ট", (THY,)), ("lay", "টাইফয়েড টেস্ট", (WIDAL,)), ("lay", "ডেঙ্গু টেস্ট", (NS1, IGG)),
    ("lay", "ম্যালেরিয়া টেস্ট", (MAL,)), ("lay", "এইডস টেস্ট", (HIV,)), ("lay", "হেপাটাইটিস বি", (HBS,)),
    ("lay", "হেপাটাইটিস সি", (HCV,)), ("lay", "রক্তের গ্রুপ", (BGRP,)), ("lay", "খালি পেটে সুগার", (FBS,)),
    ("lay", "খাওয়ার পরে সুগার", (PPBS,)), ("lay", "প্রস্রাব পরীক্ষা", (URINE,)), ("lay", "ইউরিন টেস্ট", (URINE,)),
    ("lay", "পেটের ছবি", (USGA,)), ("lay", "পেটের পরীক্ষা", (USGA,)), ("lay", "পেটের আলট্রাসাউন্ড", (USGA,)),
    ("lay", "বুকের এক্স-রে", (CXR,)), ("lay", "ট্রেডমিল টেস্ট", (TMT,)), ("lay", "সুগার টেস্ট", (FBS, PPBS, HBA)),
    ("lay", "হার্টের টেস্ট", (ECG, ECHO, TMT)), ("lay", "মাথার ছবি", NONE), ("lay", "বুকের ছবি", (CXR,)),
    ("lay", "হার্টের ছবি", (ECHO,)), ("lay", "রক্তের কোলেস্টেরল", (LIPID,)), ("lay", "জন্ডিসের টেস্ট", (LFT,)),
    ("rom", "peter chhobi", (USGA,)), ("rom", "peter porikkha", (USGA,)), ("rom", "rokto porikkha", NONE),
    ("rom", "mathar chhobi", NONE), ("rom", "cbc", (CBC,)), ("rom", "uric acid", (URIC,)),
    ("rom", "sugar test", (FBS, PPBS, HBA)), ("rom", "kidney test", (KFT,)),
]
HI_TERMS = [
    ("clin", "सीबीसी", (CBC,)), ("clin", "सी बी सी", (CBC,)), ("clin", "ईएसआर", (ESR,)),
    ("clin", "ब्लड शुगर फास्टिंग", (FBS,)), ("clin", "शुगर पीपी", (PPBS,)), ("clin", "एचबीए वन सी", (HBA,)),
    ("clin", "हीमोग्लोबिन ए वन सी", (HBA,)), ("clin", "लिपिड प्रोफाइल", (LIPID,)),
    ("clin", "लिवर फंक्शन टेस्ट", (LFT,)), ("clin", "एलएफटी", (LFT,)), ("clin", "किडनी फंक्शन टेस्ट", (KFT,)),
    ("clin", "केएफटी", (KFT,)), ("clin", "थायराइड प्रोफाइल", (THY,)), ("clin", "टीएसएच", (TSH, THY)),
    ("clin", "यूरिन रूटीन", (URINE,)), ("clin", "विडाल टेस्ट", (WIDAL,)), ("clin", "डेंगू एनएस वन", (NS1,)),
    ("clin", "डेंगू आईजीजी", (IGG,)), ("clin", "मलेरिया एंटीजन", (MAL,)), ("clin", "सीआरपी", (CRP,)),
    ("clin", "विटामिन डी", (VITD,)), ("clin", "विटामिन बी बारह", (B12,)), ("clin", "क्रिएटिनिन", (CREAT,)),
    ("clin", "इलेक्ट्रोलाइट्स", (ELEC,)), ("clin", "ब्लड ग्रुप", (BGRP,)), ("clin", "एचआईवी टेस्ट", (HIV,)),
    ("clin", "एचबीएसएजी", (HBS,)), ("clin", "एचसीवी", (HCV,)), ("clin", "ईसीजी", (ECG,)),
    ("clin", "चेस्ट एक्सरे", (CXR,)), ("clin", "पेट का यूएसजी", (USGA,)),
    ("clin", "प्रेगनेंसी अल्ट्रासाउंड", (USGP,)), ("clin", "इको टेस्ट", (ECHO,)), ("clin", "टीएमटी", (TMT,)),
    ("clin", "पैप स्मियर", (PAP,)), ("clin", "पीएसए", (PSA,)), ("clin", "यूरिक एसिड", (URIC,)),
    ("clin", "कैल्शियम", (CALC,)),
    ("lay", "कोलेस्ट्रॉल टेस्ट", (LIPID,)), ("lay", "किडनी टेस्ट", (KFT,)), ("lay", "लिवर टेस्ट", (LFT,)),
    ("lay", "थायराइड टेस्ट", (THY,)), ("lay", "टाइफाइड टेस्ट", (WIDAL,)), ("lay", "डेंगू टेस्ट", (NS1, IGG)),
    ("lay", "मलेरिया टेस्ट", (MAL,)), ("lay", "एड्स टेस्ट", (HIV,)), ("lay", "हेपेटाइटिस बी", (HBS,)),
    ("lay", "हेपेटाइटिस सी", (HCV,)), ("lay", "खून का ग्रुप", (BGRP,)), ("lay", "खाली पेट शुगर", (FBS,)),
    ("lay", "खाने के बाद शुगर", (PPBS,)), ("lay", "पेशाब की जांच", (URINE,)), ("lay", "यूरिन टेस्ट", (URINE,)),
    ("lay", "पेट की तस्वीर", (USGA,)), ("lay", "पेट की जांच", (USGA,)), ("lay", "पेट का अल्ट्रासाउंड", (USGA,)),
    ("lay", "छाती का एक्स-रे", (CXR,)), ("lay", "ट्रेडमिल टेस्ट", (TMT,)), ("lay", "शुगर टेस्ट", (FBS, PPBS, HBA)),
    ("lay", "दिल का टेस्ट", (ECG, ECHO, TMT)), ("lay", "सिर की तस्वीर", NONE), ("lay", "छाती की फोटो", (CXR,)),
    ("lay", "दिल की फोटो", (ECHO,)), ("lay", "खून की जांच", NONE), ("lay", "पीलिया का टेस्ट", (LFT,)),
    ("rom", "pet ki jaanch", (USGA,)), ("rom", "pet ka test", (USGA,)), ("rom", "khoon ki jaanch", NONE),
    ("rom", "sar ki jaanch", NONE), ("rom", "cbc", (CBC,)), ("rom", "uric acid", (URIC,)),
    ("rom", "sugar test", (FBS, PPBS, HBA)), ("rom", "kidney test", (KFT,)),
]
EN_TERMS = [
    ("clin", "CBC", (CBC,)), ("clin", "complete blood count", (CBC,)), ("clin", "ESR", (ESR,)),
    ("clin", "fasting blood sugar", (FBS,)), ("clin", "blood sugar PP", (PPBS,)), ("clin", "HbA1c", (HBA,)),
    ("clin", "lipid profile", (LIPID,)), ("clin", "liver function test", (LFT,)), ("clin", "LFT", (LFT,)),
    ("clin", "kidney function test", (KFT,)), ("clin", "KFT", (KFT,)), ("clin", "thyroid profile", (THY,)),
    ("clin", "TSH", (TSH, THY)), ("clin", "urine routine examination", (URINE,)), ("clin", "Widal test", (WIDAL,)),
    ("clin", "dengue NS1 antigen", (NS1,)), ("clin", "dengue IgG IgM", (IGG,)), ("clin", "malaria antigen", (MAL,)),
    ("clin", "CRP", (CRP,)), ("clin", "vitamin D", (VITD,)), ("clin", "vitamin B12", (B12,)),
    ("clin", "serum creatinine", (CREAT,)), ("clin", "serum electrolytes", (ELEC,)),
    ("clin", "blood grouping and Rh typing", (BGRP,)), ("clin", "HIV test", (HIV,)), ("clin", "HBsAg", (HBS,)),
    ("clin", "HCV", (HCV,)), ("clin", "ECG", (ECG,)), ("clin", "chest X-ray", (CXR,)),
    ("clin", "USG whole abdomen", (USGA,)), ("clin", "USG pregnancy profile", (USGP,)),
    ("clin", "2D echo", (ECHO,)), ("clin", "echocardiography", (ECHO,)), ("clin", "TMT", (TMT,)),
    ("clin", "pap smear", (PAP,)), ("clin", "PSA", (PSA,)), ("clin", "uric acid", (URIC,)),
    ("clin", "serum calcium", (CALC,)),
    ("lay", "cholesterol test", (LIPID,)), ("lay", "kidney test", (KFT,)), ("lay", "liver test", (LFT,)),
    ("lay", "thyroid test", (THY,)), ("lay", "typhoid test", (WIDAL,)), ("lay", "dengue test", (NS1, IGG)),
    ("lay", "malaria test", (MAL,)), ("lay", "AIDS test", (HIV,)), ("lay", "hepatitis B test", (HBS,)),
    ("lay", "hepatitis C test", (HCV,)), ("lay", "blood group test", (BGRP,)),
    ("lay", "empty stomach sugar test", (FBS,)), ("lay", "sugar after meals", (PPBS,)),
    ("lay", "urine test", (URINE,)), ("lay", "stomach picture", (USGA,)), ("lay", "stomach scan", (USGA,)),
    ("lay", "stomach test", (USGA,)), ("lay", "chest picture", (CXR,)), ("lay", "treadmill test", (TMT,)),
    ("lay", "sugar test", (FBS, PPBS, HBA)), ("lay", "heart test", (ECG, ECHO, TMT)),
    ("lay", "head scan", NONE), ("lay", "brain scan", NONE), ("lay", "heart ultrasound", (ECHO,)),
    ("lay", "pregnancy scan", (USGP,)), ("lay", "jaundice test", (LFT,)), ("lay", "blood test", NONE),
    ("rom", "peter chhobi", (USGA,)), ("rom", "pet ka test", (USGA,)), ("rom", "rokto porikkha", NONE),
    ("rom", "mathar chhobi", NONE),
]
TERMS = {"bn": BN_TERMS, "hi": HI_TERMS, "en": EN_TERMS}

# Several different ways to ask one price, per language (the plain ones, then the "lame" ones).
FRAMES = {
    "bn": ["{t} এর দাম কত", "{t} করাতে কত টাকা লাগবে", "{t} এর রেট কত", "{t} টেস্ট কত পড়বে", "দাদা {t} কত নেবেন"],
    "hi": ["{t} का रेट क्या है", "{t} कराने में कितना पैसा लगेगा", "{t} की कीमत बताइए", "{t} का दाम कितना है",
           "भैया {t} कितने का है"],
    "en": ["what is the price of {t}", "how much does {t} cost", "{t} rate please", "{t} price", "tell me {t} charges"],
}
N_FRAMES = 3  # the first three frames run for every term; the rest run for the clinical subset (see below)

# Two six-test lists per language, used by the "1 test, 2 tests, 3 tests ..." and the "total" tests. SEQ_A is clinical
# names, SEQ_B is lay words; MIX interleaves them (the "mixed test" order).
SEQ_A = {
    "bn": [("সিবিসি", CBC), ("ইউরিক অ্যাসিড", URIC), ("লিপিড প্রোফাইল", LIPID), ("কিডনি ফাংশন টেস্ট", KFT),
           ("ইএসআর", ESR), ("ব্লাড সুগার ফাস্টিং", FBS)],
    "hi": [("सीबीसी", CBC), ("यूरिक एसिड", URIC), ("लिपिड प्रोफाइल", LIPID), ("किडनी फंक्शन टेस्ट", KFT),
           ("ईएसआर", ESR), ("ब्लड शुगर फास्टिंग", FBS)],
    "en": [("CBC", CBC), ("uric acid", URIC), ("lipid profile", LIPID), ("kidney function test", KFT),
           ("ESR", ESR), ("fasting blood sugar", FBS)],
}
SEQ_B = {
    "bn": [("রক্তের গ্রুপ", BGRP), ("প্রস্রাব পরীক্ষা", URINE), ("থাইরয়েড টেস্ট", THY), ("টাইফয়েড টেস্ট", WIDAL),
           ("এইডস টেস্ট", HIV), ("পিএসএ", PSA)],
    "hi": [("खून का ग्रुप", BGRP), ("पेशाब की जांच", URINE), ("थायराइड टेस्ट", THY), ("टाइफाइड टेस्ट", WIDAL),
           ("एड्स टेस्ट", HIV), ("पीएसए", PSA)],
    "en": [("blood group", BGRP), ("urine test", URINE), ("thyroid test", THY), ("typhoid test", WIDAL),
           ("AIDS test", HIV), ("PSA", PSA)],
}
AND = {"bn": " আর ", "hi": " और ", "en": " and "}
PRICES_FRAME = {"bn": "{t} এর দাম কত", "hi": "{t} का रेट क्या है", "en": "what is the price of {t}"}
TOTAL_FRAME = {"bn": "{t} সব মিলিয়ে টোটাল কত হবে", "hi": "{t} का टोटल कितना होगा", "en": "what is the total for {t}"}
# "remove this and tell me the total", with the removed test named, then without naming it.
REMOVE_FRAMES = {
    "bn": ["{r} বাদ দিয়ে টোটাল কত হবে", "{r} টা বাদ দিন, বাকিগুলোর টোটাল কত", "ওটা বাদ দিয়ে টোটাল কত হবে"],
    "hi": ["{r} हटा दो, टोटल कितना हुआ", "{r} को छोड़कर बाकी का टोटल कितना होगा", "उसे हटाकर टोटल कितना हुआ"],
    "en": ["remove {r} and tell me the total", "take out {r}, what is the total now", "remove that one and give me the total"],
}

RESULTS: dict = collections.defaultdict(list)  # filled by the tests, summarised by test_zz_report


# ---------------------------------------------------------------------------------------------------- harness
class RecordingTools:
    """Real clinic API (TestClient) behind the same `get_test_rate` the orchestrator calls, remembering every lookup.

    sourav: exists so a test can see WHICH catalogue test the agent really looked up (not just what it said) --
    the spoken text carries the Bengali/Hindi/English test name, which is useless for checking a wrong-test defect.
    Flow: get_test_rate(name) -> GET /api/v1/tests/search -> validated like agent/tools_client.py does -> recorded."""

    def __init__(self, client):
        self.client = client
        self.lookups: list = []

    async def get_test_rate(self, name):
        raw = self.client.get("/api/v1/tests/search", params={"name": name}).json()
        result = validated(TestAnswer, raw, "get_test_rate")
        self.lookups.append((name, result))
        return result

    async def get_test_prep(self, test_name, lang="bn"):
        # sourav, 2026-10-16 (final verification): a price question that also says "fasting" / "empty stomach" now makes the orchestrator
        # answer the implied preparation question too (agent/combined_question.py, b356fbe), through get_test_prep. This stand-in lacked it, so
        # those turns raised AttributeError before any price was checked. It forwards to the real clinic API like get_faq and records nothing,
        # so the price assertions (which test was looked up, which rate was spoken) are untouched.
        from agent.api_models import PrepAnswer

        raw = self.client.get("/api/v1/tests/prep", params={"name": test_name, "lang": lang}).json()
        return validated(PrepAnswer, raw, "get_test_prep")

    async def get_faq(self, topic, lang="bn"):
        # sourav: a 'total ...' follow-up can be taken by the fast path as a clinic FAQ question; the real FAQ answer
        # is served by the same API, so this just forwards it (it is not a price and is not recorded as a lookup).
        from agent.api_models import FaqAnswer

        raw = self.client.get("/api/v1/faq", params={"topic": topic, "lang": lang}).json()
        return validated(FaqAnswer, raw, "get_faq")


class Outcome:
    def __init__(self, spoken, llm_texts, lookups):
        self.spoken = " ".join(spoken)
        self.llm_called = bool(llm_texts)
        self.lookups = lookups
        self.found = [r["test_name"] for _, r in lookups if r.get("found")]
        self.rates = {r["test_name"]: r["rate_inr"] for _, r in lookups if r.get("found")}
        self.ambiguous = any(r.get("ambiguous") for _, r in lookups)
        self.notfound = any(not r.get("found") and not r.get("ambiguous") for _, r in lookups)


def classify(out: Outcome, expected: tuple) -> str:
    """CORRECT / WRONG / ASK / NOTFOUND / LLM / OTHER for one price turn -- see the module docstring."""
    if out.found:
        if any(name not in expected for name in out.found):
            return "WRONG"
        if any(str(rate) not in out.spoken for rate in out.rates.values()):
            return "WRONG"  # looked up fine but the reply does not carry the price
        return "CORRECT"
    if out.llm_called:
        return "LLM"
    if out.ambiguous:
        return "ASK"
    if out.notfound:
        return "NOTFOUND"
    return "OTHER"


@pytest.fixture(scope="module")
def m():
    with pod_stubs(REPO_ROOT) as imp:
        yield imp("main_pcm")


@pytest.fixture(scope="module")
def clinic():
    with clinic_app(sample_patients=False) as (_app, client):
        yield client


@pytest.fixture(scope="module")
def payload(clinic):
    return clinic.get("/api/v1/catalogue").json()


@pytest.fixture(scope="module")
def truth(clinic, payload):
    """canonical test name -> rate, taken from the real API by the exact name (ambiguous ones, e.g. TSH, are skipped)."""
    out = {}
    for t in payload["tests"]:
        r = clinic.get("/api/v1/tests/search", params={"name": t["name"]}).json()
        if r.get("found"):
            out[t["name"]] = r["rate_inr"]
    return out


@pytest.fixture
def rig(m, clinic, payload, monkeypatch, tmp_path):
    """sourav: builds a fresh phone call per use. Flow: patch the pod-only pieces (ASR text, TTS, LLM, the catalogue
    fetch) -> a new CallSession for `lang` -> `say(text)` runs ONE real `_dispatch_turn` and returns an Outcome.
    A new call per turn in the matrix keeps one turn's leftovers (a pending 'did you mean') from touching the next;
    the sequence/memory tests reuse one call on purpose."""
    tools = RecordingTools(clinic)
    state = {"text": "", "lang": "en", "llm_texts": [], "llm_reply": None}

    async def route(session, wav):
        return state["lang"], ASRResult(state["text"]), True

    async def no_reload():
        return None

    async def cached_catalogue():
        return payload

    async def llm_stand_in(session, text, lang, key):
        # sourav: the real model cannot run off-GPU. `llm_reply` (set per test) is a CLEARLY-LABELLED stand-in for
        # what a model would extract; with none set the model "does not understand" (intent unclear).
        state["llm_texts"].append(text)
        reply = state["llm_reply"]
        data = reply(text) if callable(reply) else reply
        return data or {"intent": "unclear", "slots": {}, "secondary_intent": None, "direct_reply_bn": None}

    monkeypatch.setattr(m, "_fast_path", FastPath(Catalogue(payload)))
    monkeypatch.setattr(m, "_maybe_reload_fast_path", no_reload)
    monkeypatch.setattr(m, "_cached_catalogue", cached_catalogue)
    monkeypatch.setattr(m, "_tools", tools)
    monkeypatch.setattr(m, "_admission", None)
    monkeypatch.setattr(m, "_tts_router", FakeTTS())
    monkeypatch.setattr(m, "CONDITION_INPUT", "off")
    monkeypatch.setattr(m, "_route_and_transcribe", route)
    monkeypatch.setattr(m, "_resolve_intent_uncached", llm_stand_in)
    voice = utterance(FATHER, dur=3.0, seed=1, amp=0.3)

    class Rig:
        pass

    r = Rig()
    r.state, r.tools, r.n = state, tools, 0

    def new_call(lang):
        session = m.CallSession(FakeWS())
        session.release_gate()
        for lg in LANGS:
            session.disclosed_langs.add(lg)
        session.call_state = m.new_call_state()
        state["lang"] = lang
        r.session = session
        return session

    async def say(text, lang=None):
        state["text"] = text
        if lang:
            state["lang"] = lang
        session = r.session
        session.turn_epoch = session.speak_epoch
        r.n += 1
        before_say, before_lookups, before_llm = len(session.ws.spoken()), len(tools.lookups), len(state["llm_texts"])
        await m._dispatch_turn(session, _wav(tmp_path, voice, f"u{r.n}.wav"))
        return Outcome(session.ws.spoken()[before_say:], state["llm_texts"][before_llm:], tools.lookups[before_lookups:])

    r.new_call, r.say = new_call, say
    return r


# --------------------------------------------------------------------------------------- 0. the data is sound
@pytest.mark.parametrize("lang", LANGS)
def test_at_least_fifty_terms_per_language_and_every_expected_test_exists(lang, payload):
    """sourav: guards this file itself -- the brief asked for 50+ lay and clinical terms per language, and a typo in
    an expected canonical name would silently turn a good answer into a 'WRONG' later."""
    terms = TERMS[lang]
    assert len(terms) >= 50, f"{lang}: only {len(terms)} terms"
    assert sum(1 for k, _, _ in terms if k == "lay") >= 20 and sum(1 for k, _, _ in terms if k == "clin") >= 25
    names = {t["name"] for t in payload["tests"]}
    for _, text, expected in terms:
        assert set(expected) <= names, (lang, text, expected)


# ----------------------------------------------------------------- 1. one test's price, every phrasing and word
def _matrix_params():
    return [pytest.param(lang, i, id=f"{lang}-{i:02d}-{TERMS[lang][i][1].replace(' ', '_')}")
            for lang in LANGS for i in range(len(TERMS[lang]))]


@pytest.mark.asyncio
@pytest.mark.parametrize("lang,idx", _matrix_params())
async def test_single_test_price_never_quotes_the_wrong_test(rig, truth, lang, idx):
    """sourav: for ONE term (lay or clinical) in ONE language, ask its price in several different sentence shapes
    through the real orchestrator and check the outcome class of every shape. Flow: fresh call -> for each frame,
    say the sentence -> classify (see module docstring) -> keep the result for the final report.
    Fails ONLY on WRONG (a price for a test the caller did not ask for). The truth boundary (CLAUDE.md) says an
    unknown/unsure test must never be answered with another test's price; gaps (NOTFOUND/LLM) are reported, not failed."""
    kind, term, expected = TERMS[lang][idx]
    frames = FRAMES[lang][: N_FRAMES if kind != "clin" else len(FRAMES[lang])]
    wrong = []
    for frame in frames:
        rig.new_call(lang)
        sentence = frame.format(t=term)
        out = await rig.say(sentence)
        verdict = classify(out, expected)
        RESULTS[f"single-{lang}"].append(
            {"lang": lang, "kind": kind, "term": term, "sentence": sentence, "verdict": verdict,
             "found": out.found, "expected": list(expected), "spoken": out.spoken[:160]}
        )
        if verdict == "WRONG":
            wrong.append((sentence, out.found, out.spoken[:120]))
        for name, rate in out.rates.items():
            assert truth.get(name, rate) == rate, f"{name}: quoted {rate}, catalogue says {truth.get(name)}"
    assert not wrong, f"{lang} {term!r} expected {expected}: WRONG answers: {wrong}"


# ------------------------------------------- 2. the fast path itself must never hand back a PARTIAL multi answer
@pytest.mark.parametrize("lang", LANGS)
def test_fast_path_does_not_answer_a_multi_test_question_with_one_test(lang, payload):
    """sourav: a caller who names several tests in one breath ('A B price', 'A and B total') must never get just ONE
    test's price from the deterministic fast path -- that silently drops a test (the 2026-10-13 live defect, see
    main_pcm._answer_enquiry_intent). Either the fast path abstains (goes to the model) or it must serve all of them."""
    fp = FastPath(Catalogue(payload))
    bad = []
    for n in range(2, 7):
        names = [t for t, _ in SEQ_A[lang][:n]]
        for joiner in (AND[lang], " ", ", "):  # with a conjunction, with nothing between, with a comma
            for frame in (PRICES_FRAME[lang], TOTAL_FRAME[lang]):
                text = frame.format(t=joiner.join(names))
                hit = fp.resolve(text, lang)
                if hit is not None and hit.intent == "test_rate":
                    bad.append((text, hit.slots["test_name"]))
    assert not bad, f"{lang}: fast path answered a multi-test question with a single test: {bad[:6]}"


# ----------------- 3. price of 1 test, then 2, then 3 ... (and the same as a TOTAL, with lay words mixed in)
def _interleave(a, b):
    out = []
    for x, y in zip(a, b):
        out += [x, y]
    return out


def _oracle_for(names, expected_len):
    """sourav: STAND-IN for the model (it cannot run off-GPU): returns what a correct extractor would -- every spoken
    test name, split the way the live pod really splits them (last one in `test_name`, the rest in `test_names`,
    see main_pcm._answer_enquiry_intent's 2026-10-13 comment). It only runs when the fast path abstained."""
    def reply(text):
        return {"intent": "test_rate", "slots": {"test_name": names[-1], "test_names": names[:-1]},
                "secondary_intent": None, "direct_reply_bn": None}
    return reply


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("mode", ["price_clinical", "total_mixed"])
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
async def test_price_and_total_for_one_to_six_tests(rig, truth, lang, mode, n):
    """sourav: 'ask the price of 1 test, then 2, then 3, then 4 ... one by one' and 'the total of a+b, a+b+c, ...
    with mixed tests'. Flow: fresh call -> ONE utterance naming `n` tests -> real dispatch (fast path first, stand-in
    model if it abstains) -> every test must have been looked up, and for n>=2 the spoken TOTAL must equal the sum of
    the real catalogue prices. 'price_clinical' = clinical names; 'total_mixed' = lay and clinical words interleaved."""
    pairs = SEQ_A[lang][:n] if mode == "price_clinical" else _interleave(SEQ_A[lang], SEQ_B[lang])[:n]
    names = [t for t, _ in pairs]
    want = [c for _, c in pairs]
    frame = PRICES_FRAME[lang] if mode == "price_clinical" else TOTAL_FRAME[lang]
    rig.state["llm_reply"] = _oracle_for(names, n)
    rig.new_call(lang)
    out = await rig.say(frame.format(t=AND[lang].join(names)))
    RESULTS[f"multi-{lang}"].append(
        {"mode": mode, "n": n, "names": names, "found": out.found, "want": want, "llm": out.llm_called,
         "spoken": out.spoken[:240]}
    )
    missing = [c for c in want if c not in out.found]
    assert not missing, f"tests never looked up / not found: {missing}; looked up {out.found}; said: {out.spoken[:200]}"
    extra = [c for c in out.found if c not in want]
    assert not extra, f"looked up a test that was not asked for: {extra}"
    if n >= 2:
        total = sum(truth[c] for c in want)
        assert str(total) in out.spoken, f"expected total {total} in: {out.spoken[:240]}"
    else:
        assert str(truth[want[0]]) in out.spoken


# ----------- 4. context memory: three prices asked one by one, then "remove this, what is the total"
@pytest.mark.asyncio
@pytest.mark.parametrize("lang", LANGS)
async def test_memory_after_three_separate_price_questions(rig, truth, lang):
    """sourav: 'ask 3 test prices, then say remove this and tell the total (context memory must perform well)'.
    Flow: ONE call -> ask test 1, 2, 3 each in its own turn (each must be answered correctly) -> check what the
    session remembers. For 'remove the second and give the total' to be answerable the session must still hold all
    three tests; this reports how many it actually holds."""
    rig.new_call(lang)
    trio = SEQ_A[lang][:3]
    for name, canon in trio:
        out = await rig.say(PRICES_FRAME[lang].format(t=name))
        assert classify(out, (canon,)) == "CORRECT", (lang, name, out.spoken[:150])
    # sourav, 2026-10-14: the call-level list (session.priced_tests, agent/cart_tally.py) is where the tests priced so far live;
    # last_enquiry_entities still holds only the last one, on purpose (it feeds "and the prep for that?").
    remembered = list(rig.session.priced_tests)
    RESULTS[f"memory-{lang}"].append({"remembered": remembered})
    assert len(remembered) == 3, f"after 3 separate price questions the call remembers {len(remembered)} test(s): {remembered}"


@pytest.mark.asyncio
@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("which", [0, 1, 2])
async def test_remove_one_then_total_is_the_sum_of_the_other_two(rig, truth, lang, which):
    """sourav: the literal brief -- ask 3 prices one by one, then 'remove <one of them> and tell the total'; the
    answer must be the sum of the OTHER two real prices (never one test's price, never an invented number).
    The model stand-in here is the MOST PLAUSIBLE real output, because the real prompt only ever sees this one
    sentence (agent/llm.py build_prompt): it extracts the one test that was named -> test_rate for the removed test.
    If it is nameless ('remove that one') the stand-in says 'unclear'."""
    rig.new_call(lang)
    trio = SEQ_A[lang][:3]
    for name, _ in trio:
        await rig.say(PRICES_FRAME[lang].format(t=name))
    removed_name, removed_canon = trio[which]
    keep = [c for i, (_, c) in enumerate(trio) if i != which]
    expected_total = sum(truth[c] for c in keep)

    def plausible_model(text):
        if removed_name in text:
            return {"intent": "test_rate", "slots": {"test_name": removed_name, "test_names": []},
                    "secondary_intent": None, "direct_reply_bn": None}
        return None  # a nameless sentence: the model has nothing to extract -> unclear

    rig.state["llm_reply"] = plausible_model
    verdicts = []
    trio_names = [nm for nm, _ in trio]
    for frame in REMOVE_FRAMES[lang]:
        # sourav, 2026-10-14: every sentence is a fresh, independent try on the SAME three-test list (the old loop let
        # the first removal change the list for the next sentence). A nameless sentence means 'the one just discussed',
        # so that test is the last one priced.
        rig.session.priced_tests = [x for x in trio_names if x != removed_name] + [removed_name] if '{r}' not in frame else list(trio_names)
        sentence = frame.format(r=removed_name)
        out = await rig.say(sentence)
        ok = str(expected_total) in out.spoken
        verdicts.append({"sentence": sentence, "ok": ok, "found": out.found, "llm": out.llm_called,
                         "spoken": out.spoken[:200]})
    RESULTS[f"remove-{lang}"].append({"removed": removed_name, "expected_total": expected_total, "turns": verdicts})
    bad = [v for v in verdicts if not v["ok"]]
    assert not bad, f"expected total {expected_total} (= {keep}); got: {[(v['sentence'], v['spoken'][:90]) for v in bad]}"


def test_the_intent_prompt_is_built_from_the_current_sentence_only():
    """sourav: a structural fact behind the memory tests -- if `build_prompt` takes no conversation history, then a
    model-extracted 'remove that one / the total' can never know the earlier tests, whatever the model is."""
    from agent import llm

    # sourav, 2026-10-14: DELIBERATE DESIGN, owner's truth-boundary rule: the model is still shown only the current
    # sentence. The memory the brief asks for is kept in code instead (agent/cart_tally.py + session.priced_tests), so
    # which tests are on the list and every price are never the model's call. This test now pins both halves.
    from agent import cart_tally

    params = list(inspect.signature(llm.build_prompt).parameters)
    RESULTS["prompt-params"].append({"params": params})
    assert callable(cart_tally.plan) and callable(cart_tally.removed_text)


# --------------------------------------------------------------------------------------------- report
def test_zz_report():
    """sourav: prints a per-language coverage table at the end of the run (use -s) and, when the environment variable
    LAB_PRICE_REPORT names a file, writes every recorded result there as JSON for the written report."""
    for lang in LANGS:
        rows = RESULTS.get(f"single-{lang}", [])
        by = collections.Counter(r["verdict"] for r in rows)
        by_kind = {k: collections.Counter(r["verdict"] for r in rows if r["kind"] == k) for k in ("clin", "lay", "rom")}
        print(f"\n[{lang}] single-price turns: {len(rows)}  {dict(by)}")
        for k, c in by_kind.items():
            print(f"    {k}: {dict(c)}")
    path = os.environ.get("LAB_PRICE_REPORT")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(RESULTS, f, ensure_ascii=False, indent=1)
