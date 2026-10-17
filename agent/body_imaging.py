"""'A picture of my chest / head / belly / hand' -- which imaging test does the clinic actually do for that part of the body?

Callers rarely say "X-ray" or "CT". They say what they want a picture OF: "buker chhobi korbo", "mathar chhobi korbo", "peter chhobi korbo",
"haater chhobi tulbo" (Bengali), "छाती का फोटो कराना है", "हाथ की तस्वीर" (Hindi). The body part names the region; the clinic's own catalogue says
which imaging tests exist for it. The agent then ASKS, from those real tests only: "are you asking about the chest X-ray?" (one test) or "the
chest X-ray or the chest CT?" (several). If the catalogue has no imaging test for that part, it says so plainly and sends the caller to the
prescription route -- it never names a scan the clinic does not do (CLAUDE.md truth boundary), and it never recommends a test: it only
clarifies the one the caller already asked for.

The region -> test mapping is read from test NAMES in the live catalogue (a test whose name mentions the region), so a CT or MRI added to the
catalogue tomorrow is offered with no change here.
"""

from __future__ import annotations

import re
import unicodedata

# region -> (words that name the body part in bn / hi / en,   words in a catalogue TEST NAME that mean the same region)
REGIONS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "chest": (("বুকের", "বুকে", "বুক", "छाती", "सीने", "सीना", "chest", "buker", "buke", "chhati", "seene"), ("chest", "thorax", "lung", "thoracic")),
    "head": (("মাথার", "মাথা", "মস্তিষ্ক", "ব্রেন", "सिर", "दिमाग", "मस्तिष्क", "head", "brain", "mathar", "matha", "dimag"), ("head", "brain", "skull", "cranial")),
    "abdomen": (("পেটের", "পেট", "पेट", "पेट", "stomach", "abdomen", "belly", "peter", "pet ki"), ("abdomen", "abdominal", "whole abdomen", "kub")),
    "hand": (("হাতের", "হাত", "হাতে", "বাহুর", "কব্জি", "कलाई", "हाथ", "बाजू", "hand", "arm", "wrist", "haater", "hater", "haath"), ("hand", "arm", "wrist", "forearm", "elbow", "shoulder")),
    "leg": (("পায়ের", "পা", "পায়ে", "উরু", "পা'র", "टांग", "पैर", "पाँव", "leg", "foot", "ankle", "payer", "paer", "pair ki"), ("leg", "foot", "ankle", "femur", "tibia")),
    "knee": (("হাঁটুর", "হাঁটু", "घुटने", "घुटना", "knee", "hantur"), ("knee",)),
    "spine": (("পিঠের", "পিঠ", "কোমরের", "কোমর", "মেরুদণ্ডের", "মেরুদণ্ড", "पीठ", "कमर", "रीढ़", "back", "spine", "pither", "komorer"), ("spine", "spinal", "lumbar", "back")),
    "neck": (("গলার", "ঘাড়ের", "ঘাড়", "गर्दन", "गले", "neck", "ghar"), ("neck", "cervical", "thyroid")),
}

# a word that says the caller wants a PICTURE / scan of it
_PICTURE = (
    "ছবি", "ফটো", "স্ক্যান", "ছবিটা", "তুলব", "তুলতে",
    "तस्वीर", "फोटो", "इमेज", "स्कैन", "खिंचवा", "खिंचवाना",
    "chhobi", "chobi", "photo", "picture", "scan", "image", "imaging",
)
# but a question about PRICE, a doctor, a booking number etc. is a different question
_OTHER = (
    "দাম", "রেট", "কত টাকা", "ডাক্তার", "বুকিং", "কনফার্মেশন", "रेट", "दाम", "कीमत", "डॉक्टर", "price", "rate", "cost", "doctor", "booking", "confirmation",
)
# a bare catalogue test name already said ("chest x-ray", "বুকের এক্স রে") is handled by the normal test lookup, not here
_NAMES_A_TEST = ("এক্স রে", "এক্সরে", "এক্স-রে", "x-ray", "xray", "x ray", "एक्स रे", "एक्सरे")


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    return " ".join(re.sub(r"[.,;:!?\"'()।॥]+", " ", s).split())


def _has_word(text: str, words: tuple[str, ...]) -> bool:
    padded = f" {text} "
    for w in words:
        w = _norm(w)
        if w.isascii():
            if f" {w} " in padded:
                return True
        elif w in text:
            return True
    return False


def detect(text: str) -> str | None:
    """The body region a caller wants a picture of ("chest", "head", "abdomen", "hand", "leg", "knee", "spine", "neck"), or None.
    Needs BOTH a body-part word and a picture word, and no price / doctor / booking question in the same breath."""
    t = _norm(text)
    if not t or len(t.split()) > 14:
        return None
    if _has_word(t, _OTHER) or _has_word(t, _NAMES_A_TEST):
        return None
    if not _has_word(t, _PICTURE):
        return None
    hits = [(len(w), region) for region, (names, _) in REGIONS.items() for w in names if _has_word(t, (w,))]
    return max(hits)[1] if hits else None


def candidates(region: str, tests: list[dict]) -> list[str]:
    """Catalogue Imaging/Cardiac tests whose NAME mentions the region (canonical English names)."""
    keys = REGIONS[region][1]
    out = []
    for t in tests:
        if t.get("sample_type") not in ("Imaging", "Cardiac"):
            continue
        name = str(t.get("name") or "")
        low = name.lower()
        if any(re.search(rf"\b{re.escape(k)}\b", low) for k in keys):
            out.append(name)
    return out


_REGION_NOUN = {
    "bn": {"chest": "বুকের", "head": "মাথার", "abdomen": "পেটের", "hand": "হাতের", "leg": "পায়ের", "knee": "হাঁটুর", "spine": "পিঠ বা কোমরের", "neck": "গলা বা ঘাড়ের"},
    "hi": {"chest": "छाती", "head": "सिर", "abdomen": "पेट", "hand": "हाथ", "leg": "पैर", "knee": "घुटने", "spine": "पीठ या कमर", "neck": "गर्दन"},
    "en": {"chest": "chest", "head": "head", "abdomen": "abdomen", "hand": "hand", "leg": "leg", "knee": "knee", "spine": "back", "neck": "neck"},
}


def question(region: str, spoken_names: list[str], lang: str) -> str:
    """The clarifying question, from the catalogue's own test names (already in the call's language) -- or, with none, the plain 'not listed'."""
    lang = lang if lang in _REGION_NOUN else "en"
    noun = _REGION_NOUN[lang][region]
    if not spoken_names:
        return {
            "bn": f"{noun} ছবির কোনো টেস্ট আমাদের তালিকায় এখন নেই।",
            "hi": f"{noun} की तस्वीर का कोई टेस्ट अभी हमारी सूची में नहीं है।",
            "en": f"We do not have an imaging test for the {noun} listed right now.",
        }[lang]
    if len(spoken_names) == 1:
        return {
            "bn": f"{noun} ছবি বলতে কি {spoken_names[0]} বোঝাচ্ছেন?",
            "hi": f"{noun} की तस्वीर से क्या आपका मतलब {spoken_names[0]} है?",
            "en": f"By a picture of the {noun}, do you mean the {spoken_names[0]}?",
        }[lang]
    joined = {"bn": " নাকি ", "hi": " या ", "en": " or "}[lang].join(spoken_names)
    return {
        "bn": f"{noun} ছবি বলতে কোনটা বোঝাচ্ছেন -- {joined}?",
        "hi": f"{noun} की तस्वीर से आपका मतलब कौन सा है -- {joined}?",
        "en": f"By a picture of the {noun}, which do you mean -- {joined}?",
    }[lang]
