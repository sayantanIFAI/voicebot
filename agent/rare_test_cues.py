"""A rare test the clinic's table does not hold -- "USG of the thyroid gland", "TB test in the ear" -- goes to a person (owner, 2026-10-23, item 7).

The clinic's lab table (clinic-api/seed.py LAB_TESTS) holds 34 tests: the common blood and urine tests, ECG, chest X-ray, USG whole abdomen, USG pregnancy profile,
2D echo, TMT, pap smear and PSA. A scan or test of ANY OTHER body part, or a TB test, is not in it -- yet a doctor may well have advised it "at Sonoscan". The agent must
neither price it nor refuse it (a guessed answer about a test is a wrong fact in the caller's ear): a human who can look at the prescription takes the call.

Decided from the words alone, in English, Hindi and Bengali (their own scripts or Latin letters, recogniser spellings included):
  * an imaging word (USG, ultrasound, sonography, scan, CT, MRI, X-ray, doppler) within a few words of a body part the table has no scan for (thyroid, neck, ear, eye,
    nose, throat, knee, shoulder, spine, breast, joint ...);
  * a test word beside the ear / nose / throat / eye;
  * a TB (tuberculosis) test.
NOT claimed here: the head and brain (agent/lay_terms.py already answers those with the prescription-on-WhatsApp reply), the chest, the abdomen, the pregnancy scan, a
"thyroid profile" (a listed blood test -- a thyroid is only a body part when an imaging word sits beside it).
"""

from __future__ import annotations

import re

_SPLIT = re.compile(r"[^\s,.;:!?।॥\"()\[\]{}]+")
_SPELLED = (
    (re.compile(r"\bu\s*\.?\s*s\s*\.?\s*g\b"), "usg"),
    (re.compile(r"\bx\s*[-.]?\s*ray\b"), "xray"),
    (re.compile(r"\bct\s*[-.]?\s*scan\b"), "ct scan"),
    (re.compile(r"\bultra\s+sound\b|\bultra\s+sonography\b"), "ultrasound"),
    (re.compile(r"এক্স\s*-?\s*রে"), "এক্সরে"),
    (re.compile(r"एक्स\s*-?\s*रे"), "एक्सरे"),
)

_IMAGING = (
    "usg", "ultrasound", "ultrasonography", "sonography", "sonogram", "scan", "ct", "mri", "xray", "doppler", "ultrasonic",
    "ইউএসজি", "আলট্রা", "আল্ট্রা", "সনোগ্রাফি", "সোনোগ্রাফি", "স্ক্যান", "স্কান", "সিটি", "এমআরআই", "এক্সরে", "ডপলার",
    "यूएसजी", "अल्ट्रा", "अल्ट्रासाउंड", "सोनोग्राफी", "स्कैन", "सीटी", "एमआरआई", "एक्सरे", "डॉपलर",
)  # fmt: skip
_TESTWORDS = (
    "test", "tests", "culture", "swab", "biopsy", "check", "checkup", "examination", "exam",
    "পরীক্ষা", "টেস্ট", "কালচার", "বায়োপসি", "চেকআপ",
    "जांच", "जाँच", "टेस्ट", "कल्चर", "बायोप्सी", "चेकअप", "परीक्षण",
)  # fmt: skip
_TB = ("tb", "tuberculosis", "mantoux", "টিবি", "যক্ষ্মা", "टीबी", "क्षय", "तपेदिक")
# Body parts the clinic has no scan or test for. Latin stems are matched at the start of a word (kane, kaner, kaan); the English ones whole.
_PARTS_LATIN_STEMS = ("kan", "kaan", "nak", "gola", "golar", "chokh", "chok", "ghar", "hatu", "kandh", "kadh", "stan", "gardan", "gala", "aankh", "ankh", "thyroid", "thairoid", "tairoid", "thayroid", "thyrid")
_PARTS_LATIN_WHOLE = ("neck", "ear", "ears", "eye", "eyes", "nose", "throat", "knee", "shoulder", "spine", "breast", "joint", "joints", "sinus", "sinuses", "elbow", "wrist", "ankle", "hip", "salivary", "carotid", "scrotum", "testis", "tonsil")
_PARTS_SCRIPT_STEMS = (
    "থাইরয়েড", "থাইরোয়েড", "থাইরাইড", "গলা", "গলার", "ঘাড়", "কান", "চোখ", "নাক", "হাঁটু", "কাঁধ", "মেরুদণ্ড", "স্তন", "জয়েন্ট", "কনুই", "কব্জি", "গোড়ালি", "টনসিল",
    "थायराइड", "थायरॉइड", "थाइरॉयड", "गला", "गले", "गर्दन", "कान", "आँख", "आंख", "नाक", "घुटन", "कंधा", "कंधे", "रीढ़", "स्तन", "जोड़", "कोहनी", "टॉन्सिल",
)  # fmt: skip
# The ear / nose / throat / eye: a plain test beside one of these is not in the table either
_ENT_LATIN_STEMS = ("kan", "kaan", "nak", "gola", "golar", "chokh", "chok", "aankh", "ankh", "gala")
_ENT_LATIN_WHOLE = ("ear", "ears", "eye", "eyes", "nose", "throat", "tonsil")
_ENT_SCRIPT_STEMS = ("গলা", "গলার", "কান", "চোখ", "নাক", "টনসিল", "गला", "गले", "कान", "आँख", "आंख", "नाक", "टॉन्सिल")


_THYROID = ("thyroid", "thairoid", "tairoid", "thayroid", "thyrid", "থাইরয়েড", "থাইরোয়েড", "থাইরাইড", "थायराइड", "थायरॉइड", "थाइरॉयड")
_LAB_AFTER = ("profile", "test", "function", "level", "tsh", "t3", "t4", "panel", "hormone", "প্রোফাইল", "টেস্ট", "ফাংশন", "লেভেল", "হরমোন", "प्रोफाइल", "टेस्ट", "फंक्शन", "लेवल", "हार्मोन", "जांच", "जाँच", "পরীক্ষা")


def _is_thyroid(tok: str) -> bool:
    return tok.startswith(_THYROID)


def _tokens(text: str) -> list[str]:
    low = " ".join((text or "").lower().split())
    for pattern, repl in _SPELLED:
        low = pattern.sub(repl, low)
    return _SPLIT.findall(low)


def _is_part(tok: str, whole: tuple[str, ...], latin_stems: tuple[str, ...], script_stems: tuple[str, ...]) -> bool:
    if tok in whole:
        return True
    if tok.isascii():
        return len(tok) >= 3 and any(tok.startswith(s) for s in latin_stems)
    return any(tok.startswith(s) for s in script_stems)


def _near(tokens: list[str], left, right, window: int) -> bool:
    ls = [i for i, t in enumerate(tokens) if left(t)]
    rs = [i for i, t in enumerate(tokens) if right(t)]
    return any(i != j and abs(i - j) <= window for i in ls for j in rs)


def asks_for_unlisted_test(text: str, lang: str = "en") -> bool:
    """True when the sentence asks for a scan / test the clinic's table does not hold (see the module docstring)."""
    tokens = _tokens(text)
    if len(tokens) < 2:
        return False

    def imaging(t: str) -> bool:
        return t in _IMAGING or (not t.isascii() and any(t.startswith(s) for s in _IMAGING if not s.isascii()))

    def part(t: str) -> bool:
        return _is_part(t, _PARTS_LATIN_WHOLE, _PARTS_LATIN_STEMS, _PARTS_SCRIPT_STEMS)

    # "thyroid profile / test / TSH" is the listed blood test, not a scan of the gland: such a thyroid is no body part here
    for i, t in enumerate(tokens):
        if _is_thyroid(t) and i + 1 < len(tokens) and tokens[i + 1].startswith(_LAB_AFTER):
            tokens[i] = "-"

    def ent(t: str) -> bool:
        return _is_part(t, _ENT_LATIN_WHOLE, _ENT_LATIN_STEMS, _ENT_SCRIPT_STEMS)

    def testword(t: str) -> bool:
        return t in _TESTWORDS or (not t.isascii() and any(t.startswith(s) for s in _TESTWORDS if not s.isascii()))

    def tb(t: str) -> bool:
        return t in _TB or (not t.isascii() and any(t.startswith(s) for s in _TB if not s.isascii()))

    if _near(tokens, imaging, part, 4):
        return True
    if _near(tokens, testword, ent, 3) or _near(tokens, imaging, ent, 4):
        return True
    return _near(tokens, tb, testword, 3)
