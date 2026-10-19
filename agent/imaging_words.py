"""USG and "thyroid gland" as the recogniser writes them.

Real call 2026-10-19: "আপনাদের ওখানে উসগ হয়, থাইরাইড গ্লিন্ট এর মধ্যে" -- the recogniser wrote USG as "উসগ" and "thyroid gland" as "থাইরাইড গ্লিন্ট" (later "থাইল্যান্ড"). Nothing downstream
knew "উসগ" was an ultrasound, and "থাইরাইড" was taken for the blood test, so the caller was told "yes, we do the thyroid profile". The fix is at the front door: the transcript is
rewritten to the CANONICAL spelling ("ইউএসজি", "থাইরয়েড গ্ল্যান্ড") before the intent model, the test lookup or anything else reads it, so the whole pipeline sees the same word the catalogue
uses. Only whole words / whole letter-runs are rewritten, and "থাইল্যান্ড" (Thailand) only in a sentence that also mentions USG.
"""

from __future__ import annotations

import re
import unicodedata

from agent import clinical_terms

_BN = r"\u0980-\u09FF"
_DEV = r"\u0900-\u097F"

# U-S-G said as a word ("উসগ"), in letters ("ইউ এস জি") or run together ("ইউএসজি"): a vowel-ish start, an S, a G/J.
_USG_BN = re.compile(rf"(?<![{_BN}])(?:ইউ|ইয়ু|ইয়ূ|উ|ঊ|ও|অ|আ)\s*(?:এস|স|শ|ছ)\s*(?:জি|জী|গ|গি|গী|ঝি|জ)(?![{_BN}])")
_USG_HI = re.compile(rf"(?<![{_DEV}])(?:यू|यु|ऊ|उ)\s*(?:एस|स)\s*(?:जी|ग|गी)(?![{_DEV}])")
_USG_LATIN = re.compile(r"(?<![A-Za-z])u\.?\s*s\.?\s*g\.?(?![A-Za-z])", re.IGNORECASE)

# "thyroid gland": the thyroid word, then a garbled "gland" (or "Thailand" for the whole phrase)
_THYROID = r"(?:থাইরাইড|থাইরয়েড|থাইরোয়েড|থাইরইড|থাইরয়ড|থাইরোইড|থাইড|থাইরাড|থাইল্যান্ড)"
_GLAND = r"(?:গ্লিন্ট|গ্লিনট|গ্লান্ড|গ্ল্যান্ড|গ্লেন্ড|গ্লেন্ট|গ্ল্যান্ট|গ্ল্যান|গ্লান|গ্লিন্ড|গ্লিন|ল্যান্ড)"
_THYROID_GLAND = re.compile(rf"(?<![{_BN}]){_THYROID}\s+{_GLAND}(?![{_BN}])")
_THAILAND = re.compile(rf"(?<![{_BN}])থাইল্যান্ড(?![{_BN}])")
_THYROID_HI_GLAND = re.compile(rf"(?<![{_DEV}])(?:थायराइड|थायरॉइड|थायरोइड)\s+(?:ग्लैंड|ग्लांड|ग्लिंट|ग्लैन्ड)(?![{_DEV}])")

USG_BN = "ইউএসজি"
USG_HI = "यूएसजी"

# CBC said fast and written as a made-up word. Real call 2026-10-19: "ক্ব আর ফুল আপডোমেন ইউএসজি" -- "ক্ব" is CBC. Only words that have no other meaning, and only in a sentence about a test.
_CBC_GARBLES = frozenset(
    "ক্ব ক্বক ক্বকি ক্বসি ক্বিসি ক্বক্ব সিবিশি সিভিসি চিবিসি শিবিসি সীবীসী সিবিসী সিবিচি সিবিক সিপিসি क्वक क्व सीबीसी सिबिसी सिबीसी".split()
)
_TEST_CUE = ("টেস্ট", "টাকা", "রেট", "দাম", "কত", "করতে", "করাতে", "করব", "করবো", "লাগবে", "হয়", "বুক", "टेस्ट", "रेट", "दाम", "कितना", "test", "rate", "price", "cost", "book")
_SCAN_CUE = (USG_BN, USG_HI, "usg", "আলট্রাসাউন্ড", "আল্ট্রাসাউন্ড", "সোনোগ্রাফি", "अल्ट्रासाउंड", "ultrasound")
_WHOLE_WORD = re.compile(rf"(?<![{_BN}])(?:ফুল|হুল|ফুলা|হোল|হোলে|হোয়াল|ফল)(?![{_BN}])")


def _garbled_abdomen(out: str) -> str:
    """"ফুল আপডোমেন ইউএসজি" (full abdomen USG): in a sentence about a scan, a word that SOUNDS like "abdomen" is the abdomen, and "full" before it is "whole". The clinic's own
    test is "হোল অ্যাবডোমেন ইউএসজি", so the rewritten words are the ones its alias list holds. Only a sound match in the scan context (clinical_terms) -- a near match suggests, it never resolves."""
    low = out.lower()
    if not any(c in low for c in _SCAN_CUE):
        return out
    words = out.split()
    changed = False
    for i, w in enumerate(words):
        core = w.strip(".,;:!?।")
        if len(core) >= 5 and not core.isascii() and (core == "অ্যাবডোমেন" or clinical_terms._best("abdomen", core) >= 0.7):  # a scan was just named, so the bar is lower than anywhere else
            words[i] = w.replace(core, "অ্যাবডোমেন")
            changed = True
            if i > 0 and _WHOLE_WORD.fullmatch(words[i - 1].strip(".,;:!?।")):
                words[i - 1] = "হোল"
    return " ".join(words) if changed else out


def normalise(text: str) -> str:
    """The transcript with USG and "thyroid gland" in their canonical spelling (see the module docstring). Returns the text unchanged when there is nothing to rewrite."""
    if not text:
        return text
    t = unicodedata.normalize("NFC", text)
    out = _USG_BN.sub(USG_BN, t)
    out = _USG_HI.sub(USG_HI, out)
    out = _THYROID_GLAND.sub("থাইরয়েড গ্ল্যান্ড", out)
    out = _THYROID_HI_GLAND.sub("थायरॉइड ग्लैंड", out)
    if USG_BN in out or USG_HI in out or _USG_LATIN.search(out):
        out = _THAILAND.sub("থাইরয়েড গ্ল্যান্ড", out)
    out = _garbled_abdomen(out)
    if any(c in out.lower() for c in _TEST_CUE):
        parts = out.split()
        for i, w in enumerate(parts):
            core = w.strip(".,;:!?।")
            if core in _CBC_GARBLES:
                parts[i] = w.replace(core, "CBC")
        out = " ".join(parts)
    return out if out != t else text
