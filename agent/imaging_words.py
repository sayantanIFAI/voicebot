"""USG and "thyroid gland" as the recogniser writes them.

Real call 2026-10-19: "আপনাদের ওখানে উসগ হয়, থাইরাইড গ্লিন্ট এর মধ্যে" -- the recogniser wrote USG as "উসগ" and "thyroid gland" as "থাইরাইড গ্লিন্ট" (later "থাইল্যান্ড"). Nothing downstream
knew "উসগ" was an ultrasound, and "থাইরাইড" was taken for the blood test, so the caller was told "yes, we do the thyroid profile". The fix is at the front door: the transcript is
rewritten to the CANONICAL spelling ("ইউএসজি", "থাইরয়েড গ্ল্যান্ড") before the intent model, the test lookup or anything else reads it, so the whole pipeline sees the same word the catalogue
uses. Only whole words / whole letter-runs are rewritten, and "থাইল্যান্ড" (Thailand) only in a sentence that also mentions USG.
"""

from __future__ import annotations

import re
import unicodedata

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
    return out if out != t else text
