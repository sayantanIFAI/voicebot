"""Owner, 2026-10-29: "I will give the name / the day and time / the booking a little later", "I will call a little later" -- the caller is not ready to go on.

The agent thanks them, asks them to call again later, and closes the call (main.py, _clinical_and_unrelated_guards). Deterministic, no model, in Bengali (script and Roman), Hindi and English.

Two things decide it, and both are needed:
  * the caller says THEY will do it later -- "পরে নাম লেখাব", "ektu pore call korchi", "बाद में फ़ोन करूँगा", "I'll call back later";
  * what is put off is a call, or the booking's details (name, day, time, the booking itself).
A request TO the clinic ("পরে ফোন করুন", "call me later", "बाद में फ़ोन कीजिए") is the opposite and never matches. A bare "later" ("পরে", "ektu pore") never matches, and neither does a
"later" that belongs to a visit ("বিকেলে আসব", "I will come in the evening").
"""

from __future__ import annotations

import re
import unicodedata

_LATER = (
    r"(?:একটু\s*পরে|পরে|পরেই|কিছুক্ষণ\s*পরে|একটু\s*বাদে|বাদে|পরে\s*আবার|আবার\s*পরে"
    r"|बाद\s*में|बाद\s*मे|थोड़ी\s*देर\s*(?:बाद|में)|थोड़ी\s*देर\s*बाद\s*में|कुछ\s*देर\s*बाद|फिर\s*कभी|फिर\s*से\s*बाद\s*में"
    r"|ektu\s+pore|\bpore\b|\bporey\b|\bektu\s+por\b|\bbaad\s+mein\b|\bbaad\s+me\b|\blater\b|\ba\s+(?:bit|little)\s+later\b|\bin\s+a\s+(?:bit|while|few\s+minutes)\b)"
)

_CALL = r"(?:কল|ফোন|ফোনে|ফোন\s*করে|কল\s*করে|कॉल|काल|फ़ोन|फोन|\bcall\b|\bphone\b|\bring\b|\bcalling\b|\bfon\b|\bphon\b)"

# what a caller says of themself: I am calling / I will call / I will tell ... -- Bengali (script, Roman), Hindi, English
_I_WILL = (
    r"(?:করছি|করবো|করব|করছিই|করে\s*নেব|করে\s*নিচ্ছি|করি|দিচ্ছি|দেবো|দেব|বলছি|বলবো|বলব|জানাচ্ছি|জানাবো|জানাব|লেখাচ্ছি|লেখাবো|লেখাব|লিখবো|লিখব|লিখিয়ে\s*দেব|নিচ্ছি"
    r"|करता\s*हूँ|करता\s*हूं|करती\s*हूँ|करती\s*हूं|करूँगा|करूंगा|करूँगी|करूंगी|कर\s*लूँगा|कर\s*लूंगा|कर\s*लूँगी|कर\s*लूंगी|बताता\s*हूँ|बताता\s*हूं|बताती\s*हूँ|बताती\s*हूं"
    r"|बताऊँगा|बताऊंगा|बताऊँगी|बताऊंगी|दूँगा|दूंगा|दूँगी|दूंगी|देता\s*हूँ|देता\s*हूं|देती\s*हूँ|देती\s*हूं|लिखवाऊँगा|लिखवाऊंगा|लिखवाऊँगी|लिखवाऊंगी|लिखाऊँगा|लिखाऊंगा"
    r"|korchi|korbo|korchhi|kortechi|kori|korechi|dichhi|debo|bolchi|bolbo|bolchhi|janabo|janachhi|likhabo|likhchhi|lekhabo|lekhachhi|nichhi|nebo"
    r"|\bi\s*(?:will|ll|am\s+going\s+to|shall|'ll)\b|\bi'll\b|\bwill\s+(?:call|ring|phone|give|tell|say|share|book|do|send)\b|\bi\s+call\b|\blet\s+me\b)"
)

# a request TO the clinic is the opposite of this: the caller is not the one who will call
_REQUEST_TO_CLINIC = re.compile(
    r"(?:করুন|করবেন|করে\s*দিন|করে\s*দেবেন|করে\s*নেবেন|দেবেন|বলুন|বলবেন|कीजिए|कीजिये|कीजिएगा|करिए|करिये|कर\s*दीजिए|कर\s*दीजिये|कर\s*देना|कर\s*देंगे|करना|दीजिए|बताइए|बताइये"
    r"|korun|korben|diben|bolun|\bcall\s+me\b|\bring\s+me\b|\bphone\s+me\b|\byou\s+(?:call|ring|phone)\b|\bplease\s+(?:call|ring|phone)\b|\bcan\s+you\b|\bcould\s+you\b|\bwould\s+you\b)",
    re.I,
)

# what is put off: a call, or the booking's details
_THING = r"(?:নাম|সময়|দিন|তারিখ|বুকিং|অ্যাপয়েন্টমেন্ট|এপয়েন্টমেন্ট|অ্যাপয়েন্ট|নম্বর|নাম্বার|নাম\s*লেখ|नाम|समय|दिन|तारीख|बुकिंग|अपॉइंटमेंट|नंबर|नम्बर|\bnaam\b|\bnam\b|\bsomoy\b|\bsamay\b|\bdin\b|\btarikh\b|\bbooking\b|\bappointment\b|\bname\b|\btime\b|\bday\b|\bdate\b|\bnumber\b|\bdetails\b)"

_LATER_RE = re.compile(_LATER, re.I)
_CALL_RE = re.compile(_CALL, re.I)
_I_WILL_RE = re.compile(_I_WILL, re.I)
_THING_RE = re.compile(_THING, re.I)
# a visit, not a call or a detail: "বিকেলে আসব", "I will come in the evening", "कल आऊँगा"
_VISIT_RE = re.compile(r"(?:আসব|আসবো|যাব|যাবো|আসছি|आऊँगा|आऊंगा|आऊँगी|आऊंगी|आता\s*हूँ|आता\s*हूं|जाऊँगा|जाऊंगा|\bcome\b|\bvisit\b|\barrive\b|\baashbo\b|\basbo\b|\bjabo\b)", re.I)
# "I do not want to book" / "later" used as a plain answer to a different question stays out: it must name a call or a detail.


def _clean(text: str) -> str:
    return re.sub(r"[।,.!?;:\"'’]+", " ", unicodedata.normalize("NFC", text or "")).strip()


def asks_to_call_later(text: str, lang: str = "") -> bool:  # noqa: ARG001  (all three languages are checked together; the argument keeps the module's usual shape)
    """True when the caller says they will call, or will give the name / day / time / booking, a little later."""
    s = _clean(text)
    if len(s) < 4 or not _LATER_RE.search(s):
        return False
    if _REQUEST_TO_CLINIC.search(s):
        return False
    if not _I_WILL_RE.search(s):
        return False
    return bool(_CALL_RE.search(s) or _THING_RE.search(s))
