"""A caller who already HAS a booking and wants it moved: "আমার একটা অ্যাপয়েন্টমেন্ট আছে দু তারিখে, ওটা একদিন আগানো যাবে", "I can't come that day", "appointment ta ektu pichiye din", "मेरी अपॉइंटमेंट एक दिन पहले हो सकती है"
(owner, 2026-10-31: the call was answered "which doctor would you like an appointment with?" -- the move-it flow was never entered).

Recognised by its PARTS, in every language, not by a list of sentences:
  * a BOOKING word (appointment in any spelling a phone recogniser gives it, booking, serial, "my name was written") or an EXISTING-booking cue ("আছে", "already booked", "बुक है");
  * a DIRECTION: a strong one (bring forward / push back / change -- the verb says it) or a weak one (আগে, পরে, earlier, later: also the words of a NEW booking, so they count only beside an existing-booking cue);
  * or "I cannot come (that day)", which with a booking word is the same request.
It also reads HOW FAR: "একদিন আগে" = one day earlier, "এক সপ্তাহ পরে" = a week later, so the new date can be worked out from the booking that is found (shift_days).
Pure text in, answer out; no model.
"""

from __future__ import annotations

import re

# an appointment / booking word, however it is written or misheard ("অযাপ্ফোন্টমেন্ট", "এপয়েন্টমেন্ট", "ন্টমেন্ট")
_BOOKING = re.compile(
    r"ন্টমেন্ট|ন্টমেণ্ট|অ্যাপ|অ্যা\s*প|অযাপ|এপয়|আপয়|appoint|apoint|booking|bukking|बुकिंग|अपॉइंट|अपाइंट|एपॉइंट|अपोइंट|अप्वाइंट|বুকিং|বুক করা|বুক আছে|সিরিয়াল|serial|\bbook(?:ed)?\b|"
    r"নাম লেখানো|নাম লিখানো|নাম লিখিয়ে|নাম লেখা|নাম লিখে",
    re.I,
)
# the booking is already there
_EXISTING = re.compile(
    r"আছে|করা আছে|করা হয়েছে|করেছি|করে রেখেছি|করে নিয়েছি|হয়ে আছে|আগে থেকে|already|have an|have a|i have|booked|मेरी|मेरा|है\b|हुई है|कर रखी|करवा रखी|করা ছিল|করা আছে|বুক ছিল",
    re.I,
)
# STRONG directions: the verb itself says "move it"
_FORWARD_STRONG = re.compile(r"\bagan[oa]\w*|\begiye\b|\baagiye\b|\bagiye\b|\b(?:aage|agey)\s+(?:kor|ni|an)\w*|আগানো|আগাতে|আগিয়ে|আগাইয়া|আগাব|এগিয়ে|এগোনো|এগানো|এগোতে|আগে\s*(?:করা|নেওয়া|নেয়া|আনা|আনতে|করে|করতে|নিয়ে)|prepone|pre-pone|bring (?:it )?forward|move (?:it )?(?:up|earlier|forward)|advance|आगे\s*(?:करना|कर)|पहले\s*(?:करना|कर|ले)|जल्दी\s*(?:करना|कर|ले)", re.I)
_BACK_STRONG = re.compile(r"\bpich(?:i|a)\w*|\bpechi\w*|\bderi\s+kor\w*|পিছানো|পিছিয়ে|পিছাতে|পেছানো|পেছাতে|পেছিয়ে|পিছিয়ে|দেরি\s*(?:করে|করা)|পোস্টপোন|postpone|push (?:it )?(?:back|later|out)|delay|put off|पीछे|आगे बढ़ा|आगे बढा|टाल|पोस्टपोन", re.I)
_CHANGE_STRONG = re.compile(
    r"\bbod?l(?:e|a)\w*|\banyo\s*din\b|\barekdin\b|\bare?k\s*din\b|reschedule|রিশিডিউল|রিসিডিউল|রিসেডিউল|change|বদল|বদলা|পরিবর্তন|সরিয়ে|সরাতে|সরানো|অন্য\s*দিন|অন্যদিন|আরেক\s*দিন|আরেকদিন|অন্য\s*তারিখ|shift|move|रीशेड्यूल|बदल|दूसरे\s*दिन|दूसरी\s*तारीख|खिसक|सरका",
    re.I,
)
# WEAK directions (also the words of a new booking: "পরের সোমবার"): only beside an existing-booking cue
_FORWARD_WEAK = re.compile(r"\baage\b|\bagey\b|আগে|আগের|আগেই|earlier|before|पहले", re.I)
_BACK_WEAK = re.compile(r"\bpore\b|\bporey\b|পরে|পরের|পরেই|later|after|बाद|अगले", re.I)
# "I cannot come (that day)"
_CANNOT_COME = re.compile(
    r"(?:আসতে|যেতে|আসা|যাওয়া|আসার|যাবার)\s*(?:তো\s*)?(?:পারব|পারবো|পারবনা|পারব না|পারবো না|পারছি না|পারি না|পারি নাই|পারব নাহ|হবে না|হবেনা|হচ্ছে না|সম্ভব হবে না|হয়ে উঠবে না)|"
    r"(?:আসতে|যেতে)\s*পারবো?\s*না|can'?t (?:come|make it|go)|cannot (?:come|make it)|won'?t be able to (?:come|make)|unable to come|not able to come|"
    r"आ\s*नहीं\s*(?:सकता|सकती|पाऊँगा|पाऊंगा|पाऊँगी|पाऊंगी|पा रहा|पा रही)|नहीं\s*आ\s*(?:सकता|सकती|पाऊँगा|पाऊंगा|पाऊँगी|पाऊंगी)",
    re.I,
)

_NUM_WORD = {
    "এক": 1, "একটা": 1, "একদিন": 1, "দুই": 2, "দু": 2, "দুটো": 2, "দুদিন": 2, "তিন": 3, "তিনদিন": 3, "চার": 4, "চারদিন": 4, "পাঁচ": 5, "পাঁচদিন": 5, "ছয়": 6, "সাত": 7, "আট": 8, "দশ": 10,
    "ek": 1, "ekta": 1, "ekdin": 1, "dui": 2, "du": 2, "duin": 2, "tin": 3, "char": 4, "panch": 5, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5,
}
_DAY_UNIT = r"(?:দিন|din|day|days|दिन)"
_WEEK_UNIT = r"(?:সপ্তাহ|হপ্তা|সপ্তা|week|weeks|हफ्ता|हफ़्ता|हफ्ते|सप्ताह)"
_RE_SHIFT = re.compile(rf"(\d+|[^\s\d]+)\s*({_DAY_UNIT}|{_WEEK_UNIT})", re.I)


def says_booking(text: str) -> bool:
    return bool(_BOOKING.search(text or ""))


def says_cannot_come(text: str) -> bool:
    return bool(_CANNOT_COME.search(text or ""))


def direction(text: str) -> str | None:
    """"earlier" (bring forward), "later" (push back), "change" (another day) or None -- by the strong words only."""
    t = text or ""
    if _FORWARD_STRONG.search(t):
        return "earlier"
    if _BACK_STRONG.search(t):
        return "later"
    if _CHANGE_STRONG.search(t):
        return "change"
    return None


def asks_reschedule(text: str) -> bool:
    """True when the sentence asks to MOVE a booking the caller already has (or says they cannot come to it)."""
    t = (text or "").strip()
    if not t or len(t.split()) > 40:
        return False
    booking = says_booking(t)
    existing = bool(_EXISTING.search(t))
    strong = direction(t)
    if strong and booking:
        return True
    if strong and existing and re.search(r"(?:^|\s)(?:ওটা|ওটাকে|এটা|এটাকে|সেটা|সেটাকে|it|that|उसे|इसे|वो|वह)(?:\s|$)", t, re.I):
        return True  # "move it" -- the booking is "it"
    if says_cannot_come(t) and booking:
        return True
    if booking and existing and (_FORWARD_WEAK.search(t) or _BACK_WEAK.search(t)) and re.search(r"যাবে|যায়|যাবে কি|করা যাবে|হবে কি|সম্ভব|can|could|possible|सकता|सकती|हो सकत|हो पाएगा", t, re.I):
        return True  # "আছে ... একদিন আগে করা যাবে?" -- a weak direction with a question about whether it can be done
    return False


def shift_days(text: str) -> int | None:
    """How many days earlier (negative) or later (positive) the caller wants it, when they say how far: "একদিন আগে" -> -1, "দুদিন পরে" -> +2, "এক সপ্তাহ পরে" -> +7. None when no distance is
    said or there is no direction. "আগে" / "পরে" and the like give the sign; the strong verbs ("আগানো", "পিছিয়ে") give it too."""
    t = text or ""
    sign = 0
    if _FORWARD_STRONG.search(t) or (_FORWARD_WEAK.search(t) and not _BACK_STRONG.search(t) and not _BACK_WEAK.search(t)):
        sign = -1
    elif _BACK_STRONG.search(t) or _BACK_WEAK.search(t):
        sign = 1
    if not sign:
        return None
    for m in _RE_SHIFT.finditer(t.replace("একদিন", "এক দিন").replace("দুদিন", "দুই দিন").replace("তিনদিন", "তিন দিন").replace("ekdin", "ek din")):
        n_raw, unit = m.group(1), m.group(2)
        n = int(n_raw) if n_raw.isdigit() else _NUM_WORD.get(n_raw.lower())
        if not n:
            continue
        days = n * (7 if re.fullmatch(_WEEK_UNIT, unit, re.I) else 1)
        return sign * days
    return None
