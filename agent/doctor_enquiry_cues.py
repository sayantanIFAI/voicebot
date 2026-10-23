"""Three questions about doctors that are answered by rule, from the clinic's own data, never by the model (owner, 2026-10-23):

  * "valo doctor ke ache?" / "which doctor is good?"      -- no doctor is ever recommended: ask the problem or department, and name who sits in
                                                              General Medicine on the day (good_doctor_reply).
  * "kon kon din boshen?" / "which days does he sit?"      -- the weekdays, not dates: "সোম বুধ শুক্র বারে বসেন" (weekdays_reply).
  * "aj doctor na pele next kobe?" / "if not today, when?" -- the next sitting date AND the other doctors of the department who are available
                                                              (next_if_absent_reply).

This module only RECOGNISES the question and phrases the answer from facts it is handed; the facts (who sits, which weekdays, which date) come from the
clinic API in the orchestrator, so a number or a name is never composed here. Whole-sentence regexes, deterministic, in Bengali, Hindi and English,
written in their own scripts or in Latin letters. Phrase lists are provisional and were written by a non-native speaker (agent/persona.py
REVIEW_STATUS): expect a native reviewer to add more, same caveat agent/human_request.py carries.
"""

from __future__ import annotations

import re
import unicodedata

from agent import slot_proposal

_I = re.I

# ---- "which doctor is good?" -- a recommendation is asked for; none is ever given -------------------------------------------------------------------------
_GOOD_DOCTOR = [
    re.compile(r"\b(good|best|better|nice|top|famous|experienced|expert)\s+(doctors?|dr|specialists?)\b", _I),
    re.compile(r"\bwhich\s+(doctor|dr)\b.{0,15}\b(good|best|better)\b", _I),
    re.compile(r"\b(who|which)\b.{0,15}\b(best|good)\b.{0,15}\b(doctor|dr)\b", _I),
    re.compile(r"\b(recommend|suggest)\b.{0,20}\b(doctor|dr)\b|\b(doctor|dr)\b.{0,15}\b(recommend|suggest)", _I),
    re.compile(r"\b(valo|bhalo|bhal|bhalo|sera|best|accha|acha|achha|achcha)\s+(doctor|daktar|dagtar|dactar|dr)\b", _I),
    re.compile(r"\b(doctor|daktar|dagtar|dactar)\s+(valo|bhalo|sera|accha|acha|achha)\b", _I),
    re.compile(r"(ভালো|ভাল|সেরা|বেস্ট|নামকরা|অভিজ্ঞ)\s*(একজন\s*)?(ডাক্তার|ডাক্তর|ডক্টর)"),
    re.compile(r"(ডাক্তার|ডক্টর)\S*\s*.{0,12}(ভালো|ভাল|সেরা|বেস্ট)"),
    re.compile(r"(ডাক্তার|ডক্টর)\s*(সাজেস্ট|রেকমেন্ড|রিকমেন্ড)"),
    re.compile(r"(अच्छा|अच्छे|अच्छी|बढ़िया|बेहतरीन|बेस्ट|सबसे अच्छा|सबसे अच्छे)\s*(डॉक्टर|डाक्टर|डॉ)"),
    re.compile(r"(डॉक्टर|डाक्टर)\S*\s*.{0,12}(अच्छा|अच्छे|बेस्ट|बढ़िया)"),
    re.compile(r"(डॉक्टर|डाक्टर)\s*(सजेस्ट|रिकमेंड|सुझाव)"),
]

# ---- "which days does he sit?" -- the weekly pattern, plural: a single-day question ("which day will he sit") keeps its dated answer ---------------------
_WEEKDAYS = [
    re.compile(
        r"\b(which|what)\s+days\b.{0,30}\b(sit|sits|come|comes|available|visit|visits|attend|attends|chamber|consult|consults|in)\b",
        _I,
    ),
    re.compile(r"\b(weekly|week)\s+(schedule|timing|timings|days)\b", _I),
    re.compile(r"\b(sits|sit|comes|come|available|visits)\b.{0,15}\bwhich\s+days\b", _I),
    re.compile(
        r"\bkon\s*kon\s*(din|bar|dine|bare)\b|\bkon\s*(din|bar)\s*(din\s*)?(boshen|bose|asen|ashen|thaken)\b.{0,10}\bkon\s*(din|bar)\b",
        _I,
    ),
    re.compile(r"কোন\s*কোন\s*(দিন|বার|দিনে|বারে)"),
    re.compile(r"কোন\s*(কোন\s*)?(দিন|বার)\S*\s*.{0,12}(বসেন|বসে|আসেন|থাকেন)"),
    re.compile(r"সপ্তাহে\s*(কবে\s*কবে|কোন\s*কোন)"),
    re.compile(r"कौन\s*(कौन\s*)?से\s*(दिन|वार)\s*(को\s*)?(बैठते|बैठती|आते|आती|मिलते|उपलब्ध)?"),
    re.compile(r"किस\s*किस\s*(दिन|वार)|किन\s*दिनों|हफ्ते\s*में\s*कब\s*कब|सप्ताह\s*में\s*कब\s*कब"),
    re.compile(r"\bkaun\s*(kaun\s*)?se\s*din\b|\bkis\s*kis\s*din\b", _I),
]

# ---- "if the doctor is not there today, when next?" --------------------------------------------------------------------------------------------------
_NEXT_IF_ABSENT = [
    re.compile(
        r"\b(if|in case)\b.{0,30}\b(not|n'?t|no)\b.{0,25}\b(today|available|there|sitting|in)\b.{0,30}\b(next|when|which day|then)\b",
        _I,
    ),
    re.compile(
        r"\b(not|n'?t)\s+(available|there|in|sitting|get|find|able to see)\b.{0,25}\btoday\b.{0,30}\b(next|when|then|which day)\b",
        _I,
    ),
    re.compile(r"\btoday\b.{0,25}\b(not|n'?t)\b.{0,25}\b(next|when|which day)\b", _I),
    re.compile(r"\bif not today\b|\bnot today\b.{0,15}\b(then )?(when|next)\b", _I),
    re.compile(
        r"\b(aj|aji|ajke|aaj)\b.{0,30}\bna\s*(pele|thakle|boshle|hole|pai|paile)\b.{0,25}\b(next|kobe|pore|porer|abar|ar)\b",
        _I,
    ),
    re.compile(
        r"\bna\s*(pele|thakle|boshle|hole|paile)\b.{0,15}\b(next|kobe|porer|pore)\b.{0,25}\b(aj|aji|ajke|aaj)\b|\b(aj|aji|ajke|aaj)\b.{0,25}\b(doctor|dagtar|daktar|dactar)\b.{0,20}\bna\s*(pele|thakle|boshle|hole|paile)\b",
        _I,
    ),
    re.compile(
        r"আজ(কে)?\S*\s*.{0,30}(না\s*পেলে|না\s*থাকলে|না\s*বসলে|পাওয়া\s*না\s*গেলে|না\s*হলে).{0,25}(কবে|পরে|পরের|আবার|নেক্সট)"
    ),
    re.compile(r"আজ(কে)?\s*না\s*(পেলে|হলে).{0,20}(কবে|পরের|পরে)"),
    re.compile(
        r"आज\S*\s*.{0,30}(नहीं\s*मिल|नहीं\s*हो|न\s*मिल|नहीं\s*बैठ|नही\s*मिल|नही\s*बैठ).{0,25}(कब|अगला|अगली|फिर|नेक्स्ट)"
    ),
    re.compile(r"आज\s*नहीं\s*तो.{0,12}(कब|अगला|अगली|फिर)|अगर\s*आज.{0,25}नहीं.{0,20}(कब|अगला|अगली)"),
    re.compile(
        r"\baaj\b.{0,25}\bnahi\b.{0,20}\b(mile|milenge|baithte|baithenge|hai|ho)\b.{0,20}\b(kab|next|agla|agli)\b", _I
    ),
]

_PATTERNS = {"good_doctor": _GOOD_DOCTOR, "weekdays": _WEEKDAYS, "next_if_absent": _NEXT_IF_ABSENT}


def _nfc(t: str) -> str:
    return unicodedata.normalize("NFC", t or "").strip()


def _any(kind: str, text: str) -> bool:
    t = _nfc(text)
    return bool(t) and any(p.search(t) for p in _PATTERNS[kind])


def asks_good_doctor(text: str, lang: str = "en") -> bool:
    """ "valo doctor ke ache" / "best doctor" / "अच्छा डॉक्टर कौन है" -- a recommendation is asked for. All three languages are always checked."""
    return _any("good_doctor", text)


def asks_doctor_weekdays(text: str, lang: str = "en") -> bool:
    """ "kon kon din boshen" / "which days does he sit" / "कौन से दिन बैठते हैं" -- the weekly pattern (plural), not one date."""
    return _any("weekdays", text)


def asks_next_if_absent(text: str, lang: str = "en") -> bool:
    """ "aj doctor na pele next kobe" / "if not today, when next" / "आज नहीं मिले तो अगला कब"."""
    return _any("next_if_absent", text)


# ------------------------------------------------------------------------------------------------------------------------------------ the answers
WEEKDAY_NAMES = {
    # Bengali: the short form every Bengali speaker uses before "বার" ("সোম বুধ শুক্র বারে বসেন")
    "bn": ("সোম", "মঙ্গল", "বুধ", "বৃহস্পতি", "শুক্র", "শনি", "রবি"),
    "hi": ("सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"),
    "en": ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
}


def _join(names: list[str], lang: str) -> str:
    if lang == "bn":
        return " ".join(names)
    word = {"hi": "और", "en": "and"}.get(lang, "and")
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} {word} {names[-1]}"


def weekdays_reply(doctor: str, weekdays: list[int], hours: str | None, lang: str) -> str:
    """The doctor's sitting WEEKDAYS (0 = Monday), never dates: "ডাক্তার X সোম বুধ শুক্র বারে বসেন।" The hours are added only when every day has the same hours."""
    lang = lang if lang in WEEKDAY_NAMES else "en"
    days = sorted(set(weekdays))
    if not days:
        return {
            "bn": f"{doctor}-এর নির্দিষ্ট কোনো বসার দিন এখন আমার কাছে নেই। কাউন্টারে খোঁজ নিতে পারেন।",
            "hi": f"{doctor} के बैठने के दिन अभी मेरे पास नहीं हैं। काउंटर पर पूछ सकते हैं।",
            "en": f"I do not have fixed sitting days for {doctor} right now. You can ask at the counter.",
        }[lang]
    names = _join([WEEKDAY_NAMES[lang][d] for d in days], lang)
    if lang == "bn":
        out = f"{doctor} {names} বারে বসেন।"
        return out + (f" সময় {hours}।" if hours else "")
    if lang == "hi":
        out = f"{doctor} {names} को बैठते हैं।"
        return out + (f" समय {hours}।" if hours else "")
    return f"{doctor} sits on {names}." + (f" Hours {hours}." if hours else "")


_PROBLEM_ASK = {
    "bn": "কোন ডাক্তার ভালো সেটা আমি বলতে পারি না। আপনার কী সমস্যা বা কোন বিভাগে দেখাতে চান, বললে সেই বিভাগের ডাক্তার বলে দিতে পারব।",
    "hi": "कौन सा डॉक्टर अच्छा है, यह मैं नहीं बता सकती। आपको क्या समस्या है या किस विभाग में दिखाना है, बताइए तो उस विभाग के डॉक्टर बता दूँगी।",
    "en": "I cannot say which doctor is better. Tell me your problem or the department, and I will tell you the doctors of that department.",
}
_GM_TODAY = {
    "bn": " {when} জেনারেল মেডিসিনে আছেন: {names}।",
    "hi": " {when} जनरल मेडिसिन में हैं: {names}।",
    "en": " In General Medicine {when}: {names}.",
}


def general_medicine_line(doctors: list[str], date_iso: str | None, lang: str) -> str:
    """ " In General Medicine today: A, B." -- only the clinic's own list of who sits; empty when nobody does."""
    lang = lang if lang in _GM_TODAY else "en"
    if not doctors:
        return ""
    when = slot_proposal.when_phrase(date_iso, lang) if date_iso else {"bn": "আজ", "hi": "आज", "en": "today"}[lang]
    return _GM_TODAY[lang].format(when=when, names=", ".join(doctors[:5]))


def good_doctor_reply(doctors: list[str], date_iso: str | None, lang: str) -> str:
    """No doctor is recommended. The caller is asked for the problem or department, and -- only from the clinic's own list -- told who is available in
    General Medicine on the day (today unless a date is given)."""
    lang = lang if lang in _PROBLEM_ASK else "en"
    return _PROBLEM_ASK[lang] + general_medicine_line(doctors, date_iso, lang)


_NEXT = {
    "bn": "{doctor} {when} বসবেন।",
    "hi": "{doctor} {when} बैठेंगे।",
    "en": "{doctor} sits {when}.",
}
_NO_NEXT = {
    "bn": "{doctor}-এর পরের কোনো বসার দিন এখন পাচ্ছি না। কাউন্টারে খোঁজ নিতে পারেন।",
    "hi": "{doctor} का अगला बैठने का दिन अभी नहीं मिल रहा। काउंटर पर पूछ सकते हैं।",
    "en": "I cannot find {doctor}'s next sitting day right now. You can ask at the counter.",
}
_OTHERS = {
    "bn": " আজ একই বিভাগে আছেন: {names}। কাকে দেখাতে চান?",
    "hi": " आज इसी विभाग में हैं: {names}। किसे दिखाना चाहेंगे?",
    "en": " Today in the same department: {names}. Which one would you like to see?",
}


def next_if_absent_reply(doctor: str, next_date: str | None, others: list[str], lang: str) -> str:
    """The doctor's next sitting DATE, plus the other doctors of the same department who are available today -- both from the clinic's data."""
    lang = lang if lang in _NEXT else "en"
    if next_date:
        out = _NEXT[lang].format(doctor=doctor, when=slot_proposal.when_phrase(next_date, lang))
    else:
        out = _NO_NEXT[lang].format(doctor=doctor)
    if others:
        out += _OTHERS[lang].format(names=", ".join(others[:4]))
    return out


# ---- "when should I come?" (owner, 2026-10-23, item 24) ---------------------------------------------------------------------------------------------------
# Answered with the doctor's sitting START time and "keep one to two hours in hand" (slot_proposal.conclusion) -- the sentence the booking already ends with.
# A question that carries a token / serial NUMBER ("when should I come, my token is 20") keeps its more exact answer (owner_answers.token_arrival_reply).
_WHEN_TO_COME = [
    re.compile(
        r"\bwhen\s+(?:should|do|shall|can|must|will|to)\s+(?:i|we)?\s*(?:come|go|reach|arrive|visit|be there)\b", _I
    ),
    re.compile(
        r"\bwhen\s+(?:do|should|shall|can)\s+(?:i|we)\s+(?:need to\s+|have to\s+|want to\s+)?(?:go|come|reach|arrive)\b",
        _I,
    ),
    re.compile(r"\bwhat\s+time\b.{0,25}\b(?:come|reach|arrive|go|visit|be there)\b", _I),
    re.compile(r"\bwhen\b.{0,12}\b(?:i|we)\b.{0,10}\b(?:go|come)\b.{0,20}\b(?:see|doctor|him|her|clinic)\b", _I),
    re.compile(
        r"\bkokhon\s*(?:ami\s*)?(?:asbo|ashbo|asben|ashben|ashte|aste|jabo|jete|jaabo|pouchabo|pouchbo|pouchte|aschi)\b",
        _I,
    ),
    re.compile(
        r"\bkotay\s*(?:asbo|ashbo|ashte|aste|jabo|jete)\b|\bkotar\s*(?:shomoy|somoy)?\s*(?:asbo|ashbo|ashte|aste|jabo)\b",
        _I,
    ),
    re.compile(r"\bkon\s*(?:shomoy|somoy)\s*(?:asbo|ashbo|jabo|ashte|aste)\b", _I),
    re.compile(r"কখন\s*(?:আসব|আসবো|আসতে|যাব|যাবো|যেতে|পৌঁছ|পৌছ|এসে|গিয়ে)"),
    re.compile(r"কটা(?:র)?\s*(?:সময়|মধ্যে)?\s*(?:আসব|আসবো|আসতে|যাব|যাবো|যেতে|পৌঁছ|পৌছ)"),
    re.compile(r"কটায়\s*(?:আসব|আসবো|আসতে|যাব|যাবো|যেতে|যাওয়া|পৌঁছ|পৌছ)"),
    re.compile(r"কখন\s*(?:ডাক্তার|এখানে|ক্লিনিকে)?.{0,10}(?:আসা|যাওয়া)\s*(?:লাগবে|হবে|উচিত|যাবে)"),
    re.compile(r"कब\s*(?:आऊं|आऊँ|आऊंगा|आऊँगा|आऊंगी|आना|आएं|आएँ|आयें|पहुंच|पहुँच|जाना|जाऊं|जाऊँ|जाएं|जाएँ)"),
    re.compile(r"कितने\s*बजे\s*(?:आना|आऊं|आऊँ|आएं|आएँ|जाना|जाऊं|जाऊँ|पहुंच|पहुँच)"),
    re.compile(
        r"\bkab\s*(?:aaun|aun|aana|aaunga|aaungi|aaye|aaiye|jaana|jana|jaun|jaaun|pahunchu|pahunchna|pahuchna)\b", _I
    ),
    re.compile(r"\bkitne\s*baje\s*(?:aana|aaun|aun|jaana|jana|aaye|pahunch\w*|pahuch\w*)\b", _I),
]

# ---- "what is my serial number?" (owner, 2026-10-23, item 21) -----------------------------------------------------------------------------------------------
# After payment the serial number comes on WhatsApp. A caller who only says a token / serial number ("my token is 20") or asks to move it up
# (owner_answers.asks_token_push) is not asking this and keeps the existing answers.
_SERIAL = [
    re.compile(r"\b(?:what|which|where|when|how)\b.{0,30}\b(?:serial|token)\b", _I),
    re.compile(r"\b(?:want|need|give|send|tell|get)\b.{0,15}\b(?:my |the |a )?(?:serial|token)\b", _I),
    re.compile(
        r"\b(?:serial|token)\b.{0,25}\b(?:ki|koto|kobe|kokhon|kothay|kothai|pabo|paboi|jabo|janbo|jante|kemne|kemon)\b",
        _I,
    ),
    re.compile(r"\b(?:ki|koto|kobe|kokhon|kothay)\b.{0,15}\b(?:serial|token)\b", _I),
    re.compile(r"\b(?:serial|token)\b.{0,25}\b(?:kya|kitna|kab|kahan|kaise|milega|milegi|pata)\b", _I),
    re.compile(r"(?:সিরিয়াল|টোকেন).{0,20}(?:কত|কখন|কোথায়|কোথা|কবে|কী|কি|কীভাবে|কিভাবে|পাব|পাবো|জানব|জানবো|জানতে)"),
    re.compile(r"(?:কত|কখন|কোথায়|কবে|কী|কি).{0,15}(?:সিরিয়াল|টোকেন)"),
    re.compile(r"(?:सीरियल|टोकन).{0,20}(?:क्या|कितना|कब|कहां|कहाँ|कैसे|मिलेगा|मिलेगी|पता)"),
    re.compile(r"(?:क्या|कब|कहां|कहाँ|कैसे).{0,15}(?:सीरियल|टोकन)"),
]

_PATTERNS["when_to_come"] = _WHEN_TO_COME
_PATTERNS["serial_number"] = _SERIAL


def asks_when_to_come(text: str, lang: str = "en") -> bool:
    """ "kokhon asbo" / "when should I come" / "कितने बजे आना है" -- when to arrive, with no token number in it."""
    return _any("when_to_come", text)


def asks_serial_number(text: str, lang: str = "en") -> bool:
    """ "serial number kobe pabo" / "what is my serial number" / "सीरियल नंबर कब मिलेगा"."""
    return _any("serial_number", text)


_SERIAL_REPLY = {
    "bn": "পেমেন্ট করার পর আপনার সিরিয়াল নম্বর হোয়াটসঅ্যাপে পেয়ে যাবেন।",
    "hi": "पेमेंट करने के बाद आपका सीरियल नंबर व्हाट्सऐप पर मिल जाएगा।",
    "en": "After you pay, you will get your serial number on WhatsApp.",
}


def serial_number_reply(lang: str) -> str:
    return _SERIAL_REPLY.get(lang, _SERIAL_REPLY["en"])


# ======================================================================================================== round 4 (owner, 2026-10-23): items 17, 16, 22
# Colloquial and romanized wording is matched with deliberately tolerant spellings: the recogniser writes the same Bengali / Hindi sound many ways
# ("phonepe", "phonepay", "phonpe", "kota", "kouta", "kota theke", "kota thke"), so each cue lists the spellings seen rather than one canonical form.

# ---- item 17: "PhonePe / UPI / GPay hobena" -- the caller cannot pay online ----------------------------------------------------------------------------------
_PAY_APPS_RM = r"(?:phone\s?pe|phone\s?pay|phonepe|phonepay|phonpe|phon\s?pe|gpay|g\s?pay|google\s?pay|paytm|pay\s?tm|upi|u\s?p\s?i|bhim|online\s+payment|online\s+pay|net\s?banking)"
_CANNOT_RM = r"(?:hobe\s*na|hobena|hoibo\s*na|hoybena|hoyna|hobe\s*nah|nei|nai|ney|nahi|nhi|nahin|chole\s*na|cholena|cholbe\s*na|parbo\s*na|parbona|korte\s*parbo\s*na|dite\s*parbo\s*na|jabe\s*na|jabena)"
_CANNOT_PAY = [
    re.compile(r"\b" + _PAY_APPS_RM + r"\b.{0,30}\b" + _CANNOT_RM + r"\b", _I),
    re.compile(r"\b" + _CANNOT_RM + r"\b.{0,20}\b" + _PAY_APPS_RM + r"\b", _I),
    re.compile(
        r"\b(?:no|without|don'?t have|do not have|haven'?t got|not having|cannot use|can'?t use)\s+(?:a\s+|an\s+|any\s+|the\s+|my\s+)?"
        + _PAY_APPS_RM
        + r"\b",
        _I,
    ),
    re.compile(
        r"\b"
        + _PAY_APPS_RM
        + r"\b\s*(?:is\s+|are\s+|does\s+|do\s+)?(?:not|n'?t)\s+(?:available|possible|working|accepted|there|an option|work|works)\b",
        _I,
    ),
    re.compile(
        r"\b(?:can'?t|cannot|can not|unable to|won'?t be able to|not able to)\b.{0,15}\b(?:pay|payment)\b.{0,12}\b(?:online|digitally|by upi|by phone)\b",
        _I,
    ),
    re.compile(
        r"\bonline\b.{0,12}\b(?:payment|pay)\b.{0,15}\b(?:possible\s+na|hobe\s*na|hobena|parbo\s*na|kora\s+jabe\s+na|nei|nai)\b",
        _I,
    ),
    re.compile(
        r"(?:ফোন\s*পে|ফোনপে|জিপে|জি\s*পে|গুগল\s*পে|পেটিএম|ইউপিআই|ইউ\s*পি\s*আই|ভীম|অনলাইন\S*\s*(?:পেমেন্ট|পে))\S*\s*.{0,22}(?:হবে\s*না|হবেনা|নেই|নাই|চলে\s*না|চলবে\s*না|করতে\s*পারব\s*না|পারবো\s*না|পারব\s*না|যাবে\s*না)"
    ),
    re.compile(r"(?:হবে\s*না|হবেনা|নেই|নাই|পারব\s*না).{0,15}(?:ফোন\s*পে|ফোনপে|জিপে|গুগল\s*পে|পেটিএম|ইউপিআই|অনলাইন)"),
    re.compile(
        r"(?:फोन\s*पे|फोनपे|गूगल\s*पे|जी\s*पे|जीपे|पेटीएम|यूपीआई|यू\s*पी\s*आई|ऑनलाइन\s*(?:पेमेंट|भुगतान|पे))\S*\s*.{0,20}(?:नहीं|नही|ना)\s*(?:है|हो|होगा|होगी|चलता|चलेगा|कर|पाऊंगा|पाऊँगा|पाएंगे|सकता|सकते|सकती)"
    ),
    re.compile(r"(?:नहीं|नही).{0,12}(?:फोन\s*पे|फोनपे|गूगल\s*पे|जीपे|पेटीएम|यूपीआई)"),
    re.compile(
        r"\b(?:phonepe|phonepay|gpay|paytm|upi|online)\b.{0,20}\b(?:nahi|nhi|nahin|na)\s*(?:hai|ho|hoga|chalta|chalega|kar|paunga|payenge|sakta|sakte)\b",
        _I,
    ),
]
_COME_TO_CLINIC = [
    re.compile(
        r"\b(?:come|go|visit|walk|reach)\b.{0,15}\b(?:to |at |in )?(?:the |your )?(?:clinic|counter|center|centre)\b.{0,30}\b(?:book|booking|serial|appointment|register|name)\b",
        _I,
    ),
    re.compile(
        r"\b(?:book|booking|serial|appointment|register)\b.{0,25}\b(?:at|in|from)\s+(?:the |your )?(?:clinic|counter)\b",
        _I,
    ),
    re.compile(
        r"\bwalk[\s-]?in\b|\bdirectly\b.{0,12}\b(?:clinic|counter)\b|\b(?:clinic|counter)\b.{0,12}\bdirectly\b", _I
    ),
    re.compile(
        r"\b(?:clinic|counter|clinik|kaunter)\w{0,3}\s*(?:e|a)?\s*(?:giye|gie|gya|eshe|ese|asbo|ashbo|jabo|jaabo|aste|ashte)\b.{0,25}\b(?:book|serial|naam|nam|appointment|korbo|likhbo|lekhabo)\b",
        _I,
    ),
    re.compile(
        r"\b(?:book|serial|naam|nam|appointment)\b.{0,25}\b(?:clinic|counter|clinik|kaunter)\w{0,3}\s*(?:e|a)?\s*(?:giye|gie|eshe|ese|asbo|ashbo|jabo|aste|ashte)\b",
        _I,
    ),
    re.compile(
        r"\b(?:sorasori|sharasori|shorashori|seedhe|seedha|sidhe|sidha)\b.{0,20}\b(?:clinic|counter|clinik)\b", _I
    ),
    re.compile(
        r"(?:ক্লিনিকে|ক্লিনিক|কাউন্টারে|কাউন্টার)\S*\s*.{0,12}(?:এসে|গিয়ে|যেয়ে|আসব|আসবো|যাব|যাবো|এসেই).{0,25}(?:বুক|সিরিয়াল|নাম|অ্যাপয়েন্টমেন্ট|করব|করবো|লেখাব|লেখাবো|লিখব)"
    ),
    re.compile(r"(?:বুক|সিরিয়াল|নাম|অ্যাপয়েন্টমেন্ট).{0,25}(?:ক্লিনিকে|কাউন্টারে)\S*\s*(?:এসে|গিয়ে|যেয়ে|আসব|যাব|করব)"),
    re.compile(r"সরাসরি\s*(?:ক্লিনিকে|কাউন্টারে|গিয়ে|এসে)"),
    re.compile(
        r"(?:क्लिनिक|काउंटर)\S*\s*(?:पर|में)?\s*(?:आकर|जाकर|आके|जाके|आ\s*कर|जा\s*कर).{0,25}(?:बुक|सीरियल|नाम|अपॉइंटमेंट|करवा|लिखवा)"
    ),
    re.compile(r"(?:बुक|सीरियल|नाम|अपॉइंटमेंट).{0,25}(?:क्लिनिक|काउंटर)\S*\s*(?:पर|में)?\s*(?:आकर|जाकर|आके|जाके|आ\s*कर)"),
    re.compile(r"सीधे\s*(?:क्लिनिक|काउंटर)|(?:क्लिनिक|काउंटर)\s*(?:पर|में)\s*(?:आकर|जाकर)"),
]

# ---- item 16: "kota theke kota bose?" -- the doctor's sitting hours, from and to -------------------------------------------------------------------------------
_HOURS_RANGE = [
    re.compile(r"\bfrom\s+what\s+time\s+(?:to|till|until)\s+(?:what\s+time|when)\b", _I),
    re.compile(r"\b(?:till|until|up\s*to|upto)\s+what\s+time\b", _I),
    re.compile(
        r"\bwhat\s+(?:are\s+)?(?:the\s+)?(?:timing|timings|hours|sitting\s+hours|chamber\s+hours|schedule)\b.{0,25}\b(?:doctor|dr|sit|sits|chamber|consult|he|she)\b",
        _I,
    ),
    re.compile(r"\b(?:doctor|dr)\b.{0,20}\b(?:timing|timings|hours|schedule)\b", _I),
    re.compile(
        r"\b(?:start|starts|begin|begins)\b.{0,12}\b(?:and|to|till|until)\b.{0,12}\b(?:end|ends|finish|finishes|close|closes)\b",
        _I,
    ),
    re.compile(r"\bko+u?ta\w{0,2}\s*(?:theke|thake|teke|thke|thekey|theke\s+shuru|hote)\s*ko+u?ta\b", _I),
    re.compile(r"\bko+u?ta\w{0,2}\s*(?:porjonto|porjonto|obdhi|obodhi|ovdhi|aboddhi|pojonto)\b", _I),
    re.compile(
        r"\bkokhon\s*(?:theke|thake|teke|thke)\s*kokhon\b|\bkotokkhon\s*(?:porjonto\s*)?(?:bose|bosen|boshen|boshe|boshben|bosben|bosbe|thaken|achen)\b",
        _I,
    ),
    re.compile(
        r"\b(?:doctor|dagtar|daktar|dactar|dr)\w{0,4}\s*(?:er|r)?\s*(?:somoy|shomoy|timing|time)\s*(?:ki|koto|kobe|bolun|bolen)?\b.{0,15}\b(?:bose|boshen|boshe|boshbe|boshben)\b",
        _I,
    ),
    re.compile(r"(?:কটা|কোটা|কতটা|কটায়|কখন)\s*(?:থেকে|থাকে|থেকে)\s*(?:কটা|কোটা|কতটা|কটায়|কখন)"),
    re.compile(r"(?:কটা|কোটা|কতটা|কখন)\s*পর্যন্ত\s*(?:বসেন|বসবেন|বসে|থাকেন|থাকবেন|দেখেন|দেখবেন)"),
    re.compile(r"কতক্ষণ\s*(?:পর্যন্ত\s*)?(?:বসেন|বসবেন|বসে|থাকেন|থাকবেন|দেখেন)"),
    re.compile(r"(?:ডাক্তার|ডক্টর)\S*\s*(?:এর\s*|র\s*)?(?:বসার\s*)?(?:সময়সূচি|সময়সূচী|টাইমিং|সময়)\s*(?:কত|কী|কি|বলুন)?"),
    re.compile(
        r"कितने\s*बजे\s*से\s*कितने\s*बजे|कब\s*से\s*कब\s*तक|कितने\s*बजे\s*तक\s*(?:बैठते|बैठती|बैठेंगे|मिलते|देखते)|कितनी\s*देर\s*तक\s*(?:बैठते|बैठती|बैठेंगे|मिलते|देखते)"
    ),
    re.compile(r"(?:डॉक्टर|डाक्टर)\S*\s*(?:का|की|के)\s*(?:टाइमिंग|समय|टाइम|शेड्यूल)"),
    re.compile(
        r"\bkitne\s*baje\s*se\s*kitne\s*baje\b|\bkab\s*se\s*kab\s*tak\b|\bkitne\s*baje\s*tak\s*(?:baithte|baithti|baithenge|milte|dekhte)\b",
        _I,
    ),
]
# The clinic's OWN opening hours are a different question (the FAQ answers it): a sentence with an open / close / clinic word is left to it.
_CLINIC_HOURS = re.compile(
    r"\b(?:open|opens|opening|close|closes|closing|clinic|counter|khola|kholo|bondho|bandh)\b|খোলা|খোলে|বন্ধ|ক্লিনিক|কাউন্টার|खुल|बंद|क्लिनिक|काउंटर",
    _I,
)

# ---- item 22: "bolchi ekta naam lekhanor chhilo" -- a name to be written (a serial to be taken) with no doctor named ------------------------------------------------
_WRITE_NAME = [
    # Only the "write / put a NAME" wording. A plain "I want an appointment / serial" is an ordinary booking request: the booking flow asks its own questions.
    re.compile(r"\bna+m\s*(?:l[eo]kh|lik|lekh|likh)\w*", _I),
    re.compile(r"\bna+m\s+(?:ta\s+|ti\s+|ekta\s+)?(?:lekhano|lekhate|lekhabo|likhabo|likhbo|lekha|likha|lekhar|likhar|lekhanor|likhanor)\b", _I),
    re.compile(r"\b(?:put|write|add|register|enter)\s+(?:my|a|the|his|her)\s+name\b", _I),
    re.compile(r"\bna+m\s*(?:likh|likhwa|likhva|likha|likhna)\w*", _I),
    re.compile(r"\bserial\s*(?:lagano|lagate|lagabo|laganor|lagwana|lagwani|lagwa|lagana)\b", _I),
    re.compile(r"নাম\s*(?:লেখ|লিখ|লেখা|লিখা|লেখানো|লিখানো|লেখাতে|লেখাব|লেখাবো|লেখাচ্ছি|লেখার)\S*"),
    re.compile(r"সিরিয়াল\s*(?:লাগাতে|লাগাব|লাগাবো|লাগানোর|লাগিয়ে)"),
    re.compile(r"नाम\s*(?:लिख|लिखवा|लिखा|लिखाना|लिखवाना|लिखने)\S*"),
    re.compile(r"सीरियल\s*(?:लगवा|लगा)\S*"),
]
# A real doctor / test / date in the same sentence is the booking flow's business (it asks for what is missing); only the bare "write a name" is answered here.
_WRITE_NAME_GUARD = re.compile(
    r"\b(?:test|tests|usg|x-?ray|scan|mri|ct|blood|sugar|cbc|report)\b|টেস্ট|পরীক্ষা|রক্ত|রিপোর্ট|এক্স-?রে|স্ক্যান|टेस्ट|जांच|रिपोर्ट|एक्स-?रे|स्कैन",
    _I,
)

_PATTERNS["cannot_pay_online"] = _CANNOT_PAY
_PATTERNS["come_to_clinic"] = _COME_TO_CLINIC
_PATTERNS["hours_range"] = _HOURS_RANGE
_PATTERNS["write_name"] = _WRITE_NAME


def cannot_pay_online(text: str, lang: str = "en") -> bool:
    """ "PhonePe hobena" / "UPI nei" / "GPay chole na" / "ऑनलाइन पेमेंट नहीं हो पाएगा" -- the caller cannot pay through an app."""
    return _any("cannot_pay_online", text)


def asks_come_to_clinic(text: str, lang: str = "en") -> bool:
    """ "clinic e giye book korte parbo?" / "can I come and book at the clinic" / "सीधे क्लिनिक आकर बुक" -- booking in person instead of by phone."""
    return _any("come_to_clinic", text)


def asks_sitting_hours(text: str, lang: str = "en") -> bool:
    """ "doctor x kota theke kota bose" / "from what time to what time does he sit" / "कितने बजे से कितने बजे तक" -- the from-and-to hours.
    Never the clinic's own opening hours (an open / close / clinic word leaves the sentence to the FAQ)."""
    t = _nfc(text)
    return bool(t) and _CLINIC_HOURS.search(t) is None and _any("hours_range", text)


def asks_to_write_name(text: str, lang: str = "en") -> bool:
    """ "ekta naam lekhanor chhilo" / "serial nite chai" / "नाम लिखवाना है" -- a name to be written (a serial to be taken); a test or report in the sentence is left alone."""
    t = _nfc(text)
    return bool(t) and _WRITE_NAME_GUARD.search(t) is None and _any("write_name", text)


_CANNOT_PAY_REPLY = {
    "bn": "পেমেন্ট ছাড়া নাম লেখানো যাবে না। যার কাছে হোয়াটসঅ্যাপ আর ফোনপে আছে, তাঁর সাহায্য নিয়ে আবার ফোন করুন।",
    "hi": "पेमेंट के बिना नाम नहीं लिखा जा सकता। जिनके पास व्हाट्सऐप और फोनपे है, उनकी मदद लेकर दोबारा फ़ोन कीजिए।",
    "en": "The name cannot be registered without the payment. Please take help from someone who has WhatsApp and PhonePe, and call again.",
}
_COME_TO_CLINIC_REPLY = {
    "bn": "হ্যাঁ, আপনি ক্লিনিকে এসে বুক করতে পারেন। তবে বুকিংয়ের স্লট ভরে গেলে একই তারিখ নাও পেতে পারেন।",
    "hi": "जी, आप क्लिनिक आकर बुक कर सकते हैं। लेकिन बुकिंग के स्लॉट भर गए हों तो हो सकता है उसी तारीख़ का स्लॉट न मिले।",
    "en": "Yes, you can come to the clinic and book. But if the booking slots are full, you may not get the same date.",
}


def cannot_pay_online_reply(lang: str) -> str:
    return _CANNOT_PAY_REPLY.get(lang, _CANNOT_PAY_REPLY["en"])


def come_to_clinic_reply(lang: str) -> str:
    return _COME_TO_CLINIC_REPLY.get(lang, _COME_TO_CLINIC_REPLY["en"])


_HOURS = {
    "bn": "ডাক্তার {doctor} {when} {start} থেকে {end} পর্যন্ত বসবেন।",
    "hi": "डॉक्टर {doctor} {when} {start} से {end} तक बैठेंगे।",
    "en": "Doctor {doctor} sits {when} from {start} to {end}.",
}


def sitting_hours_reply(doctor: str, date_iso: str, start: str, end: str, lang: str) -> str:
    """The doctor's sitting hours, FROM and TO, on the day told (or his next sitting day) -- both from the clinic's schedule table."""
    lang = lang if lang in _HOURS else "en"
    doctor = (
        doctor.replace("Dr. ", "", 1)
        .replace("Dr ", "", 1)
        .replace("ডাঃ ", "", 1)
        .replace("ডাক্তার ", "", 1)
        .replace("डॉ. ", "", 1)
    )
    return _HOURS[lang].format(doctor=doctor, when=slot_proposal.when_phrase(date_iso, lang), start=start, end=end)

# ======================================================================================================== round 6 (owner, 2026-10-23): phrasings found missing in testing
# Added to the lists above (extend, never replaced), after a sweep of about 390 sentences through the real turn handler left these unrecognised.
_GOOD_DOCTOR.extend(
    [
        re.compile(r"\b(?:bhala|bhale|badhiya|behtar|behtareen|bahut\s+accha|bahut\s+achha)\s+(?:doctor|daktar|dagtar|dr)\b", _I),
        re.compile(r"\bsabse\s+(?:accha|acha|achha|bhala|badhiya|behtar)\s+(?:doctor|daktar|dagtar|dr)\b", _I),
        re.compile(r"\b(?:doctor|daktar|dagtar)\s+(?:bhala|bhale|badhiya|behtar)\s+(?:kaun|kon|kaunsa|kaun\s+sa|kaunsi|kya)\b", _I),
        re.compile(r"(?:भला|भले|बेहतर|बेहतरीन)\s*(?:डॉक्टर|डाक्टर|डॉ)"),
    ]
)
_WEEKDAYS.extend(
    [
        re.compile(r"\bki\s*ki\s*(?:bar|bare|barey|baar|din|dine)\b", _I),
        re.compile(r"\bko(?:y|i)(?:ta|ti)?\s*(?:din|dine|bar)\b.{0,14}\b(?:bose|bosen|boshen|boshe|asen|ashen|thaken|achen)\b", _I),
        re.compile(r"\bkitne\s*din\b.{0,14}\b(?:baithte|baithti|baithenge|aate|aati|milte|milti|milenge)\b", _I),
        re.compile(r"কয়\s*(?:টা\s*)?দিন.{0,14}(?:বসেন|বসে|আসেন|থাকেন)"),
        re.compile(r"कितने\s*दिन.{0,14}(?:बैठते|बैठती|आते|आती|मिलते|मिलेंगे)"),
    ]
)
_NEXT_IF_ABSENT.extend(
    [
        re.compile(
            r"\b(?:if|in case|when)\b.{0,25}\b(?:absent|on leave|unavailable|not\s+available|not\s+there|not\s+sitting|not\s+in)\b.{0,30}\b(?:when|next|which day|then)\b",
            _I,
        ),
        re.compile(
            r"\b(?:doctor|daktar|dagtar|dactar|dr)\b.{0,15}\bna\s*(?:pele|thakle|boshle|hole|paile|aste|ashle)\b.{0,20}\b(?:next|kobe|pore|porer|abar|arekdin)\b",
            _I,
        ),
        re.compile(
            r"\b(?:aaj|aj)\b.{0,25}\b(?:na|nahi|nahin|nhi)\s*(?:mile|milenge|milen|baithe|baithenge|aaye|aayenge)\b.{0,25}\b(?:kab|next|agla|agli|phir|fir)\b",
            _I,
        ),
        re.compile(r"\b(?:doctor|daktar)\b.{0,12}\b(?:na|nahi|nahin)\s*(?:mile|milen|ho|hon|baithe)\b.{0,15}\b(?:to\s+)?(?:kab|next|agla|agli)\b", _I),
        re.compile(r"(?:যদি)?\s*(?:ডাক্তার|ডক্টর)\S*\s*.{0,15}(?:না\s*পেলে|না\s*থাকলে|না\s*বসলে).{0,20}(?:কবে|পরে|পরের|আবার|নেক্সট)"),
        re.compile(r"(?:अगर|यदि)\s*(?:डॉक्टर|डाक्टर).{0,20}(?:नहीं|न)\s*(?:हैं|है|मिले|हों|हो).{0,15}(?:तो\s*)?(?:कब|अगला|अगली)"),
    ]
)
_WHEN_TO_COME.append(
    re.compile(r"\bko(?:ita|yta|yeta|ta|yti)\s*(?:r|er)?\s*(?:moddhe|modhye|mddhe|shomoy|somoy)?\s*(?:asbo|ashbo|asben|ashben|ashte|aste|jabo|aschi)\b", _I)
)
_CANNOT_PAY.append(
    re.compile(
        r"\b" + _PAY_APPS_RM + r"\b\s*(?:will\s+not|won'?t|wouldn'?t|doesn'?t|does\s+not|isn'?t\s+going\s+to)\s+(?:work|go\s+through|be\s+possible|be\s+accepted)\b",
        _I,
    )
)
_COME_TO_CLINIC.extend(
    [
        re.compile(
            r"\b(?:clinic|counter|clinik)\b\s*(?:pe|par|me|mein)?\s*(?:aakar|aake|aaker|aa\s*kar|jakar|jaakar|jaake)\b.{0,25}\b(?:book|booking|serial|naam|nam|appointment|likhwa|karwa)\w*",
            _I,
        ),
        re.compile(
            r"\b(?:book|booking|serial|naam|nam|appointment)\b.{0,25}\b(?:clinic|counter|clinik)\b\s*(?:pe|par|me|mein)?\s*(?:aakar|aake|aaker|jakar|jaakar|jaake)\b",
            _I,
        ),
    ]
)
