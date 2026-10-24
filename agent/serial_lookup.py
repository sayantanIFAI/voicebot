"""'I called, booked, paid -- but the serial number never came on WhatsApp.' Confirm the mobile number, look it up in the database, tell the caller.

sourav, 2026-10-24: the owner's rule. When a caller says they booked and paid but did not get the serial number on WhatsApp, the agent (1) asks for the mobile number the
booking was made with, (2) READS IT BACK and waits for a yes -- a serial is only looked up for a number the caller has confirmed, never a misheard one -- (3) asks the
database (clinic-api /api/v1/bookings/serial; payment is SIMULATED there today) and (4) tells the caller what the database holds: the serial number if the booking was paid,
or that no payment is recorded yet, or that no booking was found. The serial is NEVER composed by the model (CLAUDE.md: the LLM may decide what the caller wants, never what is true).

This module is the pure half: the cue, the little state machine for the number (ask -> confirm -> look up), and every sentence. main.py owns the session and the tool calls.
Spoken digits are read with agent.spoken_codes.parse_phone (digit words in English / Hindi / Bengali, numerals, +91 and 0 prefixes), the same reader the booking flow trusts.
"""

from __future__ import annotations

import re

from agent import spoken_codes

MAX_TRIES = 3  # asking for / confirming the number: after this many misses the caller is pointed to the counter instead of being asked forever

# ------------------------------------------------------------------------------------------------------------------------- the cue
_SERIAL_WORDS = (
    "serial", "sirial", "seriyal", "sireal", "token", "টোকেন", "সিরিয়াল", "সিরিয়েল",
    # real call 2026-10-24: "সেরিয়াল নম্বার আসেনি" (সে, not সি) was not read as a serial-number report at all,
    # and fell through to a fresh booking offer instead of a database lookup for the EXISTING one.
    "সেরিয়াল", "সেরিয়েল", "सीरियल", "सिरियल", "सेरियल", "सेरियाल", "टोकन", "टोकेन",
)  # fmt: skip
# "did not come / did not get" in the three languages, Latin letters included (the recogniser writes them every way)
_NOT_RECEIVED = re.compile(
    r"(?:did\s*n'?o?t|didnt|haven'?t|have\s+not|has\s*n'?o?t|hasn't|not\s+yet|never|still\s+(?:not|waiting|no)|no\s+(?:serial|token|message)|not\s+(?:get|got|receive|received|come|came|arrive|arrived|see|seen))"
    r"|(?:paini|pai\s*ni|pai\s*nai|pacchi\s*na|pachchi\s*na|pachhi\s*na|aseni|ase\s*ni|ashe\s*ni|ashchhe\s*na|aschhe\s*na|ase\s*na|ashena|ashe\s*na|asheni|ayni)"
    r"|(?:nahi|nahin|nahee|nhi|nai)\s*(?:aaya|aya|aayi|ayi|aai|mila|mili|mil\s*raha|aa\s*raha|aa\s*rahi|aaye)"
    r"|(?:abhi\s*tak|ab\s*tak)\s*(?:nahi|nahin)"
    r"|পাইনি|পাই\s*নি|পাইনা|পাচ্ছি\s*না|পাচ্ছি\s*নাই|আসেনি|আসে\s*নি|আসছে\s*না|আসেনা|আসে\s*না|আসেনাই|এখনো\s*(?:আসেনি|পাইনি)"
    r"|नहीं\s*(?:आया|आई|आयी|मिला|मिली|आ\s*रहा|आ\s*रही|आए|आये)|नही\s*(?:आया|आई|आयी|मिला|मिली)|अभी\s*तक\s*नहीं|कुछ\s*नहीं\s*आया",
    re.IGNORECASE,
)


def reports_serial_not_received(text: str, lang: str = "en") -> bool:
    """sourav, 2026-10-24: True for "I paid but the serial number did not come on WhatsApp" in English, Hindi and Bengali (their own scripts or Latin letters). Why: this report
    must reach the database lookup instead of the generic "after you pay you will get your serial on WhatsApp" reply, which tells a caller who HAS paid nothing. Flow: the sentence
    must name a serial / token AND say it did not come / was not got. A plain "what is my serial number?" is not this and keeps its old answer."""
    t = " ".join((text or "").lower().split())
    if not t or len(t.split()) > 40:
        return False
    return any(w in t for w in _SERIAL_WORDS) and bool(_NOT_RECEIVED.search(t))


# -------------------------------------------------------------------------------------------------------------------- the state machine
def new_flow(turn: int) -> dict:
    """sourav, 2026-10-24: the state of one "find my serial" conversation. stage: "ask" (waiting for the number) or "confirm" (number read back, waiting for yes / no)."""
    return {"stage": "ask", "phone": None, "buf": "", "tries": 0, "turn": turn}


def advance(flow: dict, text: str, yes_no: str | None) -> tuple[str, str | None]:
    """sourav, 2026-10-24: one caller turn inside the flow -> (action, payload). Why: keeps every rule about the number out of main.py, where it could not be tested. Actions:
      "confirm"       payload = the 10-digit number to read back and have confirmed
      "lookup"        payload = the confirmed number to look up in the database
      "ask_again"     the number was not understood / was refused: ask for it again
      "repeat"        the readback got neither yes nor no: ask yes or no again
      "give_up"       MAX_TRIES misses: stop asking
    Flow ("ask"): read a mobile number from the sentence (digits may arrive over several turns, kept in flow["buf"]); a complete valid one moves to "confirm". Flow ("confirm"):
    yes -> look it up; no -> back to "ask" for a new number; a different complete number in the sentence replaces the one being confirmed; anything else -> repeat the question."""
    spoken = spoken_codes.parse_phone(text)
    if flow["stage"] == "confirm":
        if spoken and spoken != flow["phone"]:
            flow["phone"] = spoken  # "no, it is 98..." -- the corrected number is read back in its turn
            return "confirm", spoken
        if yes_no == "yes":
            return "lookup", flow["phone"]
        if yes_no == "no":
            flow.update(stage="ask", phone=None, buf="")
            flow["tries"] += 1
            return ("give_up", None) if flow["tries"] >= MAX_TRIES else ("ask_again", None)
        flow["tries"] += 1
        return ("give_up", None) if flow["tries"] >= MAX_TRIES else ("repeat", flow["phone"])
    # stage "ask"
    if spoken is None:
        digits = spoken_codes._digit_run(text)
        if digits:
            flow["buf"] += digits
            spoken = spoken_codes.parse_phone(flow["buf"])
            if spoken is None and len(flow["buf"]) >= 10:
                flow["buf"] = ""  # ten or more digits that are not a mobile number: start over rather than keep adding
    if spoken:
        flow.update(stage="confirm", phone=spoken, buf="")
        return "confirm", spoken
    flow["tries"] += 1
    return ("give_up", None) if flow["tries"] >= MAX_TRIES else ("ask_again", None)


# ------------------------------------------------------------------------------------------------------------------------ the sentences
def _lang(lang: str) -> str:
    return lang if lang in ("bn", "hi") else "en"


def _digits_spoken(number: str) -> str:
    return " ".join(number)


ASK_NUMBER = {
    "bn": "দুঃখিত, আমি দেখে দিচ্ছি। যে মোবাইল নম্বর দিয়ে বুক করেছিলেন, সেটা একটু বলবেন?",
    "hi": "माफ़ कीजिए, मैं देखती हूँ। जिस मोबाइल नंबर से बुक किया था, वो बताइए?",
    "en": "I am sorry about that, let me check. Could you tell me the mobile number you booked with?",
}
ASK_AGAIN = {
    "bn": "নম্বরটা আরেকবার বলবেন? দশটা সংখ্যা, ধীরে ধীরে।",
    "hi": "नंबर एक बार फिर बताइए? दस अंक, धीरे-धीरे।",
    "en": "Could you say the number once more? Ten digits, slowly please.",
}
GIVE_UP = {
    "bn": "দুঃখিত, নম্বরটা ঠিকমতো বুঝতে পারছি না। অনুগ্রহ করে ক্লিনিকের কাউন্টারে বুকিং আইডি দেখিয়ে সিরিয়াল নম্বরটা নিয়ে নেবেন।",
    "hi": "माफ़ कीजिए, मैं नंबर ठीक से समझ नहीं पा रही। कृपया क्लिनिक के काउंटर पर बुकिंग आईडी दिखाकर सीरियल नंबर ले लीजिए।",
    "en": "I am sorry, I cannot make out the number. Please show your booking ID at the clinic counter and they will give you the serial number.",
}
CHECK_FAILED = {
    "bn": "দুঃখিত, এই মুহূর্তে সিরিয়াল নম্বরটা দেখতে পারছি না। একটু পরে আবার ফোন করবেন, বা ক্লিনিকের কাউন্টারে বুকিং আইডি দেখাবেন।",
    "hi": "माफ़ कीजिए, अभी सीरियल नंबर नहीं देख पा रही। थोड़ी देर बाद फिर फ़ोन कीजिए, या क्लिनिक के काउंटर पर बुकिंग आईडी दिखाइए।",
    "en": "I am sorry, I cannot check the serial number right now. Please call again in a little while, or show your booking ID at the clinic counter.",
}


def ask_number(lang: str) -> str:
    return ASK_NUMBER[_lang(lang)]


def ask_again(lang: str) -> str:
    return ASK_AGAIN[_lang(lang)]


def give_up(lang: str) -> str:
    return GIVE_UP[_lang(lang)]


def check_failed(lang: str) -> str:
    return CHECK_FAILED[_lang(lang)]


def confirm_number(number: str, lang: str) -> str:
    """sourav, 2026-10-24: the number read back digit by digit, for a yes. Why: a serial is told only for a number the caller has confirmed."""
    d = _digits_spoken(number)
    return {
        "bn": f"আপনার নম্বর {d}। ঠিক আছে?",
        "hi": f"आपका नंबर {d}। सही है?",
        "en": f"Your number is {d}. Is that right?",
    }[_lang(lang)]


def _slot(b: dict, lang: str) -> str:
    doctor = b.get("doctor_name") or {"bn": "ডাক্তার", "hi": "डॉक्टर", "en": "the doctor"}[_lang(lang)]
    return f"{doctor}, {b.get('date')}, {b.get('time_slot')}"


def result_reply(bookings: list[dict], lang: str) -> str:
    """sourav, 2026-10-24: what the database says, in words. Why: this is the whole point of the flow, and every figure in it (serial, doctor, date, time) is the database's own.
    Flow: no booking -> say so; paid bookings -> their serial numbers; an unpaid booking -> the booking was found but no payment is recorded, so no serial exists yet."""
    lg = _lang(lang)
    if not bookings:
        return {
            "bn": "এই নম্বরে কোনো আসন্ন অ্যাপয়েন্টমেন্ট পেলাম না। অন্য নম্বর দিয়ে বুক করে থাকলে সেই নম্বরটা বলুন, না হলে ক্লিনিকের কাউন্টারে বুকিং আইডি দেখান।",
            "hi": "इस नंबर पर कोई आने वाली अपॉइंटमेंट नहीं मिली। किसी और नंबर से बुक किया हो तो वो नंबर बताइए, वरना क्लिनिक के काउंटर पर बुकिंग आईडी दिखाइए।",
            "en": "I could not find an upcoming appointment for this number. If you booked with another number, tell me that one, or show your booking ID at the clinic counter.",
        }[lg]
    paid = [b for b in bookings if b.get("paid") and b.get("serial_number") is not None]
    unpaid = [b for b in bookings if not (b.get("paid") and b.get("serial_number") is not None)]
    parts = []
    for b in paid[:2]:
        n = b["serial_number"]
        who = b.get("patient_name") or ""
        parts.append(
            {
                "bn": f"{who}-এর পেমেন্ট পেয়েছি। {_slot(b, lg)} -- আপনার সিরিয়াল নম্বর {n}।",
                "hi": f"{who} का पेमेंट मिल गया है। {_slot(b, lg)} -- आपका सीरियल नंबर {n} है।",
                "en": f"I can see the payment for {who}. For {_slot(b, lg)}, your serial number is {n}.",
            }[lg]
        )
    for b in unpaid[:1]:
        parts.append(
            {
                "bn": f"{_slot(b, lg)} -- বুকিংটা আছে, কিন্তু পেমেন্ট এখনো আমাদের সিস্টেমে আসেনি, তাই আপনার সিরিয়াল এখনো নিশ্চিত হয়নি। পেমেন্ট এলে হোয়াটসঅ্যাপে সিরিয়াল চলে যাবে।",
                "hi": f"{_slot(b, lg)} -- बुकिंग है, लेकिन पेमेंट अभी हमारे सिस्टम में नहीं आया, इसलिए आपका सीरियल अभी पक्का नहीं हुआ। पेमेंट आते ही व्हाट्सऐप पर सीरियल आ जाएगा।",
                "en": f"For {_slot(b, lg)}, the booking is there, but I do not see the payment in our system yet, so your serial is not confirmed yet. It will come on WhatsApp once the payment is received.",
            }[lg]
        )
    return " ".join(parts)


def resend_note(last4: str | None, lang: str) -> str:
    """sourav, 2026-10-24: said after the serial was queued to WhatsApp again. The channel is a placeholder, so the words say "sending", never "sent"; the last four digits let a caller notice a wrongly-typed WhatsApp number."""
    tail = f" {_digits_spoken(last4)}" if last4 else ""
    return {
        "bn": f"আমি সিরিয়াল নম্বরটা আবার হোয়াটসঅ্যাপে পাঠাচ্ছি। নম্বরটা{tail} দিয়ে শেষ হয়, তাই তো?" if last4 else "আমি সিরিয়াল নম্বরটা আবার হোয়াটসঅ্যাপে পাঠাচ্ছি।",
        "hi": f"मैं सीरियल नंबर दोबारा व्हाट्सऐप पर भेज रही हूँ। नंबर{tail} पर खत्म होता है, सही है?" if last4 else "मैं सीरियल नंबर दोबारा व्हाट्सऐप पर भेज रही हूँ।",
        "en": f"I am sending the serial number to WhatsApp again, to the number ending{tail}." if last4 else "I am sending the serial number to WhatsApp again.",
    }[_lang(lang)]


def unpaid_followup(lang: str) -> str:
    """sourav, 2026-10-24: said after an unpaid booking was found AND a callback was queued: a person will check the payment. Only said when the callback was really queued."""
    return {
        "bn": "আপনি যদি পেমেন্ট করে থাকেন, আমাদের একজন সহকর্মী আপনাকে ফোন করে দেখে নেবেন।",
        "hi": "अगर आपने पेमेंट कर दिया है, तो हमारे एक साथी आपको फ़ोन करके देख लेंगे।",
        "en": "If you have already paid, one of our colleagues will call you back and check it.",
    }[_lang(lang)]
