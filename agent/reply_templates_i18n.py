"""Hindi and English reply templates.

Same discipline as reply_templates.py (which owns Bengali and dispatches
here): every fact in a spoken sentence is a substitution from a live API
response, never model text. Only the sentence around the fact changes with
the language.

Spoken-name rule, same as the Bengali `_spoken_test_name`: the voice for a
language cannot pronounce another language's script, so the name spoken is
the one in the caller's own script -- Devanagari for Hindi (from the API's
seeded `*_hi` alias), the catalogue's English label for English -- and
never a name in a script the TTS would silently drop.
"""
from __future__ import annotations

import re

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")

_HI_MISSING = {
    ("test_rate", "test_name"): "किस टेस्ट की कीमत जाननी है, ज़रा बताइए?",
    ("doctor_availability", "doctor_name"): "किन डॉक्टर के बारे में पूछ रहे हैं?",
    ("book_appointment", "doctor_name"): "किस डॉक्टर के साथ अपॉइंटमेंट लेना चाहते हैं?",
    ("book_appointment", "date"): "किस दिन के लिए अपॉइंटमेंट चाहिए?",
    ("book_appointment", "time_slot"): "किस समय के लिए अपॉइंटमेंट चाहिए?",
    ("book_appointment", "patient_name"): "मरीज़ का नाम बताइए?",
    ("book_appointment", "phone"): "एक फ़ोन नंबर दीजिए, ताकि मैं कन्फ़र्मेशन भेज सकूँ?",
    ("test_prep", "test_name"): "किस टेस्ट की तैयारी जाननी है, ज़रा बताइए?",
    ("clinic_faq", "faq_topic"): "माफ़ कीजिए, ठीक से समझ नहीं पाई। क्या आप फिर से बताएँगे?",
}
_HI_MISSING_DEFAULT = "माफ़ कीजिए, क्या आप थोड़ा साफ़ बोलेंगे?"

_EN_MISSING = {
    ("test_rate", "test_name"): "Which test would you like the price of?",
    ("doctor_availability", "doctor_name"): "Which doctor are you asking about?",
    ("book_appointment", "doctor_name"): "Which doctor would you like to book with?",
    ("book_appointment", "date"): "Which day would you like the appointment for?",
    ("book_appointment", "time_slot"): "What time would you like the appointment?",
    ("book_appointment", "patient_name"): "May I have the patient's name?",
    ("book_appointment", "phone"): "Could you give me a phone number so I can send the confirmation?",
    ("test_prep", "test_name"): "Which test would you like the preparation for?",
    ("clinic_faq", "faq_topic"): "Sorry, I didn't quite get that. Could you say it again?",
}
_EN_MISSING_DEFAULT = "Sorry, could you say that a little more clearly?"

# Sample types that are a specimen a person gives. "Imaging"/"Cardiac" are
# test categories, not samples -- "sample: imaging" would be nonsense.
_SPECIMENS = {"blood", "urine", "stool", "serum", "saliva", "swab", "plasma"}


def missing_slot_prompt(intent: str, missing: str, lang: str) -> str:
    if lang == "hi":
        return _HI_MISSING.get((intent, missing), _HI_MISSING_DEFAULT)
    return _EN_MISSING.get((intent, missing), _EN_MISSING_DEFAULT)


def _script_ok(text: str, lang: str) -> bool:
    if lang == "hi":
        return bool(_DEVANAGARI.search(text))
    return bool(_LATIN.search(text)) and not _DEVANAGARI.search(text)


def _suggestions(result: dict, lang: str) -> list[str]:
    """did_you_mean mixes English names and aliases in every script. Keep
    only the ones this language's voice can say, or the caller hears a
    list with the names missing."""
    return [s for s in (result.get("did_you_mean") or []) if _script_ok(s, lang)]


def _test_name(slots: dict, result: dict, lang: str) -> str:
    if lang == "hi":
        heard = slots.get("test_name") or ""
        return (result.get("test_name_hi")
                or (heard if _DEVANAGARI.search(heard) else "")
                or "यह")
    return result.get("test_name") or slots.get("test_name") or "the test"


def _doctor_name(slots: dict, result: dict, lang: str) -> str:
    if lang == "hi":
        alias = result.get("doctor_name_hi")
        if alias:
            return f"डॉक्टर {alias}"
        heard = slots.get("doctor_name") or ""
        return heard if _DEVANAGARI.search(heard) else "डॉक्टर"
    full = result.get("doctor_name") or slots.get("doctor_name") or ""
    surname = full.replace("Dr.", "").replace("Dr", "").split()[-1:] if full else []
    return f"Doctor {surname[0]}" if surname else "the doctor"


def _query(slots: dict, key: str, lang: str) -> str:
    q = slots.get(key) or ""
    return q if _script_ok(q, lang) else ""


def test_rate_reply(slots: dict, result: dict, lang: str) -> str:
    hi = lang == "hi"
    if not result.get("found"):
        return _not_found_test(slots, result, lang)
    name = _test_name(slots, result, lang)
    rate = result["rate_inr"]
    hours = result.get("report_time_hours")
    sample = (result.get("sample_type") or "").strip()
    if hi:
        reply = f"{name} टेस्ट की कीमत {rate} रुपये है।"
        if sample.lower() in _SPECIMENS:
            reply += f" सैंपल: {sample}।"
        if hours:
            reply += f" रिपोर्ट {hours} घंटे में मिल जाएगी।"
        return reply
    reply = f"The price of the {name} test is {rate} rupees."
    if sample.lower() in _SPECIMENS:
        reply += f" Sample: {sample}."
    if hours:
        reply += f" You will get the report within {hours} hours."
    return reply


def _not_found_test(slots: dict, result: dict, lang: str) -> str:
    q = _query(slots, "test_name", lang)
    sugg = _suggestions(result, lang)
    if lang == "hi":
        if sugg:
            return (f"'{q}' नाम का टेस्ट नहीं मिला। क्या आप यह कहना चाह रहे हैं: {', '.join(sugg)}?"
                    if q else f"वह टेस्ट नहीं मिला। क्या आप यह कहना चाह रहे हैं: {', '.join(sugg)}?")
        return (f"माफ़ कीजिए, '{q}' नाम का कोई टेस्ट हमारी सूची में नहीं है।" if q
                else "माफ़ कीजिए, वह टेस्ट हमारी सूची में नहीं है।")
    if sugg:
        return (f"I couldn't find a test called '{q}'. Did you mean: {', '.join(sugg)}?" if q
                else f"I couldn't find that test. Did you mean: {', '.join(sugg)}?")
    return (f"Sorry, there is no test called '{q}' on our list." if q
            else "Sorry, that test is not on our list.")


def doctor_availability_reply(slots: dict, result: dict, lang: str) -> str:
    hi = lang == "hi"
    if not result.get("found"):
        q = _query(slots, "doctor_name", lang)
        if hi:
            return (f"माफ़ कीजिए, '{q}' नाम के कोई डॉक्टर हमारे यहाँ नहीं हैं।" if q
                    else "माफ़ कीजिए, इस नाम के कोई डॉक्टर हमारे यहाँ नहीं हैं।")
        return (f"Sorry, we don't have a doctor named '{q}'." if q
                else "Sorry, we don't have a doctor by that name.")

    name = _doctor_name(slots, result, lang)
    if result.get("available"):
        hours = result.get("chamber_hours", "")
        date = result.get("date")
        if hi:
            when = f" {date} को" if date else " आज"
            return f"जी हाँ,{when} {name} चेंबर में रहेंगे। समय: {hours}।"
        when = f" on {date}" if date else " today"
        return f"Yes, {name} will be in the chamber{when}. Timing: {hours}."

    next_date = result.get("next_available_date")
    if next_date:
        if hi:
            return f"{name} उस दिन नहीं बैठेंगे। अगला उपलब्ध दिन: {next_date}।"
        return f"{name} does not sit on that day. The next available day is {next_date}."
    if hi:
        return f"{name} अभी किसी तय दिन नहीं बैठ रहे हैं। आप हमारे काउंटर पर पूछ सकते हैं।"
    return f"{name} is not sitting on any fixed day right now. You can ask at our counter."


def test_prep_reply(slots: dict, result: dict, lang: str) -> str:
    if not result.get("found"):
        return _not_found_test(slots, result, lang)
    name = _test_name(slots, result, lang)
    default = ("इस टेस्ट के लिए किसी ख़ास तैयारी की ज़रूरत नहीं है।" if lang == "hi"
               else "No special preparation is needed for this test.")
    instructions = result.get("prep_instructions") or default
    return (f"{name} टेस्ट के लिए: {instructions}" if lang == "hi"
            else f"For the {name} test: {instructions}")


def clinic_faq_reply(slots: dict, result: dict, lang: str) -> str:
    fallback = ("माफ़ कीजिए, इस बारे में अभी सही जानकारी नहीं दे सकती। कृपया काउंटर पर संपर्क करें।"
                if lang == "hi"
                else "Sorry, I can't give you accurate information on that right now. "
                     "Please contact the counter.")
    if not result.get("found"):
        return fallback
    return result.get("answer") or fallback


def booking_reply(slots: dict, result: dict, lang: str) -> str:
    hi = lang == "hi"
    if result.get("success"):
        doctor = _doctor_name(slots, result, lang)
        if hi:
            return (f"आपकी अपॉइंटमेंट कन्फ़र्म हो गई है। {doctor}, {result['date']}, "
                    f"समय {result['time_slot']}। कन्फ़र्मेशन नंबर: {result['confirmation_id']}।")
        return (f"Your appointment is confirmed. {doctor}, {result['date']}, "
                f"at {result['time_slot']}. Your confirmation number is {result['confirmation_id']}.")

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return (f"वह समय बुक हो चुका है। ये समय खाली हैं: {', '.join(alts)}। कौन सा चाहिए?" if hi
                    else f"That time is already booked. These times are free: {', '.join(alts)}. "
                         f"Which one would you like?")
        return ("वह समय बुक हो चुका है, और आसपास कोई समय खाली नहीं है।" if hi
                else "That time is already booked, and nothing nearby is free.")
    if reason == "doctor_not_found":
        q = _query(slots, "doctor_name", lang)
        if hi:
            return (f"माफ़ कीजिए, '{q}' नाम के कोई डॉक्टर नहीं मिले।" if q
                    else "माफ़ कीजिए, इस नाम के कोई डॉक्टर नहीं मिले।")
        return (f"Sorry, I couldn't find a doctor named '{q}'." if q
                else "Sorry, I couldn't find a doctor by that name.")
    return ("माफ़ कीजिए, अपॉइंटमेंट बुक नहीं हो सका। थोड़ी देर बाद फिर कोशिश करें, या काउंटर पर संपर्क करें।"
            if hi else
            "Sorry, I couldn't book the appointment. Please try again shortly, or contact the counter.")
