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

from agent import slot_proposal
from agent.report_time_wording import report_time_phrase, report_time_safety_note
from agent.sample_wording import sample_sentence

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")

_HI_MISSING = {
    ("test_rate", "test_name"): "किस टेस्ट की कीमत जाननी है, ज़रा बताइए?",
    ("doctor_availability", "doctor_name"): "किन डॉक्टर के बारे में पूछ रहे हैं?",
    ("department_query", "symptom_description"): "किस समस्या के लिए डॉक्टर को दिखाना चाहते हैं, ज़रा बताइए?",
    ("book_appointment", "doctor_name"): "किस डॉक्टर के साथ अपॉइंटमेंट लेना चाहते हैं?",
    ("book_appointment", "date"): "किस दिन के लिए अपॉइंटमेंट चाहिए?",
    ("book_appointment", "time_slot"): "किस समय के लिए अपॉइंटमेंट चाहिए?",
    ("book_appointment", "patient_name"): "अपना नाम बताइए, प्लीज़?",
    ("book_appointment", "phone"): "एक फ़ोन नंबर दीजिए, ताकि मैं कन्फ़र्मेशन भेज सकूँ?",
    ("test_prep", "test_name"): "किस टेस्ट की तैयारी जाननी है, ज़रा बताइए?",
    ("clinic_faq", "faq_topic"): "माफ़ कीजिए, ठीक से समझ नहीं पाई। क्या आप फिर से बताएँगे?",
    ("book_test", "test_names"): "कौन से टेस्ट बुक करना चाहते हैं, ज़रा बताइए?",
    ("book_test", "date"): "किस दिन टेस्ट करवाना चाहते हैं?",
    ("book_test", "patient_name"): "मरीज़ का नाम बताइए?",
    ("book_test", "phone"): "एक फ़ोन नंबर दीजिए, ताकि मैं कन्फ़र्मेशन भेज सकूँ?",
    ("reschedule_appointment", "confirmation_id"): "आपका कन्फ़र्मेशन नंबर या जिस नंबर से बुक किया था वह बताइए?",
    ("reschedule_appointment", "new_date"): "किस दिन ले जाना चाहते हैं?",
    ("reschedule_appointment", "new_time_slot"): "किस समय ले जाना चाहते हैं?",
    ("cancel_appointment", "confirmation_id"): "आपका कन्फ़र्मेशन नंबर या जिस नंबर से बुक किया था वह बताइए?",
    ("add_test_booking", "confirmation_id"): "आपका कन्फ़र्मेशन नंबर बताइए?",
    ("add_test_booking", "test_name"): "कौन सा टेस्ट जोड़ना चाहते हैं?",
    ("lookup_booking", "phone"): "जिस नंबर से बुक किया था वह बताइए?",
}
_HI_MISSING_DEFAULT = "माफ़ कीजिए, क्या आप थोड़ा साफ़ बोलेंगे?"

_EN_MISSING = {
    ("test_rate", "test_name"): "Which test would you like the price of?",
    ("doctor_availability", "doctor_name"): "Which doctor are you asking about?",
    ("department_query", "symptom_description"): "What problem would you like a doctor for, could you tell me?",
    ("book_appointment", "doctor_name"): "Which doctor would you like to book with?",
    ("book_appointment", "date"): "Which day would you like the appointment for?",
    ("book_appointment", "time_slot"): "What time would you like the appointment?",
    ("book_appointment", "patient_name"): "Could you tell me your name, please?",
    ("book_appointment", "phone"): "Could you give me a phone number so I can send the confirmation?",
    ("test_prep", "test_name"): "Which test would you like the preparation for?",
    ("clinic_faq", "faq_topic"): "Sorry, I didn't quite get that. Could you say it again?",
    ("book_test", "test_names"): "Which tests would you like to book?",
    ("book_test", "date"): "Which day would you like the tests for?",
    ("book_test", "patient_name"): "May I have the patient's name?",
    ("book_test", "phone"): "Could you give me a phone number so I can send the confirmation?",
    (
        "reschedule_appointment",
        "confirmation_id",
    ): "Could you give me your confirmation number, or the number you booked from?",
    ("reschedule_appointment", "new_date"): "Which day would you like to move it to?",
    ("reschedule_appointment", "new_time_slot"): "Which time would you like to move it to?",
    (
        "cancel_appointment",
        "confirmation_id",
    ): "Could you give me your confirmation number, or the number you booked from?",
    ("add_test_booking", "confirmation_id"): "Could you give me your confirmation number?",
    ("add_test_booking", "test_name"): "Which test would you like to add?",
    ("lookup_booking", "phone"): "Could you give me the number you booked from?",
}
_EN_MISSING_DEFAULT = "Sorry, could you say that a little more clearly?"

# Sample types that are a specimen a person gives. "Imaging"/"Cardiac" are
# test categories, not samples -- "sample: imaging" would be nonsense.
_SPECIMENS = {"blood", "urine", "stool", "serum", "saliva", "swab", "plasma"}


def insufficient_information_reply(lang: str) -> str:
    if lang == "hi":
        return "माफ़ कीजिए, ठीक से सुन नहीं पाई। क्या आप फिर से थोड़ा साफ़ बोलेंगे?"
    return "Sorry, I didn't catch that clearly enough. Could you say it again?"


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
    scripted = result.get(f"did_you_mean_{lang}") or []
    if scripted:
        return list(scripted)
    return [s for s in (result.get("did_you_mean") or []) if _script_ok(s, lang)]


def _test_name(slots: dict, result: dict, lang: str) -> str:
    if lang == "hi":
        heard = slots.get("test_name") or ""
        return result.get("test_name_hi") or (heard if _DEVANAGARI.search(heard) else "") or "यह"
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
        sentence = sample_sentence(sample, "hi")
        if sentence:
            reply += f" {sentence}"
        time_sentence = report_time_phrase(hours, "hi")
        if time_sentence:
            reply += f" {time_sentence} {report_time_safety_note('hi')}"
        return reply
    reply = f"The price of the {name} test is {rate} rupees."
    sentence = sample_sentence(sample, "en")
    if sentence:
        reply += f" {sentence}"
    time_sentence = report_time_phrase(hours, "en")
    if time_sentence:
        reply += f" {time_sentence} {report_time_safety_note('en')}"
    return reply


def multi_test_rate_reply(results: list[tuple[str, dict]], lang: str) -> str:
    """Hindi/English counterpart of reply_templates.multi_test_rate_reply -- see its docstring for why this
    exists and the truth-boundary rule (a test not found is named and left OUT of the total, never
    guessed)."""
    hi = lang == "hi"
    priced, not_found, total = [], [], 0
    for requested_name, result in results:
        if result.get("found"):
            name = _test_name({"test_name": requested_name}, result, lang)
            rate = result["rate_inr"]
            total += rate
            priced.append(f"{name} टेस्ट की कीमत {rate} रुपये" if hi else f"the {name} test is {rate} rupees")
        else:
            not_found.append(requested_name)
    if not priced:
        return (
            "माफ़ कीजिए, इनमें से कोई भी टेस्ट हमारी लिस्ट में नहीं मिला। कोई और टेस्ट बताएँगे?"
            if hi
            else "Sorry, I could not find any of these tests in our list. Would you like to ask about a different one?"
        )
    if hi:
        reply = "; ".join(priced) + f"। सब मिलाकर टोटल {total} रुपये है।"
        if not_found:
            reply += f" '{', '.join(not_found)}' नाम का टेस्ट नहीं मिला, इसलिए यह टोटल में शामिल नहीं है।"
        return reply
    reply = "; ".join(priced) + f". The total comes to {total} rupees."
    if not_found:
        reply += f" I could not find '{', '.join(not_found)}', so that is not included in the total."
    return reply


def _not_found_test(slots: dict, result: dict, lang: str) -> str:
    q = _query(slots, "test_name", lang)
    sugg = _suggestions(result, lang)
    if result.get("ambiguous") and sugg:
        # KCD-446: exists, but matched more than one row equally well --
        # distinct framing from "not found" below, which would be false.
        return (
            f"एक से ज़्यादा टेस्ट मिले -- कौन सा: {' या '.join(sugg)}?"
            if lang == "hi"
            else f"I found more than one test -- which one did you mean: {' or '.join(sugg)}?"
        )
    # KCD-lay-terms: see reply_templates.test_rate_reply's Bengali counterpart for why this invitation is here.
    if lang == "hi":
        if sugg:
            return (
                f"'{q}' नाम का टेस्ट नहीं मिला। क्या आप {', '.join(sugg)} कहना चाह रहे हैं?"
                if q
                else f"वह टेस्ट नहीं मिला। क्या आप {', '.join(sugg)} कहना चाह रहे हैं?"
            )
        base = f"माफ़ कीजिए, '{q}' नाम का कोई टेस्ट हमारी सूची में नहीं है।" if q else "माफ़ कीजिए, वह टेस्ट हमारी सूची में नहीं है।"
        return f"{base} कोई और टेस्ट या डॉक्टर की अपॉइंटमेंट बुक करनी है?"
    if sugg:
        return (
            f"I couldn't find a test called '{q}'. Did you mean: {', '.join(sugg)}?"
            if q
            else f"I couldn't find that test. Did you mean: {', '.join(sugg)}?"
        )
    base = f"Sorry, there is no test called '{q}' on our list." if q else "Sorry, that test is not on our list."
    return f"{base} Is there any other test or doctor's appointment you'd like to book?"


def _not_found_doctor(slots: dict, result: dict, lang: str) -> str:
    hi = lang == "hi"
    q = _query(slots, "doctor_name", lang)
    sugg = _suggestions(result, lang)
    if result.get("ambiguous") and sugg:
        # Doctor-side counterpart of test_rate_reply's ambiguous framing.
        return (
            f"एक से ज़्यादा डॉक्टर मिले -- किनकी बात कर रहे हैं, {' या '.join(sugg)}?"
            if hi
            else f"I found more than one doctor -- did you mean {' or '.join(sugg)}?"
        )
    # KCD-lay-terms: see reply_templates.doctor_availability_reply's Bengali counterpart for why this invitation
    # is here, and why it needs no new state -- department_query already handles the caller's next turn.
    if hi:
        if sugg:
            return (
                f"'{q}' नाम के डॉक्टर नहीं मिले। क्या आप {', '.join(sugg)} कहना चाह रहे हैं?"
                if q
                else f"वह डॉक्टर नहीं मिले। क्या आप {', '.join(sugg)} कहना चाह रहे हैं?"
            )
        base = (
            f"माफ़ कीजिए, '{q}' नाम के कोई डॉक्टर हमारे यहाँ नहीं हैं।"
            if q
            else "माफ़ कीजिए, इस नाम के कोई डॉक्टर हमारे यहाँ नहीं हैं।"
        )
        return f"{base} आपको क्या तकलीफ़ है बताइए, तो मैं सही विभाग के डॉक्टर बता सकूँगी।"
    if sugg:
        return (
            f"I couldn't find a doctor named '{q}'. Did you mean {', '.join(sugg)}?"
            if q
            else f"I couldn't find that doctor. Did you mean {', '.join(sugg)}?"
        )
    base = f"Sorry, we don't have a doctor named '{q}'." if q else "Sorry, we don't have a doctor by that name."
    return f"{base} Tell me what's troubling you, and I can suggest the right doctor."


def doctor_availability_reply(slots: dict, result: dict, lang: str) -> str:
    hi = lang == "hi"
    if not result.get("found"):
        return _not_found_doctor(slots, result, lang)

    name = _doctor_name(slots, result, lang)
    if result.get("available"):
        hours = result.get("chamber_hours", "")
        date = result.get("date")
        if hi:
            return f"जी हाँ, {name} {slot_proposal.when_phrase(date, 'hi')} बैठेंगे। समय: {hours}।"
        return f"Yes, {name} sits {slot_proposal.when_phrase(date, 'en')}. Timing: {hours}."

    next_date = result.get("next_available_date")
    if next_date:
        if hi:
            return f"{name} उस दिन नहीं बैठेंगे। वे {slot_proposal.when_phrase(next_date, 'hi')} बैठेंगे।"
        return f"{name} does not sit on that day. The next sitting is {slot_proposal.when_phrase(next_date, 'en')}."
    if hi:
        return f"{name} अभी किसी तय दिन नहीं बैठ रहे हैं। आप हमारे काउंटर पर पूछ सकते हैं।"
    return f"{name} is not sitting on any fixed day right now. You can ask at our counter."


def booking_doctor_sitting_hint(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if not result.get("found") or not result.get("available"):
        return "किस दिन के लिए अपॉइंटमेंट चाहिए?" if hi else "Which day would you like the appointment for?"
    name = _doctor_name({}, result, lang)
    date, hours = result.get("date"), result.get("chamber_hours", "")
    if hi:
        return f"{name} {slot_proposal.when_phrase(date, 'hi')} बैठेंगे, {hours} के बीच। उसी दिन अपॉइंटमेंट कर दूँ?"
    return f"{name} sits {slot_proposal.when_phrase(date, 'en')}, between {hours}. Shall I book that day?"


def test_prep_reply(slots: dict, result: dict, lang: str) -> str:
    if not result.get("found"):
        return _not_found_test(slots, result, lang)
    name = _test_name(slots, result, lang)
    # The lab-test table decides: its preparation text if it has one; else its fasting_required column.
    if result.get("fasting_required"):
        default = (
            "इस टेस्ट के लिए खाली पेट रहना होगा। कितने घंटे, यह काउंटर पर पूछ लें।"
            if lang == "hi"
            else "Fasting is required for this test. Please ask the counter how many hours."
        )
    else:
        default = (
            "इस टेस्ट के लिए किसी ख़ास तैयारी की ज़रूरत नहीं है।"
            if lang == "hi"
            else "No special preparation is needed for this test."
        )
    instructions = result.get("prep_instructions") or default
    return f"{name} टेस्ट के लिए: {instructions}" if lang == "hi" else f"For the {name} test: {instructions}"


def clinic_faq_reply(slots: dict, result: dict, lang: str) -> str:
    fallback = (
        "माफ़ कीजिए, इस बारे में अभी सही जानकारी नहीं दे सकती। कृपया काउंटर पर संपर्क करें।"
        if lang == "hi"
        else "Sorry, I can't give you accurate information on that right now. Please contact the counter."
    )
    if not result.get("found"):
        return fallback
    return result.get("answer") or fallback


def booking_reply(slots: dict, result: dict, lang: str) -> str:
    hi = lang == "hi"
    if result.get("success"):
        return "धन्यवाद।" if hi else "Thank you."  # the day, hour and "come with one to two hours in hand" were the question just answered with a yes

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return (
                f"वह समय बुक हो चुका है। ये समय खाली हैं: {', '.join(alts)}। कौन सा चाहिए?"
                if hi
                else f"That time is already booked. These times are free: {', '.join(alts)}. Which one would you like?"
            )
        return (
            "वह समय बुक हो चुका है, और आसपास कोई समय खाली नहीं है।"
            if hi
            else "That time is already booked, and nothing nearby is free."
        )
    if reason == "doctor_ambiguous":
        names = result.get("did_you_mean_hi" if hi else "did_you_mean") or result.get("did_you_mean") or []
        if hi:
            return (
                f"एक से ज़्यादा डॉक्टर मिले -- आप किसके बारे में पूछ रहे हैं, {' या '.join(names)}?"
                if names
                else "एक से ज़्यादा डॉक्टर मिले। आप किस डॉक्टर के बारे में पूछ रहे हैं, पूरा नाम बताएँगे?"
            )
        return (
            f"I found more than one doctor -- which do you mean, {' or '.join(names)}?"
            if names
            else "I found more than one doctor. Which one do you mean? Please say the full name."
        )
    if reason == "doctor_not_found":
        q = _query(slots, "doctor_name", lang)
        if hi:
            return f"माफ़ कीजिए, '{q}' नाम के कोई डॉक्टर नहीं मिले।" if q else "माफ़ कीजिए, इस नाम के कोई डॉक्टर नहीं मिले।"
        return f"Sorry, I couldn't find a doctor named '{q}'." if q else "Sorry, I couldn't find a doctor by that name."
    if reason == "doctor_not_available_that_day":
        return (
            "माफ़ कीजिए, उस दिन डॉक्टर नहीं बैठते। कोई और दिन बताएँगे?"
            if hi
            else "Sorry, the doctor doesn't sit that day. Could you say another day?"
        )
    if reason == "invalid_slot":
        # sourav, 2026-10-16: say the real chamber hours (clinic's own valid_slots) so the caller can pick a time that works.
        from agent.booking_capture import hours_span

        span = hours_span(result.get("valid_slots"))
        if span:
            return (
                f"माफ़ कीजिए, डॉक्टर {span[0]} से {span[1]} तक बैठते हैं। इसके भीतर कोई समय बताएँगे?"
                if hi
                else f"Sorry, the doctor sits from {span[0]} to {span[1]}. Could you give a time within those hours?"
            )
        return (
            "माफ़ कीजिए, वह समय ठीक नहीं है। चेंबर के समय के भीतर कोई समय बताएँगे?"
            if hi
            else "Sorry, that time isn't valid. Could you give a time within the chamber hours?"
        )
    if reason == "date_in_past":
        return (
            "वह तारीख तो निकल चुकी है। अगले हफ़्ते उसी दिन के लिए बुक कर दूँ?"
            if hi
            else "That date has already passed. Shall I book the same weekday next week instead?"
        )
    if reason == "too_far_ahead":
        from agent.reply_templates import too_far_ahead_reply

        return too_far_ahead_reply(result, lang)
    if reason == "hold_expired":
        return (
            "माफ़ कीजिए, समय रोका नहीं जा सका, थोड़ी देर हो गई। फिर कोशिश करें?"
            if hi
            else "Sorry, that slot couldn't be held in time. Shall we try again?"
        )
    return (
        "माफ़ कीजिए, अपॉइंटमेंट बुक नहीं हो सका। थोड़ी देर बाद फिर कोशिश करें, या काउंटर पर संपर्क करें।"
        if hi
        else "Sorry, I couldn't book the appointment. Please try again shortly, or contact the counter."
    )


def _shown(slots: dict, held: str | None = None) -> str:
    """The hour the doctor sits, unless the caller asked for a time themselves (agent.reply_templates.shown_time)."""
    if slots.get("_time_explicit"):
        return slots.get("time_slot") or held or ""
    return slots.get("_sits_start") or slots.get("time_slot") or held or ""


def booking_confirmation_readback(slots: dict, action: str, lang: str) -> str:
    hi = lang == "hi"
    if action == "book_appointment":
        heard = ""
        if slots.get("patient_address"):
            heard = f"पता {slots['patient_address']} सुना। " if hi else f"I have the address as {slots['patient_address']}. "
        return heard + slot_proposal.arrival_question(slots.get("date"), _shown(slots), "hi" if hi else "en")
    if action == "book_test":
        tests = ", ".join(slots.get("_test_names_display", [])) or ("टेस्ट" if hi else "the tests")
        if hi:
            return (
                f"तो {slots.get('date')} को {tests} -- मरीज़ का नाम {slots.get('patient_name')}, "
                f"फ़ोन नंबर {slots.get('phone')} -- यह बुकिंग कन्फ़र्म कर दूँ?"
            )
        return (
            f"So {tests} on {slots.get('date')}, for {slots.get('patient_name')}, "
            f"phone number {slots.get('phone')} -- shall I confirm this booking?"
        )
    if action == "reschedule_appointment":
        return (
            f"तो अपॉइंटमेंट {slots.get('new_date')} को, समय {slots.get('new_time_slot')} पर ले जाऊँ?"
            if hi
            else f"So I'll move the appointment to {slots.get('new_date')} at {slots.get('new_time_slot')} -- confirm?"
        )
    if action == "cancel_appointment":
        return "आपकी अपॉइंटमेंट रद्द कर दूँ?" if hi else "Shall I cancel your appointment?"
    if action == "add_test_booking":
        return (
            f"तो आपकी बुकिंग में {slots.get('test_name')} टेस्ट जोड़ दूँ?"
            if hi
            else f"Shall I add the {slots.get('test_name')} test to your booking?"
        )
    return "क्या यह कन्फ़र्म कर दूँ?" if hi else "Shall I confirm this?"


def reschedule_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if result.get("success"):
        if hi:
            return (
                f"आपकी अपॉइंटमेंट {result['date']} को, समय {result['time_slot']} पर बदल दी गई है। "
                f"नया कन्फ़र्मेशन नंबर: {result['confirmation_id']}।"
            )
        return (
            f"Your appointment has been moved to {result['date']} at {result['time_slot']}. "
            f"New confirmation number: {result['confirmation_id']}."
        )
    reason = result.get("reason")
    if reason == "too_far_ahead":
        from agent.reply_templates import too_far_ahead_reply

        return too_far_ahead_reply(result, lang, rescheduling=True)
    if reason == "not_found":
        return (
            "माफ़ कीजिए, इस कन्फ़र्मेशन नंबर पर कोई अपॉइंटमेंट नहीं मिली।"
            if hi
            else "Sorry, I couldn't find an appointment with that confirmation number."
        )
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return (
                f"वह समय बुक हो चुका है। ये समय खाली हैं: {', '.join(alts)}। कौन सा चाहिए?"
                if hi
                else f"That time is already booked. These times are free: {', '.join(alts)}. Which one would you like?"
            )
        return (
            "वह समय बुक हो चुका है। आपकी पुरानी अपॉइंटमेंट वैसी ही है।"
            if hi
            else "That time is already booked. Your original appointment is unchanged."
        )
    return (
        "माफ़ कीजिए, अपॉइंटमेंट बदली नहीं जा सकी। आपकी पुरानी अपॉइंटमेंट वैसी ही है।"
        if hi
        else "Sorry, I couldn't move the appointment. Your original appointment is unchanged."
    )


def cancel_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    reason = result.get("reason")
    if reason == "charge_confirmation_required":
        charge = result["charge_inr"]
        return (
            f"इस समय रद्द करने पर {charge} रुपये कटेंगे। फिर भी रद्द कर दूँ?"
            if hi
            else f"Cancelling now means a charge of {charge} rupees. Shall I still cancel it?"
        )
    if result.get("success"):
        charge = result.get("charge_inr") or 0
        if charge:
            return (
                f"आपकी अपॉइंटमेंट रद्द कर दी गई है। {charge} रुपये काटे गए हैं।"
                if hi
                else f"Your appointment has been cancelled. A charge of {charge} rupees has been applied."
            )
        return (
            "आपकी अपॉइंटमेंट बिना किसी शुल्क के रद्द कर दी गई है।"
            if hi
            else "Your appointment has been cancelled with no charge."
        )
    if reason == "not_found":
        return (
            "माफ़ कीजिए, इस कन्फ़र्मेशन नंबर पर कोई अपॉइंटमेंट नहीं मिली।"
            if hi
            else "Sorry, I couldn't find an appointment with that confirmation number."
        )
    return (
        "माफ़ कीजिए, रद्द नहीं हो सकी। काउंटर से संपर्क करें।"
        if hi
        else "Sorry, I couldn't cancel it. Please contact the counter."
    )


def lookup_reply(bookings: list[dict], lang: str) -> str:
    hi = lang == "hi"
    if not bookings:
        return (
            "माफ़ कीजिए, आपके नाम कोई आने वाली अपॉइंटमेंट नहीं मिली।"
            if hi
            else "Sorry, I couldn't find any upcoming appointment for you."
        )
    b = bookings[0]
    if hi:
        return (
            f"आपकी अगली अपॉइंटमेंट: {b.get('doctor_name')} डॉक्टर के पास, {b['date']} को, "
            f"समय {b['time_slot']} पर। कन्फ़र्मेशन नंबर: {b['confirmation_id']}। "
            f"चाहें तो लिखित कन्फ़र्मेशन फिर से भेज सकती हूँ।"
        )
    return (
        f"Your next appointment: with {b.get('doctor_name')}, on {b['date']} at {b['time_slot']}. "
        f"Confirmation number: {b['confirmation_id']}. I can resend the written confirmation if you'd like."
    )


def multiple_bookings_reply(bookings: list[dict], lang: str) -> str:
    hi = lang == "hi"
    parts = (
        [f"{b.get('doctor_name')} डॉक्टर के पास {b['date']} की" for b in bookings[:3]]
        if hi
        else [f"with {b.get('doctor_name')} on {b['date']}" for b in bookings[:3]]
    )
    if hi:
        return f"आपके नाम कई बुकिंग हैं -- {', '.join(parts)}। कौन सी बुकिंग, कन्फ़र्मेशन नंबर बताइए?"
    return (
        f"You have more than one booking -- {', '.join(parts)}. Which one -- could you give me the confirmation number?"
    )


def multi_test_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if not result.get("success"):
        return (
            "माफ़ कीजिए, टेस्ट बुक नहीं हो सके। थोड़ी देर बाद फिर कोशिश करें।"
            if hi
            else "Sorry, I couldn't book the tests. Please try again shortly."
        )
    names = ", ".join(result["test_names"])
    if hi:
        reply = (
            f"{names} -- ये टेस्ट {result['date']} को बुक कर दिए गए हैं। "
            f"कुल खर्च {result['total_rate_inr']} रुपये। कन्फ़र्मेशन नंबर: {result['confirmation_id']}।"
        )
        if result.get("combined_prep"):
            reply += f" तैयारी: {result['combined_prep']}"
        return reply
    reply = (
        f"{names} have been booked for {result['date']}. Total cost {result['total_rate_inr']} rupees. "
        f"Confirmation number: {result['confirmation_id']}."
    )
    if result.get("combined_prep"):
        reply += f" Preparation: {result['combined_prep']}"
    return reply


def add_test_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if result.get("success"):
        return (
            f"{result['test_name']} टेस्ट आपकी {result['date']} की बुकिंग में जोड़ दिया गया है।"
            if hi
            else f"The {result['test_name']} test has been added to your booking for {result['date']}."
        )
    reason = result.get("reason")
    if reason == "not_found":
        return (
            "माफ़ कीजिए, इस कन्फ़र्मेशन नंबर पर कोई टेस्ट बुकिंग नहीं मिली।"
            if hi
            else "Sorry, I couldn't find a test booking with that confirmation number."
        )
    if reason == "test_not_found":
        return "माफ़ कीजिए, यह टेस्ट हमारी सूची में नहीं है।" if hi else "Sorry, that test is not on our list."
    if reason == "already_booked":
        return "यह टेस्ट तो पहले से ही आपकी बुकिंग में है।" if hi else "That test is already part of your booking."
    return "माफ़ कीजिए, टेस्ट जोड़ा नहीं जा सका।" if hi else "Sorry, I couldn't add that test."


def resend_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if result.get("success"):
        return (
            f"कन्फ़र्मेशन दोबारा भेज दिया गया है, नंबर खत्म होता है {result['sent_to_last4']} से।"
            if hi
            else f"The confirmation has been resent, to the number ending in {result['sent_to_last4']}."
        )
    if result.get("reason") == "rate_limited":
        return (
            "अभी थोड़ी देर पहले ही भेजा गया था। थोड़ी देर बाद फिर कहिए।"
            if hi
            else "It was already sent a moment ago. Please ask again shortly."
        )
    if result.get("reason") == "no_phone_on_file":
        return (
            "माफ़ कीजिए, इस बुकिंग के लिए कोई फ़ोन नंबर दर्ज नहीं है, इसलिए भेज नहीं पा रही।"
            if hi
            else "Sorry, there's no phone number on file for this booking, so I can't send it."
        )
    return (
        "माफ़ कीजिए, इस कन्फ़र्मेशन नंबर पर कोई बुकिंग नहीं मिली।"
        if hi
        else "Sorry, I couldn't find a booking with that confirmation number."
    )


_NAMES_PER_GROUP = 5


def blood_test_list_reply(spoken_names: list[str], lang: str) -> str:
    """`spoken_names` are already the short codes (reply_templates.blood_test_list_reply calls
    agent.catalogue_forms.lab_test_code before delegating here).

    Grouped into short runs, each ending in a sentence break -- same fix, same reason, as the Bengali version:
    one long comma-separated run with no internal sentence break became one over-long TTS clause on a live call
    (2026-09-28) and was never actually spoken."""
    hi = lang == "hi"
    if not spoken_names:
        return (
            "माफ़ कीजिए, अभी हमारी सूची में कोई ब्लड टेस्ट नहीं है। काउंटर पर पूछ सकते हैं।"
            if hi
            else "Sorry, we don't have any blood tests listed right now. You could ask at the counter."
        )
    groups = [spoken_names[i : i + _NAMES_PER_GROUP] for i in range(0, len(spoken_names), _NAMES_PER_GROUP)]
    if hi:
        listed = "। ".join(", ".join(g) for g in groups)
        return f"हमारे यहाँ ये ब्लड टेस्ट होते हैं -- {listed}। इनमें से कौन सा करवाना है?"
    listed = ". ".join(", ".join(g) for g in groups)
    return f"These are the blood tests we do -- {listed}. Which one would you like?"


_CATEGORY_LABEL = {
    "urine": ("यूरिन", "urine"),
    "imaging": ("स्कैन/एक्स-रे/अल्ट्रासाउंड जैसे", "scan/X-ray/ultrasound type"),
}


def category_test_list_reply(spoken_names: list[str], category: str, lang: str) -> str:
    """Generalises blood_test_list_reply above to any sample_type -- see
    agent.reply_templates.category_test_list_reply's own docstring for why this exists as a sibling rather
    than a rewrite of that one."""
    hi = lang == "hi"
    hi_label, en_label = _CATEGORY_LABEL.get(category, ("इस तरह के", "this kind of"))
    if not spoken_names:
        return (
            f"माफ़ कीजिए, अभी हमारी सूची में कोई {hi_label} टेस्ट नहीं है। काउंटर पर पूछ सकते हैं।"
            if hi
            else f"Sorry, we don't have any {en_label} tests listed right now. You could ask at the counter."
        )
    groups = [spoken_names[i : i + _NAMES_PER_GROUP] for i in range(0, len(spoken_names), _NAMES_PER_GROUP)]
    if hi:
        listed = "। ".join(", ".join(g) for g in groups)
        return f"हमारे यहाँ ये {hi_label} टेस्ट होते हैं -- {listed}। इनमें से कौन सा करवाना है?"
    listed = ". ".join(", ".join(g) for g in groups)
    return f"These are the {en_label} tests we do -- {listed}. Which one would you like?"


def department_route_reply(result: dict, symptom: str, lang: str) -> str:
    hi = lang == "hi"
    if result.get("matched"):
        # Owner, 2026-10-17: who is in the department TODAY, and the caller chooses. No "it would be best to see ..." advice.
        doctors, today = result.get("doctors") or [], result.get("doctors_today") or []
        dept = result["department_name"]
        if today:
            if hi:
                return f"आज {dept} विभाग में {len(today)} डॉक्टर हैं: {', '.join(today)}। आप किसे दिखाना चाहेंगे?"
            return f"Today the {dept} department has {len(today)} doctor{'s' if len(today) != 1 else ''}: {', '.join(today)}. Which one would you like to see?"
        if doctors:
            if hi:
                return f"{dept} विभाग में {', '.join(doctors)} हैं, पर आज कोई नहीं बैठ रहा। किसके बारे में जानना चाहेंगे?"
            return f"The {dept} department has {', '.join(doctors)}, but none of them sits today. Whose timings would you like?"
        if hi:
            return f"{dept} विभाग के डॉक्टरों की सूची अभी मेरे पास नहीं है। काउंटर पर पूछ सकते हैं।"
        return f"I do not have the {dept} department's doctors listed right now. You can ask at the counter."
    if result.get("ambiguous"):
        by_dept = result.get("doctors_by_department") or {}

        def _with_doctors(name: str) -> str:
            docs = by_dept.get(name) or []
            return f"{name} ({', '.join(docs)})" if docs else name

        parts = [_with_doctors(n) for n in result["candidates"]]
        return (
            f"यह {' या '.join(parts)} -- इनमें से कोई एक विभाग हो सकता है। कौन सा बताएँगे?"
            if hi
            else f"This could be either {' or '.join(parts)}. Which would you like?"
        )
    return (
        "माफ़ कीजिए, ठीक से समझ नहीं पाई किस विभाग में दिखाना होगा। काउंटर पर पूछ सकते हैं।"
        if hi
        else "Sorry, I couldn't quite tell which department this needs. You could ask at the counter."
    )


def booking_department_suggestion_reply(department_name: str, doctors: list[str], lang: str) -> str:
    hi = lang == "hi"
    who = (f" हमारे यहाँ {', '.join(doctors)} हैं।" if hi else f" We have {', '.join(doctors)} here.") if doctors else ""
    base = (
        f"तो इसके लिए {department_name} विभाग में दिखाना ठीक रहेगा।"
        if hi
        else f"Then it would be best to see the {department_name} department for this."
    )
    return base + who


def booking_general_checkup_first_reply(doctors: list[str], lang: str) -> str:
    hi = lang == "hi"
    who = (f" {doctors[0]} हैं।" if hi else f" {doctors[0]} is available.") if doctors else ""
    if hi:
        return (
            "ठीक से समझ नहीं आया कि कौन सी समस्या है, इसलिए पहले जनरल मेडिसिन में एक चेकअप करा लेना अच्छा रहेगा।"
            + who
            + " ज़रूरत पड़ने पर वो आगे किसी विशेषज्ञ डॉक्टर के पास भेज देंगे।"
        )
    return (
        "It isn't quite clear what the problem is, so it would be best to first get a general checkup done."
        + who
        + " They will refer you to a specialist if needed."
    )


def conflict_reply(conflict: dict, lang: str) -> str:
    hi = lang == "hi"
    if hi:
        return (
            f"उस समय आपकी पहले से एक अपॉइंटमेंट है -- {conflict.get('doctor_name')} डॉक्टर के पास, "
            f"{conflict['date']} को, समय {conflict['time_slot']} पर। पुरानी रखूँ, बदलूँ, या नई जोड़ दूँ?"
        )
    return (
        f"You already have an appointment at that time -- with {conflict.get('doctor_name')}, "
        f"on {conflict['date']} at {conflict['time_slot']}. Shall I keep it, move it, or add a new one?"
    )


def earliest_available_reply(result: dict, lang: str) -> str:
    hi = lang == "hi"
    if not result.get("found"):
        q = result.get("query")
        return f"माफ़ कीजिए, '{q}' नाम के कोई डॉक्टर नहीं मिले।" if hi else f"Sorry, I couldn't find a doctor named '{q}'."
    if not result.get("available"):
        return (
            f"माफ़ कीजिए, {result.get('doctor_name')} डॉक्टर के पास नज़दीकी समय में कोई जगह खाली नहीं है। "
            f"बाद में फिर फ़ोन करें तो बता पाऊँगी।"
            if hi
            else f"Sorry, {result.get('doctor_name')} has nothing free in the near term. "
            f"Please call again later and I can check for you."
        )
    if hi:
        reply = f"सबसे जल्दी खाली है {result['date']} को, समय {result['time_slot']} पर।"
        alts = result.get("alternatives") or []
        if alts:
            reply += f" इसके अलावा: {', '.join(alts)}।"
        return reply
    reply = f"The earliest opening is on {result['date']} at {result['time_slot']}."
    alts = result.get("alternatives") or []
    if alts:
        reply += f" Also available: {', '.join(alts)}."
    return reply


def draft_resume_reply(draft: dict, lang: str) -> str:
    hi = lang == "hi"
    return (
        "पिछली बार कॉल कट गई थी, आपकी बुकिंग पूरी नहीं हुई थी। जहाँ से छोड़ा था वहीं से जारी रखूँ, या नए सिरे से शुरू करूँ?"
        if hi
        else "Your call dropped last time before the booking finished. Shall I continue where we left off, or start fresh?"
    )


def spelling_prompt(lang: str) -> str:
    hi = lang == "hi"
    return "नाम एक-एक अक्षर करके बताएँगे?" if hi else "Could you spell the name out for me, letter by letter?"


def spelling_readback(spelled: str, lang: str) -> str:
    hi = lang == "hi"
    return f"मैंने सुना {spelled.upper()} -- क्या यह ठीक है?" if hi else f"I heard {spelled.upper()} -- is that right?"
