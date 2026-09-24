"""Composes the spoken Bengali reply from TOOL DATA, never from the LLM's
own words, for any intent where a fact (a price, a date, a confirmation
ID) is at stake.

This is the same discipline voicerx/gate.py already applies to drug names
("the SLM proposes, the gazetteer decides") ported to this domain: the LLM
may decide WHAT the caller wants and WHICH slots it heard, but the actual
number in the caller's ear always comes from the Spring Boot response,
substituted into a fixed template. The model never gets a chance to
misremember or round a price it was merely shown a moment ago.

Only "smalltalk" skips this file entirely and uses the LLM's own
direct_reply_bn -- there is no fact to get wrong in "নমস্কার" or "ধন্যবাদ".
"""
from __future__ import annotations

from agent import reply_templates_i18n as _i18n


def _spoken_test_name(slots: dict, result: dict) -> str:
    """What the caller HEARS as the test's name.

    Order matters. The API's `test_name` is the catalogue's English label
    ("Uric Acid") and the Bengali TTS tokenizer drops Latin script
    outright, so putting it in a spoken sentence removes the name from the
    reply entirely -- the caller hears a price attached to nothing. Prefer
    the seeded Bengali alias; failing that, echo the caller's own words
    back, which is what a person at the counter would do anyway.
    """
    return (result.get("test_name_bn")
            or slots.get("test_name")
            or result.get("test_name")
            or "টেস্ট")


def _spoken_doctor_name(slots: dict, result: dict) -> str:
    """Same problem, same order. Aliases are seeded as surnames ("সেন"),
    so this adds the honorific the English label already carried."""
    alias = result.get("doctor_name_bn")
    if alias:
        return f"ডাঃ {alias}"
    return slots.get("doctor_name") or result.get("doctor_name") or "ডাক্তার"


def missing_slot_prompt(intent: str, missing: str, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.missing_slot_prompt(intent, missing, lang)
    prompts = {
        ("test_rate", "test_name"): "কোন টেস্টের রেট জানতে চান, একটু বলবেন?",
        ("doctor_availability", "doctor_name"): "কোন ডাক্তারের কথা জিজ্ঞেস করছেন?",
        ("book_appointment", "doctor_name"): "কোন ডাক্তারের সাথে অ্যাপয়েন্টমেন্ট করতে চান?",
        ("book_appointment", "date"): "কোন দিনের জন্য অ্যাপয়েন্টমেন্ট চাই?",
        ("book_appointment", "time_slot"): "কোন সময়ের জন্য অ্যাপয়েন্টমেন্ট চাই?",
        ("book_appointment", "patient_name"): "রোগীর নামটা বলবেন?",
        ("book_appointment", "phone"): "একটা ফোন নম্বর দেবেন, যাতে কনফার্মেশন পাঠাতে পারি?",
        ("test_prep", "test_name"): "কোন টেস্টের প্রস্তুতি জানতে চান, একটু বলবেন?",
        ("clinic_faq", "faq_topic"): "দুঃখিত, ঠিক বুঝতে পারলাম না। আর একটু বলবেন?",
        ("book_test", "test_names"): "কোন টেস্টগুলো বুক করতে চান, একটু বলবেন?",
        ("book_test", "date"): "কোন দিনের জন্য টেস্ট করাতে চান?",
        ("book_test", "patient_name"): "রোগীর নামটা বলবেন?",
        ("book_test", "phone"): "একটা ফোন নম্বর দেবেন, যাতে কনফার্মেশন পাঠাতে পারি?",
        ("reschedule_appointment", "confirmation_id"): "আপনার কনফার্মেশন নম্বরটা বা যে নম্বর থেকে বুক করেছিলেন সেটা বলবেন?",
        ("reschedule_appointment", "new_date"): "কোন দিনে নিয়ে যেতে চান?",
        ("reschedule_appointment", "new_time_slot"): "কোন সময়ে নিয়ে যেতে চান?",
        ("cancel_appointment", "confirmation_id"): "আপনার কনফার্মেশন নম্বরটা বা যে নম্বর থেকে বুক করেছিলেন সেটা বলবেন?",
        ("add_test_booking", "confirmation_id"): "আপনার কনফার্মেশন নম্বরটা বলবেন?",
        ("add_test_booking", "test_name"): "কোন টেস্টটা যোগ করতে চান?",
        ("lookup_booking", "phone"): "যে নম্বর থেকে বুক করেছিলেন সেটা বলবেন?",
    }
    return prompts.get((intent, missing), "দুঃখিত, একটু স্পষ্ট করে বলবেন?")


def insufficient_information_reply(lang: str = "bn") -> str:
    """KCD-442: a distinct THIRD outcome, neither "this does not exist"
    (KCD-443's not-found path) nor "the system is unreachable"
    (phrase("tool_failure", lang)) -- the turn itself was heard too
    unclearly (agent/confidence_gate.py) to trust running a lookup on
    what was extracted from it at all. Saying so plainly beats a
    confident answer about the wrong test."""
    if lang != "bn":
        return _i18n.insufficient_information_reply(lang)
    return "দুঃখিত, ঠিক শুনতে পাইনি। আপনি কি আবার একটু স্পষ্ট করে বলবেন?"


def test_rate_reply(slots: dict, result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.test_rate_reply(slots, result, lang)
    if not result.get("found"):
        suggestions = result.get("did_you_mean_bn") or result.get("did_you_mean") or []
        if result.get("ambiguous") and suggestions:
            # KCD-446: this test EXISTS -- several rows matched equally
            # well -- distinct from the not-found framing below, which
            # would tell the caller something untrue.
            return f"একাধিক টেস্ট পেলাম -- কোনটার কথা বলছেন: {' নাকি '.join(suggestions)}?"
        if suggestions:
            return (f"'{slots.get('test_name')}' নামে টেস্ট খুঁজে পাইনি। "
                     f"আপনি কি {', '.join(suggestions)} বলতে চাইছেন?")
        return f"দুঃখিত, '{slots.get('test_name')}' নামে কোনো টেস্ট আমাদের তালিকায় নেই।"

    rate = result["rate_inr"]
    name = _spoken_test_name(slots, result)
    sample = result.get("sample_type")
    hours = result.get("report_time_hours")
    reply = f"{name} টেস্টের রেট {rate} টাকা।"
    if sample:
        reply += f" স্যাম্পল: {sample}।"
    if hours:
        reply += f" রিপোর্ট {hours} ঘণ্টার মধ্যে পাবেন।"
    return reply


def doctor_availability_reply(slots: dict, result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.doctor_availability_reply(slots, result, lang)
    if not result.get("found"):
        suggestions = result.get("did_you_mean_bn") or result.get("did_you_mean") or []
        if result.get("ambiguous") and suggestions:
            # Doctor-side counterpart of KCD-446's test-ambiguity framing.
            return f"একাধিক ডাক্তার পেলাম -- কার কথা বলছেন, {' নাকি '.join(suggestions)}?"
        if suggestions:
            return (f"'{slots.get('doctor_name')}' নামে ডাক্তার খুঁজে পাইনি। "
                     f"আপনি কি {', '.join(suggestions)} বলতে চাইছেন?")
        return f"দুঃখিত, '{slots.get('doctor_name')}' নামে কোনো ডাক্তার আমাদের এখানে নেই।"

    name = _spoken_doctor_name(slots, result)
    if result.get("available"):
        hours = result.get("chamber_hours", "")
        date_txt = f" {result.get('date')} তারিখে" if result.get("date") else " আজ"
        return f"হ্যাঁ,{date_txt} {name} চেম্বারে থাকবেন। সময়: {hours}।"

    next_date = result.get("next_available_date")
    if next_date:
        return f"{name} ওই দিন বসবেন না। পরবর্তী উপলব্ধ দিন: {next_date}।"
    return f"{name} এখন কোনো নির্দিষ্ট দিন বসছেন না। আমাদের কাউন্টারে খোঁজ নিতে পারেন।"


def test_prep_reply(slots: dict, result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.test_prep_reply(slots, result, lang)
    if not result.get("found"):
        suggestions = result.get("did_you_mean_bn") or result.get("did_you_mean") or []
        if result.get("ambiguous") and suggestions:
            return f"একাধিক টেস্ট পেলাম -- কোনটার কথা বলছেন: {' নাকি '.join(suggestions)}?"
        if suggestions:
            return (f"'{slots.get('test_name')}' নামে টেস্ট খুঁজে পাইনি। "
                     f"আপনি কি {', '.join(suggestions)} বলতে চাইছেন?")
        return f"দুঃখিত, '{slots.get('test_name')}' নামে কোনো টেস্ট আমাদের তালিকায় নেই।"

    name = _spoken_test_name(slots, result)
    instructions = result.get("prep_instructions") or "এই টেস্টের জন্য বিশেষ কোনো প্রস্তুতির প্রয়োজন নেই।"
    return f"{name} টেস্টের জন্য: {instructions}"


def clinic_faq_reply(slots: dict, result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.clinic_faq_reply(slots, result, lang)
    if not result.get("found"):
        return "দুঃখিত, এই বিষয়ে এখন সঠিক তথ্য দিতে পারছি না। কাউন্টারে যোগাযোগ করুন।"
    return result.get("answer") or "দুঃখিত, এই বিষয়ে এখন সঠিক তথ্য দিতে পারছি না। কাউন্টারে যোগাযোগ করুন।"


def booking_reply(slots: dict, result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.booking_reply(slots, result, lang)
    if result.get("success"):
        return (f"আপনার অ্যাপয়েন্টমেন্ট কনফার্ম হয়েছে। "
                f"{_spoken_doctor_name(slots, result)}, {result['date']}, সময় {result['time_slot']}। "
                f"কনফার্মেশন নম্বর: {result['confirmation_id']}।")

    reason = result.get("reason")
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return f"ওই সময়টা বুক হয়ে গেছে। এই সময়গুলো ফাঁকা আছে: {', '.join(alts)}। কোনটা চান?"
        return "ওই সময়টা বুক হয়ে গেছে, এবং কাছাকাছি কোনো সময় ফাঁকা নেই।"
    if reason == "doctor_not_found":
        return f"দুঃখিত, '{slots.get('doctor_name')}' নামে কোনো ডাক্তার খুঁজে পেলাম না।"
    if reason == "doctor_not_available_that_day":
        return "দুঃখিত, ওই দিন ডাক্তার বসেন না। অন্য কোনো দিন বলবেন?"
    if reason == "invalid_slot":
        return "দুঃখিত, ওই সময়টা ঠিক নেই। চেম্বারের সময়ের মধ্যে একটা সময় বলবেন?"
    if reason == "date_in_past":
        # KCD-363: never silently rolled forward -- stated plainly, and
        # the caller is offered the same weekday next week.
        return "ওই তারিখটা তো চলে গেছে। আগামী সপ্তাহে ওই দিনের জন্য বুক করব?"
    if reason == "hold_expired":
        return "দুঃখিত, সময়টা ধরে রাখা যায়নি, একটু দেরি হয়ে গেছে। আবার চেষ্টা করি?"
    return "দুঃখিত, অ্যাপয়েন্টমেন্ট বুক করা গেল না। একটু পরে আবার চেষ্টা করুন, অথবা কাউন্টারে যোগাযোগ করুন।"


def booking_confirmation_readback(slots: dict, action: str, lang: str = "bn") -> str:
    """Read back what is about to be written BEFORE it is written -- the
    step KCD-367/369 depend on: a caller who spots a mistake here corrects
    it before anything is committed, not after."""
    if lang != "bn":
        return _i18n.booking_confirmation_readback(slots, action, lang)
    if action == "book_appointment":
        # KCD-448: read back the number the confirmation actually goes to
        # (contact_phone wins over phone, same precedence as
        # booking_flow.effective_phone) -- never the raw `phone` slot,
        # which is None whenever the caller gave a different contact
        # number or declined one, both of which would otherwise be read
        # back as the literal word "None". A declined phone has already
        # been announced separately (phrase("no_confirmation_number", lang)
        # in main.py, before this readback runs) so it is omitted here
        # rather than repeated.
        phone = slots.get("contact_phone") or slots.get("phone")
        phone_clause = f", ফোন নম্বর {phone} " if phone else " "
        return (f"তাহলে {slots.get('doctor_name')} ডাক্তারের কাছে {slots.get('date')} তারিখে, "
                f"সময় {slots.get('time_slot')}-এ, রোগীর নাম {slots.get('patient_name')}"
                f"{phone_clause}-- এই অ্যাপয়েন্টমেন্টটা কনফার্ম করব?")
    if action == "book_test":
        # ", " not the ideographic "、" -- a stray full-width character
        # from an earlier edit, inconsistent with every other list-join
        # in this file and in reply_templates_i18n.py.
        tests = ", ".join(slots.get("_test_names_display", [])) or "টেস্ট"
        phone = slots.get("contact_phone") or slots.get("phone")
        phone_clause = f", ফোন নম্বর {phone} " if phone else " "
        return (f"তাহলে {slots.get('date')} তারিখে {tests} -- রোগীর নাম {slots.get('patient_name')}"
                f"{phone_clause}-- এই বুকিংটা কনফার্ম করব?")
    if action == "reschedule_appointment":
        return f"তাহলে অ্যাপয়েন্টমেন্টটা {slots.get('new_date')} তারিখে, সময় {slots.get('new_time_slot')}-এ নিয়ে যাব?"
    if action == "cancel_appointment":
        return "আপনার অ্যাপয়েন্টমেন্টটা বাতিল করে দেব?"
    if action == "add_test_booking":
        return f"তাহলে আপনার বুকিং-এ {slots.get('test_name')} টেস্টটা যোগ করে দেব?"
    return "এটা কনফার্ম করব?"


def reschedule_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.reschedule_reply(result, lang)
    if result.get("success"):
        return (f"আপনার অ্যাপয়েন্টমেন্টটা {result['date']} তারিখে, সময় {result['time_slot']}-এ "
                f"পাল্টে দেওয়া হয়েছে। নতুন কনফার্মেশন নম্বর: {result['confirmation_id']}।")
    reason = result.get("reason")
    if reason == "not_found":
        return "দুঃখিত, এই কনফার্মেশন নম্বরে কোনো অ্যাপয়েন্টমেন্ট খুঁজে পেলাম না।"
    if reason == "slot_taken":
        alts = result.get("alternative_slots") or []
        if alts:
            return f"ওই সময়টা বুক হয়ে গেছে। এই সময়গুলো ফাঁকা আছে: {', '.join(alts)}। কোনটা চান?"
        return "ওই সময়টা বুক হয়ে গেছে। আপনার আগের অ্যাপয়েন্টমেন্টটা ঠিক আগের মতোই আছে।"
    return "দুঃখিত, অ্যাপয়েন্টমেন্টটা পাল্টানো গেল না। আপনার আগের অ্যাপয়েন্টমেন্টটা ঠিক আগের মতোই আছে।"


def cancel_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.cancel_reply(result, lang)
    reason = result.get("reason")
    if reason == "charge_confirmation_required":
        charge = result["charge_inr"]
        return (f"এই সময়ে বাতিল করলে {charge} টাকা কাটা যাবে। তাও কি বাতিল করব?")
    if result.get("success"):
        charge = result.get("charge_inr") or 0
        if charge:
            return f"আপনার অ্যাপয়েন্টমেন্টটা বাতিল করা হয়েছে। {charge} টাকা কাটা হয়েছে।"
        return "আপনার অ্যাপয়েন্টমেন্টটা কোনো চার্জ ছাড়াই বাতিল করা হয়েছে।"
    if reason == "not_found":
        return "দুঃখিত, এই কনফার্মেশন নম্বরে কোনো অ্যাপয়েন্টমেন্ট খুঁজে পেলাম না।"
    return "দুঃখিত, বাতিল করা গেল না। কাউন্টারে যোগাযোগ করুন।"


def lookup_reply(bookings: list[dict], lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.lookup_reply(bookings, lang)
    if not bookings:
        return "দুঃখিত, আপনার নামে কোনো আসন্ন অ্যাপয়েন্টমেন্ট খুঁজে পেলাম না।"
    b = bookings[0]
    return (f"আপনার পরবর্তী অ্যাপয়েন্টমেন্ট: {b.get('doctor_name')} ডাক্তারের কাছে, "
            f"{b['date']} তারিখে, সময় {b['time_slot']}-এ। কনফার্মেশন নম্বর: {b['confirmation_id']}। "
            f"লিখিত কনফার্মেশনটা আবার পাঠিয়ে দিতে পারি, চাইলে বলবেন।")


def multi_test_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.multi_test_reply(result, lang)
    if not result.get("success"):
        return "দুঃখিত, টেস্টগুলো বুক করা গেল না। একটু পরে আবার চেষ্টা করুন।"
    names = ", ".join(result["test_names"])
    reply = (f"{names} -- এই টেস্টগুলো {result['date']} তারিখে বুক করা হয়েছে। "
             f"মোট খরচ {result['total_rate_inr']} টাকা। কনফার্মেশন নম্বর {result['confirmation_id']}।")
    if result.get("combined_prep"):
        reply += f" প্রস্তুতি: {result['combined_prep']}"
    return reply


def add_test_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.add_test_reply(result, lang)
    if result.get("success"):
        return f"{result['test_name']} টেস্টটা আপনার {result['date']} তারিখের বুকিং-এ যোগ করা হয়েছে।"
    reason = result.get("reason")
    if reason == "not_found":
        return "দুঃখিত, এই কনফার্মেশন নম্বরে কোনো টেস্ট বুকিং খুঁজে পেলাম না।"
    if reason == "test_not_found":
        return "দুঃখিত, এই নামে কোনো টেস্ট আমাদের তালিকায় নেই।"
    if reason == "already_booked":
        return "এই টেস্টটা তো আগে থেকেই আপনার বুকিং-এ আছে।"
    return "দুঃখিত, টেস্টটা যোগ করা গেল না।"


def resend_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.resend_reply(result, lang)
    if result.get("success"):
        return f"কনফার্মেশনটা আবার পাঠিয়ে দেওয়া হয়েছে, নম্বর শেষ হচ্ছে {result['sent_to_last4']} দিয়ে।"
    reason = result.get("reason")
    if reason == "rate_limited":
        return "একটু আগেই পাঠানো হয়েছে। একটু অপেক্ষা করে আবার বলবেন।"
    if reason == "no_phone_on_file":
        # Distinct from "booking not found" (below) -- the booking DOES
        # exist, there is simply no number on file to send anything to,
        # since the caller declined to give one at booking time.
        return "দুঃখিত, এই বুকিং-এর জন্য কোনো ফোন নম্বর রাখা নেই, তাই পাঠাতে পারছি না।"
    return "দুঃখিত, এই কনফার্মেশন নম্বরে কোনো বুকিং খুঁজে পেলাম না।"


def department_route_reply(result: dict, symptom: str, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.department_route_reply(result, symptom, lang)
    if result.get("matched"):
        # Administrative routing only, never a diagnosis.
        return f"এই সমস্যার জন্য {result['department_name']} বিভাগে দেখানো ভালো হবে। ডাক্তারের অ্যাপয়েন্টমেন্ট করে দেব?"
    if result.get("ambiguous"):
        return f"এটা {' অথবা '.join(result['candidates'])} -- দুটোর যেকোনো একটা বিভাগ হতে পারে। কোনটা বলবেন?"
    return "দুঃখিত, ঠিক কোন বিভাগে দেখাবেন বুঝতে পারলাম না। কাউন্টারে জিজ্ঞেস করে নিতে পারেন।"


def conflict_reply(conflict: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.conflict_reply(conflict, lang)
    return (f"আপনার তো ওই সময়ে আগে থেকেই একটা অ্যাপয়েন্টমেন্ট আছে -- {conflict.get('doctor_name')} ডাক্তারের কাছে, "
            f"{conflict['date']} তারিখে, সময় {conflict['time_slot']}-এ। আগেরটা রাখব, পাল্টাব, নাকি নতুন করে যোগ করব?")


def earliest_available_reply(result: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.earliest_available_reply(result, lang)
    if not result.get("found"):
        return f"দুঃখিত, '{result.get('query')}' নামে কোনো ডাক্তার খুঁজে পেলাম না।"
    if not result.get("available"):
        return f"দুঃখিত, {result.get('doctor_name')} ডাক্তারের কাছাকাছি সময়ে কোনো ফাঁকা সময় নেই। পরে আবার ফোন করলে জানাতে পারব।"
    reply = f"সবচেয়ে তাড়াতাড়ি ফাঁকা আছে {result['date']} তারিখে, সময় {result['time_slot']}-এ।"
    alts = result.get("alternatives") or []
    if alts:
        reply += f" এছাড়াও: {', '.join(alts)}।"
    return reply


def draft_resume_reply(draft: dict, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.draft_resume_reply(draft, lang)
    return "গত বার কল কেটে গিয়েছিল, আপনার বুকিং শেষ হয়নি। যেখানে ছিলাম সেখান থেকে চালিয়ে যাব, নাকি নতুন করে শুরু করব?"


def multiple_bookings_reply(bookings: list[dict], lang: str = "bn") -> str:
    """KCD/CodeRabpit-flagged: resolving a reschedule/cancel/resend target
    by phone alone used to silently take bookings[0] with no
    disambiguation and no statement of which one -- a caller with
    several bookings (their own, or a proxy's) could have a "yes" act on
    the WRONG one. Lists up to three by doctor+date+time (same cap as
    KCD-446's near-match offer) and asks for the confirmation number,
    the same deterministic bearer-token identifier every other lookup
    path in this codebase already uses to pick exactly one booking."""
    if lang != "bn":
        return _i18n.multiple_bookings_reply(bookings, lang)
    parts = [f"{b.get('doctor_name')} ডাক্তারের {b['date']} তারিখের" for b in bookings[:3]]
    return (f"আপনার নামে একাধিক বুকিং আছে -- {', '.join(parts)}। "
            f"কোনটার কথা বলছেন, কনফার্মেশন নম্বরটা বলবেন?")


def spelling_prompt(lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.spelling_prompt(lang)
    return "নামটা একটু বানান করে বলবেন, এক একটা অক্ষর করে?"


def spelling_readback(spelled: str, lang: str = "bn") -> str:
    if lang != "bn":
        return _i18n.spelling_readback(spelled, lang)
    return f"আমি শুনলাম {spelled.upper()} -- ঠিক আছে?"
