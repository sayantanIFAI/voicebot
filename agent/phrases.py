"""Fixed, non-factual phrases the agent says, per language.

Everything here is a sentence with no fact in it -- a greeting, an apology,
a hand-off notice. Facts (prices, times, instructions) never live in this
file; they are template substitutions from a live API response
(agent/reply_templates.py).

Kept as one table so the set of things the agent can say on a fixed path
exists in exactly one place: the same strings are spoken, pre-synthesized
at startup (TTSClient.prewarm) and pinned by tests/test_i18n_phrases.py to
be identical across languages in coverage.
"""
from __future__ import annotations

PHRASES: dict[str, dict[str, str]] = {
    "bn": {
        "greeting": "নমস্কার। বলুন, কীভাবে সাহায্য করতে পারি?",
        "asr_empty": "দুঃখিত। আমি শুনতে পাইনি। আবার বলবেন?",
        "unclear": "দুঃখিত। আমি বুঝতে পারিনি। আবার বলবেন?",
        "llm_failure": "একটু সমস্যা হচ্ছে, একটু ধরুন।",
        "tool_failure": "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
        "idle_close": "লাইনে কোনো সাড়া পাচ্ছি না, কল শেষ করছি। ধন্যবাদ।",
        "smalltalk_default": "নমস্কার, কী সাহায্য করতে পারি?",
        "handoff": "আমি আপনাকে আমাদের স্টাফের সাথে যুক্ত করছি, দয়া করে লাইনে থাকুন।",
        "cancel_aborted": "ঠিক আছে, আপনার অ্যাপয়েন্টমেন্টটা যেমন ছিল তেমনই আছে।",
        "no_confirmation_number": "ঠিক আছে, তাহলে বুকিংটা হয়ে যাবে, কিন্তু কোনো লিখিত কনফার্মেশন পাঠাতে পারব না।",
        "language_switched": "ঠিক আছে, এখন থেকে বাংলায় বলছি।",
        # KCD-461: spoken only when a lookup exceeds FILLER_THRESHOLD_S
        # (main.py's _await_with_filler) -- pre-warmed and cached like
        # every other line here, so it costs no synthesis time at the
        # exact moment the system is already running slow. Wording per
        # explicit request: the short, informal "আচ্ছা ঠিক আছে" ("okay,
        # alright") rather than a longer "please hold" phrase -- closer to
        # what a person at the counter actually says while checking.
        "please_wait": "আচ্ছা, ঠিক আছে।",
        # KCD-486: spoken instead of a confirmation number when the write
        # could not be verified in the system of record -- never a false
        # confirmation, and never the generic tool_failure apology either,
        # since the write likely DID happen and a retry could double-book.
        "booking_hold_for_verification": "আপনার বুকিংটা প্রসেস হচ্ছে, একটু দেখে নিশ্চিত করছি।",
        # Empathetic re-asks, spoken BEFORE any hand-off when the caller could
        # not be heard or understood (agent/reask_policy.py). The wording
        # takes the blame ("that's on me"), never the caller's -- a person who
        # is faint, mumbling or in a noisy room is usually already
        # embarrassed, and being told they are the problem makes it worse.
        # One question each, short sentences: they must satisfy the senior
        # policy (agent/speech_policy.py) too. Wording pending clinical-lead
        # review, like every acknowledgement in this codebase.
        "reask_low_volume": "দুঃখিত। আপনার গলা একটু আস্তে শোনাচ্ছে। একটু জোরে বলবেন, প্লিজ?",
        "reask_noisy": "দুঃখিত। লাইনে একটু শব্দ আছে। শান্ত জায়গা থেকে আবার বলবেন?",
        "reask_crosstalk": "দুঃখিত। পাশে আরও কারও গলা শোনা যাচ্ছে। আপনি আবার বলবেন?",
        "reask_mumbled": "দুঃখিত। আমি ঠিক বুঝতে পারিনি। কোনও তাড়া নেই, একটু ধীরে বলবেন?",
        "reask_generic": "দুঃখিত। আমি ঠিক শুনতে পাইনি। আবার বলবেন?",
        "reask_final": "দুঃখিত। আমি আপনাকে বুঝতে পারলাম না।",
    },
    "hi": {
        "greeting": "नमस्कार। बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
        "asr_empty": "माफ़ कीजिए। मैं सुन नहीं पाई। क्या आप फिर बोलेंगे?",
        "unclear": "माफ़ कीजिए। मैं समझ नहीं पाई। क्या आप फिर बताएँगे?",
        "llm_failure": "थोड़ी दिक्कत आ रही है, कृपया एक पल रुकिए।",
        "tool_failure": "अभी मैं यह देख नहीं पा रही हूँ। कृपया काउंटर पर संपर्क करें।",
        "idle_close": "लाइन पर कोई आवाज़ नहीं आ रही, कॉल समाप्त कर रही हूँ। धन्यवाद।",
        "smalltalk_default": "नमस्कार, मैं आपकी क्या मदद कर सकती हूँ?",
        "handoff": "मैं आपको हमारे स्टाफ़ से जोड़ रही हूँ, कृपया लाइन पर बने रहिए।",
        "cancel_aborted": "ठीक है, आपकी अपॉइंटमेंट जैसी थी वैसी ही है।",
        "no_confirmation_number": "ठीक है, बुकिंग हो जाएगी, लेकिन मैं कोई लिखित कन्फ़र्मेशन नहीं भेज पाऊँगी।",
        "language_switched": "ठीक है, अब हिंदी में बात करती हूँ।",
        "please_wait": "ठीक है।",
        "booking_hold_for_verification": "आपकी बुकिंग प्रोसेस हो रही है, मैं अभी पुष्टि करके बताती हूँ।",
        "reask_low_volume": "माफ़ कीजिए। आपकी आवाज़ थोड़ी धीमी है। क्या आप थोड़ा ज़ोर से बोलेंगे?",
        "reask_noisy": "माफ़ कीजिए। लाइन पर थोड़ा शोर है। क्या आप शांत जगह से फिर बोलेंगे?",
        "reask_crosstalk": "माफ़ कीजिए। पास से किसी और की आवाज़ आ रही है। क्या आप फिर बोलेंगे?",
        "reask_mumbled": "माफ़ कीजिए। मैं ठीक से समझ नहीं पाई। कोई जल्दी नहीं, क्या आप धीरे बोलेंगे?",
        "reask_generic": "माफ़ कीजिए। मैं ठीक से सुन नहीं पाई। क्या आप फिर बोलेंगे?",
        "reask_final": "माफ़ कीजिए। मैं आपको समझ नहीं पाई।",
    },
    "en": {
        "greeting": "Hello. How can I help you?",
        "asr_empty": "Sorry. I did not catch that. Could you say it again?",
        "unclear": "Sorry. I did not understand. Could you say that again?",
        "llm_failure": "We're having a small problem, please hold for a moment.",
        "tool_failure": "I can't check that right now. Please contact the counter.",
        "idle_close": "I can't hear anything on the line, so I'm ending the call. Thank you.",
        "smalltalk_default": "Hello, how can I help you?",
        "handoff": "Let me connect you to our staff, please stay on the line.",
        "cancel_aborted": "Okay, your appointment is unchanged.",
        "no_confirmation_number": "Okay, I'll go ahead with the booking, but I won't be able to send you a written confirmation.",
        "language_switched": "Okay, switching to English now.",
        "please_wait": "Let me check, please.",
        "booking_hold_for_verification": "Your booking is being processed, let me confirm it for you.",
        "reask_low_volume": "Sorry. Your voice is a little soft. Could you please speak a little louder?",
        "reask_noisy": "Sorry. There is some noise on the line. Could you try somewhere quieter?",
        "reask_crosstalk": "Sorry. I can hear someone else near you. Could you say that again?",
        "reask_mumbled": "Sorry. I did not understand. There is no rush. Could you say it slowly?",
        "reask_generic": "Sorry. I did not catch that. Could you say it again?",
        "reask_final": "Sorry. I could not understand you.",
    },
}

DEFAULT_LANGUAGE = "bn"

# KCD-353: the greeting says, before anything else, WHO the caller is speaking with and that it is an
# automated assistant, and that a person is available. It is composed at call time from parts the
# operator can change in the database (agent/messages.py): the welcome, the disclosure
# (agent/disclosure.py, versioned), and a one-line pointer to 112 for emergencies. The text kept in
# PHRASES["greeting"] is the built-in composition, used for the start-up pre-synthesis cache.
from agent import messages as _messages
from agent.disclosure import disclosure_for as _disclosure_for

EMERGENCY_HINT = {
    "bn": "জরুরি অবস্থায় সরাসরি ১১২ নম্বরে ফোন করবেন।",
    "hi": "आपात स्थिति में कृपया सीधे 112 पर कॉल करें।",
    "en": "In an emergency, please call 112 directly.",
}


def greeting_text(lang: str) -> str:
    """welcome. disclosure. emergency pointer. question -- the welcome and the question are the two
    sentences of the `greeting` phrase; the disclosure and the pointer go between them."""
    import re
    base = _messages.text("greeting", lang, _BASE_GREETING.get(lang) or _BASE_GREETING["bn"])
    parts = [x for x in re.split(r"(?<=[।.!?])\s+", base.strip()) if x]
    middle = [_disclosure_for(lang), _messages.text("emergency_hint", lang, EMERGENCY_HINT.get(lang) or EMERGENCY_HINT["bn"])]
    if len(parts) < 2:
        return " ".join(parts + middle)
    return " ".join([parts[0]] + middle + parts[1:])


_BASE_GREETING = {lang: table["greeting"] for lang, table in PHRASES.items()}
for _lang, _table in PHRASES.items():
    _table["greeting"] = greeting_text(_lang)

# KCD-054: said when the voice on the line changes after the caller was verified.
# Deliberately says nothing about WHY (the agent cannot know who is speaking, only
# that the voice differs) and does not accuse anyone.
PHRASES["bn"]["reverify_notice"] = "নিরাপত্তার জন্য ব্যক্তিগত তথ্য বলার আগে আমাকে আবার পরিচয় যাচাই করতে হবে।"
PHRASES["hi"]["reverify_notice"] = "सुरक्षा के लिए, निजी जानकारी बताने से पहले मुझे दोबारा पहचान की पुष्टि करनी होगी।"
# A possible emergency (agent/emergency.py). DRAFT: the wording, and the number it names, need
# clinical sign-off before this is relied on. It states no condition and gives no advice beyond
# where to call; a person joins the call straight after.
PHRASES["bn"]["emergency_notice"] = "এটা জরুরি অবস্থা হতে পারে। এখনই ১১২ নম্বরে ফোন করুন, অথবা কাছের হাসপাতালে যান।"
PHRASES["hi"]["emergency_notice"] = "यह आपात स्थिति हो सकती है। कृपया अभी 112 पर कॉल करें, या नज़दीकी अस्पताल जाएँ।"
PHRASES["en"]["emergency_notice"] = "This may be an emergency. Please call 112 now, or go to the nearest hospital."
PHRASES["en"]["reverify_notice"] = "For your security, I need to verify who I am speaking with again before I share any personal details."

# Spoken when a call is turned away before the caller has said a word, so
# there is no language to choose: all three, shortest first is not worth the
# complexity -- they are pre-synthesized and cached, so this costs no GPU
# time at exactly the moment the system is overloaded.
HANDOFF_ALL_LANGUAGES = ("bn", "hi", "en")


def phrase(key: str, lang: str = DEFAULT_LANGUAGE) -> str:
    table = PHRASES.get(lang) or PHRASES[DEFAULT_LANGUAGE]
    if key == "greeting":
        return greeting_text(lang if lang in PHRASES else DEFAULT_LANGUAGE)
    # the operator can change any phrase from the database (agent/messages.py); the built-in text
    # is what is spoken whenever the database has no row or cannot be reached
    return _messages.text(key, lang if lang in PHRASES else DEFAULT_LANGUAGE, table[key])


def prewarm_lines() -> dict[str, list[str]]:
    return {lang: list(table.values()) for lang, table in PHRASES.items()}
