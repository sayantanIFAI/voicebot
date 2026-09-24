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
        "greeting": "নমস্কার, কলকাতা কেয়ার ডায়াগনস্টিকসে স্বাগতম। কীভাবে সাহায্য করতে পারি?",
        "asr_empty": "দুঃখিত, শুনতে পাইনি। আবার বলবেন?",
        "unclear": "দুঃখিত, বুঝতে পারিনি। আবার একটু বলবেন?",
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
        "reask_low_volume": "দুঃখিত, আপনার গলাটা একটু আস্তে শোনাচ্ছে। একটু জোরে বলবেন, বা ফোনটা মুখের কাছে ধরবেন?",
        "reask_noisy": "দুঃখিত, লাইনে একটু শব্দ হচ্ছে, তাই ঠিক শুনতে পাচ্ছি না। একটু শান্ত জায়গা থেকে আবার বলবেন?",
        "reask_crosstalk": "দুঃখিত, পাশ থেকে আরও কারও গলা শোনা যাচ্ছে। তাই আপনার কথা আলাদা করতে পারছি না। একটু আবার বলবেন?",
        "reask_mumbled": "দুঃখিত, আপনার কথাটা ঠিক বুঝতে পারিনি, দোষটা আমারই। তাড়া নেই, একটু ধীরে আবার বলবেন?",
        "reask_generic": "দুঃখিত, আমি ঠিক শুনতে পাইনি। একটু আবার বলবেন?",
        "reask_final": "আপনাকে ঠিকমতো বুঝতে না পেরে আমি সত্যিই দুঃখিত।",
    },
    "hi": {
        "greeting": "नमस्कार, कोलकाता केयर डायग्नोस्टिक्स में आपका स्वागत है। मैं आपकी क्या मदद कर सकती हूँ?",
        "asr_empty": "माफ़ कीजिए, मैं सुन नहीं पाई। क्या आप दोबारा बोलेंगे?",
        "unclear": "माफ़ कीजिए, मैं समझ नहीं पाई। क्या आप ज़रा फिर से बताएँगे?",
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
        "reask_low_volume": "माफ़ कीजिए, आपकी आवाज़ धीमी आ रही है। फ़ोन मुँह के पास लाकर ज़ोर से बोलेंगे?",
        "reask_noisy": "माफ़ कीजिए, लाइन पर शोर है। मैं ठीक से सुन नहीं पा रही। क्या आप शांत जगह से दोबारा बोलेंगे?",
        "reask_crosstalk": "माफ़ कीजिए, पास से किसी और की आवाज़ आ रही है। इसलिए आपकी बात अलग नहीं कर पा रही। क्या आप फिर बोलेंगे?",
        "reask_mumbled": "माफ़ कीजिए, मैं ठीक से समझ नहीं पाई, यह मेरी कमी है। कोई जल्दी नहीं, क्या आप धीरे से दोबारा बोलेंगे?",
        "reask_generic": "माफ़ कीजिए, मैं ठीक से सुन नहीं पाई। क्या आप एक बार फिर बोलेंगे?",
        "reask_final": "आपको ठीक से न समझ पाने के लिए मुझे सचमुच खेद है।",
    },
    "en": {
        "greeting": "Hello, welcome to Kolkata Care Diagnostics. How can I help you?",
        "asr_empty": "Sorry, I didn't catch that. Could you say it again?",
        "unclear": "Sorry, I didn't understand. Could you please say that again?",
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
        "reask_low_volume": "Sorry, your voice is coming through quite faint. Could you speak up a little, or hold the phone closer?",
        "reask_noisy": "Sorry, there's noise on the line and I can't hear you clearly. Could you try somewhere quieter?",
        "reask_crosstalk": "Sorry, I can hear someone else near you. I can't pick out your words. Could you say that once more?",
        "reask_mumbled": "Sorry, I couldn't quite make that out, and that's on me. There's no rush, could you say it again a little slower?",
        "reask_generic": "Sorry, I didn't catch that. Could you say it once more?",
        "reask_final": "I'm really sorry I haven't been able to understand you properly.",
    },
}

DEFAULT_LANGUAGE = "bn"

# KCD-353: the greeting says, before anything else, that the caller is talking to
# an automated assistant and that a person is available. Composed here from
# agent/disclosure.py (versioned, pending clinical and legal review) so the
# wording lives in one place and every greeting -- spoken, pre-synthesised and
# cached -- carries it.
from agent.disclosure import insert_into_greeting as _with_disclosure

for _lang, _table in PHRASES.items():
    _table["greeting"] = _with_disclosure(_table["greeting"], _lang)

# KCD-054: said when the voice on the line changes after the caller was verified.
# Deliberately says nothing about WHY (the agent cannot know who is speaking, only
# that the voice differs) and does not accuse anyone.
PHRASES["bn"]["reverify_notice"] = "নিরাপত্তার জন্য ব্যক্তিগত তথ্য বলার আগে আমাকে আবার পরিচয় যাচাই করতে হবে।"
PHRASES["hi"]["reverify_notice"] = "सुरक्षा के लिए, निजी जानकारी बताने से पहले मुझे दोबारा पहचान की पुष्टि करनी होगी।"
PHRASES["en"]["reverify_notice"] = "For your security, I need to verify who I am speaking with again before I share any personal details."

# Spoken when a call is turned away before the caller has said a word, so
# there is no language to choose: all three, shortest first is not worth the
# complexity -- they are pre-synthesized and cached, so this costs no GPU
# time at exactly the moment the system is overloaded.
HANDOFF_ALL_LANGUAGES = ("bn", "hi", "en")


def phrase(key: str, lang: str = DEFAULT_LANGUAGE) -> str:
    table = PHRASES.get(lang) or PHRASES[DEFAULT_LANGUAGE]
    return table[key]


def prewarm_lines() -> dict[str, list[str]]:
    return {lang: list(table.values()) for lang, table in PHRASES.items()}
