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
        "greeting": "বলুন, কীভাবে সাহায্য করতে পারি?",
        "asr_empty": "দুঃখিত। আমি শুনতে পাইনি। আবার বলবেন?",
        "unclear": "দুঃখিত। আমি বুঝতে পারিনি। আবার বলবেন?",
        "llm_failure": "একটু সমস্যা হচ্ছে, একটু ধরুন।",
        "tool_failure": "এই মুহূর্তে দেখতে পারছি না। কাউন্টারে যোগাযোগ করুন, দয়া করে।",
        "idle_close": "লাইনে কোনো সাড়া পাচ্ছি না, কল শেষ করছি। ধন্যবাদ।",
        "smalltalk_default": "সোনোস্ক্যান বাণী। বলুন, কীভাবে সাহায্য করতে পারি?",
        "handoff": "আমি আপনাকে আমাদের স্টাফের সাথে যুক্ত করছি, দয়া করে লাইনে থাকুন।",
        "cancel_aborted": "ঠিক আছে, আপনার অ্যাপয়েন্টমেন্টটা যেমন ছিল তেমনই আছে।",
        "no_confirmation_number": "ঠিক আছে, তাহলে বুকিংটা হয়ে যাবে, কিন্তু কোনো লিখিত কনফার্মেশন পাঠাতে পারব না।",
        "language_switched": "ঠিক আছে, এখন থেকে বাংলায় বলছি।",
        # Asked only after two turns IN A ROW in this language, spoken in THIS language itself (the one the
        # caller has actually been using) -- see main.py's offer-after-two-turns check, next to lang_history.
        "offer_switch": "আমি কি বাংলায় কথা বলব?",
        # KCD-461: spoken only when the turn is STILL unanswered FILLER_THRESHOLD_S (0.7 s) after the caller stopped
        # (main.py's _await_with_filler); an answer that arrives sooner is given with no filler at all.
        # OWNER'S WORDING, 2026-09-25: "আচ্ছা বলছি" ("okay, I'm telling you"), in all three languages -- pre-warmed
        # and cached like every other line here, so it costs no synthesis time when the system is already slow.
        "please_wait": "আচ্ছা, বলছি।",
        # The caller has said nothing for SILENCE_PROMPT_S after the agent finished speaking: ask once whether there is
        # more (one question), and say that the call ends if not. If still silent, idle_close ends the call.
        "silence_prompt": "আর কিছু জানতে চান? না চাইলে আমি কল শেষ করে দেব।",
        "silence_go_on": "হ্যাঁ, বলুন।",
        "silence_goodbye": "ঠিক আছে, ধন্যবাদ। ভালো থাকবেন।",
        # Abuse from the caller (agent/abuse.py): calm, no imitation, no lecture; the third time the call is closed.
        "abuse_first": "আমি আপনাকে সাহায্য করতে এখানে আছি। ভদ্রভাবে বললে আমি আরও ভালো সাহায্য করতে পারব। বলুন, কী জানতে চান?",
        "abuse_second": "ভদ্রভাবে কথা বললেই আমি সাহায্য করতে পারব। আপনি কি কোনো তথ্য জানতে চান?",
        "abuse_final": "আমি এভাবে কথা চালিয়ে যেতে পারছি না। দরকার হলে আবার ফোন করবেন। ধন্যবাদ।",
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
        "greeting": "बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
        "asr_empty": "माफ़ कीजिए। मैं सुन नहीं पाई। क्या आप फिर बोलेंगे?",
        "unclear": "माफ़ कीजिए। मैं समझ नहीं पाई। क्या आप फिर बताएँगे?",
        "llm_failure": "थोड़ी दिक्कत आ रही है, कृपया एक पल रुकिए।",
        "tool_failure": "अभी मैं यह देख नहीं पा रही हूँ। कृपया काउंटर पर संपर्क करें।",
        "idle_close": "लाइन पर कोई आवाज़ नहीं आ रही, कॉल समाप्त कर रही हूँ। धन्यवाद।",
        "smalltalk_default": "सोनोस्कैन वाणी। बताइए, मैं आपकी क्या मदद कर सकती हूँ?",
        "handoff": "मैं आपको हमारे स्टाफ़ से जोड़ रही हूँ, कृपया लाइन पर बने रहिए।",
        "cancel_aborted": "ठीक है, आपकी अपॉइंटमेंट जैसी थी वैसी ही है।",
        "no_confirmation_number": "ठीक है, बुकिंग हो जाएगी, लेकिन मैं कोई लिखित कन्फ़र्मेशन नहीं भेज पाऊँगी।",
        "language_switched": "ठीक है, अब हिंदी में बात करती हूँ।",
        "offer_switch": "क्या मैं हिंदी में बात करूँ?",
        "please_wait": "अच्छा, बताती हूँ।",
        "silence_prompt": "क्या आप कुछ और जानना चाहते हैं? नहीं तो मैं कॉल समाप्त कर दूँगी।",
        "silence_go_on": "जी, बताइए।",
        "silence_goodbye": "ठीक है, धन्यवाद। अपना ख़याल रखिए।",
        "abuse_first": "मैं आपकी मदद के लिए यहाँ हूँ। सम्मान से बात करेंगे तो मैं और अच्छी मदद कर पाऊँगी। बताइए, आप क्या जानना चाहते हैं?",
        "abuse_second": "सम्मान से बात करने पर ही मैं मदद कर पाऊँगी। क्या आप कोई जानकारी चाहते हैं?",
        "abuse_final": "मैं इस तरह बात जारी नहीं रख सकती। ज़रूरत हो तो दोबारा फ़ोन करें। धन्यवाद।",
        "booking_hold_for_verification": "आपकी बुकिंग प्रोसेस हो रही है, मैं अभी पुष्टि करके बताती हूँ।",
        "reask_low_volume": "माफ़ कीजिए। आपकी आवाज़ थोड़ी धीमी है। क्या आप थोड़ा ज़ोर से बोलेंगे?",
        "reask_noisy": "माफ़ कीजिए। लाइन पर थोड़ा शोर है। क्या आप शांत जगह से फिर बोलेंगे?",
        "reask_crosstalk": "माफ़ कीजिए। पास से किसी और की आवाज़ आ रही है। क्या आप फिर बोलेंगे?",
        "reask_mumbled": "माफ़ कीजिए। मैं ठीक से समझ नहीं पाई। कोई जल्दी नहीं, क्या आप धीरे बोलेंगे?",
        "reask_generic": "माफ़ कीजिए। मैं ठीक से सुन नहीं पाई। क्या आप फिर बोलेंगे?",
        "reask_final": "माफ़ कीजिए। मैं आपको समझ नहीं पाई।",
    },
    "en": {
        "greeting": "How can I help you?",
        "asr_empty": "Sorry. I did not catch that. Could you say it again?",
        "unclear": "Sorry. I did not understand. Could you say that again?",
        "llm_failure": "We're having a small problem, please hold for a moment.",
        "tool_failure": "I can't check that right now. Please contact the counter.",
        "idle_close": "I can't hear anything on the line, so I'm ending the call. Thank you.",
        "smalltalk_default": "Sonoscan Vaani. How can I help you?",
        "handoff": "Let me connect you to our staff, please stay on the line.",
        "cancel_aborted": "Okay, your appointment is unchanged.",
        "no_confirmation_number": "Okay, I'll go ahead with the booking, but I won't be able to send you a written confirmation.",
        "language_switched": "Okay, switching to English now.",
        "offer_switch": "Would you like me to continue in English?",
        "please_wait": "Alright, let me tell you.",
        "silence_prompt": "Is there anything else you would like to know? If not, I will end the call.",
        "silence_go_on": "Yes, please go ahead.",
        "silence_goodbye": "Alright, thank you. Take care.",
        "abuse_first": "I am here to help you. I can help you better if we speak politely. What would you like to know?",
        "abuse_second": "I can only help if we speak politely. Is there some information you need?",
        "abuse_final": "I cannot continue the conversation like this. Please call again whenever you need. Thank you.",
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
from agent import messages as _messages  # noqa: E402  (deliberately after the tables above: see the comment)

EMERGENCY_HINT = {
    "bn": "জরুরি অবস্থায় সরাসরি ১১২ নম্বরে ফোন করবেন।",
    "hi": "आपात स्थिति में कृपया सीधे 112 पर कॉल करें।",
    "en": "In an emergency, please call 112 directly.",
}


# OWNER'S INSTRUCTION, 2026-09-25 (after the first live call): the first thing a caller hears is only
#   "Namaskar. I am Manorama Vaani. Tell me, how can I help you?"
# so the spoken greeting is the welcome, this one identity sentence, and the question. It replaces the longer
# KCD-353 opening (that it is an automated assistant, that staff are available at any time, and the pointer to 112).
# The full disclosure text still exists (the text-message notice) and can be brought back with no deploy: change
# the `greeting_identity` row to the longer text, or add an `emergency_hint` row (the pointer is spoken only when
# such a row exists).
#
# OWNER'S INSTRUCTION, 2026-10-12: reversed the above for Bengali only, for one day -- see the owner's 2026-10-13
# instruction below, which reverses it back.
#
# OWNER'S INSTRUCTION, 2026-10-13: restores the 2026-09-25 short wording for ALL THREE languages -- the opening
# greeting (Bengali) and the once-only language-switch restatement (Hindi/English, agent/disclosure.py's
# DISCLOSURE) say ONLY the identity sentence and the question, nothing about being automated or staff being
# reachable. Kept verbatim-identical per language with DISCLOSURE so the two call sites never say two different
# things about what the agent is.
#
# OWNER'S INSTRUCTION, 2026-10-14 (sourav): the brand is "Sonoscan Vaani" and the opening is ONLY
#   "Sonoscan Vaani. How can I help you?"  -- no "Namaskar/Hello", no "you are speaking with ...", nothing about being
# automated. Same sentence, in each language, for the opening greeting, the bare-greeting reply (agent/fast_path_cues.py)
# and the smalltalk fallback (PHRASES["smalltalk_default"]). The language-switch path no longer repeats it at all
# (main.py / main_pcm.py: the once-only disclosure line was removed from the turn loop).
GREETING_IDENTITY = {
    "bn": "সোনোস্ক্যান বাণী।",
    "hi": "सोनोस्कैन वाणी।",
    "en": "Sonoscan Vaani.",
}


def greeting_text(lang: str) -> str:
    """welcome. who you are speaking with. question -- the welcome and the question are the two sentences of the
    `greeting` phrase; the identity sentence (and, only if the operator has set one, the 112 pointer) go between."""
    import re

    base = _messages.text("greeting", lang, _BASE_GREETING.get(lang) or _BASE_GREETING["bn"])
    parts = [x for x in re.split(r"(?<=[।.!?])\s+", base.strip()) if x]
    middle = [_messages.text("greeting_identity", lang, GREETING_IDENTITY.get(lang) or GREETING_IDENTITY["bn"])]
    if _messages.has("emergency_hint", lang):
        middle.append(_messages.text("emergency_hint", lang, EMERGENCY_HINT.get(lang) or EMERGENCY_HINT["bn"]))
    if len(parts) < 2:
        # sourav: the built-in greeting is now ONE sentence (the question), so the identity goes FIRST:
        # "Sonoscan Vaani. How can I help you?" -- never "How can I help you? Sonoscan Vaani."
        return " ".join(middle + parts)
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
# The turn was heard but the NAME in it was not trusted (the recognisers disagreed): say so and ask for just the name.
PHRASES["bn"]["name_not_caught"] = "দুঃখিত, নামটা ঠিক শুনতে পাইনি।"
PHRASES["hi"]["name_not_caught"] = "माफ़ कीजिए, नाम ठीक से सुन नहीं पाई।"
PHRASES["en"]["name_not_caught"] = "Sorry, I did not catch the name."
PHRASES["en"]["reverify_notice"] = (
    "For your security, I need to verify who I am speaking with again before I share any personal details."
)

# ------------------------------------------------------------------------------------------- lay terms / prescription
# Said whenever the caller is asked to send a prescription photo over WhatsApp so a human colleague can call back
# (agent/lay_terms.py's HEAD category, and both blood/abdomen flows in main.py). The number is fixed and spoken
# plainly -- digits, not a link, since this is a voice call.
PHRASES["bn"]["send_prescription_whatsapp"] = (
    "প্রেসক্রিপশনের একটা ছবি এই হোয়াটসঅ্যাপ নম্বরে পাঠিয়ে দিন, ৯৬৩৫৫৮৮৩০৬। আমাদের একজন সহকর্মী দেখে আপনাকে কল করবেন।"
)
PHRASES["hi"]["send_prescription_whatsapp"] = (
    "पर्चे की एक फोटो इस व्हाट्सऐप नंबर पर भेज दीजिए, 9635588306। हमारा एक सहकर्मी देखकर आपको कॉल करेगा।"
)
PHRASES["en"]["send_prescription_whatsapp"] = (
    "Please send a photo of the prescription to this WhatsApp number, 9635588306. A colleague will look at it "
    "and call you back."
)

# OWNER'S INSTRUCTION, 2026-10-12: a caller asked for a test's price/prep but does not know the test's name at
# all (agent/lay_terms.expresses_not_knowing_test_name). Asked ONCE, before giving up -- a caller who at least
# knows the broad kind still gets a real, catalogue-backed answer (main.py's new category-answer handling); only
# a caller who cannot say even this gets the safety-check message below. No guessing past this point.
PHRASES["bn"]["ask_test_category"] = "ঠিক আছে, এটা কি রক্ত পরীক্ষা, মূত্র পরীক্ষা, নাকি স্ক্যান/এক্স-রে/আল্ট্রাসাউন্ডের মতো কিছু?"
PHRASES["hi"]["ask_test_category"] = "ठीक है, क्या यह ब्लड टेस्ट है, यूरिन टेस्ट है, या स्कैन/एक्स-रे/अल्ट्रासाउंड जैसा कुछ?"
PHRASES["en"]["ask_test_category"] = (
    "Okay, is it a blood test, a urine test, or something like a scan, X-ray, or ultrasound?"
)

# Spoken when the caller does not know even the broad category -- CLAUDE.md's truth boundary: never guess a
# test name, hand off to a human with the prescription instead. Same WhatsApp number as send_prescription_
# whatsapp above (kept as a separate phrase, not a rewrite of that one, so the existing lay-term/head-imaging
# flows that already use it are untouched); this version adds the "or come to the counter" option the owner
# asked for, which that older phrase never offered.
PHRASES["bn"]["test_unknown_safety_check"] = (
    "ঠিক আছে, এখন ঠিক করে বলা সম্ভব নয়। প্রেসক্রিপশনের একটা ছবি এই হোয়াটসঅ্যাপ নম্বরে পাঠিয়ে দিন, ৯৬৩৫৫৮৮৩০৬। "
    "অথবা প্রেসক্রিপশন নিয়ে সরাসরি আমাদের কাউন্টারে চলে আসুন।"
)
PHRASES["hi"]["test_unknown_safety_check"] = (
    "ठीक है, अभी ठीक से बता पाना संभव नहीं है। पर्चे की एक फोटो इस व्हाट्सऐप नंबर पर भेज दीजिए, 9635588306। "
    "या पर्चा लेकर सीधे हमारे काउंटर पर आ जाइए।"
)
PHRASES["en"]["test_unknown_safety_check"] = (
    "I'm not able to say without the exact test name. Please send a photo of the prescription to this WhatsApp "
    "number, 9635588306. Or come directly to our clinic counter with the prescription."
)

# The catalogue has no head/brain imaging test today (agent/lay_terms.py's module docstring); this says so plainly
# instead of naming a service the clinic does not offer, then hands off to send_prescription_whatsapp above.
PHRASES["bn"]["head_imaging_not_listed"] = (
    "মাথার স্ক্যান বা এমআরআই এখন আমাদের তালিকায় নেই। ডাক্তার প্রেসক্রিপশনে ঠিক কী বলেছেন সেটা দেখে বলা ভালো।"
)
PHRASES["hi"]["head_imaging_not_listed"] = (
    "सिर का स्कैन या एमआरआई अभी हमारी सूची में नहीं है। पर्चे में डॉक्टर ने ठीक क्या लिखा है, वह देखकर बताना बेहतर होगा।"
)
PHRASES["en"]["head_imaging_not_listed"] = (
    "A head scan or MRI is not on our list right now. It's best a colleague checks exactly what the doctor's "
    "prescription says."
)

# ------------------------------------------------------------------------------------------------------- lab tests
# A lab test is paid and booked at the counter, never over the call (main.py's book_test/add_test_booking intents).
PHRASES["bn"]["lab_test_counter_only"] = (
    "টেস্টের বুকিং আর পেমেন্ট আমাদের কাউন্টারে গিয়ে করতে হয়, ফোনে হয় না। প্রেসক্রিপশন নিয়ে কাউন্টারে চলে আসুন।"
)
PHRASES["hi"]["lab_test_counter_only"] = (
    "टेस्ट की बुकिंग और पेमेंट काउंटर पर आकर करनी होती है, फ़ोन पर नहीं होती। पर्चा लेकर काउंटर पर आ जाइए।"
)
PHRASES["en"]["lab_test_counter_only"] = (
    "A lab test has to be booked and paid for at the counter, not over the phone. Please come to the counter with "
    "your prescription."
)

# ---------------------------------------------------------------------------------------------------- payment link
# Said as the closing line after a doctor's appointment is confirmed: the payment link has been sent, and the call
# ends here (main.py's _end_call) -- the caller pays on their own phone, never a number spoken or taken on the call.
PHRASES["bn"]["payment_link_sent"] = (
    "আপনার অ্যাপয়েন্টমেন্ট বুক হয়েছে। বুকিং নম্বর আর পেমেন্ট লিংক আপনার মোবাইলে পাঠানো হয়েছে, ওখান থেকে পেমেন্ট করে আপনার স্লট নিশ্চিত করুন। ধন্যবাদ।"
)
PHRASES["hi"]["payment_link_sent"] = (
    "आपकी अपॉइंटमेंट बुक हो गई है। पेमेंट लिंक आपके मोबाइल पर भेज दिया गया है, वहाँ से पेमेंट करके बुकिंग पक्की कर लीजिए। धन्यवाद।"
)
PHRASES["en"]["payment_link_sent"] = (
    "Your appointment is booked. A payment link has been sent to your phone -- please pay there to confirm it. "
    "Thank you."
)

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
