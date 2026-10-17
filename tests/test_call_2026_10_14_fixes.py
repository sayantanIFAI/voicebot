"""Regression tests for the four problems found in the real call of 2026-10-14 (debug transcript).

Every sentence below is taken from that call or is the same request worded differently. Pure functions only: no GPU,
no network, no model. (The model-side half of each fix was checked separately against the pod's real Qwen.)
"""

from __future__ import annotations

import pytest

from agent.endpointing import classify_completeness
from agent.language_switch import detect_language_switch_request
from agent.slot_grounding import ground_slots
from agent.test_availability import asks_if_test_is_available, availability_reply

# ------------------------------------------------ 1. "speak / reply in English" was never recognised


@pytest.mark.parametrize(
    "text, recognised_as, target",
    [
        ("speaking in english", "en", "en"),
        ("Reply in English", "en", "en"),
        ("Speak in english", "en", "en"),
        ("reply in english.", "en", "en"),
        ("speak in English please sir", "en", "en"),
        ("can you talk english", "en", "en"),
        ("english", "en", "en"),
        # the Hindi / Bengali recognisers write the same English words in their own script
        ("स्पेकिंग इंग्लिश", "hi", "en"),
        ("स्पीक इन इंग्लिश", "hi", "en"),
        ("रिप्लाई इन इंग्लिश", "hi", "en"),
        ("इंग्लिश में जवाब दीजिए", "hi", "en"),
        ("স্পিকিং ইন ইংলিশ", "bn", "en"),
        ("ইংলিশে উত্তর দিন", "bn", "en"),
        # the other targets work the same way
        ("please reply in hindi", "en", "hi"),
        ("हिंदी में जवाब दो", "hi", "hi"),
        ("রিপ্লাই ইন বাংলা", "bn", "bn"),
        ("bengali please not hindi", "en", "bn"),
    ],
)
def test_a_request_to_speak_in_a_language_is_recognised(text, recognised_as, target):
    assert detect_language_switch_request(text, recognised_as) == target


@pytest.mark.parametrize(
    "text, recognised_as",
    [
        ("speak in hindi", "hi"),  # long-standing rule: a Latin-script request the Hindi recogniser produced is ignored
        ("What is the price of uric acid", "en"),
        ("my report is in english can you send it", "en"),
        ("send me the report on whatsapp", "en"),
        ("the parking parking is free", "en"),
        ("CBC test rate", "en"),
        ("আমার একটা টেস্ট বুক করতে হবে", "bn"),
        ("मुझे अपॉइंटमेंट बुक करनी है", "hi"),
        ("বাংলাদেশ থেকে এসেছি বলুন", "bn"),
    ],
)
def test_ordinary_sentences_are_not_a_language_switch(text, recognised_as):
    assert detect_language_switch_request(text, recognised_as) is None


# ------------------------------------------------ 2. an invented CBC + ECG total nobody asked for


@pytest.mark.parametrize("text", ["reply in english", "speaking in english", "thank you very much"])
def test_test_names_the_model_made_up_are_dropped_from_a_sentence_with_no_test_in_it(text):
    slots, dropped = ground_slots(text, {"test_names": ["CBC", "ECG"], "test_name": None})
    assert not slots["test_names"]
    assert dropped == ["CBC", "ECG"]


def test_a_doctor_the_model_made_up_is_dropped_too():
    slots, dropped = ground_slots("reply in english", {"doctor_name": "Dr. Sen"})
    assert slots["doctor_name"] is None and dropped == ["Dr. Sen"]


@pytest.mark.parametrize(
    "text, slots",
    [
        ("what is the price of CBC", {"test_names": ["CBC"]}),
        # a garbled name inside a real price question: the model's reading must survive
        ("আচ্ছা আর ক্বক র সাথে যদি আমি উরিক এসিদ টেস্ট করি একসাথে টোটাল কত পরবে", {"test_names": ["CBC", "Uric Acid"]}),
        ("সিবিসি", {"test_name": "সিবিসি"}),
        ("ডক্টর সেন", {"doctor_name": "Dr. Sen"}),
    ],
)
def test_real_questions_keep_what_the_model_read(text, slots):
    kept, dropped = ground_slots(text, slots)
    assert kept == slots and dropped == []


def test_a_name_the_catalogue_finds_in_the_sentence_is_never_dropped():
    kept, dropped = ground_slots("hmm uh yes", {"test_name": "Calcium"}, names_a_test=True)
    assert kept == {"test_name": "Calcium"} and dropped == []


def test_grounding_never_edits_the_dict_it_is_given():
    original = {"test_names": ["CBC", "ECG"]}
    ground_slots("reply in english", original)
    assert original == {"test_names": ["CBC", "ECG"]}


# ------------------------------------------------ 3. "is the lipid profile test done?" is not a booking


@pytest.mark.parametrize(
    "text, lang",
    [
        ("লিপিড প্রোফাইল টেস্ট হয়", "bn"),
        ("লিপিড প্রোফাইল টেস্ট হয়", "bn"),  # the same words with the dot under য written as one letter
        ("লিপিড প্রোফাইল টেস্ট কি হয়?", "bn"),
        ("ECG টেস্ট করা যায়", "bn"),
        ("ইউরিক অ্যাসিড টেস্ট আছে", "bn"),
        ("लिपिड प्रोफाइल टेस्ट होता है", "hi"),
        ("सीबीसी टेस्ट होता है क्या", "hi"),
        ("आपके यहाँ थायराइड टेस्ट होता है", "hi"),
        ("do you do the lipid profile test", "en"),
        ("is the CBC test available", "en"),
        ("is the thyroid test done here", "en"),
    ],
)
def test_an_availability_question_is_recognised(text, lang):
    assert asks_if_test_is_available(text, lang)


@pytest.mark.parametrize(
    "text, lang",
    [
        ("লিপিড প্রোফাইল টেস্ট হয় আর দাম কত", "bn"),  # also asks the price: priced as before
        ("सीबीसी टेस्ट होता है कितना लगेगा", "hi"),
        ("I want to book the lipid profile test", "en"),  # a real booking still gets the counter reply
        ("লিপিড প্রোফাইল টেস্ট করাতে চাই", "bn"),
        ("सीबीसी टेस्ट बुक करना है", "hi"),
        ("CBC test rate", "en"),
        ("what is the preparation for the lipid test", "en"),
        ("is the report ready", "en"),
        ("হয়", "bn"),  # a bare yes-word with no test named is an answer, not a question
        ("available", "en"),
    ],
)
def test_booking_price_and_prep_questions_are_not_availability_questions(text, lang):
    assert not asks_if_test_is_available(text, lang)


def test_the_availability_reply_says_yes_and_asks_what_more_in_every_language():
    en = availability_reply(["Lipid Profile"], "en")
    # SPEC CHANGE 2026-10-17 (owner): the answer opens with the plain yes -- "Yes, we do." / "হ্যাঁ, হয়।" / "जी हाँ, होता है।"
    assert en.startswith("Yes, we do. We do the Lipid Profile test.") and "What would you like to know" in en
    hi = availability_reply(["लिपिड प्रोफाइल टेस्ट"], "hi")
    assert "होता है" in hi and "जानना चाहेंगे" in hi
    bn = availability_reply(["লিপিড প্রোফাইল টেস্ট"], "bn")
    assert "হয়" in bn and "জানতে চান" in bn
    # it states no price, preparation or time: the only fact is that the test exists
    for reply in (en, hi, bn):
        assert not any(ch.isdigit() for ch in reply)
        assert "counter" not in reply and "কাউন্টার" not in reply and "काउंटर" not in reply


def test_two_tests_are_both_confirmed():
    reply = availability_reply(["CBC", "ECG"], "en")
    assert "CBC test and ECG test" in reply and "them" in reply


# ------------------------------------------------ 4. sentences cut off mid-thought


@pytest.mark.parametrize(
    "text, lang",
    [
        ("ठीक है मेरे को एक चीज बोलिए इधर", "hi"),
        ("ঠিক আছে আমাকে বলুন তো এই ক্ব টেস্ট", "bn"),
        ("मुझे एक सवाल है", "hi"),
        ("একটা কথা বলুন", "bn"),
        ("tell me one thing", "en"),
        ("I have a quick question", "en"),
        ("this test", "en"),
    ],
)
def test_a_sentence_that_announces_more_is_unfinished(text, lang):
    assert classify_completeness(text, lang) == "incomplete"


@pytest.mark.parametrize(
    "text, lang",
    [
        ("What is the price of this test", "en"),
        ("CBC test", "en"),
        ("এই টেস্টের দাম কত", "bn"),
        ("সিবিসি টেস্ট", "bn"),
        ("আমার নাম রাহুল", "bn"),
    ],
)
def test_a_finished_or_neutral_sentence_is_not_held_open(text, lang):
    assert classify_completeness(text, lang) != "incomplete"


# ------------------------------------------------ 5. the English recogniser's "Reply in English" was thrown away


def test_an_english_request_to_change_language_is_not_discarded_by_the_language_gate():
    """Pod-tested 2026-10-14: the English engine heard "Reply in English" with full agreement, language ID gave English
    0.04 (just under ENGLISH_MIN_LID), and only 1 of the 3 words was in the English word list, so the transcript was
    dropped and a Bengali mis-hearing won. The request's own words must count as English."""
    from dataclasses import dataclass

    from agent.lang_select import english_word_share, pick_candidate

    @dataclass
    class Heard:
        text: str
        decoder_agreement: float

    assert english_word_share("reply in english") >= 0.5
    assert english_word_share("speaking in english") >= 0.5
    assert english_word_share("ami kal ashbo ekhane") < 0.5  # romanised Bengali still must not pass as English
    heard = [("bn", Heard("হুম", 0.0)), ("hi", Heard("", 1.0)), ("en", Heard("reply in english", 1.0))]
    lang, result = pick_candidate(heard, {"bn": 0.52, "en": 0.04, "hi": 0.44}, None)
    assert lang == "en" and result.text == "reply in english"


def test_a_garbled_verb_does_not_hide_an_in_english_request():
    # pod-tested: the English recogniser heard "Reply in English" as "apply in english"
    assert detect_language_switch_request("apply in english", "en") == "en"
    assert detect_language_switch_request("ripley in hindi", "en") == "hi"
    assert detect_language_switch_request("the report is in english and long", "en") is None  # 6 words: not a request


def test_a_low_agreement_english_switch_command_still_wins_the_language_choice():
    from dataclasses import dataclass

    from agent.lang_select import pick_candidate

    @dataclass
    class Heard:
        text: str
        decoder_agreement: float

    heard = [("bn", Heard("হুম", 0.0)), ("hi", Heard("", 1.0)), ("en", Heard("apply in english", 0.25))]
    lang, result = pick_candidate(heard, {"bn": 0.52, "en": 0.04, "hi": 0.44}, None)
    assert lang == "en" and result.text == "apply in english"
    # an English transcript that is NOT a switch command keeps the old, stricter rule
    heard = [("bn", Heard("হুম", 0.0)), ("en", Heard("where is your klein lo", 0.25))]
    assert pick_candidate(heard, {"bn": 0.52, "en": 0.04, "hi": 0.44}, None)[0] == "bn"


@pytest.mark.parametrize(
    "text, recognised_as, target",
    [
        ("apply in english", "en", "en"),
        ("apply english please", "en", "en"),
        ("ripley english", "en", "en"),
        ("एप्लाई इंग्लिश", "hi", "en"),
        ("ইংলিশ প্লিজ", "bn", "en"),
        ("some hindi", "en", "hi"),
    ],
)
def test_the_keyword_alone_is_the_request_in_a_short_utterance(text, recognised_as, target):
    assert detect_language_switch_request(text, recognised_as) == target


# ------------------------------------------------ 6. "Would you like me to continue in English?" -- "yes" was ignored


@pytest.mark.parametrize("word", ["yes", "Yes", "हाँ", "हा", "হ্যাঁ", "ওকে", "ok", "no", "না", "नहीं"])
def test_a_one_word_yes_or_no_is_never_a_stray_fragment_however_long_the_clip(word):
    """Pod log, call b3aea9d2: a clear English "yes" (all three recognisers agreed) was logged `jumbled transcript
    (fragment)` and dropped, because the clip was over 2.5 s and the word is under 4 letters."""
    from agent.audio_quality import transcript_problem

    assert transcript_problem(word, "en", 3.4, 1.0) is None


def test_a_real_stray_syllable_is_still_a_fragment():
    from agent.audio_quality import transcript_problem

    assert transcript_problem("ক", "bn", 3.4, 1.0) == "fragment"


@pytest.mark.parametrize(
    "heard, expected",
    [
        ("yes", "yes"), ("Yes please", "yes"), ("হ্যাঁ", "yes"), ("हाँ", "yes"), ("ইয়েস", "yes"), ("यस", "yes"),
        ("জি", "yes"), ("ok", "yes"), ("no", "no"), ("না", "no"), ("नहीं", "no"), ("নো", "no"),
        ("what is the price", None), ("", None),
    ],
)
def test_the_answer_to_the_switch_offer_is_understood_in_any_script(heard, expected):
    import ast
    import pathlib

    # main.py cannot be imported off the pod (it needs torch / NeMo), so pull the two small names out of its source.
    source = pathlib.Path(__file__).resolve().parents[1].joinpath("main.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {"_SPOKEN_YES", "_SPOKEN_NO", "_answer_to_switch_offer"}
    chunk = [n for n in tree.body if (isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") in wanted)
             or (isinstance(n, ast.FunctionDef) and n.name in wanted)]
    assert len(chunk) == 3
    import re

    from agent.booking_flow import classify_yes_no

    namespace = {"re": re, "classify_yes_no": classify_yes_no}
    exec(compile(ast.Module(body=chunk, type_ignores=[]), "main.py", "exec"), namespace)
    assert namespace["_answer_to_switch_offer"](heard, "en", "bn") == expected


# ------------------------------------------------ 7. "price AND fasting?" in one sentence: both answered


def test_a_price_question_that_also_asks_about_fasting_gets_its_second_question_added():
    from agent.combined_question import add_implied_second_question

    data = {"intent": "test_rate", "slots": {"test_name": "সিবিসি"}, "secondary_intent": None, "secondary_slots": None}
    out = add_implied_second_question("সিবিসি টেস্টের দাম কত ফাস্টিং করতে হবে", data)
    assert out["secondary_intent"] == "test_prep" and out["secondary_slots"] == {"test_name": "সিবিসি"}
    assert data["secondary_intent"] is None  # the original is not edited


@pytest.mark.parametrize(
    "text, intent, expected",
    [
        ("CBC test price and do I need to fast", "test_rate", "test_prep"),
        ("सीबीसी का रेट क्या है और खाली पेट आना है", "test_rate", "test_prep"),
        ("how should I prepare for the CBC test and how much is it", "test_prep", "test_rate"),
        ("সিবিসি টেস্টের দাম কত", "test_rate", None),  # price only: nothing added
        ("what is the CBC test rate", "test_rate", None),
    ],
)
def test_the_second_question_is_added_only_when_the_sentence_asks_it(text, intent, expected):
    from agent.combined_question import add_implied_second_question

    out = add_implied_second_question(text, {"intent": intent, "slots": {"test_name": "CBC"}})
    assert out.get("secondary_intent") == expected


def test_an_existing_second_question_is_never_replaced_and_no_test_means_no_addition():
    from agent.combined_question import add_implied_second_question

    keep = {"intent": "test_rate", "slots": {"test_name": "CBC"}, "secondary_intent": "clinic_faq", "secondary_slots": {}}
    assert add_implied_second_question("price and fasting", keep) is keep
    none = {"intent": "test_rate", "slots": {}}
    assert add_implied_second_question("price and fasting", none) is none
