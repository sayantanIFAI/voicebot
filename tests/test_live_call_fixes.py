"""Regressions from the first live call on the pod (call c7eb4b15, 2026-09-25), through the real orchestrator.

    python -m pytest tests/test_live_call_fixes.py -v

What the caller experienced, and what the log showed:
  1. A Bengali call was answered in Hindi. Language ID said bn 0.95 / hi 0.05, all three recognisers ran, and the
     Hindi one (writing Bengali speech out in Devanagari) reported a higher agreement than the Bengali one.
  2. The agent "could not hear" and made the caller repeat the whole sentence, when the intent ("the price of a
     test") had been understood and only the name was not trusted.
  3. The price reply spoke a label and a colon ("sample: Blood") and, in Bengali, any catalogue category as a sample.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO_ROOT, os.path.join(REPO_ROOT, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent.lid import LIDResult
from agent.phrases import phrase
from agent.reply_templates import missing_slot_prompt
from agent.reply_templates import test_rate_reply as price_reply
from agent.sample_wording import is_specimen, sample_sentence
from test_orchestrator_booking_flow import env, m       # noqa: F401  (the harness: real dispatch, fake tools)


class Result:
    def __init__(self, text, agreement, decoder_used="rnnt"):
        self.text, self.decoder_agreement, self.decoder_used = text, agreement, decoder_used


# ================================================================================ 1. language

class FakeLID:
    def __init__(self, language, scores):
        self._r = LIDResult(language=language, confidence=scores[language], scores=scores)

    def identify_path(self, path):
        return self._r


class FakeRouter:
    def __init__(self, results):
        self.results = results

    async def transcribe_many(self, languages, path):
        return [(lang, self.results[lang]) for lang in languages]

    async def transcribe(self, language, path):
        return self.results[language]


@pytest.mark.asyncio
async def test_the_pod_call_a_bengali_caller_is_answered_in_bengali_not_hindi(m, env, monkeypatch):
    """The exact numbers from the log: LID bn 0.95 / en 0.00 / hi 0.05; agreement bn 0.20, hi 0.50, en 0.33."""
    monkeypatch.undo()
    monkeypatch.setattr(m, "_lid", FakeLID("bn", {"bn": 0.95, "en": 0.0, "hi": 0.05}))
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi", "en"))
    monkeypatch.setattr(m, "_asr_router", FakeRouter({
        "bn": Result("এইচবিএ ওয়ান সি টেস্টের দাম কত", 0.20),
        "hi": Result("एइबी वन एसी ए टेस्टर्ड दाम को तो", 0.50),
        "en": Result("hb a one c test her dam co", 0.33)}))
    env.session.lang_router.note_response_language("bn")          # the call opened in Bengali
    lang, result = await m._route_and_transcribe(env.session, "x.wav")
    assert lang == "bn" and result.text.startswith("এইচবিএ")


@pytest.mark.asyncio
async def test_a_call_in_bengali_does_not_flip_to_hindi_on_a_weak_hindi_read_even_with_no_prior_turn(m, env, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(m, "_lid", FakeLID("bn", {"bn": 0.84, "en": 0.0, "hi": 0.15}))
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi", "en"))
    monkeypatch.setattr(m, "_asr_router", FakeRouter({"bn": Result("বাংলা", 0.30), "hi": Result("हिंदी", 0.80),
                                                       "en": Result("english", 0.10)}))
    lang, _ = await m._route_and_transcribe(env.session, "x.wav")
    assert lang == "bn"


@pytest.mark.asyncio
async def test_a_genuine_hindi_caller_is_still_answered_in_hindi(m, env, monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(m, "_lid", FakeLID("hi", {"bn": 0.02, "en": 0.0, "hi": 0.98}))
    monkeypatch.setattr(m, "_languages_active", ("bn", "hi", "en"))
    monkeypatch.setattr(m, "_asr_router", FakeRouter({"bn": Result("बंगाली", 0.10), "hi": Result("सीबीसी का रेट", 0.90),
                                                       "en": Result("x", 0.0)}))
    lang, _ = await m._route_and_transcribe(env.session, "x.wav")
    assert lang == "hi"


# ================================================================ 2. ask for the name, not the whole sentence

@pytest.mark.asyncio
@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
async def test_when_only_the_name_was_not_trusted_the_agent_asks_for_the_name(m, env, monkeypatch, lang):
    heard = {"bn": "ইউরিন টেস্টের দাম কত", "hi": "यूरिन टेस्ट की कीमत क्या है", "en": "what is the price of a urine test"}[lang]

    async def route(session, wav):
        return lang, Result(heard, 0.44)                 # a well-formed sentence; the recognisers just disagreed
    monkeypatch.setattr(m, "_route_and_transcribe", route)
    env.session.disclosed_langs.add(lang)
    said = await env.say(heard, "test_rate", {"test_name": "ইউরিন"})
    assert said[-1] == f"{phrase('name_not_caught', lang)} {missing_slot_prompt('test_rate', 'test_name', lang)}"
    assert "?" in said[-1] and said[-1].count("?") == 1                   # one question: which test


@pytest.mark.asyncio
async def test_a_doctor_question_asks_for_the_doctor_and_an_unnamed_intent_keeps_the_general_reply(m, env, monkeypatch):
    from agent.reply_templates import insufficient_information_reply

    async def route(session, wav):
        return "en", Result("garbled", 0.44)
    monkeypatch.setattr(m, "_route_and_transcribe", route)
    said = await env.say("garbled", "doctor_availability", {"doctor_name": "x"})
    assert said[-1] == f"{phrase('name_not_caught', 'en')} {missing_slot_prompt('doctor_availability', 'doctor_name', 'en')}"
    said = await env.say("garbled", "clinic_faq", {"faq_topic": "hours"})
    assert said[-1] == insufficient_information_reply("en")               # no name to ask for: the general reply


@pytest.mark.asyncio
async def test_a_trusted_turn_is_still_answered_straight_away(m, env, monkeypatch):
    async def route(session, wav):
        return "en", Result("what is the price of a CBC test", 0.90)
    monkeypatch.setattr(m, "_route_and_transcribe", route)
    said = await env.say("x", "test_rate", {"test_name": "CBC"})
    assert "350" in " ".join(said) or "three hundred" in " ".join(said)


# ================================================================================================ 3. the sample

@pytest.mark.parametrize("sample", ["Blood", "Urine", "Stool", "Serum", "Saliva", "Swab", "Plasma"])
def test_every_specimen_has_a_sentence_in_every_language(sample):
    for lang in ("bn", "hi", "en"):
        s = sample_sentence(sample, lang)
        assert s and ":" not in s and "{" not in s
        assert not any("a" <= c.lower() <= "z" for c in s) or lang == "en"     # no Latin word inside bn/hi speech


@pytest.mark.parametrize("category", ["Imaging", "Cardiac", "Sample (Cervical)", "", None])
def test_a_category_that_is_not_a_specimen_says_nothing_about_a_sample(category):
    assert sample_sentence(category, "bn") == "" and not is_specimen(category)


def test_the_price_reply_is_a_sentence_in_all_three_languages():
    result = {"found": True, "test_name": "HbA1c", "test_name_bn": "এইচবিএ১সি", "test_name_hi": "एचबीए वन सी",
              "rate_inr": 650, "sample_type": "Blood", "report_time_hours": 24}
    bn, hi, en = (price_reply({"test_name": "x"}, result, lang) for lang in ("bn", "hi", "en"))
    assert "রক্তের নমুনা" in bn and "ब्लड का सैंपल" in hi and "A blood sample is needed." in en
    for text in (bn, hi, en):
        assert ":" not in text


def test_bengali_no_longer_reads_a_scan_category_out_as_a_sample():
    result = {"found": True, "test_name": "Chest X-Ray", "test_name_bn": "বুকের এক্স-রে", "rate_inr": 400,
              "sample_type": "Imaging", "report_time_hours": 4}
    assert "নমুনা" not in price_reply({"test_name": "x"}, result, "bn")
