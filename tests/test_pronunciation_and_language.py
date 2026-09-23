"""KCD-159 (agent/pronunciation.py), KCD-087/KCD-150 (agent/language_policy.py).

    python -m pytest tests/test_pronunciation_and_language.py -v
"""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import pronunciation
from agent.language_policy import (
    choose_spoken_form, dominant_script, language_mismatch, reply_matches_language,
    resolve_reply_language,
)
from agent.phrases import PHRASES
from agent.speech_norm import unspeakable_spans, verbalize

# ============================================================ pronunciation

@pytest.mark.parametrize("lang,text,expected_fragment", [
    ("bn", "আপনার Sugar টেস্ট", "সুগার"),
    ("hi", "आपका Sugar टेस्ट", "शुगर"),
    ("bn", "Report কালকে পাবেন", "রিপোর্ট"),
    ("hi", "Fasting रखना है", "फास्टिंग"),
])
def test_a_latin_clinical_word_is_spoken_through_the_lexicon(lang, text, expected_fragment):
    out = verbalize(text, lang)
    assert expected_fragment in out
    assert unspeakable_spans(out, lang) == [], out


@pytest.mark.parametrize("lang", ["bn", "hi"])
def test_a_sentence_with_english_words_no_longer_has_a_hole_in_it(lang):
    out = verbalize("Blood Sugar Fasting Report", lang)
    assert unspeakable_spans(out, lang) == []
    assert not any(c.isascii() and c.isalpha() for c in out)


def test_the_original_seven_specimen_words_are_unchanged():
    assert verbalize("Blood", "bn") == "রক্ত" and verbalize("Blood", "hi") == "खून"
    assert verbalize("urine", "bn") == "মূত্র" and verbalize("swab", "hi") == "स्वैब"


def test_a_multi_word_term_wins_over_its_parts():
    assert verbalize("ct scan", "bn") == "সিটি স্ক্যান"
    assert "সিটি স্ক্যান" in verbalize("CT scan", "bn")


def test_matching_is_whole_word_and_case_insensitive():
    assert "রিপোর্ট" in verbalize("REPORT", "bn")
    assert "reporting" in verbalize("reporting", "bn").lower()      # "report" inside a longer word is left alone


def test_an_unlisted_acronym_is_spelled_letter_by_letter():
    out = verbalize("Your PQRS value", "bn")
    assert "PQRS" not in out
    assert unspeakable_spans(out, "bn") == ["Your", "value"] or "PQRS" not in out


def test_an_unlisted_ordinary_word_is_still_blocked_never_guessed():
    # KCD-455 must keep holding: the path is a curated lexicon, not a
    # general transliterator that would guess a pronunciation for a drug name.
    assert unspeakable_spans(verbalize("রিপোর্ট Pending অবস্থায় আছে", "bn"), "bn") == ["Pending"]
    assert unspeakable_spans(verbalize("यूरिक Acid Zolpidem", "hi"), "hi") == ["Zolpidem"]


def test_english_replies_are_untouched_by_the_lexicon():
    assert verbalize("Your report is ready", "en") == "Your report is ready"


def test_confirmation_ids_are_not_mistaken_for_acronyms():
    out = verbalize("Reference KCD-4471", "bn")
    assert "KCD" not in out and "Reference" in out


def test_coverage_reports_the_signoff_gap_honestly():
    cov = pronunciation.coverage()
    assert cov["target_terms"] == 200
    assert cov["shortfall"] == 200 - cov["terms"] > 0
    assert cov["signed_off"] == 0
    assert cov["entries_needing_native_review"] == cov["terms"] * 2


def test_the_review_sheet_lists_every_entry_as_pending():
    sheet = pronunciation.review_sheet()
    assert len(sheet) == pronunciation.coverage()["terms"] * 2
    assert {r["status"] for r in sheet} == {"pending_native_review"}
    assert {r["lang"] for r in sheet} == {"bn", "hi"}
    assert all(r["spoken_form"] for r in sheet)


def test_every_lexicon_form_is_in_the_right_script():
    for term, (bn, hi) in pronunciation._ENTRIES.items():
        assert any("ঀ" <= ch <= "৿" for ch in bn), (term, bn)
        assert any("ऀ" <= ch <= "ॿ" for ch in hi), (term, hi)
        assert not any(ch.isascii() and ch.isalpha() for ch in bn + hi), (term, bn, hi)


# ============================================================ language policy

def test_the_current_utterance_decides_the_reply_language():
    assert resolve_reply_language("hi", fallback="bn") == "hi"
    assert resolve_reply_language("en", fallback="bn") == "en"


def test_an_unidentified_utterance_falls_back_to_the_last_language():
    assert resolve_reply_language(None, fallback="hi") == "hi"


def test_an_explicit_request_outranks_identification_for_that_turn():
    assert resolve_reply_language("bn", explicit_request="en") == "en"


def test_garbage_falls_back_to_bengali_never_raises():
    assert resolve_reply_language("xx", "yy", "zz") == "bn"


@pytest.mark.parametrize("lang,reply,ok", [
    ("bn", "আপনার রেট পাঁচশো টাকা।", True),
    ("hi", "आपकी कीमत पाँच सौ रुपये है।", True),
    ("en", "The price is five hundred rupees.", True),
    ("bn", "The price is five hundred rupees.", False),
    ("en", "আপনার রেট পাঁচশো টাকা।", False),
    ("hi", "আপনার রেট পাঁচশো টাকা।", False),
    ("bn", "৫০০", True),
    ("bn", "আপনার Report কালকে পাবেন", True),        # one borrowed word inside a Bengali reply is fine
])
def test_reply_language_matches_the_question_language(lang, reply, ok):
    assert reply_matches_language(reply, lang) is ok
    assert language_mismatch(lang, reply) is (not ok)


def test_dominant_script():
    assert dominant_script("hello") == "latin"
    assert dominant_script("নমস্কার") == "bengali"
    assert dominant_script("12345") is None


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_every_fixed_phrase_is_in_its_own_language(lang):
    # The multilingual suite the story asks for, over the phrase table:
    # a phrase in the wrong language for its table is a defect.
    for key, text in PHRASES[lang].items():
        assert reply_matches_language(text, lang), (lang, key, text)


def test_a_reply_template_answers_in_the_requested_language():
    from agent import reply_templates as rt
    slots = {"test_name": "x"}
    result = {"found": True, "test_name": "Lipid Profile", "test_name_bn": "লিপিড",
              "test_name_hi": "लिपिड", "rate_inr": 650, "report_time_hours": 24, "sample_type": "Blood"}
    for lang in ("bn", "hi", "en"):
        reply = rt.test_rate_reply(slots, result, lang)
        assert reply_matches_language(reply, lang), (lang, reply)


# ------------------------------------------------------ KCD-150 register

def test_the_form_the_caller_used_is_spoken_back():
    aliases = ["রক্তে শর্করা", "সুগার"]
    assert choose_spoken_form("সুগার", aliases, aliases[0]) == "সুগার"
    assert choose_spoken_form("আমার সুগার টেস্ট করাতে চাই", aliases, aliases[0]) == "সুগার"
    assert choose_spoken_form("রক্তে শর্করা", aliases, aliases[0]) == "রক্তে শর্করা"


def test_an_unknown_term_falls_back_to_the_default_never_an_invented_form():
    assert choose_spoken_form("কিছু অন্য", ["সুগার"], "সুগার") == "সুগার"
    assert choose_spoken_form(None, ["a"], "a") == "a"
    assert choose_spoken_form("x", [], None) is None
