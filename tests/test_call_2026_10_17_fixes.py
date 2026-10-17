"""Regression tests for the two failures in the real call of 2026-10-17 (a sentence cut off mid-word; a three-test total
where the recogniser garbled the names). Sentences are the caller's own. Pure functions: no GPU, no model, no network."""

from __future__ import annotations

import pytest

from agent import multi_test_confirm
from agent.endpointing import EndpointConfig, classify_completeness, never_shorter_than
from agent.pause_profile import PauseProfile
from agent.slot_grounding import ground_slots, is_plausible_test_name

# ------------------------------------------------ 1. cut off while speaking


@pytest.mark.parametrize(
    "text, lang",
    [
        ("আমাকে বলুন তো এই ক্ব", "bn"),  # the exact fragment that was cut
        ("आप मुझे ये ट्र", "hi"),
        ("ঠিক আছে আমাকে বলুন তো এই ক্ব টেস্ট", "bn"),
    ],
)
def test_a_sentence_ending_on_a_half_spoken_word_is_unfinished(text, lang):
    assert classify_completeness(text, lang) == "incomplete"


@pytest.mark.parametrize(
    "text, lang",
    [("আমার নাম রাহুল", "bn"), ("সিবিসি", "bn"), ("না", "bn"), ("yes", "en"), ("সিবিসি টেস্টের দাম কত", "bn"), ("क्या आप कल आ सकते हैं", "hi")],
)
def test_ordinary_endings_are_not_mistaken_for_a_half_spoken_word(text, lang):
    assert classify_completeness(text, lang) != "incomplete"


@pytest.mark.parametrize("gap", [0.15, 0.25, 0.35, 0.5])
def test_the_adaptive_pause_of_a_fast_speaker_can_no_longer_cut_below_the_baseline(gap):
    base = EndpointConfig()
    profile = PauseProfile()
    t = 0.0
    for _ in range(3):
        spans = []
        for _ in range(4):
            spans.append({"start": t, "end": t + 0.5})
            t += 0.5 + gap
        profile.observe_utterance(spans, upto_s=t)
    adapted = profile.config(base)
    assert adapted.complete_confirm_s < base.silence_confirm_s  # without the floor this caller WOULD be cut early
    fixed = never_shorter_than(adapted, base, 2.5)
    assert fixed.silence_confirm_s >= base.silence_confirm_s
    assert fixed.complete_confirm_s >= base.silence_confirm_s
    assert fixed.incomplete_confirm_s >= 2.5


def test_a_slow_speakers_longer_adapted_wait_is_kept():
    base = EndpointConfig()
    slow = EndpointConfig(silence_confirm_s=1.8, complete_confirm_s=1.2, incomplete_confirm_s=2.7)
    kept = never_shorter_than(slow, base, 2.5)
    assert kept.silence_confirm_s == 1.8 and kept.incomplete_confirm_s == 2.7


# ------------------------------------------------ 2. the three-test total with garbled names

SAID = "আমাকে বলুন তো এই ক্রিকাসিড আর র আর ওই টেস্ট টা ওই যে আপনাদের থাইরাইড টেস্ট এই তিনটের টোটাল কত টাকা পড়বে"


@pytest.mark.parametrize("junk", ["র", "ওই", "আর", "টেস্ট", "the", "এই"])
def test_function_words_and_single_letters_are_never_test_names(junk):
    assert not is_plausible_test_name(junk)


@pytest.mark.parametrize("real", ["ক্রিকাসিড", "থাইরাইড টেস্ট", "CBC", "সিবিসি", "ইউরিক অ্যাসিড", "ESR", "TSH"])
def test_real_and_garbled_test_names_are_kept(real):
    assert is_plausible_test_name(real)


def test_the_junk_the_model_returned_is_dropped_even_though_the_sentence_is_about_tests():
    slots, dropped = ground_slots(SAID, {"test_names": ["ক্রিকাসিড", "র", "ওই", "থাইরাইড টেস্ট"]})
    assert slots["test_names"] == ["ক্রিকাসিড", "থাইরাইড টেস্ট"]
    assert dropped == ["র", "ওই"]


URIC = {
    "found": False,
    "did_you_mean": ["Uric Acid", "Serum Creatinine"],
    "did_you_mean_bn": ["ইউরিক অ্যাসিড", "ক্রিয়াটিনিন"],
    "did_you_mean_hi": ["यूरिक एसिड", "क्रिएटिनिन"],
}
THYROID = {
    "found": False,
    "needs_confirmation": True,
    "did_you_mean": ["Thyroid Profile (T3 T4 TSH)"],
    "did_you_mean_bn": ["থাইরয়েড প্রোফাইল"],
    "did_you_mean_hi": ["थायराइड प्रोफाइल"],
}


def test_garbled_names_with_sound_alikes_are_read_back_once_and_priced_only_after_a_yes():
    question, canonical = multi_test_confirm.plan([("ক্রিকাসিড", URIC), ("থাইরাইড টেস্ট", THYROID)], "bn")
    assert "ইউরিক অ্যাসিড" in question and "থাইরয়েড প্রোফাইল" in question and "টোটাল" in question
    assert canonical == ["Uric Acid", "Thyroid Profile (T3 T4 TSH)"]  # exact catalogue names: the replay after "yes" matches exactly
    assert not any(ch.isdigit() for ch in question)  # nothing priced before the caller confirms


def test_the_question_is_asked_in_the_callers_language():
    for lang, marker in (("hi", "यूरिक एसिड"), ("en", "Uric Acid")):
        question, _ = multi_test_confirm.plan([("x", URIC), ("y", THYROID)], lang)
        assert marker in question


def test_a_test_that_matched_exactly_is_kept_alongside_the_suggested_ones():
    cbc = {"found": True, "test_name": "Complete Blood Count (CBC)", "test_name_bn": "সিবিসি", "rate_inr": 400}
    question, canonical = multi_test_confirm.plan([("সিবিসি", cbc), ("ক্রিকাসিড", URIC)], "bn")
    assert canonical == ["Complete Blood Count (CBC)", "Uric Acid"] and "সিবিসি" in question and "ইউরিক অ্যাসিড" in question


def test_a_name_with_no_sound_alike_keeps_the_ordinary_not_found_reply():
    assert multi_test_confirm.plan([("ক্রিকাসিড", URIC), ("zzzz", {"found": False})], "bn") is None


def test_an_ambiguous_name_is_never_guessed():
    ambiguous = {"found": False, "ambiguous": True, "did_you_mean": ["ESR", "Blood Sugar PP"]}
    assert multi_test_confirm.plan([("র", ambiguous), ("ক্রিকাসিড", URIC)], "bn") is None


def test_when_everything_matched_by_its_written_form_there_is_nothing_to_confirm():
    a = {"found": True, "test_name": "ECG", "test_name_bn": "ইসিজি"}
    b = {"found": True, "test_name": "Uric Acid", "test_name_bn": "ইউরিক অ্যাসিড"}
    assert multi_test_confirm.plan([("ইসিজি", a), ("ইউরিক অ্যাসিড", b)], "bn") is None


def test_one_test_heard_twice_is_asked_about_once():
    twice = multi_test_confirm.plan([("ক্রিকাসিড", URIC), ("ক্রিকাসিড", URIC), ("থাইরাইড টেস্ট", THYROID)], "bn")
    assert twice is not None and twice[1] == ["Uric Acid", "Thyroid Profile (T3 T4 TSH)"]


# ------------------------------------------------ 3. "what's your location" answered in Hindi


class _Heard:
    def __init__(self, text, agreement):
        self.text, self.decoder_agreement = text, agreement


@pytest.mark.parametrize(
    "text, agreement",
    [
        # measured on the pod: language ID called every one of these Hindi (0.72-0.99); the English recogniser heard them
        ("what is your location", 0.6),
        ("what is your location", 1.0),
        ("where is your klein lo", 0.67),
        ("what are your timings", 0.6),
        ("how much is the", 0.8),
        ("i want to book an appointment", 1.0),
        ("you take", 1.0),
    ],
)
def test_a_clean_english_sentence_overrules_a_confident_hindi_language_id(text, agreement):
    from agent.lang_select import english_overrules_language_id

    assert english_overrules_language_id(_Heard(text, agreement))


@pytest.mark.parametrize(
    "text, agreement",
    [
        ("sibisi test rate koto", 0.9),  # romanised Bengali the English engine wrote out: too few real English words
        ("ami kal ashbo ekhane", 1.0),
        ("what is your location", 0.3),  # the English decoders disagree: not trusted
        ("ok", 1.0),  # one word is not enough to overrule anything
        ("वाट्स योर लोकेशन", 1.0),  # not Latin script
        ("hmm uh", 1.0),
        ("", 0.0),
    ],
)
def test_noise_romanised_indic_and_weak_results_do_not_overrule_language_id(text, agreement):
    from agent.lang_select import english_overrules_language_id

    assert not english_overrules_language_id(_Heard(text, agreement))


def test_a_garbled_first_utterance_that_tie_breaks_to_english_is_still_not_decisive():
    # real call 2026-10-12: "hano parking a chair" must not lock a whole Bengali call to English
    from agent.lang_select import english_overrules_language_id

    assert not english_overrules_language_id(_Heard("hano parking a chair", 1.0))
    assert not english_overrules_language_id(_Heard("sob mila ke koto hobe", 0.9))


# ------------------------------------------------ 4. wrong quick answers: the address for a price question and for "timings"

from agent import fastpath_guards  # noqa: E402
from agent.lang_select import language_of_script  # noqa: E402

HINDI_TOTAL = "मेरे को बोलिए ये सी बी सी यूरी का सीट थाइरॉइड टेस्ट तीनों का भाव एक साथ कितना पड़ेगा"


@pytest.mark.parametrize(
    "text, fallback, expected",
    [
        (HINDI_TOTAL, "bn", "hi"),  # the real sentence: said in Hindi inside a call whose reply language was Bengali
        ("what is your location", "bn", "en"),
        ("আপনাদের ঠিকানা কী", "hi", "bn"),
        ("ECG টেস্টের দাম কত", "hi", "bn"),  # mixed: the Bengali script is the majority
        ("12345", "hi", "hi"),  # no letters: keep the fallback
        ("", "bn", "bn"),
    ],
)
def test_the_language_of_a_sentence_is_the_script_it_is_written_in(text, fallback, expected):
    assert language_of_script(text, fallback) == expected


def test_a_location_answer_needs_a_word_about_a_place():
    loc = {"faq_topic": "location"}
    assert fastpath_guards.veto("clinic_faq", loc, "what are your timings", lambda: 0)  # the live mistake: answered with the address
    assert fastpath_guards.veto("clinic_faq", loc, "what are your timing", lambda: 0)
    for ok in ("what is your location", "where is the clinic", "ঠিকানাটা কী", "आपका पता क्या है", "how do I reach you", "আপনারা কোথায়"):
        assert fastpath_guards.veto("clinic_faq", loc, ok, lambda: 0) is None, ok


def test_other_topics_and_intents_are_never_vetoed_by_the_location_rule():
    assert fastpath_guards.veto("clinic_faq", {"faq_topic": "hours"}, "what are your timings", lambda: 0) is None
    assert fastpath_guards.veto("test_rate", {"test_name": "CBC"}, "what is the price of the CBC test", lambda: 1) is None


def test_a_total_question_that_names_only_one_test_goes_to_the_model():
    assert fastpath_guards.veto("test_rate", {"test_name": "सी बी सी"}, HINDI_TOTAL, lambda: 1)
    assert fastpath_guards.veto("test_rate", {"test_name": "CBC"}, "CBC and ECG together how much", lambda: 1)
    assert fastpath_guards.veto("test_rate", {"test_name": "CBC"}, "CBC আর ইউরিক অ্যাসিডের টোটাল কত", lambda: 0)


def test_a_total_question_that_names_every_test_is_answered_normally():
    assert fastpath_guards.veto("test_rate", {"test_name": "CBC"}, "CBC and ECG total how much", lambda: 2) is None


def test_a_plain_price_question_is_never_treated_as_a_total():
    for text in ("CBC টেস্টের দাম কত", "सीबीसी का रेट क्या है", "what is the price of the CBC test", "how much is a thyroid test"):
        assert not fastpath_guards.asks_for_total(text), text


def test_a_failing_test_count_never_lets_the_shortcut_answer_a_total():
    def boom():
        raise RuntimeError("catalogue not ready")

    assert fastpath_guards.veto("test_rate", {"test_name": "CBC"}, "CBC and ECG total", boom)


# ---- a test lost entirely: "all three together" but fewer names understood -> ask back, never price a partial list
from agent import multi_test_confirm as _mtc


def _found(name, bn=None, hi=None):
    return {"found": True, "test_name": name, "test_name_bn": bn or name, "test_name_hi": hi or name}


def test_expected_count_needs_a_total_cue():
    assert _mtc.expected_count("cbc uric acid thyroid all three together price") == 3
    assert _mtc.expected_count("মেরে को बोलिए सीबीसी यूरिक एसिड थाइरॉइड तीनों का भाव एक साथ कितना") == 3
    assert _mtc.expected_count("এই তিনটের টোটাল কত টাকা পড়বে") == 3
    assert _mtc.expected_count("book me in three days") is None  # no total asked for
    assert _mtc.expected_count("what is the total") is None


def test_shortfall_asks_back_when_a_test_was_lost():
    results = [("uric acid", _found("Uric Acid")), ("thyroid", _found("Thyroid Profile (T3 T4 TSH)"))]
    q = _mtc.shortfall(results, 3, "en")
    assert q and "Uric Acid" in q and "Thyroid" in q and "three" in q
    assert _mtc.shortfall(results, 3, "hi") and _mtc.shortfall(results, 3, "bn")


def test_shortfall_none_when_everything_understood_or_no_count():
    results = [("a", _found("Uric Acid")), ("b", _found("Thyroid Profile"))]
    assert _mtc.shortfall(results, 2, "en") is None
    assert _mtc.shortfall(results, None, "en") is None


def test_shortfall_with_no_understood_name_still_asks():
    q = _mtc.shortfall([("zzz", {"found": False})], 3, "en")
    assert q and "three" in q


def test_shortfall_counts_a_sound_alike_as_understood():
    r = [("ক্রিকাসিড", {"found": False, "did_you_mean": ["Uric Acid"]}), ("থাইরাইড", {"found": False, "did_you_mean": ["Thyroid Profile"]})]
    assert _mtc.shortfall(r, 2, "en") is None
    assert _mtc.shortfall(r, 3, "en")


# ---- a test code said letter by letter that the Bengali recogniser wrote as a made-up word (agent/spelled_letters.py)
from agent import spelled_letters as _sl

_CODES = {"CBC": "Complete Blood Count (CBC)", "HIV": "HIV Test (ELISA)", "ESR": "ESR", "TSH": "TSH"}


def test_spelled_letters_become_a_code_candidate():
    assert _sl.match_codes("c b c tasted them", _CODES) == [("CBC", "Complete Blood Count (CBC)", True)]
    assert _sl.match_codes("amar c b c tist coroth", _CODES)[0][0] == "CBC"


def test_a_near_spelled_code_is_a_suggestion_not_exact():
    # real English-recogniser output for a spoken "H I V" was "h i b"
    assert _sl.match_codes("h i b test kot", _CODES) == [("HIB", "HIV Test (ELISA)", False)]


def test_ordinary_english_words_never_match_a_code():
    assert _sl.match_codes("he hit them", _CODES) == []
    assert _sl.match_codes("his test", _CODES) == []
    assert _sl.match_codes("tasted them", _CODES) == []


def test_a_whole_token_code_matches_only_exactly():
    assert _sl.match_codes("what is the esr", _CODES)[0][:3:2] == ("ESR", True)
    assert _sl.match_codes("what is the esp", _CODES) == []


def test_a_code_close_to_two_tests_is_no_evidence():
    codes = {"HIV": "A", "HIB": "B"}
    assert _sl.match_codes("h i c", codes) == [] or len(_sl.match_codes("h i c", codes)) == 0


# ---- English question frames written in Bengali/Devanagari letters; spelled letters do not count against English; how-to-reach cues
from agent import lang_select as _ls
from agent import test_availability as _ta
from agent import translit_frames as _tf


def test_do_you_do_frame_in_bengali_and_hindi_letters():
    assert _tf.match("দো ইউ ডু চেস্ট এক্স রে") == "চেস্ট এক্স রে"
    assert _tf.match("दो यू डू चेस्ट एक्स रे") == "चेस्ट एक्स रे"
    assert _tf.match("वू यू डू चेस्ट रेट") == "चेस्ट रेट"
    assert _tf.match("डू यू हैव थाइरॉइड टेस्ट") == "थाइरॉइड"


def test_frame_needs_the_whole_frame_and_a_name():
    assert _tf.match("আপনারা কি এক্স রে করেন") is None
    assert _tf.match("दो यू डू") is None
    assert _tf.match("what is the rate") is None


def test_the_frame_makes_it_an_availability_question():
    assert _ta.asks_if_test_is_available("দো ইউ ডু চেস্ট এক্স রে", "en")
    assert not _ta.asks_if_test_is_available("চেস্ট এক্স রে কত", "bn")


def test_spelled_letters_do_not_count_against_english():
    assert _ls.english_word_share("what is the rate of c b c and eurica it together") >= 0.6
    assert _ls.english_word_share("sibisi test rate koto") < 0.6
    assert _ls.english_word_share("i want a test") == 1.0  # "i" and "a" are words and still count


# ---- "d s h test" is handed to the catalogue lookup, hesitations are not a finished sentence, no Hindi জি in Bengali replies
from agent.endpointing import classify_completeness as _cc


def test_a_sentence_of_only_spelled_letters_and_test_is_a_code_request():
    assert _sl.bare_code_request("d s h test") == "DSH"
    assert _sl.bare_code_request("ডি এস এইচ টেস্ট") == "DSH"
    assert _sl.bare_code_request("डी एस एच टेस्ट") == "DSH"
    assert _sl.bare_code_request("c b c") == "CBC"


def test_more_than_a_code_is_not_a_bare_code_request():
    assert _sl.bare_code_request("what is the rate of c b c") is None
    assert _sl.bare_code_request("d s h test price") is None
    assert _sl.bare_code_request("a b") is None  # too short to be a code
    assert _sl.bare_code_request("hello there") is None


def test_a_hesitation_or_opener_is_not_a_finished_sentence():
    for text, lang in [("it's", "en"), ("it's uh", "en"), ("what", "en"), ("आपके", "hi"), ("आप लोगों के साथ", "hi"), ("আপনাদের", "bn")]:
        assert _cc(text, lang) == "incomplete", text
    assert _cc("yes", "en") == "complete" and _cc("thank you", "en") == "complete"


def test_no_hindi_ji_in_bengali_replies():
    """"জি" is Hindi "ji"; West Bengal says "হ্যাঁ" / "আচ্ছা". Checked on the spoken text of the three places it used to be."""
    import pathlib
    from agent import phrases, test_availability
    assert not _re_ji.search(test_availability.availability_reply(["Lipid Profile"], "bn"))
    assert not _re_ji.search(phrases.phrase("silence_go_on", "bn"))
    src = pathlib.Path(__file__).resolve().parents[1].joinpath("agent", "senior_care.py").read_text(encoding="utf-8")
    assert not any(_re_ji.search(line) for line in src.splitlines() if '"bn":' in line)


import re as _re0
_re_ji = _re0.compile("(?<![\u0980-\u09FF])জি(?![\u0980-\u09FF])")


def test_a_code_inside_a_named_test_is_not_a_further_test():
    # "T S H" next to a named Thyroid Profile (T3 T4 TSH) must not become a fourth test
    codes = {"TSH": "TSH", "CBC": "Complete Blood Count (CBC)"}
    found = _sl.match_codes("c b c ya rosa t s h complete blood profile", codes)
    already = {"Lipid Profile", "Thyroid Profile (T3 T4 TSH)"}
    kept = [c for c, name, _ in found if name not in already and not any(c.upper() in a.upper() for a in already)]
    assert kept == ["CBC"]


# ---- long clips are cut at their pauses; the ask-back says "তিন টেস্ট" not "তিনটিটি"
import numpy as _np
from agent import clip_chunks as _cl


def _speech_with_pauses(sr=16000, seg_s=(7.0, 6.0, 4.0), gap_s=0.5):  # pauses fall inside each 6.6-11 s search window
    rng = _np.random.default_rng(1)
    parts = []
    for i, s in enumerate(seg_s):
        parts.append((rng.standard_normal(int(s * sr)) * 3000).astype(_np.int16))
        if i < len(seg_s) - 1:
            parts.append(_np.zeros(int(gap_s * sr), dtype=_np.int16))
    return _np.concatenate(parts)


def test_a_short_clip_is_not_cut():
    assert _cl.split_points(_speech_with_pauses(seg_s=(4.0, 4.0)), 16000, 11.0) == []


def test_a_long_clip_is_cut_in_a_pause_not_in_speech():
    x = _speech_with_pauses()
    cuts = _cl.split_points(x, 16000, 11.0)
    assert cuts, "a 17.5 s clip must be cut"
    for c in cuts:
        assert _np.abs(x[c - 800:c + 800]).max() < 1000, "the cut must fall in the silence between segments"
    pieces = _np.diff([0, *cuts, len(x)])
    assert (pieces <= 11.0 * 16000 + 1).all()


def test_ask_back_does_not_double_the_bengali_ti():
    q = _mtc.shortfall([("a", _found("Uric Acid")), ("b", _found("TSH"))], 3, "bn")
    assert "তিনটিটি" not in q and "তিনটি টেস্ট" in q


# ---- an availability question whose test NAME was lost is still an availability question
def test_availability_question_without_a_readable_name():
    for text, lang in [("ক্গ টেস্ট হয় ওখানে", "bn"), ("কি টেস্ট হয় আপনাদের ওখানে", "bn"), ("तम्त टेस्ट हो अपना दरोखा ना", "hi"), ("ইক টেস্ট হয় ওখানে", "bn")]:
        assert _ta.asks_if_test_is_available(text, lang), text
    # but not when it is about something else
    for text, lang in [("টেস্ট হয়ে যাওয়ার পর রিপোর্ট কবে", "bn"), ("সিবিসি টেস্টের দাম কত", "bn"), ("আমি টেস্ট করাতে চাই", "bn")]:
        assert not _ta.asks_if_test_is_available(text, lang), text


# ---- "book that time" after an availability answer; the doctor title is always the word "doctor"
from agent import booking_carryover as _bc
from agent import speech_norm as _sn


def test_that_time_after_an_availability_answer_fills_date_and_time():
    ctx = {"doctor_name": "Sen", "date": "2026-10-17", "window": "10:00-12:00", "turn": 3}
    out = _bc.carry_over(ctx, {}, "ওই সময় একটা বুকিং দিয়ে দেন তাহলে আমাকে", None, 4)
    assert out == {"doctor_name": "Sen", "date": "2026-10-17", "time_slot": "10:00"}


def test_that_day_that_time_still_works_when_the_doctor_is_already_in_the_booking():
    ctx = {"doctor_name": "Sen", "date": "2026-10-17", "window": "10:00-12:00", "turn": 3}
    out = _bc.carry_over(ctx, {}, "ওই দিনেরই ওই সময়েই বুকিং নেন", {"doctor_name": "Sen"}, 5)
    assert out == {"date": "2026-10-17", "time_slot": "10:00"}


def test_nothing_is_carried_without_the_pointing_words_or_when_the_caller_names_a_time():
    ctx = {"doctor_name": "Sen", "date": "2026-10-17", "window": "10:00-12:00", "turn": 3}
    assert "time_slot" not in _bc.carry_over(ctx, {}, "বুকিং করুন", None, 4)
    assert _bc.carry_over(ctx, {"time_slot": "11:00"}, "ওই সময়", None, 4)["time_slot"] == "11:00"
    assert _bc.carry_over(ctx, {}, "ওই সময়", None, 9) == {}  # too long ago


def test_a_doctor_title_is_always_said_as_doctor():
    assert _sn.expand_doctor_title("ডাঃ সেন", "bn") == "ডাক্তার সেন"
    assert _sn.expand_doctor_title("Dr. Sen", "en") == "Doctor Sen"
    assert _sn.expand_doctor_title("डॉ. सेन", "hi") == "डॉक्टर सेन"
    assert _sn.expand_doctor_title("drive", "en") == "drive"


def test_decoder_agreement_ignores_how_spelled_letters_are_spaced():
    import difflib
    src = open(__import__("pathlib").Path(__file__).resolve().parents[1] / "agent" / "asr.py", encoding="utf-8").read()
    ns = {"difflib": difflib, "SPACING_AGREEMENT_MIN": 0.8}
    exec(src[src.index("def _word_agreement"): src.index("logger = logging")], ns)
    f = ns["_word_agreement"]
    assert f("सीबीसी ई सी जी टी एस एच", "सी बी स  स जी टीएसएच") >= 0.8
    assert f("আমি যাব", "আমি যাব") == 1.0
    assert f("ক্ব ইক", "কগ ত") < 0.5  # two different mis-hearings still disagree


# ---- a noisy English reading must not beat clean Bengali inside an established Bengali call
def test_noisy_english_does_not_beat_clean_bengali_in_a_bengali_call():
    from types import SimpleNamespace as R

    bn = R(text="তো আমার একটা বুকিং করে দিন কালকে", decoder_agreement=1.0)
    en = R(text="the booking go to then galva", decoder_agreement=0.62)
    lid = {"bn": 0.97, "en": 0.0, "hi": 0.03}
    assert _ls.pick_candidate([("bn", bn), ("en", en)], lid, "bn")[0] == "bn"
    # clearly English still takes the turn, even mid-Bengali-call
    clear = R(text="what is the price of the blood test", decoder_agreement=0.9)
    assert _ls.pick_candidate([("bn", bn), ("en", clear)], lid, "bn")[0] == "en"
    # and on the first turn (no established call language) the old rule stands
    assert _ls.pick_candidate([("bn", bn), ("en", en)], lid, None)[0] == "en"


# ---- "<your place> X <is done>" with a garbled X that is close to a test the clinic does
def test_the_have_frame_finds_the_words_between_the_place_and_the_verb():
    assert _ta.have_frame_name("আপনাদের ওখানে চিস্টিক্স রে হয়") == "চিস্টিক্স রে"
    assert _ta.have_frame_name("आपके यहाँ चेस्ट एक्स रे होता है") == "चेस्ट एक्स रे"
    assert _ta.have_frame_name("আমি কাল যাব") is None
    assert _ta.asks_if_test_is_available("আপনাদের ওখানে চিস্টিক্স রে হয়", "bn")
