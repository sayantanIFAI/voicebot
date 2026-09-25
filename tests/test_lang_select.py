"""lang_select: the LID-unsure path, pinned to the numbers measured on the pod."""
import dataclasses

from agent.lang_select import languages_to_verify, pick_candidate, script_share, speakable


@dataclasses.dataclass
class R:
    text: str
    decoder_agreement: float = 1.0


ACTIVE = ("bn", "hi", "en")


def test_measured_indian_english_scores_trigger_verification_of_every_language():
    # LID said hi:0.91 for English speech; en:0.09 is the tell.
    assert languages_to_verify("hi", {"bn": 0.00, "hi": 0.91, "en": 0.09}, ACTIVE) == ["hi", "en", "bn"]
    assert languages_to_verify("hi", {"bn": 0.01, "hi": 0.62, "en": 0.36}, ACTIVE) == ["hi", "en", "bn"]


def test_english_that_lid_ranks_third_is_still_tried():
    # measured on the pod: English speech came back bn 0.83 / hi 0.16 / en 0.01
    got = languages_to_verify("unknown", {"bn": 0.83, "hi": 0.16, "en": 0.01}, ACTIVE)
    assert got == ["bn", "hi", "en"] and "en" in got


def test_decisive_lid_skips_extra_asr():
    assert languages_to_verify("bn", {"bn": 1.00, "hi": 0.00, "en": 0.00}, ACTIVE) is None
    assert languages_to_verify("hi", {"bn": 0.00, "hi": 1.00, "en": 0.00}, ACTIVE) is None


def test_bengali_hindi_confusion_is_also_verified():
    assert languages_to_verify("bn", {"bn": 0.51, "hi": 0.49, "en": 0.00}, ACTIVE)[:2] == ["bn", "hi"]


def test_only_active_languages_are_verified():
    assert languages_to_verify("hi", {"bn": 0.1, "hi": 0.8, "en": 0.1}, ("bn", "hi")) == ["hi", "bn"]


def test_unknown_language_with_no_scores_still_tries_everything():
    assert languages_to_verify("unknown", {}, ACTIVE) == ["bn", "hi", "en"]


def test_english_audio_wins_even_when_the_bengali_engine_agrees_with_itself_perfectly():
    # measured: "What is the price of the uric acid test?" (clean)
    #   bn ASR agreement 1.00 (transliterated), hi 0.60, en 0.78
    bn = R("হোয়াট ইজ দ্য প্রাইজ অফ দ্য উইক অ্যাসিড টেস্ট", 1.00)
    hi = R("वॉट इज़ द प्राइस ऑफ द वीक एसिड टेस्ट", 0.60)
    en = R("what is the price of the rick acid test", 0.78)
    assert pick_candidate([("bn", bn), ("hi", hi), ("en", en)])[0] == "en"


def test_bengali_audio_is_not_mistaken_for_english():
    # measured: bn 1.00 / hi 0.50 / en 0.00 on Bengali audio
    bn = R("আমাদের ক্লিনিক প্রতিদিন সকাল আটটা", 1.00)
    hi = R("आमादर क्लिनिक प्रतिदिन सकाल आठ टा", 0.50)
    en = R("ahmad kenic protyin shokhal", 0.00)
    assert pick_candidate([("bn", bn), ("hi", hi), ("en", en)])[0] == "bn"


def test_hindi_audio_is_not_mistaken_for_english_or_bengali():
    # measured: bn 0.40 / hi 0.75 / en 0.38 on "यूरिक एसिड टेस्ट की कीमत क्या है"
    bn = R("ইউরিক অ্যাসিড টেস্ট কি কীমত ক্যা হ্য", 0.40)
    hi = R("यूरिक एसिड टेस्ट की क़ीमत क्या है", 0.75)
    en = R("euric as it restkee kemah", 0.38)
    assert pick_candidate([("hi", hi), ("en", en), ("bn", bn)])[0] == "hi"


def test_english_engine_below_the_gate_does_not_win():
    assert pick_candidate([("hi", R("हाँ", 0.5)), ("en", R("yes", 0.59))])[0] == "hi"
    assert pick_candidate([("hi", R("हाँ", 0.5)), ("en", R("yes", 0.60))])[0] == "en"


def test_english_at_the_lowest_observed_agreement_still_wins():
    # live pod: LID hi 0.63 / en 0.36, agreements hi 0.78, en 0.67, bn 0.60
    hi = R("वॉट इज़ द प्राइस ऑफ़ द वीक एसिड टेस्ट", 0.78)
    en = R("what is the price of the rick acid test", 0.67)
    bn = R("হোয়াট ইজ দ্য প্রাইস", 0.60)
    assert pick_candidate([("hi", hi), ("en", en), ("bn", bn)])[0] == "en"


def test_english_gate_needs_latin_text():
    assert pick_candidate([("hi", R("हाँ", 0.5)), ("en", R("हाँ", 1.0))])[0] == "hi"


def test_bengali_engine_on_hindi_audio_loses_on_measured_agreement():
    # measured: bn ASR on Hindi audio 0.40-0.67; hi ASR on Hindi audio 0.75-1.00
    bn_on_hi = R("হম ক্লিনিক হরদিন সুবহ আট বজে", 0.67)
    hi_on_hi = R("हमारा क्लिनिक हर दिन सुबह आठ बजे", 1.00)
    assert pick_candidate([("bn", bn_on_hi), ("hi", hi_on_hi)])[0] == "hi"


def test_ties_go_to_the_lid_preferred_candidate():
    a, b = R("হ্যাঁ", 0.8), R("हाँ", 0.8)
    assert pick_candidate([("bn", a), ("hi", b)])[0] == "bn"


def test_empty_transcripts_are_never_chosen():
    assert pick_candidate([("hi", R("", 1.0)), ("en", R("yes", 0.9))])[0] == "en"
    assert pick_candidate([("hi", R("", 1.0)), ("en", R("", 1.0))])[0] == "hi"


def test_speakable_rejects_wrong_script_and_untalkable_spans():
    assert speakable("नमस्कार, मैं क्या मदद कर सकती हूँ?", "hi")
    assert not speakable("নমস্কার", "hi")
    assert not speakable("hello there", "hi")
    assert not speakable("नमस्कार", "en")
    assert speakable("Hello, how can I help?", "en")
    assert not speakable("", "bn")


def test_a_bengali_reply_ending_in_a_danda_is_speakable_and_devanagari_letters_still_are_not():
    """The danda is the full stop of Bengali too; it lives in the Devanagari block and used to make every Bengali
    reply that ended in one 'the wrong script'."""
    assert speakable("ধন্যবাদ" + chr(0x0964) + " আর কিছু জানতে চান?", "bn")
    assert speakable("ধন্যবাদ" + chr(0x0965), "bn")
    assert not speakable("नमस्कार, मैं क्या मदद कर सकती हूँ?", "bn")       # real Devanagari letters: still rejected
    assert speakable("नमस्कार" + chr(0x0964), "hi")
