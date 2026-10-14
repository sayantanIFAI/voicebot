"""agent/bn_respell.py: a Bengali word the voice says badly is respelled; nothing else in the sentence is touched."""

from __future__ import annotations

from agent import bn_respell


def test_a_listed_word_is_respelled_in_place():
    out = bn_respell.apply("দুঃখিত, সময়টা ধরে রাখা যায়নি।", "bn")
    assert out == "দুঃখিত, শমোইটা ধরে রাখা যায়নি।"


def test_only_whole_words_are_replaced():
    # "সময়টাও" (also) is a different word and must not be rewritten by the entry for "সময়টা"
    assert bn_respell.apply("সময়টাও", "bn") == "সময়টাও"


def test_other_languages_and_unlisted_text_are_untouched():
    assert bn_respell.apply("সময়টা", "hi") == "সময়টা"
    assert bn_respell.apply("Your report is ready", "en") == "Your report is ready"
    assert bn_respell.apply("আপনার রিপোর্ট তৈরি।", "bn") == "আপনার রিপোর্ট তৈরি।"
    assert bn_respell.apply("", "bn") == ""


def test_the_switch_turns_it_off(monkeypatch):
    monkeypatch.setenv("BN_RESPELL", "0")
    assert bn_respell.apply("সময়টা", "bn") == "সময়টা"


def test_an_entry_is_never_the_word_itself_or_empty():
    for word, spelling in bn_respell.table().items():
        assert spelling and spelling != word


def test_the_words_a_native_listener_chose_are_in_the_table():
    chosen = {"প্রয়োজন": "প্রোজোন", "সাড়া": "শাঁড়া", "সময়টা": "শমোইটা", "নেওয়া": "নেওয়াা"}
    for word, spelling in chosen.items():
        assert bn_respell.respell(word) == spelling
    # the three the listener rejected every candidate for, and the one where the plain spelling was the best, stay as written
    for word in ("দুঃখিত", "কয়েক", "হয়ে", "যাওয়ার"):
        assert bn_respell.respell(word) == word


# ---- the "sh" rule: native Bengali words only -------------------------------------------------------------------------


def test_native_words_with_a_plain_sa_get_sha():
    for word, spelling in {
        "সময়": "শময়", "সব": "শব", "সেটা": "শেটা", "সাথে": "শাথে", "সকাল": "শকাল", "সুবিধা": "শুবিধা",
        "সাত": "শাত", "সত্তর": "শত্তর", "সেন": "শেন", "সৌরভ": "শৌরভ", "মাস": "মাশ", "হাসপাতাল": "হাশপাতাল",
    }.items():
        assert bn_respell.respell(word) == spelling


def test_a_conjunct_sa_is_never_touched():
    # স্ন in স্নেহাশিস, স্থ, স্প ... stay as written; only a plain স changes
    assert bn_respell.respell("স্নেহাশিস") == "স্নেহাশিশ"
    assert "স্" in bn_respell.respell("সংস্থাগুলোর")


def test_english_origin_words_keep_their_real_s():
    # Bengalis say these with a true "s": the rule must never reach them
    for loan in ("সিবিসি", "সিটি", "সুগার", "ক্যালসিয়াম", "প্রেসক্রিপশন", "সোনোস্ক্যান", "সেন্টার", "অ্যাসিড", "সরি", "সিরাম"):
        assert bn_respell.respell(loan) == loan


def test_every_word_in_the_native_list_really_has_a_plain_sa():
    for word in bn_respell._NATIVE_SH:
        assert bn_respell._sh(word) != word or word in bn_respell.table(), word


def test_the_rule_can_be_switched_off_without_losing_the_chosen_words(monkeypatch):
    monkeypatch.setenv("BN_RESPELL_RULES", "0")
    assert bn_respell.respell("সব") == "সব"
    assert bn_respell.respell("সাড়া") == "শাঁড়া"  # an explicit, listener-chosen entry still applies


def test_a_whole_sentence_is_respelled_word_by_word():
    out = bn_respell.apply("সাত দিনের মধ্যে সময়ের সাথে সিবিসি রিপোর্ট পাবেন।", "bn")
    assert out == "শাত দিনের মধ্যে শময়ের শাথে সিবিসি রিপোর্ট পাবেন।"
