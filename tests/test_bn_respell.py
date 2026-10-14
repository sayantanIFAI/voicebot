"""agent/bn_respell.py: a Bengali word the voice says badly is respelled; nothing else in the sentence is touched."""

from __future__ import annotations

from agent import bn_respell


def test_a_listed_word_is_respelled_in_place():
    out = bn_respell.apply("দুঃখিত, সময়টা ধরে রাখা যায়নি।", "bn")
    assert out == "দুঃখিত, সমোইটা ধরে রাখা যায়নি।"


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
