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
