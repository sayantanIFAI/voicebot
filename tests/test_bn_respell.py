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
        assert spelling and (spelling != word or word == "সাত")  # সাত: pinned as written by the listener (7), so the sh rule leaves it


def test_the_words_a_native_listener_chose_are_in_the_table():
    # SPEC CHANGE 2026-10-17: the listener's second round chose নেওয়া -> নেওআ (was নেওয়াা), যাওয়ার -> যাওয়াার, স্যাম্পল -> স্যাম্পেল;
    # কয়েক was chosen from the user's own recording. আরও stays as written (candidate 1 was the best).
    chosen = {"প্রয়োজন": "প্রোজোন", "সাড়া": "শাঁড়া", "সময়টা": "শমোইটা", "নেওয়া": "নেওআ", "যাওয়ার": "যাওয়াার",
              "স্যাম্পল": "স্যাম্পেল", "কয়েক": "কয়এক"}
    for word, spelling in chosen.items():
        assert bn_respell.respell(word) == spelling
    # the words the listener rejected every candidate for, and the one where the plain spelling was the best, stay as written
    for word in ("দুঃখিত", "হয়ে", "আরও"):
        assert bn_respell.respell(word) == word


def test_the_chosen_spelling_carries_over_to_the_same_word_with_an_ending():
    assert bn_respell.respell("নেওয়ার") == "নেওআর"
    assert bn_respell.respell("যাওয়াটা") == "যাওয়াাটা"
    assert bn_respell.respell("কয়েকটা") == "কয়একটা"
    assert bn_respell.respell("স্যাম্পলের") == "স্যাম্পেলের"
    # round 3: নিয়ে chosen (candidate 4); গিয়ে and হয় kept as written
    assert bn_respell.respell("নিয়ে") == "নিইয়ে"
    assert bn_respell.respell("গিয়ে") == "গিয়ে" and bn_respell.respell("হয়") == "হয়"
    # a longer word that merely starts the same is left alone
    assert bn_respell.respell("কয়েকশো") == "কয়েকশো"
    # round 4
    assert bn_respell.respell("ঠিকানা") == "ঠিকানাহ" and bn_respell.respell("ঠিকানাটা") == "ঠিকানাটা"
    assert bn_respell.respell("নদিয়া") == "নোদিয়া" and bn_respell.respell("নদিয়ার") == "নোদিয়ার"


# ---- the "sh" rule: native Bengali words only -------------------------------------------------------------------------


def test_native_words_with_a_plain_sa_get_sha():
    for word, spelling in {
        "সময়": "শময়", "সব": "শব", "সেটা": "শেটা", "সাথে": "শাথে", "সকাল": "শকাল", "সুবিধা": "শুবিধা",
        "সাত": "সাত", "সত্তর": "শত্তর", "সেন": "শেন", "সৌরভ": "শৌরভ", "মাস": "মাশ", "হাসপাতাল": "হাশপাতাল",
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
    assert out == "সাত দিনের মধ্যে শময়ের শাথে সিবিসি রিপোর্ট পাবেন।"


def test_days_numbers_and_inflected_forms_are_covered_too():
    # found missing by scanning every source file, not only the template files
    assert bn_respell.apply("সোমবার সকাল সাড়ে সাতটায় সময়টায় জিনিস সর্দি", "bn") == "শোমবার শকাল শাঁড়ে শাতটায় শময়টায় জিনিশ শর্দি"
    assert bn_respell.respell("সোমবারে") == "শোমবারে"


def test_the_reviewed_list_never_contains_an_english_origin_word():
    loans = {"সিবিসি", "সিটি", "সুগার", "সেন্টার", "ক্যালসিয়াম", "প্রেসক্রিপশন", "সোনোস্ক্যান", "সিরাম", "অ্যাসিড", "সরি", "সিরিয়াল", "সেপ্টেম্বর", "ডিসেম্বর"}
    assert not (loans & bn_respell._NATIVE_SH)


def test_joy_names_are_said_as_two_words_and_ordinary_words_with_second_letter_ya_are_not_cut():
    # Owner, 2026-10-28: জয়দেব -> জয় দেব; the rule is the জয় + consonant-start shape, never বয়স / শয়তান / নয়ন
    for word, spelling in {"জয়দেব": "জয় দেব", "জয়দীপ": "জয় দীপ", "জয়দেবের": "জয় দেবের", "জয়কৃষ্ণ": "জয় কৃষ্ণ", "জয়গোপাল": "জয় গোপাল", "জয়দীপকে": "জয় দীপকে"}.items():
        assert bn_respell.respell(word) == spelling
    for word in ("জয়", "জয়ন্ত", "জয়ন্তী", "জয়শ্রী", "জয়িতা", "জয়ী", "জয়ের", "বয়স", "বয়সের", "শয়তান", "নয়ন", "পয়সা", "ময়দা", "দয়াল", "হয়েছে"):
        assert bn_respell.respell(word) == word
