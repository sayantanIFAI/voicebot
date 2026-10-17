"""GATE TEST (owner's rule, 2026-10-17): letters pronounced separately must stay separate and be read as the test code they spell -- for EVERY
test code, in Bengali, Hindi and English, in any position in the sentence. Not a hand-picked few: the codes below are the owner's list plus every
letters-only code in the clinic catalogue; add a code to CODES and the whole matrix runs for it."""

import importlib.util
import pathlib
import sys

import pytest

from agent import spelled_letters as sl

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("spelled_forms", ROOT / "clinic-api" / "spelled_forms.py")
sf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sf)

# the owner's list, then the catalogue's own codes
CODES = ["CBC", "TSH", "PSA", "ESR", "EEG", "ECG", "HIV", "FFS", "CRP", "KFT", "LFT", "TMT", "HCV"]
CATALOGUE = {c: f"Test {c}" for c in CODES}


def _spell(code: str, table: dict[str, str], joiner: str = " ") -> str:
    return joiner.join(table[c] for c in code)


FRAMES_BN = ["{x}", "{x} টেস্ট কত", "{x} টেস্টের দাম কত", "আমার {x} করাতে হবে", "আপনাদের ওখানে {x} হয়", "এই যে {x} এর রিপোর্ট কবে পাব"]
FRAMES_HI = ["{x}", "{x} का रेट क्या है", "मुझे {x} करवाना है", "आपके यहाँ {x} होता है"]
FRAMES_EN = ["{x}", "what is the rate of {x}", "do you do {x}", "i need {x} test"]


@pytest.mark.parametrize("code", CODES)
def test_bengali_letter_names_are_read_as_the_code(code):
    for frame in FRAMES_BN:
        text = frame.format(x=_spell(code, sf.BN_LETTER))
        out, found = sl.normalise_spelled_codes(text, CATALOGUE)
        assert found == [code], (text, out, found)
        assert code in out.split(), (text, out)


@pytest.mark.parametrize("code", CODES)
def test_hindi_letter_names_are_read_as_the_code(code):
    for frame in FRAMES_HI:
        text = frame.format(x=_spell(code, sf.HI_LETTER))
        out, found = sl.normalise_spelled_codes(text, CATALOGUE)
        assert found == [code], (text, out, found)


@pytest.mark.parametrize("code", CODES)
def test_latin_single_letters_are_read_as_the_code(code):
    for frame in FRAMES_EN:
        text = frame.format(x=_spell(code.lower(), {c: c for c in code.lower()}))
        out, found = sl.normalise_spelled_codes(text, CATALOGUE)
        assert found == [code], (text, out, found)


def test_two_codes_in_one_sentence_with_the_word_and_between():
    # Bengali "আর" is both "and" and the letter R: it must stay a word between two codes
    text = f"{_spell('CBC', sf.BN_LETTER)} আর {_spell('ESR', sf.BN_LETTER)} কত"
    out, found = sl.normalise_spelled_codes(text, CATALOGUE)
    assert found == ["CBC", "ESR"] and "আর" in out.split()


def test_ordinary_sentences_are_left_alone():
    for text in ["আমি আর ও যাব", "আর ই", "আপনাদের ক্লিনিক কোথায়", "मैं आर ओ जाऊँगा", "what is the address", "a b"]:
        assert sl.normalise_spelled_codes(text, CATALOGUE) == (text, [])


def test_a_near_miss_becomes_latin_letters_for_the_lookup_to_question():
    out, found = sl.normalise_spelled_codes("ডি এস এইচ টেস্ট", CATALOGUE)
    assert found == [] and out.split()[0] == "DSH"


@pytest.mark.parametrize("code", CODES)
def test_the_clinic_catalogue_generates_the_spelled_aliases(code):
    name = f"Some Test ({code})"
    assert _spell(code, sf.BN_LETTER) in sf.forms_bn(name)
    assert _spell(code, sf.BN_LETTER, "") in sf.forms_bn(name)
    assert _spell(code, sf.HI_LETTER) in sf.forms_hi(name)


def test_agent_and_clinic_letter_tables_agree():
    for letter, bn in sf.BN_LETTER.items():
        assert sl._LETTER[bn] == letter.lower(), (letter, bn)
    for letter, hi in sf.HI_LETTER.items():
        assert sl._LETTER[hi] == letter.lower(), (letter, hi)
