"""agent/asr_refine.py: choosing between texts the recogniser already heard, against a small synthetic catalogue (no pod, no model)."""

from agent.asr_refine import AsrRefiner, parse_mode

CAT = {
    "tests": [
        {"name": "Thyroid Profile", "aliases_bn": ["থাইরয়েড"], "aliases_hi": ["थायराइड"]},
        {"name": "Complete Blood Count (CBC)", "aliases_bn": ["সিবিসি"], "aliases_hi": []},
    ],
    "doctors": [
        {"name": "Dr. A. Sen", "full_name": "Dr. Arindam Sen", "surname": "Sen", "aliases_bn": ["সেন"], "aliases_hi": []},
        {"name": "Dr. B. Sen", "full_name": "Dr. Biswajit Sen", "surname": "Sen", "aliases_bn": ["সেন"], "aliases_hi": []},
        {"name": "Dr. P. Ghosh", "full_name": "Dr. Prabir Ghosh", "surname": "Ghosh", "aliases_bn": ["ঘোষ"], "aliases_hi": []},
    ],
}
R = AsrRefiner(CAT)


def test_one_text_heard_is_nothing_to_choose():
    assert R.decide("থাইরয়েড টেস্টের দাম কত", [], "bn").action == "none"
    assert R.decide("থাইরয়েড টেস্টের দাম কত", ["থাইরয়েড টেস্টের দাম কত"], "bn").action == "none"


def test_chosen_text_that_already_names_an_entry_is_kept():
    d = R.decide("থাইরয়েড টেস্টের দাম কত", ["সিবিসি টেস্টের দাম কত"], "bn")
    assert d.action == "keep"


def test_switch_only_when_the_other_text_names_exactly_one_entry():
    d = R.decide("থাইর টেস্টের দাম কত", ["থাইরয়েড টেস্টের দাম কত"], "bn")
    assert (d.action, d.kind, d.entry, d.text) == ("switch", "test", "Thyroid Profile", "থাইরয়েড টেস্টের দাম কত")


def test_two_doctors_fit_means_ask_never_choose():
    d = R.decide("ডাক্তারের অ্যাপয়েন্টমেন্ট", ["সেন ডাক্তারের অ্যাপয়েন্টমেন্ট"], "bn")
    assert d.action == "ask" and "Sen" in d.entry


def test_nothing_named_anywhere_is_kept():
    assert R.decide("আজকের আবহাওয়া কেমন", ["আজকের আবহাওয়া কেমন আছে"], "bn").action == "keep"


def test_a_sound_alike_does_not_switch():
    # near the written form but not it: agent/gazetteer.py only suggests, and so does this layer's absence of a switch
    assert R.decide("থাইরয় টেস্ট", ["থাইরোয় টেস্ট"], "bn").action in ("keep", "none")


def test_mode_parsing_defaults_to_off():
    assert [parse_mode(v) for v in (None, "", "SHADOW", "on", "nonsense")] == ["off", "off", "shadow", "on", "off"]
