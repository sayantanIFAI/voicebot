from agent import test_overview as ov

CAT = [
    {"name": "Complete Blood Count (CBC)", "sample_type": "Blood", "aliases_bn": ["সিবিসি"], "aliases_hi": ["सीबीसी"]},
    {"name": "Lipid Profile", "sample_type": "Blood", "aliases_bn": ["লিপিড প্রোফাইল"], "aliases_hi": ["लिपिड प्रोफाइल"]},
    {"name": "Urine Routine Examination", "sample_type": "Urine", "aliases_bn": ["ইউরিন টেস্ট"], "aliases_hi": ["यूरिन टेस्ट"]},
    {"name": "ECG", "sample_type": "Cardiac", "aliases_bn": ["ইসিজি"], "aliases_hi": ["ईसीजी"]},
    {"name": "Chest X-Ray (PA view)", "sample_type": "Imaging", "aliases_bn": ["বুকের এক্স রে"], "aliases_hi": ["छाती का एक्सरे"]},
]


def test_general_questions_in_three_languages_are_recognised():
    for t in [
        "আপনাদের ওখানে কি কি টেস্ট হয়",
        "আপনি বলুন আগে আপনাদের ওখানে কি ক্দ টেস্ট ত এইসব ধরনের টেস্ট হয়",
        "कौन कौन से टेस्ट होते हैं",
        "क्या क्या टेस्ट होते हैं आपके यहाँ",
        "what tests do you do",
        "what kind of tests do you have",
        "list of tests",
    ]:
        assert ov.asks_which_tests(t), t


def test_other_questions_are_not_the_overview():
    for t in ["সিবিসি টেস্টের দাম কত", "what is the price of the tests", "আমি টেস্ট বুক করতে চাই", "आपका पता क्या है", "ECG test hoy"]:
        assert not ov.asks_which_tests(t), t


def test_the_reply_lists_only_catalogue_kinds_and_names():
    r = ov.reply(CAT, "bn")
    assert "সিবিসি" in r and "লিপিড প্রোফাইল" in r and "ইসিজি" in r and "বুকের এক্স রে" in r and "প্রস্রাবের" in r
    assert r.endswith("কোন টেস্টের কথা জানতে চান?")
    assert "Complete Blood Count" in ov.reply(CAT, "en")
    assert "सीबीसी" in ov.reply(CAT, "hi")


def test_a_kind_the_catalogue_lacks_is_not_mentioned():
    r = ov.reply([CAT[0], CAT[1]], "en")
    assert "urine" not in r and "heart" not in r and "imaging" not in r
    assert ov.reply([], "en") is None
