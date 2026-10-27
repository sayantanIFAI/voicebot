from agent.distress import DistressEvidence, asks_for_person, soft_words, strong_words


def test_plain_distress_words_are_strong_in_all_three_languages():
    assert strong_words("আমি খুব ভয় পাচ্ছি")
    assert strong_words("मुझे बहुत डर लग रहा है")
    assert strong_words("I am so scared please")
    assert not strong_words("ডাঃ সেনের অ্যাপয়েন্টমেন্ট চাই")
    assert not strong_words("quickly please book me")


def test_one_strong_turn_switches_on_and_three_quiet_turns_switch_off():
    e = DistressEvidence()
    assert e.observe("আমি আর পারছি না") == "on"
    assert [e.observe("ঠিক আছে") for _ in range(3)] == [None, None, "off"]


def test_urgency_alone_needs_two_turns():
    e = DistressEvidence()
    assert soft_words("প্লিজ তাড়াতাড়ি করুন")
    assert e.observe("প্লিজ তাড়াতাড়ি করুন") is None
    assert e.observe("এখুনি দরকার") == "on"


def test_repeating_and_unanswered_turns_count():
    e = DistressEvidence()
    assert e.observe("আমার বাবার রিপোর্ট কবে পাব বলুন") is None
    assert e.observe("আমার বাবার রিপোর্ট কবে পাব বলুন") is None  # said again: one soft turn
    assert e.observe("আমার বাবার রিপোর্ট কবে পাব বলুন", unanswered_in_a_row=2) == "on"


def test_louder_and_higher_than_own_opening_turns():
    e = DistressEvidence()
    assert e.observe("hello there friend", near_end=(-30.0, 150.0)) is None
    assert e.observe("I want to see a doctor", near_end=(-30.0, 150.0)) is None
    assert e.observe("কিছু একটা বলছি", near_end=(-18.0, 220.0)) is None  # one turn is not enough
    assert e.observe("আবার একটা কথা বলছি", near_end=(-17.0, 230.0)) == "on"


def test_loud_alone_without_pitch_rise_is_not_distress():
    e = DistressEvidence()
    e.observe("hello there friend", near_end=(-30.0, 150.0))
    e.observe("I want to see a doctor", near_end=(-30.0, 150.0))
    assert e.observe("book me one", near_end=(-15.0, 155.0)) is None
    assert e.observe("for tomorrow morning", near_end=(-15.0, 155.0)) is None


def test_asking_for_a_person():
    assert asks_for_person("আমি কারও সাথে কথা বলিয়ে দিন, মানুষের সাথে কথা বলব")
    assert asks_for_person("मुझे किसी इंसान से बात करनी है")
    assert asks_for_person("please connect me to a person")
    assert not asks_for_person("স্টাফ কখন আসেন")
