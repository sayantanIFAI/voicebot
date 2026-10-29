from agent import head_pain, whatsapp_cues


def test_plain_headache_is_asked_about_and_an_injury_is_not():
    for t in ["আমার মাথায় ব্যথা", "mathar betha doctor", "मेरे सिर में दर्द है", "I have a headache"]:
        assert head_pain.asks_about_head_pain(t), t
    for t in ["মাথায় লেগেছে", "মাথা ঘোরে", "পেটে ব্যথা", "my head hit the wall"]:
        assert not head_pain.asks_about_head_pain(t), t


def test_the_answer_decides():
    assert head_pain.hurt("মাথায় চোট লেগেছে")
    assert not head_pain.hurt("মাথায় ব্যথা অনেক")
    assert head_pain.a_pain_answer("mathai betha onek")


def test_small_phone_means_pay_at_the_counter():
    for t in ["আমার ছোট ফোন, পারব না", "keypad phone, I can't do that", "मेरा फोन छोटा है"]:
        assert whatsapp_cues.cannot_pay_online(t), t
    for t in ["না নেই", "হ্যাঁ আছে", "no whatsapp", "हाँ है"]:
        assert not whatsapp_cues.cannot_pay_online(t), t
