"""Owner, 2026-10-29: a caller who will give the name / day / time or will call a little later is thanked and the call is closed -- and nothing else is."""

import pytest

from agent import call_later
from agent.phrases import phrase


@pytest.mark.parametrize(
    "text",
    [
        "somoy din ektu pore naam likhabo",
        "ektu pore call korchi",
        "ektu pore call korchi naam lekhate",
        "সময় দিন একটু পরে নাম লিখব",
        "একটু পরে ফোন করছি নাম লেখাতে",
        "আমি পরে বুকিং করব",
        "একটু পরে কল করছি",
        "बाद में फोन करता हूँ",
        "मैं थोड़ी देर बाद नाम बताऊँगी",
        "बाद में बुकिंग करूँगा",
        "I'll call back later",
        "I will give the name later",
        "let me call a bit later",
    ],
)
def test_the_caller_will_do_it_later(text):
    assert call_later.asks_to_call_later(text)


@pytest.mark.parametrize(
    "text",
    [
        "পরে ফোন করুন",
        "একটু পরে ফোন করবেন",
        "बाद में फ़ोन कीजिए",
        "please call me later",
        "can you call later",
        "ektu pore",
        "পরে",
        "বিকেলে আসব",
        "I will come in the evening",
        "can I book for later",
        "আমার নাম রহিম",
        "ডাক্তার সাহা কখন বসেন",
        "হ্যাঁ বুকিং করে দিন",
        "",
    ],
)
def test_nothing_else_closes_the_call(text):
    assert not call_later.asks_to_call_later(text)


def test_the_closing_line_exists_in_all_three_languages():
    for lang in ("bn", "hi", "en"):
        assert phrase("call_later_final", lang)
