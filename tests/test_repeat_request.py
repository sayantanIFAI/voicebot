"""Owner, 2026-10-30: when the caller says they could not hear or follow ("sunte pelam na", "bujhlam na", "ki bolchen?", "ar ekbar bolben please" ...), the agent says its LAST LINE again, exactly."""

import pytest

from agent import repeat_request

YES = [
    "সুনতে পেলাম না", "শুনতে পারছি না", "বুঝতে পেলাম না", "বুঝতে পারছি না", "বুঝলাম না", "কী বলছেন", "কি বলছেন", "কী?", "কী বলছেন শুনতে পারছি না", "কি বলছেন শুনতে পেলাম না",
    "আর একবার বলুন", "আরেকবার বলবেন প্লিজ", "আরেকবার বলবেন", "আবার বলুন", "আবার বলেন", "আবার একবার বলুন", "আরেকবার বলুন",
    "sunte pelam na", "sunte parchi na", "bujhte pelam na", "bujhte parchi na", "bujhlam na", "ki bolchen", "ki?", "ki bolchen?", "ki bolchis?", "ki bolli?",
    "ki bolchen sunte parchi na", "ki bolchen sunte pelam na", "ar ekbar bolun", "ar ekbar bolben please", "ar ekbar bolben", "abar bolun", "abar bolen", "abar ekbar bolun", "arekbar bolun", "arekbar bolben",
    "फिर से बोलिए", "दोबारा बोलिए", "क्या कहा", "सुनाई नहीं दे रहा", "say that again", "could you repeat that please", "sorry I didn't hear you", "pardon",
]
NO = ["আরেকবার বলুন ডাক্তার সেন কবে বসেন", "কি টেস্ট", "ki test", "ki korte hobe", "হ্যাঁ", "না", "আমার নাম কিশোর", "ki khabo", "what is the fee", "আমি কী করব", "book kore din", "বুঝেছি", "na", "to", "সেন"]


@pytest.mark.parametrize("text", YES)
def test_a_request_to_hear_it_again_is_recognised(text):
    assert repeat_request.asks_repeat(text), text


@pytest.mark.parametrize("text", NO)
def test_other_sentences_are_not_taken_for_a_repeat_request(text):
    assert not repeat_request.asks_repeat(text), text
