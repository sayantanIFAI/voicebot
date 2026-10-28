"""Review 2026-10-28: socket validation and limits, negated booking requests, and the exact-only cache entries."""

import pytest

from agent import booking_carryover, ws_guard
from agent.semantic_cache import SemanticCache


@pytest.mark.parametrize("raw", ["null", "[]", "7", '"hello"', "{}", '{"type": 3}', '{"type": "launch_missiles"}', "not json", None])
def test_a_control_frame_that_is_not_a_known_object_is_ignored(raw):
    assert ws_guard.parse_control(raw) is None


def test_a_known_control_frame_passes():
    assert ws_guard.parse_control('{"type": "playback_done"}') == {"type": "playback_done"}


def test_an_oversized_control_frame_is_rejected():
    with pytest.raises(ws_guard.Rejected) as e:
        ws_guard.parse_control('{"type": "hello", "pad": "' + "x" * 20000 + '"}')
    assert e.value.code == ws_guard.CLOSE_TOO_BIG


@pytest.mark.parametrize("value,expected", [(16000, 16000), (48000, 48000), (16000.0, 16000), (0, None), (10**9, None), ("16000", None), (True, None), (None, None), (-5, None), (12345, None)])
def test_sample_rate_whitelist(value, expected):
    assert ws_guard.valid_sample_rate(value) == expected


def test_audio_and_time_limits():
    lim = ws_guard.CallLimits(0.0)
    lim.check_audio(1000, 1.0)
    with pytest.raises(ws_guard.Rejected):
        lim.check_audio(ws_guard.MAX_CHUNK_BYTES + 1, 2.0)
    with pytest.raises(ws_guard.Rejected):
        lim.check_time(ws_guard.MAX_CALL_SECONDS + 1)
    big = ws_guard.CallLimits(0.0)
    with pytest.raises(ws_guard.Rejected):
        for i in range(10_000):
            big.check_audio(ws_guard.MAX_CHUNK_BYTES, 1.0)


@pytest.mark.parametrize("text", ["Don't book an appointment", "বুক করবেন না", "बुक मत करो", "আমি বুক করব না", "no need to book", "অ্যাপয়েন্টমেন্ট লাগবে না"])
def test_a_refused_booking_is_not_a_booking_request(text):
    assert not booking_carryover.asks_to_book(text)


@pytest.mark.parametrize("text", ["please book an appointment", "একটা বুকিং দিন", "बुकिंग कर दो", "বুক তো করিনি", "don't book Dr Sen, but book Dr Das", "I do not want to cancel, book it"])
def test_a_real_booking_request_still_is(text):
    assert booking_carryover.asks_to_book(text)


def test_transactional_and_dated_entries_are_not_fuzzy_matched():
    assert not SemanticCache._is_l2_eligible({"intent": "book_appointment", "slots": {}})
    assert not SemanticCache._is_l2_eligible({"intent": "doctor_availability", "slots": {"doctor_name": "Sen", "date": "tomorrow"}})
    assert not SemanticCache._is_l2_eligible({"intent": "test_rate", "slots": {"test_name": "CBC", "relationship": "mother"}})
    assert SemanticCache._is_l2_eligible({"intent": "test_rate", "slots": {"test_name": "CBC"}})


def test_negation_is_detected_across_languages():
    assert SemanticCache._negated("don't book") and SemanticCache._negated("বুক করবেন না") and SemanticCache._negated("बुक मत करो")
    assert not SemanticCache._negated("book it")


def test_a_group_of_digits_is_repeated_back_in_the_pause():
    from agent import booking_capture as bc

    assert bc.partial_prompt("", "9530", "en") == "9 5 3 0. Go on."
    assert bc.partial_prompt("9530", "95304", "bn") == "0 4। বলুন।".replace("0 4", "4")  # only the new digit group
    assert bc.partial_prompt("95304", "95304", "hi") == "ठीक है, बोलिए।"  # nothing new: the plain "go on"
    step = bc.absorb("9530", "four one three two", True)
    assert step.status == "partial" and step.buffer == "95304132"[:8]
