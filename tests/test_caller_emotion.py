from agent.caller_emotion import SKIPPED, EmotionEvidence, EmotionResult, result_from_output

LABELS = {0: "neu", 1: "ang", 2: "hap", 3: "sad"}


def feed(labels):
    e = EmotionEvidence()
    return e, [e.add(EmotionResult(label, score)) for label, score in labels]


def test_one_unsure_stressed_clip_changes_nothing():
    _, out = feed([("sad", 0.8), ("neu", 0.9)])
    assert out == [None, None]


def test_two_stressed_clips_in_three_switch_on_then_three_calm_switch_off():
    e, out = feed([("neu", 0.9), ("ang", 0.8), ("sad", 0.8)])
    assert out[-1] == "on" and e.active
    assert [e.add(EmotionResult("neu", 0.9)) for _ in range(3)] == [None, None, "off"]


def test_one_very_sure_clip_is_enough():
    _, out = feed([("ang", 0.95)])
    assert out == ["on"]


def test_short_clips_and_missing_verdicts_are_ignored():
    e = EmotionEvidence()
    assert e.add(None) is None and e.add(SKIPPED) is None and not e.active


def test_happy_is_not_stress():
    _, out = feed([("hap", 0.99), ("hap", 0.99)])
    assert out == [None, None]


def test_log_softmax_and_raw_scores_both_become_probabilities():
    r = result_from_output([-0.1, -3.0, -4.0, -5.0], LABELS)
    assert r.label == "neu" and abs(sum(r.scores.values()) - 1) < 1e-9
    r = result_from_output([0.2, 2.0, 0.1, 0.0], LABELS)
    assert r.label == "ang" and abs(sum(r.scores.values()) - 1) < 1e-9
