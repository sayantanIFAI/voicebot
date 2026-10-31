"""agent/humanize.py: the checker that lets the model rephrase a finished reply only if every fact survives."""

from agent import humanize

VOCAB = {"সৌরভ", "মুখার্জী", "অরুণ", "কুমার", "সাহা"}
ORIG = "আজ জেনারেল মেডিসিন বিভাগে সৌরভ মুখার্জী বসছেন, সময় 10:00 থেকে 12:00। কার কাছে অ্যাপয়েন্টমেন্ট করতে চান?"


def test_a_rewrite_that_keeps_every_fact_is_accepted():
    new = "আজ জেনারেল মেডিসিন বিভাগে সৌরভ মুখার্জী আছেন, উনি 10:00 থেকে 12:00 বসবেন। কার কাছে অ্যাপয়েন্টমেন্ট করতে চান?"
    assert humanize.acceptable(ORIG, new, "bn", VOCAB)


def test_a_changed_time_or_number_is_refused():
    assert not humanize.acceptable(ORIG, ORIG.replace("12:00", "13:00") + " ", "bn", VOCAB)
    assert not humanize.acceptable(ORIG, ORIG.replace("10:00", "11:00").replace("বসছেন", "আছেন"), "bn", VOCAB)


def test_a_dropped_or_invented_doctor_is_refused():
    assert not humanize.acceptable(ORIG, ORIG.replace("সৌরভ মুখার্জী", "ওনার"), "bn", VOCAB)
    assert not humanize.acceptable(ORIG, ORIG.replace("সৌরভ মুখার্জী", "সৌরভ মুখার্জী এবং অরুণ কুমার সাহা"), "bn", VOCAB)


def test_a_question_stays_a_question_and_a_statement_stays_one():
    assert not humanize.acceptable(ORIG, ORIG.rstrip("?") + "।", "bn", VOCAB)
    stmt = "আজ জেনারেল মেডিসিন বিভাগে সৌরভ মুখার্জী বসছেন, সময় 10:00 থেকে 12:00।"
    assert not humanize.acceptable(stmt, stmt.rstrip("।") + "?", "bn", VOCAB)


def test_the_wrong_script_a_link_or_a_runaway_length_is_refused():
    assert not humanize.acceptable(ORIG, "Dr Sourav sits today 10:00 to 12:00 whom do you want?", "bn", VOCAB)
    assert not humanize.acceptable(ORIG, ORIG + " https://example.com", "bn", VOCAB)
    assert not humanize.acceptable(ORIG, ORIG + " " + "আরও কথা " * 30, "bn", VOCAB)


def test_off_by_default_and_a_failing_model_returns_the_original(monkeypatch):
    assert humanize.rewrite(ORIG, "bn", VOCAB, call=lambda *a, **k: '{"reply": "x"}') == ORIG  # disabled: untouched
    monkeypatch.setattr(humanize, "ENABLED", True)
    def boom(*a, **k):
        raise TimeoutError("slow")
    assert humanize.rewrite(ORIG, "bn", VOCAB, call=boom) == ORIG
    good = "আজ জেনারেল মেডিসিন বিভাগে সৌরভ মুখার্জী আছেন, উনি 10:00 থেকে 12:00 বসবেন। কার কাছে অ্যাপয়েন্টমেন্ট করতে চান?"
    assert humanize.rewrite(ORIG, "bn", VOCAB, call=lambda *a, **k: '{"reply": "%s"}' % good) == good
