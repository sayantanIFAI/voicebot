"""agent/dialogue_state.py: the context block given to the intent model, and the 'I had asked ...' line."""

from agent import dialogue_state


def test_nothing_to_say_on_the_first_turn():
    assert dialogue_state.render({}) == ""
    assert dialogue_state.signature({}) == ""


def test_the_context_names_what_was_just_said_and_asked():
    snap = {"last_ai": "আজ কার্ডিওলজি বিভাগে সুব্রত ব্যানার্জী আছেন। ওনার সাথে অ্যাপয়েন্টমেন্ট করতে চান?", "open_question": "yes or no: shall I book Subrata Banerjee on that day?", "doctor": "Dr. S. Banerjee", "department": "Cardiology"}
    out = dialogue_state.render(snap)
    assert "last sentence" in out and "সুব্রত ব্যানার্জী" in out and "yes or no" in out and "Cardiology" in out and "repeat_last" in out
    assert "never a source of facts" in out


def test_the_cache_key_changes_with_the_open_question():
    a = dialogue_state.signature({"open_question": "yes or no: shall I book X?"})
    b = dialogue_state.signature({"open_question": "which of the doctors just offered"})
    assert a and b and a != b


def test_the_open_question_is_asked_again_by_name():
    snap = {"open_question_spoken": "কার কাছে অ্যাপয়েন্টমেন্ট করতে চান?"}
    assert dialogue_state.open_question_line(snap, "bn") == "আমি জানতে চেয়েছিলাম: কার কাছে অ্যাপয়েন্টমেন্ট করতে চান?"
    assert dialogue_state.open_question_line({}, "bn") == ""
