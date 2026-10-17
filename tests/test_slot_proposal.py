import datetime

from agent import slot_proposal as sp

SLOTS = ["10:00", "10:15", "10:30", "10:45", "11:00", "11:15", "11:30", "11:45"]


def _at(h, m, day="2026-10-17"):
    y, mo, d = (int(x) for x in day.split("-"))
    return datetime.datetime(y, mo, d, h, m)


def test_a_future_day_gets_the_first_free_slot():
    assert sp.pick(SLOTS, "2026-10-18", _at(9, 0)) == "10:00"
    assert sp.pick(["10:30", "10:45"], "2026-10-18", _at(23, 0)) == "10:30"


def test_today_leaves_time_to_get_there():
    # 10:12 now + 30 minutes lead -> the first slot at or after 10:42 is 10:45 ("at 10:45 we can book a slot, come 15 minutes early")
    assert sp.pick(SLOTS, "2026-10-17", _at(10, 12)) == "10:45"
    assert sp.pick(SLOTS, "2026-10-17", _at(8, 0)) == "10:00"


def test_no_slot_is_offered_when_none_is_late_enough_or_free():
    assert sp.pick(SLOTS, "2026-10-17", _at(11, 40)) is None
    assert sp.pick([], "2026-10-18", _at(9, 0)) is None


def test_the_day_is_said_the_way_people_say_it():
    now = _at(9, 0)
    assert sp.day_kind("2026-10-17", now) == "today"
    assert sp.day_kind("2026-10-18", now) == "tomorrow"
    assert sp.day_kind("2026-10-20", now) == "other"
    assert sp.day_line("সেন", "2026-10-17", "bn", now) == "ডাক্তার সেন আজই বসবেন।"
    assert "আগামীকাল" in sp.day_line("সেন", "2026-10-18", "bn", now) and "2026-10-18" in sp.day_line("সেন", "2026-10-18", "bn", now)
    assert "2026-10-20 তারিখে বসবেন" in sp.day_line("সেন", "2026-10-20", "bn", now)
    assert "tomorrow" in sp.day_line("Sen", "2026-10-18", "en", now)


def test_the_proposal_names_the_time_and_asks_to_come_early():
    p = sp.proposal("সেন", "2026-10-17", "10:45", "bn", _at(10, 12))
    assert "আজই বসবেন" in p and "10:45-এ আপনার জন্য স্লট বুক করতে পারি" in p and "15 মিনিট আগে চলে আসবেন" in p
    assert "10:45" in sp.proposal("Sen", "2026-10-17", "10:45", "en", _at(10, 12))
    assert "10:45" in sp.proposal("सेन", "2026-10-17", "10:45", "hi", _at(10, 12))
