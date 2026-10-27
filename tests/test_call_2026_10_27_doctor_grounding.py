"""Regression tests for the real calls of 2026-10-27: a doctor enquiry naming a bare, ambiguous surname ("Dr. Sen"
-- two real doctors share it) or a name the recogniser garbled badly enough to miss the catalogue entirely ("ঋষা"
for "ত্রিষা"/"Trisha") each came back with a DIFFERENT, unrelated doctor's real schedule confidently reported as
fact -- the exact CLAUDE.md "Doctor Nobody" class of bug this architecture exists to prevent.

Root cause: agent/slot_grounding.ground_slots's doctor_name check treated "the sentence contains the word doctor"
as support on its own for ANY name the model filled in -- true of nearly every doctor sentence there is, so it was
no check at all. The fix is catalogue-driven, not a list of doctor names, so it holds for any surname or alias two
or more doctors share, now or added later -- not just "Sen".

    python -m pytest tests/test_call_2026_10_27_doctor_grounding.py -v
"""

from __future__ import annotations

from agent.slot_grounding import ground_slots


def test_a_name_the_catalogue_never_offered_as_a_candidate_is_dropped():
    # "डॉक्टर सेन को दिखाना है क्या वो आज बैठे है" -- real transcript, Hindi: the caller named only the shared
    # surname. Whichever doctor the model fills in, if it is not one of the two real Sens the catalogue says this
    # surname could mean, it is dropped so the turn can ask "which one?" instead.
    kept, dropped = ground_slots(
        "डॉक्टर सेन को दिखाना है क्या वो आज बैठे है",
        {"doctor_name": "Dr. Biswajit Chanda"},
        doctor_candidates=("Dr. A. Sen", "Dr. B. Sen"),
    )
    assert kept["doctor_name"] is None and dropped == ["Dr. Biswajit Chanda"]

    kept, dropped = ground_slots(
        "डॉक्टर सेन का फर्स्ट नेम केवल अरेंदम है क्या वो आज बैठ रहे हैं",
        {"doctor_name": "Dr. Partha Ray"},
        doctor_candidates=("Dr. A. Sen", "Dr. B. Sen"),
    )
    assert kept["doctor_name"] is None and dropped == ["Dr. Partha Ray"]


def test_a_name_sharing_a_real_token_with_a_candidate_still_survives():
    # A plain surname reading, or the caller's own given name for one of the two real Sens, DOES share a token
    # with a candidate -- kept, so the ambiguity still reaches the clinic API's own "which one?" question.
    kept, dropped = ground_slots("डॉक्टर सेन को दिखाना है", {"doctor_name": "Dr. Sen"}, doctor_candidates=("Dr. A. Sen", "Dr. B. Sen"))
    assert kept == {"doctor_name": "Dr. Sen"} and dropped == []

    kept, dropped = ground_slots(
        "डॉक्टर अरिंदम सेन की फीस कितनी", {"doctor_name": "Dr. Arindam Sen"}, doctor_candidates=("Dr. A. Sen", "Dr. B. Sen")
    )
    assert kept == {"doctor_name": "Dr. Arindam Sen"} and dropped == []


def test_an_unmatched_garbled_name_is_dropped_even_though_the_sentence_says_doctor():
    # "डॉक्टर ऋषा को दिखाना चाहता हूँ क्या वो आज बैठ रही है" -- real transcript: the recogniser's "ऋषा" for the
    # real doctor "त्रिषा" (Trisha) matched NOTHING in the catalogue at all (no names_a_doctor, no candidates).
    # The word "doctor" being in the sentence is no longer enough on its own to trust an unrelated name.
    kept, dropped = ground_slots(
        "डॉक्टर ऋषा को दिखाना चाहता हूँ क्या वो आज बैठ रही है", {"doctor_name": "Dr. Kalyan Halder"}
    )
    assert kept["doctor_name"] is None and dropped == ["Dr. Kalyan Halder"]


def test_this_is_general_to_any_future_shared_surname_not_special_cased_to_sen():
    # Same shape, different names entirely -- the fix is catalogue-driven (doctor_candidates), so it holds for
    # whichever surname a clinic happens to seed two doctors under.
    kept, dropped = ground_slots(
        "ডাক্তার রায় কে দেখাতে চাই আজ বসবেন কি",
        {"doctor_name": "Dr. Somebody Unrelated"},
        doctor_candidates=("Dr. N. Roy", "Dr. P. Ray"),
    )
    assert kept["doctor_name"] is None and dropped == ["Dr. Somebody Unrelated"]

    kept, dropped = ground_slots(
        "ডাক্তার রায় কে দেখাতে চাই আজ বসবেন কি", {"doctor_name": "Dr. Roy"}, doctor_candidates=("Dr. N. Roy", "Dr. P. Ray")
    )
    assert kept == {"doctor_name": "Dr. Roy"} and dropped == []
