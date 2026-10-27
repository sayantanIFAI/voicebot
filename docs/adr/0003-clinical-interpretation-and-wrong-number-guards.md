# ADR 0003 — Two rule guards before the model: "what does my result mean" and "wrong number"

**Status:** Accepted, deployed 2026-10-27.
**Source:** `agent/clinical_safety.py` and `agent/unrelated_caller.py` come from the sibling repository `innofissionaigit-design/ivr_bookingdoc_lab-3-` (which has no licence file; the owner asked for them to be added). Both were adapted here, not copied blindly.

## Why

* A caller who asks "is my report dangerous?" or "am I dying?" names no test, doctor or date, so nothing in the old turn path caught it: the model answered with a greeting or "say again". What a result means is never ours to say. A **rule** (not a prompt) now sends the caller a fixed sentence that a doctor will explain it and offers to connect the clinician. The yes / no that follows is read first on the next turn: yes hands the call to a person (`_handoff_to_human`, reason `clinical_interpretation`), no goes on.
* A wrong-number caller was re-greeted until the silence timer ended the call, and the call was recorded as `completed`, which counts as a served call. A clear wrong-number sentence now gets one kind reply and, on the second consecutive one, a polite close. The call's outcome is `unrelated_caller`.

## Where it is wired

`main.py` (and so the generated `main_pcm.py`): `_clinical_and_unrelated_guards`, called in `_dispatch_turn` right after the "caller ends the call" checks and **before** early routing, the fast path, the cache and the model. Phrases: `clinical_interpretation`, `unrelated_first`, `unrelated_final` in `agent/phrases.py`.

## Changes made to the ported code (and why)

1. **Lab-value rule narrowed** (`clinical_safety._reading_a_lab_value`). The original treated any lab term near a number as a reading. Checked against this clinic's own catalogue (846 test names and aliases x 11 question shapes), it flagged "Vitamin D (25-OH)", "hb a1c" and, worse, ordinary booking talk such as "আমার সুগার টেস্ট করাতে ১২ টার সময় আসব" (I will come at 12 for the sugar test), "TSH test 8 tar por hobe", "hba1c 6 mash por test". That rule now stands down when the sentence is about a time, a duration, a price or a booking, and a test's own digits ("25-OH", "A1c") are not readings. The direct phrases ("dangerous", "serious", "am I dying", roleplay and forced yes/no) are untouched. `tests/test_clinical_safety.py` (103 tests, including the original adversarial set) all pass, and the catalogue check flags 0 of 9,306 sentences.
2. **Only the wrong-number family is used.** The module's second family, "a name being asked for" (`PERSON_DIRECTED`), also matched ordinary Bengali answers in this clinic's call corpus ("thik ache", "whatsapp ache", "হ্যাঁ আছে", "স্টাফের সঙ্গে কথা বলব"). It is left in the module but not acted on.
3. **Gate:** the wrong-number guard runs only when no booking is open and no question is waiting (`_nothing_is_open`), and the clinic's catalogue (a test, an FAQ topic, a full doctor name, or a doctor's surname) vetoes it.
4. **Not ported:** the model-side `unrelated_caller` intent, the prompt text, the score flag and the metrics counter. The intent model and its accuracy are unchanged.

## Known limits

* The Bengali and Hindi cue lists are "reasoned, not measured". A native speaker should read `agent/clinical_safety.py` and `agent/unrelated_caller.py` and add the phrasings real callers use; both sets of tests are the place to record them.
* A caller who says "wrong number" and then, in the same call, a real request is served normally; the streak resets.
