"""The real turn logic of BOTH orchestrator entry points -- main.py and main_pcm.py (the one the Twilio phone bridge runs) -- driven with typed sentences.

This is what the unit tests of the cue functions cannot show: that the detector is actually CALLED, that its result is RESPECTED by the routing branch, and that
exactly ONE appropriate reply comes back. Each scenario is a failure seen in testing:

  A. "There is an orthopedic doctor."   -> was "we have no doctor called 'Orthopaedics Doctor'"
  B. "Do you have any health packages?" -> was the insurance / cashless FAQ
  C. "Do I have any money left to pay?"  -> was "I didn't understand, say that again"
  D. a request to talk to the doctor / to complain -> never reached a human

The intent model is SCRIPTED to make the mistakes it made on those calls (so the repair is shown, not assumed); the clinic API is the real clinic-api business logic
over a seeded throw-away database; the voice stack is stood in for (subprocess, tests/_turn_harness_runner.py). What this cannot show -- the live recogniser, the
live Ollama model, real audio -- is listed in the final report as pod verification.

    python -m pytest tests/test_turn_routing_harness.py -v
"""

import json
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO_ROOT, "tests", "_turn_harness_runner.py")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import doctor_enquiry_cues, owner_answers  # noqa: E402
from agent.reply_templates import missing_slot_prompt  # noqa: E402

ENTRY_POINTS = ("main", "main_pcm")

CASES = [
    # --- C: money still to pay
    ("Do I have any money left to pay?", "en"),
    ("क्या मेरा कोई पैसा बाकी है?", "hi"),
    ("আমার কি কোনো টাকা বাকি আছে?", "bn"),
    ("Amar kono taka baki ache?", "bn"),
    # --- B: health package (the model, scripted, would have said clinic_faq / insurance)
    ("Do you have any health packages?", "en"),
    ("फुल बॉडी चेकअप पैकेज है क्या?", "hi"),
    ("আপনাদের এখানে হেলথ প্যাকেজ আছে?", "bn"),
    ("Health package-er dam koto?", "bn"),
    # --- genuine insurance must keep using the insurance flow
    ("Do you accept cashless insurance?", "en"),
    # --- D: the doctor himself / a complaint / a person
    ("I want to speak directly to the doctor.", "en"),
    ("मुझे डॉक्टर से सीधे बात करनी है।", "hi"),
    ("আমি সরাসরি ডাক্তারের সঙ্গে কথা বলতে চাই।", "bn"),
    ("Ami doctor-er sathe directly kotha bolte chai.", "bn"),
    ("I want to make a complaint.", "en"),
    ("मुझे शिकायत करनी है।", "hi"),
    ("আমি অভিযোগ জানাতে চাই।", "bn"),
    ("Ami complaint korte chai.", "bn"),
    ("I want to talk to a person", "en"),
    # --- A: a specialty is not a doctor's name
    ("There is an orthopedic doctor.", "en"),
    ("Orthopaedics Doctor", "en"),
    ("Orthopaedics please.", "en"),
    ("Dentistry please.", "en"),
    # --- symptoms the model could not place
    ("My leg hurts.", "en"),
    ("আমার পায়ে ব্যথা।", "bn"),
    ("मेरे कान में दर्द है।", "hi"),
    ("Amar paye fracture hoyeche.", "bn"),
    # --- a department the clinic does not have is never invented
    ("My eyes hurt.", "en"),
    ("Ami jhapsa dekhchi.", "bn"),
    # --- 2026-10-23: payment link not received / which doctor is good / which days / next if absent (no doctor named)
    ("I did not get the payment link", "en"),
    ("লিংক আসেনি", "bn"),
    ("valo doctor ke ache", "en"),
    ("কোন কোন দিন বসেন", "bn"),
    ("aj doctor na pele next kobe", "en"),
    # --- items 24 and 21
    ("what time should I come", "en"),
    ("What is my serial number?", "en"),
    ("সিরিয়াল নম্বর কখন পাব", "bn"),
    ("सीरियल नंबर कब मिलेगा", "hi"),
    ("serial number kobe pabo", "en"),
    ("what time do I need to reach", "en"),
]


# item 24: a doctor was just told about, who sits from 10:00 (the runner presets his chamber hours)
WINDOW = "10:00-12:00"
ROUND4_CASES = [
    ("PHONEPAY HOBENA", "en"),
    ("ফোনপে হবে না", "bn"),
    ("फोनपे नहीं है", "hi"),
    ("can I come and book at the clinic", "en"),
    ("bolchi ekta naam lekhanor chhilo", "en"),
    ("নাম লেখানোর ছিল", "bn"),
    ("নাম লেখানো যাবে আজকে", "bn"),
    ("ওই যেটা নাম লেখাবো আজকে", "bn"),
    ("কালকে নাম লেখাতে চাই", "bn"),
]
ROUND4_WINDOW_CASES = [("kota theke kota bose", "en"), ("কটা থেকে কটা বসেন", "bn"), ("कितने बजे से कितने बजे तक बैठते हैं", "hi")]
AVAILABLE_TODAY_CASES = [
    ("বলছি আজকে আজকে কে কে অবৈলাবল আছে জেনারেল মেডিসিন এ", "bn"),
    ("জেনারেল মেডিসিনে কে কে অ্যাভেইলেবল আছে", "bn"),
    ("जनरल मेडिसिन में कौन कौन उपलब्ध है", "hi"),
    ("who is available in general medicine", "en"),
    ("who is available today in general medicine", "en"),
]
HR1_CASES = [("বলছিলাম কালকে কি কোনো হাইড্রাস্ট এ বুঝবে", "bn"), ("বলছিলাম কালকে কি কোনো হাইডুস্ট এ বুঝবে", "bn")]  # the recogniser's miss of "বলছিলাম কি কালকে কোনো হাড়ের ডাক্তার বসবে" (hr1.wav, 2026-10-30)
ROUND5_CASES = [
    ("usg of thyroid gland", "en"),
    ("tb test in ear", "en"),
    ("থাইরয়েড গ্ল্যান্ডের ইউএসজি", "bn"),
    ("कान का टीबी टेस्ट", "hi"),
    ("USG whole abdomen koto", "en"),
]
ROUND5_WINDOW_CASES = [
    ("samner robibar doctor sen ke pabo", "en"),
    ("ei robibar", "en"),
    ("সামনের রবিবার ডাক্তার সেন কি বসবেন", "bn"),
    ("अगले इतवार डॉक्टर सेन बैठेंगे", "hi"),
]
ROUND6_CASES = [
    ("achha thik ache", "bn"),
    ("ok thik ache", "en"),
    ("আচ্ছা ঠিক আছে", "bn"),
    ("अच्छा ठीक है", "hi"),
    ("okay thank you", "en"),
    ("achha thik ache, ar ekta kotha chilo", "bn"),
]
ROUND6_OPEN_CASES = [("Thik ache!", "en"), ("ঠিক আছে!", "bn")]  # the agent had just asked a question: the same words are a YES
ROUND7_CASES = [
    ("আমার কোমরে ব্যথা হচ্ছে কোন ডাক্তার দেখালে ভালো হয়", "bn"),
    ("কানে ব্যথা কোন ডাক্তার ভালো", "bn"),
    ("मेरी कमर में दर्द है कौन सा डॉक्टर अच्छा रहेगा", "hi"),
    ("I have back pain which doctor is good", "en"),
    ("I have ear pain which doctor is good", "en"),
]
ROUND7_WINDOW_CASES = [("ami ekta naam likhabo.", "bn"), ("amar naam likhabo .", "bn"), ("नाम लिखवाना है।", "hi"), ("I want to write a name.", "en")]  # a doctor was just discussed
WINDOW_CASES = [
    ("When should I come?", "en"),
    ("কখন আসতে হবে", "bn"),
    ("कितने बजे आना है", "hi"),
    ("kokhon asbo", "en"),
    ("kab aaun", "hi"),
]


@pytest.fixture(scope="module")
def turns():
    payload = json.dumps(
        [{"text": t, "lang": lg} for t, lg in CASES + ROUND4_CASES + AVAILABLE_TODAY_CASES + HR1_CASES + ROUND5_CASES + ROUND6_CASES + ROUND7_CASES]
        + [{"text": t, "lang": lg, "answering": True} for t, lg in ROUND6_OPEN_CASES]
        + [{"text": t, "lang": lg, "window": WINDOW} for t, lg in WINDOW_CASES + ROUND4_WINDOW_CASES + ROUND5_WINDOW_CASES + ROUND7_WINDOW_CASES],
        ensure_ascii=False,
    )
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, RUNNER], input=payload.encode("utf-8"), capture_output=True, env=env, timeout=600
    )
    out = proc.stdout.decode("utf-8", "replace")
    assert "@@RESULT@@" in out, "harness did not finish:\n" + proc.stderr.decode("utf-8", "replace")[-3000:]
    raw = json.loads(out.split("@@RESULT@@", 1)[1].strip().splitlines()[0])
    return {entry: {r["text"]: r for r in rows} for entry, rows in raw.items()}


def _run(turns, entry, text):
    r = turns[entry][text]
    assert r["error"] is None, (entry, text, r["error"])
    return r


def _one_reply(r):
    assert len(r["said"]) == 1, ("exactly one response per caller turn", r["said"])
    return r["said"][0]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", CASES[0:4])
def test_C_money_still_to_pay_gets_the_billing_redirect_and_never_a_balance(turns, entry, text, lang):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert said == owner_answers.billing_dues_reply(lang)
    assert not r["model_called"] and not r["handoffs"]
    assert "didn't understand" not in said.lower() and not any(ch.isdigit() for ch in said)  # no invented figure


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", CASES[4:8])
def test_B_a_health_package_question_is_answered_no_packages_never_with_the_insurance_faq(turns, entry, text, lang):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert said == owner_answers.health_package_reply(lang)
    assert "get_faq" not in r["clinic_calls"] and not r["model_called"]
    for word in ("cashless", "insurance", "ক্যাশলেস", "बीमा", "कैशलेस"):
        assert word not in said.lower()


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_B_a_genuine_insurance_question_still_uses_the_insurance_answer(turns, entry):
    r = _run(turns, entry, "Do you accept cashless insurance?")
    assert "Cashless facility is available" in _one_reply(r)
    assert not r["handoffs"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", CASES[9:13])
def test_D_a_request_to_talk_to_the_doctor_reaches_the_human_handoff(turns, entry, text, lang):
    r = _run(turns, entry, text)
    assert r["handoffs"] == ["wants_doctor_personally"], r
    assert not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", CASES[13:17])
def test_D_a_complaint_reaches_the_human_handoff(turns, entry, text, lang):
    r = _run(turns, entry, text)
    assert r["handoffs"] == ["complaint"], r
    assert not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_D_a_request_for_a_person_still_reaches_the_human_handoff(turns, entry):
    assert _run(turns, entry, "I want to talk to a person")["handoffs"] == ["caller_requested"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text", ["There is an orthopedic doctor.", "Orthopaedics Doctor"])
def test_A_a_specialty_in_a_sentence_lists_that_departments_real_doctors(turns, entry, text):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert "Orthopaedics" in said and "Dr." in said, said
    assert "don't have a doctor" not in said and "no doctor" not in said.lower()
    assert "get_doctor_availability" not in r["clinic_calls"], "a specialty is never looked up as a doctor's name"
    assert not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,dept", [("Orthopaedics please.", "Orthopaedics"), ("Dentistry please.", "Dentistry")])
def test_A_a_department_the_model_put_in_the_doctor_slot_is_repaired_into_the_department_list(turns, entry, text, dept):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert r["model_called"] and dept in said and "Dr." in said, (
        said
    )  # the (scripted) model made the mistake; the fallback repaired it
    assert "don't have a doctor" not in said


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize(
    "text,needle",
    [
        ("My leg hurts.", "Orthopaedics"),
        ("আমার পায়ে ব্যথা।", None),
        ("मेरे कान में दर्द है।", None),
        ("Amar paye fracture hoyeche.", None),
    ],
)
def test_a_symptom_the_model_could_not_place_is_routed_to_a_department_with_real_doctors(turns, entry, text, needle):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert "clinic_calls" in r and "route_department" in r["clinic_calls"], r
    assert "Dr." in said or "ডা" in said or "डॉ" in said, said
    if needle:
        assert needle in said


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text", ["My eyes hurt.", "Ami jhapsa dekhchi."])
def test_a_department_the_clinic_does_not_have_is_never_invented(turns, entry, text):
    said = _one_reply(_run(turns, entry, text))
    assert "Dr." not in said and "ডা" not in said, said


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", [("I did not get the payment link", "en"), ("লিংক আসেনি", "bn")])
def test_a_missing_payment_link_is_a_network_issue_wait_fifteen_minutes(turns, entry, text, lang):
    r = _run(turns, entry, text)
    assert _one_reply(r) == owner_answers.payment_link_missing_reply(lang)
    assert not r["model_called"] and not r["handoffs"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_which_doctor_is_good_recommends_no_one_and_asks_the_problem(turns, entry):
    r = _run(turns, entry, "valo doctor ke ache")
    said = _one_reply(r)
    assert "cannot say which doctor" in said and "problem" in said and not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_which_days_with_no_doctor_named_asks_which_doctor(turns, entry):
    r = _run(turns, entry, "কোন কোন দিন বসেন")
    assert len(r["said"]) == 1 and not r["model_called"] and not r["handoffs"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_next_if_absent_with_no_doctor_named_asks_the_problem(turns, entry):
    r = _run(turns, entry, "aj doctor na pele next kobe")
    said = _one_reply(r)
    assert "problem" in said and not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_when_to_come_with_no_doctor_known_asks_which_doctor(turns, entry):
    r = _run(turns, entry, "what time should I come")  # no doctor context on this turn: hours are never invented
    assert _one_reply(r) == "Which doctor are you asking about?" and not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", WINDOW_CASES)
def test_when_to_come_tells_the_sitting_start_time_and_to_keep_one_to_two_hours_in_hand(turns, entry, text, lang):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert not r["model_called"] and "10:00" in said and "12:00" not in said, said  # the START time only
    hours = {"en": "one to two hours", "bn": "এক থেকে দুই ঘণ্টা", "hi": "एक से दो घंटे"}[lang]
    assert hours in said, said


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize(
    "text,lang",
    [
        ("What is my serial number?", "en"),
        ("সিরিয়াল নম্বর কখন পাব", "bn"),
        ("सीरियल नंबर कब मिलेगा", "hi"),
        ("serial number kobe pabo", "en"),
    ],
)
def test_the_serial_number_question_is_answered_whatsapp_after_payment(turns, entry, text, lang):
    r = _run(turns, entry, text)
    assert _one_reply(r) == doctor_enquiry_cues.serial_number_reply(lang) and not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_what_time_do_i_reach_is_no_longer_answered_with_the_insurance_text(turns, entry):
    r = _run(turns, entry, "what time do I need to reach")
    assert "insurance" not in " ".join(r["said"]).lower() and "cashless" not in " ".join(r["said"]).lower()


# ----------------------------------------------------------------------------------- item 20: three "hello"s, then the call is cut
@pytest.fixture(scope="module")
def silence():
    env = dict(os.environ, PYTHONIOENCODING="utf-8", HARNESS_SILENCE="1")
    proc = subprocess.run([sys.executable, RUNNER], input=b"[]", capture_output=True, env=env, timeout=600)
    out = proc.stdout.decode("utf-8", "replace")
    assert "@@RESULT@@" in out, proc.stderr.decode("utf-8", "replace")[-2000:]
    return json.loads(out.split("@@RESULT@@", 1)[1].strip().splitlines()[0])


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_caller_who_never_speaks_hears_hello_three_times_and_the_call_is_cut(silence, entry):
    steps = silence[entry]["never_spoke"]
    spoken = [s["said"][0] for s in steps if s["said"]]
    assert spoken == ["Hello? Can you hear me?"] * 3, spoken
    assert steps[-1]["ended"] is True and not steps[-1]["said"]  # cut at once, nothing said
    assert not any(s["ended"] for s in steps[:-1])


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_caller_who_has_spoken_keeps_the_anything_else_prompt_then_close(silence, entry):
    steps = silence[entry]["had_spoken"]
    assert any("anything else" in t for s in steps for t in s["said"]), steps  # the existing prompt, unchanged
    assert not any("Hello? Can you hear me?" in s["said"] for s in steps)
    assert steps[-1]["ended"] is True


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", [("PHONEPAY HOBENA", "en"), ("ফোনপে হবে না", "bn"), ("फोनपे नहीं है", "hi")])
def test_item_17_a_caller_who_cannot_pay_online_is_told_to_take_help_and_call_again(turns, entry, text, lang):
    r = _run(turns, entry, text)
    assert _one_reply(r) == doctor_enquiry_cues.cannot_pay_online_reply(lang) and not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_item_17_coming_to_the_clinic_to_book_is_allowed_but_the_date_may_be_full(turns, entry):
    r = _run(turns, entry, "can I come and book at the clinic")
    assert _one_reply(r) == doctor_enquiry_cues.come_to_clinic_reply("en") and not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", ROUND4_WINDOW_CASES)
def test_item_16_kota_theke_kota_tells_the_hours_from_and_to(turns, entry, text, lang):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert "10:00" in said and "12:00" in said and not r["model_called"], said


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text", ["bolchi ekta naam lekhanor chhilo", "নাম লেখানোর ছিল", "নাম লেখানো যাবে আজকে", "ওই যেটা নাম লেখাবো আজকে", "কালকে নাম লেখাতে চাই"])
def test_item_22_a_name_to_write_with_no_doctor_asks_which_doctor(turns, entry, text):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert (
        said in ("Which doctor would you like to book with?", "কোন ডাক্তারের সাথে অ্যাপয়েন্টমেন্ট করতে চান?")
        and not r["model_called"]
    ), said


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text", [c[0] for c in ROUND5_CASES[:4]])
def test_item_7_a_rare_test_not_in_the_table_reaches_the_human_handoff(turns, entry, text):
    r = _run(turns, entry, text)
    assert r["handoffs"] == ["unlisted_test"], r
    assert not r["model_called"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_item_7_a_listed_scan_is_not_sent_to_a_person(turns, entry):
    assert _run(turns, entry, "USG whole abdomen koto")["handoffs"] == []


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text", [c[0] for c in ROUND5_WINDOW_CASES])
def test_item_25_a_doctor_asked_about_on_a_weekday_is_answered_for_that_day(turns, entry, text):
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert "10:00" in said and not r["model_called"], said


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text", [c[0] for c in ROUND6_CASES[:5]])
def test_item_6_a_bare_okay_with_nothing_open_says_goodbye_and_cuts_the_call(turns, entry, text):
    r = _run(turns, entry, text)
    assert r["closed"] and len(r["said"]) == 1 and not r["model_called"], r


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_item_6_an_okay_with_more_in_the_sentence_does_not_cut_the_call(turns, entry):
    assert not _run(turns, entry, ROUND6_CASES[5][0])["closed"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text", [c[0] for c in ROUND6_OPEN_CASES])
def test_item_6_the_same_words_after_a_question_are_a_yes_and_do_not_cut_the_call(turns, entry, text):
    assert not _run(turns, entry, text)["closed"]


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", ROUND7_CASES)
def test_item_8_a_problem_in_the_sentence_is_routed_to_its_department_not_asked_for_again(turns, entry, text, lang):
    said = _one_reply(_run(turns, entry, text))
    assert "General Medicine" not in said and "জেনারেল মেডিসিন" not in said and "जनरल मेडिसिन" not in said, said  # the generic "tell me your problem" answer
    assert any(n in said for n in ("Orthopaedics", "অর্থোপেডিক্স", "ऑर्थोपेडिक्स", "ENT", "ইএনটি", "ই এন টি", "ईएनटी", "ई एन टी")), said  # the department the symptom belongs to, with its doctors


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text,lang", ROUND7_WINDOW_CASES)
def test_item_22_a_name_to_write_with_a_doctor_just_discussed_still_asks_which_doctor(turns, entry, text, lang):
    r = _run(turns, entry, text)
    assert _one_reply(r) == missing_slot_prompt("book_appointment", "doctor_name", lang) and not r["model_called"], r


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_caller_who_is_quiet_while_a_booking_question_is_open_is_told_to_take_their_time_not_asked_anything_else(silence, entry):
    from agent.phrases import phrase

    rounds = silence[entry]["booking_open"]
    first = next(r for r in rounds if r["said"])
    assert first["said"] == [phrase("booking_wait", "en")] and not first["ended"], rounds
    assert all(phrase("silence_prompt", "en") not in r["said"] for r in rounds), rounds  # "anything else?" is never asked: a later "no" would end the call and lose the booking


@pytest.mark.parametrize("entry", ENTRY_POINTS)
@pytest.mark.parametrize("text", [c[0] for c in AVAILABLE_TODAY_CASES])
def test_who_is_available_today_lists_only_doctors_with_a_free_slot_today(turns, entry, text):
    """Owner, 2026-10-30: "আজকে কে কে অ্যাভেইলেবল আছে জেনারেল মেডিসিনে" was answered with every doctor of the department. Dr. Biswajit Sen sits today but is fully booked (the
    harness's free_slots says so): only the two doctors with a free slot are named, with their hours; no model."""
    r = _run(turns, entry, text)
    said = _one_reply(r)
    assert not r["model_called"], said
    assert "10:00-12:00" in said and "18:00-20:00" in said, said
    assert not any(w in said for w in ("বিশ্বজিৎ", "बिस्वजीत", "Biswajit")), said


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_hr1_a_misheard_bone_doctor_question_still_lists_tomorrows_orthopaedic_doctors(turns, entry):
    """Owner, 2026-10-30: the recogniser heard "হাড়ের ডাক্তার বসবে" as "হাইড্রাস্ট এ বুঝবে"; on the pod that became a lab-test guess ("PSA?"). It is the orthopaedic doctors of tomorrow."""
    r = _run(turns, entry, HR1_CASES[0][0])
    said = _one_reply(r)
    assert not r["model_called"] and "তপন বসু" in said and "শুভঙ্কর নন্দী" in said and "আগামীকাল" in said, said


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_a_sitting_type_verb_is_never_a_test_question_so_the_doctor_is_asked_for(turns, entry):
    """Owner, 2026-10-30: nobody asks a lab-test question with "বসবে" / the recogniser's "বুঝবে" (tests are হবে / করা যাবে / হচ্ছে). Whatever the recogniser made of the specialty, the answer is
    "which doctor are you asking about?", never a test guess ("PSA?")."""
    r = _run(turns, entry, HR1_CASES[1][0])
    said = _one_reply(r)
    assert "কোন ডাক্তারের কথা" in said and "টেস্ট" not in said and "পিএসএ" not in said, said
