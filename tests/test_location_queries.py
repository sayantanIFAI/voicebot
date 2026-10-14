"""sourav, 2026-10-14: every way a caller may ask WHERE the clinic is / HOW to get there -- Bengali, Hindi, English, mixed
(Hinglish/Banglish) and badly spelt/phonetic Latin -- must be caught by the fast path as clinic_faq "location", whatever
language the call is locked to (so before AND after a language change). Why: a live caller's Latin-script/mixed question
matched no keyword of the call's own language and went to the model. Flow: real clinic API catalogue -> real FastPath ->
assert the intent and topic for each sentence in each of the three call languages. The sentences are the owner's list."""

import datetime
import os
import sys

import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "clinic-api"))

from agent.fast_path import Catalogue, FastPath  # noqa: E402

TODAY = datetime.date(2026, 9, 25)

BENGALI = [
    "apnader location ta kothai?", "apnader clinic ta kothay?", "okhane kivabe jabo?", "okhane ki bhabe pouchabo?",
    "apnader address ta ektu bolben?", "rasta ta kothay bolte parben?", "apnader hospital/center ta kothai obosthito?",
    "City Centre 2 er kache kothai?", "kothay aste hobe amake?", "kon sthan e apnader clinic?",
]
HINDI = [
    "aapka location kaha hai?", "location kahan hai aapka?", "waha kaise jayenge?", "aapke clinic kaise pahunche?",
    "waha ka address kya hai?", "address batayiye na?", "waha jaane ka rasta kya hai?", "aapka center kahan par hai?",
    "city centre 2 ke pas kahan hai?", "kidhar aana padega humko?",
]
ENGLISH = [
    "Where is your location?", "What is your clinic address?", "How can I reach your center?", "How to get to your location?",
    "Where are you located?", "Can you give me directions to your clinic?", "How do I reach there?",
    "What is the exact location?", "Where is the nearest landmark?", "Send me the location address.",
]
MIXED = [
    "location kaha hai apnader?", "address ta kidhar hai?", "waha ka address ya location bolben?",
    "acha ye batao apka clinic kaha par hai?", "kibhave jabo waha par?", "how to reach apnader center?",
    "address ta kya hai apka?", "apnader location kahan hai batao?", "kaese jayenge okhane?", "where is your clinic ta?",
    "apnader address kahan par padega?", "waha par kaise pahuchbo?",
]
TYPOS = [
    "adress ta kothai", "kayse jayneg waha pr", "waha ka dress ya location", "acha hai kaha par ye apka clicnin",
    "locaton ka hai", "adress batao", "kase jau waha", "hot to reach center", "clnic kothaye", "adrez kya hai",
]
# sourav: real call -- Bengali speech written in Devanagari by a recogniser that heard Hindi
CROSS = ["ओकाने की भव्य जाबू", "okane's grand jabu", "की भाभी जाबो खाने", "उखाने जाबो की भावे", "ओखाने कीभाबे जाबो", "আপকা লোকেশন কাহা হ্যায়", "ওয়াহা কৈসে জায়েঙ্গে"]
# sourav: native-script questions must be caught in every call language (the call may be locked to another one)
NATIVE = ["ওখানে কিভাবে যাব", "ওখানে কীভাবে যাব", "আপনাদের ঠিকানাটা একটু বলবেন", "ক্লিনিক কোথায়", "वहाँ कैसे जाएँगे", "आपका लोकेशन कहाँ है", "पता बताइए"]
ALL = NATIVE + BENGALI + HINDI + ENGLISH + MIXED + TYPOS + CROSS


@pytest.fixture(scope="module")
def payload():
    from _clinic_app import clinic_app

    with clinic_app(sample_patients=False) as (_app, client):
        return client.get("/api/v1/catalogue").json()


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_every_location_question_is_caught_in_every_call_language(payload, lang):
    fp = FastPath(Catalogue(payload), today=TODAY)
    missed = []
    for s in ALL:
        r = fp.resolve(s, lang)
        got = None if r is None else (r.intent, r.slots.get("faq_topic"))
        if got != ("clinic_faq", "location"):
            missed.append((s, got))
    assert not missed, f"[{lang}] {len(missed)}/{len(ALL)} not answered as location: {missed}"


@pytest.mark.parametrize("lang", ["bn", "hi", "en"])
def test_location_is_caught_even_when_the_clinic_api_is_an_old_copy(payload, lang):
    """sourav, 2026-10-14: the pod ran an OLD clinic API, so none of the new keywords reached the agent. The agent now carries
    the phrases itself: strip them from the catalogue (an old clinic API's faq rows) and every sentence must still match."""
    old = {**payload, "faq_topics": [
        {**f, **{f"keywords_{l}": ["कोथाय"] if f["topic"] == "location" else f.get(f"keywords_{l}", []) for l in ("bn", "hi", "en")}}
        for f in payload["faq_topics"]
    ]}
    fp = FastPath(Catalogue(old), today=TODAY)
    missed = [s for s in ALL if (lambda r: r is None or (r.intent, r.slots.get("faq_topic")) != ("clinic_faq", "location"))(fp.resolve(s, lang))]
    assert not missed, f"[{lang}] {missed}"
