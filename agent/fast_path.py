"""Answer the common turn without waking the 7B model.

WHY THIS EXISTS
---------------
Measuring the semantic cache produced a finding that reaches further than
caching. Against bge-m3 on real Bengali clinic questions:

    same question, reworded / ASR-garbled     cosine 0.78 - 0.82
    DIFFERENT test, same sentence frame       cosine 0.7492
    same entity, character-level comparison   0.696 - 1.000
    different entity, character-level          0.231 - 0.381

The embedding separates "which test" by 0.03. Character overlap separates
it by 0.32 -- an order of magnitude better, because the embedding is
dominated by the sentence frame ("X টেস্টের রেট কত") while the part that
decides which price a patient is quoted is a handful of characters.

Follow that through and the conclusion is not "tune the cache". It is
that for a fixed 74-row catalogue, identifying the entity was never a
language-modelling problem. A caller asking "ইউরিক অ্যাসিড টেস্টের রেট কত"
needs two things recognised: an intent drawn from a set of four, and one
row out of 74. Both are decidable locally, in microseconds, with a wider
correctness margin than the 7B model's output was ever checked against.

So the LLM is demoted to what it is genuinely needed for: utterances this
module is NOT confident about.

Extended beyond price/availability to also cover test preparation
("test_prep") and a small fixed set of clinic FAQ topics ("clinic_faq" --
hours, location, payment, insurance, parking, report collection, contact
number, home collection): same reasoning, same commit-floor discipline,
same abstain-when-ambiguous default. See Catalogue's docstring for the
FAQ table's shape and its growth limit.

WHAT IT DELIBERATELY REFUSES
----------------------------
`book_appointment` always goes to the LLM. It needs a date, a time, a
patient name and a phone number pulled out of free speech -- open-ended
extraction with caller-specific data in it, which is exactly the case
where a pattern-matcher's failure mode is silent and wrong. The fast path
handles the questions with one entity and no PII, and hands over anything
else. Abstaining is a first-class result here, not a failure.

PER LANGUAGE (KCD-095)
----------------------
The cue words that say "this is a price question" live in agent/fast_path_cues.py, one table per language behind one
interface; `resolve(text, lang)` uses the table for `lang` and abstains for a language it has none for (including
"unknown"). Serve rate is counted per language and `snapshot()["language_gap"]` flags a gap wider than
LANGUAGE_GAP_MARGIN as a defect. Adding a language is a `fast_path_cues.register(...)` call plus catalogue columns,
not code.

FAST AT ANY CATALOGUE SIZE (KCD-096)
------------------------------------
Matching used to score every form against every word-window with difflib: 17-156 ms per turn at 74 rows and seconds
at thousands. It now goes through agent/form_index.py: an exact pruning step (a proven upper bound skips comparisons
that cannot reach the commit floor; results at or above the floor are unchanged) and, on large tables only, a bigram
shortlist whose only failure mode is an abstain. The commit gate stays a CHARACTER similarity on purpose: a
fast-path answer has no model behind it, and a sound-alike is a suggestion for the caller to confirm
(agent/gazetteer.py), never evidence to act on.
"""

from __future__ import annotations

import datetime
import difflib

from agent.location_cues import ALL_LOCATION_PHRASES as _LOCATION_PHRASES  # sourav 2026-10-14
from agent.doctor_cues import doctor_extra_forms as _doctor_extra_forms  # sourav 2026-10-14
from agent import cart_tally as _cart_tally  # sourav 2026-10-14: shared 'total' cue words
import logging
import re
import unicodedata
from collections import defaultdict

from agent import fast_path_cues as cues
from agent.enquiry_followup import is_reference_only
from agent.form_index import INDEX_MIN_FORMS, FormTable
from agent.outcome_metrics import abstentions, fast_path_served

logger = logging.getLogger("fast_path")

# Same floor as semantic_cache.ENTITY_MATCH_FLOOR, and for the same
# measured reason -- 0.696 was a real ASR variant of the right test, so a
# floor of 0.70 would have rejected it by four thousandths.
ENTITY_MATCH_FLOOR = 0.55

# A second, higher bar for committing WITHOUT the model. The cache could
# afford 0.55 because a wrong hit there still ran a live lookup against a
# name the LLM had produced; here nothing downstream re-checks the entity,
# so the margin has to carry the whole decision. 0.72 sits above every
# measured different-entity score (max 0.381) by a wide margin while
# staying under the worst same-entity score (0.696) -- anything between
# the two abstains to the LLM rather than guessing.
COMMIT_FLOOR = 0.72

# Same commit-without-a-model bar, applied to FAQ topic keyword phrases.
# REASONED, not measured -- there is no real-call FAQ-question corpus yet
# to calibrate against, unlike COMMIT_FLOOR above. Recalibrate against
# real Kolkata call audio before trusting this past the pilot; see
# agent/lid.py's module docstring for what "measured vs reasoned" means
# in this codebase.
FAQ_COMMIT_FLOOR = 0.72
FAQ_COMPLEX_FLOOR = 0.9  # owner, 2026-10-26: the stricter floor for an FAQ answered although the sentence holds a complexity word
# Two DIFFERENT tests that fit what was said nearly equally well ("blood sugar" fits both the fasting and the PP test)
# is not something to pick between: a price or a preparation rule for the wrong one is a confident wrong answer. When
# the best match is not exact and the runner-up is within this of it, the fast path abstains, and the model's turn
# reaches the clinic API, which answers "which one -- A or B?" (KCD-446). REASONED, not measured: the measured
# margins were 0.08-0.12 for the ambiguous phrasings and 0.24 or more for a name that is one test's.
AMBIGUITY_MARGIN = 0.2

# A gap in serve rate between two languages wider than this is a DEFECT in the cue data of the slower one (Blueprint
# 4.4: no language may be a second-class citizen). REASONED, not measured.
LANGUAGE_GAP_MARGIN = 0.25
# A language with fewer than this many turns is too thin to compare: its rate is noise.
MIN_TURNS_FOR_GAP = 20

_RE_WS = re.compile(r"\s+")
_RE_PUNCT = re.compile(r"[।?!,.;:'\"()\-]+")


_RE_BN_FINAL_T = re.compile(r"ত্(?=\s|$)")


def _normalize(text: str) -> str:
    """NFC first: Bengali conjuncts and vowel signs have multiple valid
    encodings, and two visually identical strings compare unequal if one
    is composed and the other is not. ASR output and seeded aliases come
    from different sources, so this is a live risk, not a theoretical one.
    The Devanagari nukta (U+093C) is dropped too: the Hindi recogniser
    writes a consonant with and without it for the same spoken word.
    """
    text = unicodedata.normalize("NFC", text).replace(chr(0x093C), "")
    text = _RE_BN_FINAL_T.sub("ৎ", text)  # a word-final ত + hasanta is the letter ৎ ("বিশ্বজিত্" = "বিশ্বজিৎ"): one spelling for the matcher
    return _RE_WS.sub(" ", _RE_PUNCT.sub(" ", text)).strip().lower()


# Words that say "a test" and nothing about WHICH test. Found on the live pod: the garbled Bengali turn "hon ak ei
# test dam koto" matched the HIV test's alias "AIDS test" at 0.89 -- on the generic word and one shared syllable --
# and only the recogniser's low-agreement gate stopped a wrong price being quoted. A match on a test form must now
# ALSO hold on the words that name the test (FormTable's `generic_words`): the forms themselves stay whole, so a
# specific alias is never confused with a shorter one that is merely its prefix. The same list, for the same
# reason, is in clinic-api/main.py (_GENERIC_WORDS).
_GENERIC_TEST_WORDS = frozenset(
    {
        "test",
        "tests",
        "টেস্ট",
        "টেস্টের",
        "টেস্টটা",
        "টেস্টটি",
        "টেস্টগুলো",
        "टेस्ट",
        "जांच",
        "जाँच",
    }
)


def _name_forms(name: str) -> list[str]:
    """A test called "Complete Blood Count (CBC)" is said as the whole name, without the bracket, or as the code."""
    forms = [name]
    if "(" in name and ")" in name:
        forms += [name.split("(")[0].strip(), name.split("(")[1].split(")")[0].strip()]
    return forms


def _dedupe(forms) -> list[str]:
    return list(dict.fromkeys(f for f in forms if f))


# sourav, 2026-10-14: "कल"/"kal" = yesterday or tomorrow in Hindi; a future-tense verb ending (बैठेंगे, मिलेंगे, होगा, baithenge, milegi ...)
# settles it as tomorrow. Used only by _resolve_date.
# sourav, 2026-10-16: the word "doctor" in any of the three languages / spellings (whole word), used by the "no doctor named" guard in resolve().
# sourav, 2026-10-16: phrases that name no particular date, so they must not send a doctor question to the model: "next available", "this week",
# "weekends", a time of day. The answer then lists the doctor's real sitting days / hours. "this evening" / "tonight" mean today.
_SOFT_DATE = re.compile(
    r"(?:^|\s)(?:next available|next appointment|next slot|next time|next visit|next day|first available|earliest|soonest|this week|weekly|weekends?|"
    r"mornings?|afternoons?|evenings?|nights?|शाम|सुबह|दोपहर|रात|shaam|subah|dopahar|raat|সকালে|বিকেলে|বিকালে|সন্ধ্যায়|দুপুরে|রাতে|sokale|bikele|sondhye|dupure)(?=\s|$)"
)
_THIS_TIME_OF_DAY = re.compile(r"(?:^|\s)(?:this (?:morning|afternoon|evening)|tonight|aaj raat|aaj shaam|aaj subah)(?=\s|$)")
_COMMON_WORD_ALIASES = frozenset({"कर", "कर्", "কর", "করে", "kor", "kar"})  # a doctor's surname that is also an everyday word (do / tax)
_BEFORE_BOOKING = re.compile(r"(?:^|\s)(?:before|pehle|age|aage)(?:\s|$)|আগে|पहले")  # sourav 2026-10-16: "is he free BEFORE I book" asks, it does not book
_SAYS_DOCTOR = re.compile(r"(?:^|\s)(?:doctor|doctors|dr|doc|daktar|doktor|dactar|ডাক্তার|ডক্টর|ডাঃ|ডাক্তারবাবু|डॉक्टर|डॉक्टर्|डाक्टर|डाक्टर्|डॉ|डॉक्टर्स)(?:\s|$)")
_KAL = re.compile(r"(?:^|\s)(?:कल|kal)(?:\s|$)")
_PARSO = re.compile(r"(?:^|\s)(?:परसों|परसो|parso|parson)(?:\s|$)")  # day after tomorrow OR two days ago; same tense rule
_AVAIL_NOW = re.compile(r"(?:^|\s)(?:available|उपलब्ध|upalabdh|achhen|achen|আছেন)(?:\s|$)")  # sourav 2026-10-16: "kal ... available hain kya" asks about tomorrow
_PAST_WORD = re.compile(r"(?:^|\s)(?:था|थे|थी|tha|thi|the|thhe|ছিলেন|chilen|chhilen)(?:\s|$)")
_FUTURE_WORD = re.compile(r"\S*(?:ेंगे|ेंगी|ेगा|ेगी|ोगा|ोगी|ोगे|ूँगा|ूंगा|ूँगी|ूंगी|enge|engi|ega|egi|oga|ogi|oge|bosben|bosbe|boshben|boshbe|asben|asbe|ashben|ashbe|achhen|achen)(?:\s|$)")

from agent.name_script import initial_letter as _initial_letter  # noqa: E402
from agent.name_script import same_given_name  # noqa: E402  (also used by main.py and the reply templates)


class Catalogue:
    """The 74 rows, with every spoken form that maps to each -- plus the
    FAQ topic table, which is the same shape (one canonical key, several
    spoken forms) even though its keys are topics ("hours") rather than
    tests or doctors. When this table outgrows a few dozen topics or needs
    ranked/partial matches, that is the signal to promote it to a real
    retrieval index (Epic E25) -- a flat keyword scan is the right tool
    only while it stays small and exhaustively enumerable, same trade-off
    this file's own docstring makes for the 74-row catalogue itself.

    One FormTable per (kind, language). The payload keys are `aliases_<lang>` (tests, doctors) and
    `keywords_<lang>` (FAQ); every language with a cue table gets a table, and a payload that carries another
    language's columns gets one for it too. The Bengali forms are exactly what they always were; Hindi and English
    also learn a test's own name, its bracket-less form and its code.
    """

    def __init__(self, payload: dict, index_min_forms: int = INDEX_MIN_FORMS):
        self._n_rows = (
            len(payload.get("tests", [])) + len(payload.get("doctors", [])) + len(payload.get("faq_topics", []))
        )
        langs = set(cues.languages()) | {"bn", "hi", "en"}
        for group in ("tests", "doctors"):
            for row in payload.get(group, []):
                langs |= {k[len("aliases_") :] for k in row if k.startswith("aliases_")}
        for row in payload.get("faq_topics", []):
            langs |= {k[len("keywords_") :] for k in row if k.startswith("keywords_")}
        # Which doctors (by position in the payload) a spoken form belongs to: "sen" belongs to every Sen. A form with
        # more than one owner is ambiguous, whatever its similarity score, and the fast path never answers on it.
        self._doctor_owners: dict[tuple[str, str], set[int]] = defaultdict(set)
        # Every word of each doctor's own names (whole name in English, Bengali and Hindi), lower-cased: what a GIVEN name said before a surname must resemble (see Catalogue._given_name_ok).
        self._doctor_words: dict[str, set[str]] = {}
        self._doctor_dept: dict[str, str] = {}  # short name AND whole name -> department (English name, as the clinic keeps it)
        self._doctor_initials: dict[str, set[str]] = {}
        self._doctor_info: dict[str, dict] = {}  # short name AND whole name -> the catalogue row (names in every script, from_kolkata ...)
        self._doctor_given: dict[str, set[str]] = {}  # the given-name words only (every word of the whole names but the last), for "was a first name said?"
        for d in payload.get("doctors", []):
            if d.get("department"):
                self._doctor_dept[d["name"]] = d["department"]
                if d.get("full_name"):
                    self._doctor_dept[d["full_name"]] = d["department"]
            given: set[str] = set()
            for f in (d.get("full_name"), d.get("full_name_bn"), d.get("full_name_hi")):
                parts = [w.strip(".,").lower() for w in (f or "").split() if w.strip(".,")]
                parts = [w for w in parts if w not in ("dr", "ডাঃ", "ডা", "डॉ", "डॉक्टर")]
                given |= set(parts[:-1])
            self._doctor_info[d["name"]] = d
            if d.get("full_name"):
                self._doctor_info[d["full_name"]] = d
            self._doctor_given[d["name"]] = given
            if d.get("full_name"):
                self._doctor_given[d["full_name"]] = given
            # the initials the doctor's Latin names carry ("Dr. Ziaul Haq" -> Z; "Dr. Z. Haq" -> Z): what a spoken initial ("জেড", "जेड", "Z") before the surname must be
            latin = [w.strip(".,") for f in (d.get("name"), d.get("full_name")) for w in (f or "").split()]
            latin = [w for w in latin if w and w.isascii() and w.lower() not in ("dr", "doctor")]
            self._doctor_initials[d["name"]] = {w[0].upper() for w in latin[:-1]}
            words: set[str] = set()
            for f in (d.get("name"), d.get("full_name"), d.get("full_name_bn"), d.get("full_name_hi")):
                words |= {w.strip(".,").lower() for w in (f or "").split() if w.strip(".,")}
            self._doctor_words[d["name"]] = words
        self._tables: dict[tuple[str, str], FormTable] = {}
        for lang in sorted(langs):
            for i, d in enumerate(payload.get("doctors", [])):
                for form in self._doctor_forms_of(d, lang):
                    self._doctor_owners[(lang, form)].add(i)
        for lang in sorted(langs):
            cue_table = cues.table_for(lang)
            exact_below = cue_table.exact_below_chars if cue_table else 5
            for kind, rows in (
                ("test", self._test_rows(payload, lang)),
                ("doctor", self._doctor_rows(payload, lang)),
                ("doctorfull", self._doctor_full_rows(payload, lang)),
                ("faq", self._faq_rows(payload, lang)),
            ):
                self._tables[(kind, lang)] = FormTable(
                    rows,
                    exact_below_chars=exact_below,
                    index_min_forms=index_min_forms,
                    generic_words=_GENERIC_TEST_WORDS if kind == "test" else frozenset(),
                )
        # the Bengali rows, under the names other code and tests have always read
        self.tests = self._tables[("test", "bn")].rows
        self.doctors = self._tables[("doctor", "bn")].rows
        self.faq_topics = self._tables[("faq", "bn")].rows

    @staticmethod
    def _test_rows(payload: dict, lang: str):
        rows = []
        for t in payload.get("tests", []):
            aliases = [_normalize(a) for a in t.get(f"aliases_{lang}", [])]
            names = [_normalize(t["name"])] if lang == "bn" else [_normalize(f) for f in _name_forms(t["name"])]
            rows.append((t["name"], _dedupe(aliases + names)))
        return rows

    @staticmethod
    def _doctor_forms_of(d: dict, lang: str) -> list[str]:
        forms = [_normalize(a) for a in d.get(f"aliases_{lang}", [])]
        forms.append(_normalize(d.get("surname") or d["name"].split()[-1]))
        whole = Catalogue._full_name_form(d, lang)
        if whole:
            forms.append(whole)  # owned by this doctor: a shared surname's owners are counted with it
        # sourav, 2026-10-14: every spelling/script a caller or the recogniser may use for this doctor (agent/doctor_cues.py)
        forms += [_normalize(x) for x in _doctor_extra_forms(d)]
        return _dedupe(forms)

    @staticmethod
    def _full_name_form(d: dict, lang: str) -> str:
        latin = " ".join(w for w in (d.get("full_name") or "").split() if w.strip(".").lower() not in ("dr", "doctor"))
        whole = {"bn": d.get("full_name_bn"), "hi": d.get("full_name_hi"), "en": latin}.get(lang)
        return _normalize(whole) if whole else ""

    @staticmethod
    def _doctor_full_rows(payload: dict, lang: str):
        """(doctor, [whole name]) for each doctor that has one: "পার্থ রায়" is one doctor's even where "রায়" is two's."""
        return [(d["name"], [f]) for d in payload.get("doctors", []) if (f := Catalogue._full_name_form(d, lang))]

    @staticmethod
    def _doctor_rows(payload: dict, lang: str):
        return [(d["name"], Catalogue._doctor_forms_of(d, lang)) for d in payload.get("doctors", [])]

    # Words that may stand right before a surname without being a given name: titles, honorifics, "I / this / that", day words, particles, the letter names used as initials ("আর" = R).
    _NOT_A_GIVEN_NAME = frozenset(
        (
            "dr doctor doc mr mrs ms sir madam the a of for with and or to about "
            "ডাক্তার ডাঃ ডা ডক্টর ডাক্তর স্যার সার সির ম্যাডাম মিস্টার মিসেস মহাশয় বাবু আমি আমার আমাকে বলছি বলছিলাম থেকে তো আর ও এবং এই ওই সেই ঐ একটা একজন আজ আজকে আজই কাল কালকে পরশু আগামীকাল এখন তারপর তাহলে মানে না হ্যাঁ ওকে ঠিক আছে কি কী যে "
            "डॉक्टर डॉ डाक्टर सर सिर मैडम मिस्टर मैं मेरा मुझे बोल रहा रही से तो और एक ये वो वह आज कल परसों अभी फिर तब मतलब नहीं हाँ हां ठीक है कि क्या जो"
        ).split()
    )

    def given_name_ok(self, text: str, name: str, form: str | None) -> bool:
        """Owner, 2026-10-29: "ডাক্তার অভিজিৎ চৌধুরী" was answered as Dr. Rajib Chowdhury, "গোপাল পান্ডে" as Dr. Ranjan Pal: the surname alone was matched and the GIVEN name said before it
        was never looked at. When a name-like word stands right before the matched surname, it must be the same name as a word of that doctor's own names; if it is not, the sentence names a
        doctor that is not in the list and is left to the model and the clinic API, which say so (and offer who they might mean) instead of reading out someone else's day."""
        f = form or ""
        words = self._doctor_words.get(name)
        if not f or not words:
            return True
        tokens = [t.strip(".,?!।") for t in text.split()]
        idx = next((i for i, t in enumerate(tokens) if t == f or (len(f) >= 3 and (t.startswith(f) or difflib.SequenceMatcher(None, f, t).ratio() >= 0.75))), None)
        if idx is None or idx == 0:
            return True
        prev = tokens[idx - 1]
        # an INITIAL before the surname ("এম পাল" = M. Pal, "जेड हक" = Z. Haq): it must be the first letter of one of the doctor's given names. "আর" / "आर" can also be the word "and": a mismatch there passes.
        letter = _initial_letter(prev)
        if letter is not None:
            initials = self._doctor_initials.get(name) or set()
            if not initials or letter in initials:
                return True
            return prev in ("আর", "आर")
        if not prev or prev.lower() in self._NOT_A_GIVEN_NAME or any(ch.isdigit() for ch in prev):
            return True
        # Owner, 2026-10-29 ("দেড় হক" for "জেড হক", Z. Haq): a very short unknown word right before a surname that only ONE doctor has is a misheard initial, not another person's first name
        # (a surname two doctors share never reaches this check: _named_doctor refuses it).
        if len(prev) <= 3:
            return True
        return any(same_given_name(prev, w) for w in words)

    def doctor_info(self, doctor_name: str | None) -> dict | None:
        return self._doctor_info.get(doctor_name or "")

    def doctor_department(self, doctor_name: str | None) -> str | None:
        return self._doctor_dept.get(doctor_name or "")

    def given_name_said(self, text: str, doctor_name: str | None) -> bool:
        """Did the sentence carry a FIRST name of this doctor (not only the surname)? True when the doctor has no first name on file."""
        given = self._doctor_given.get(doctor_name or "")
        if not given:
            return True
        return any(same_given_name(tok.strip(".,?!।"), g) for tok in text.split() for g in given)

    def doctor_owner_count(self, lang: str, form: str | None) -> int:
        """How many doctors the spoken form `form` belongs to (0 when unknown)."""
        return len(self._doctor_owners.get((lang, form), ())) if form else 0

    @staticmethod
    def _faq_rows(payload: dict, lang: str):
        # sourav, 2026-10-14: the topic "location" also gets the agent's own phrase list (agent/location_cues.py) in EVERY
        # language, so a stale clinic-api copy cannot leave a Bengali/Hindi/English/mixed/mis-spelt direction question unmatched.
        return [
            (
                f["topic"],
                _dedupe(
                    _normalize(k)
                    for k in list(f.get(f"keywords_{lang}", []))
                    # sourav, 2026-10-16 (real call): an English question said in a call locked to Bengali/Hindi ("Parking is there") found no
                    # FAQ topic because the Bengali/Hindi tables held only their own words. The English phrases are added to both.
                    + ([k for k in f.get("keywords_en", []) if " " in k.strip() or len(k.strip()) >= 6] if lang in ("bn", "hi") else [])  # sourav 2026-10-16: no short English words (cash/gpay) -- they fuzzy-matched noise like "caha"
                    + (_LOCATION_PHRASES if f["topic"] == "location" else [])
                ),
            )
            for f in payload.get("faq_topics", [])
        ]

    def __len__(self) -> int:
        return self._n_rows

    def languages(self) -> set[str]:
        return {lang for (_kind, lang) in self._tables}

    def table(self, kind: str, lang: str = "bn") -> FormTable | None:
        return self._tables.get((kind, lang))

    def match(
        self, text: str, kind: str, lang: str = "bn", floor: float = 0.0, skip_name: str | None = None
    ) -> tuple[str | None, str | None, float]:
        """-> (canonical_key, matched_spoken_form, score). `canonical_key`
        is a test/doctor name for kind in {"test", "doctor"}, or a FAQ
        topic key for kind="faq".

        floor=0 scores everything and returns the best score however low (what a caller that applies its own
        threshold, such as the history flow, has always seen). With a floor, comparisons that provably cannot
        reach it are skipped, which is what keeps this fast (agent/form_index.py); a best score at or above the
        floor is unchanged, and below it the answer is (None, None, 0.0)."""
        table = self._tables.get((kind, lang))
        if table is None:
            return None, None, 0.0
        return table.best(_normalize(text).split(), floor, skip_name)


class FastPathResult:
    __slots__ = ("intent", "slots", "direct_reply_bn", "confidence", "matched_form")

    def __init__(self, intent, slots, confidence, matched_form=None, direct_reply_bn=None):
        self.intent = intent
        self.slots = slots
        self.confidence = confidence
        self.matched_form = matched_form
        self.direct_reply_bn = direct_reply_bn

    def as_llm_shape(self) -> dict:
        """Same dict shape agent/llm.py returns, so callers cannot tell
        which path produced it and no downstream code needs a branch.
        (`direct_reply_bn` is the historical name: it holds the reply in the caller's language.)"""
        return {
            "intent": self.intent,
            "slots": self.slots,
            "direct_reply_bn": self.direct_reply_bn,
        }


def _empty_slots(**kw) -> dict:
    slots = {
        "test_name": None,
        "doctor_name": None,
        "date": None,
        "time_slot": None,
        "patient_name": None,
        "phone": None,
        "faq_topic": None,
    }
    slots.update(kw)
    return slots


def _complexity(text: str, table: cues.CueTable) -> bool:
    """Whole-word match, for cues where a substring hit would be a false positive.

    A real one this caught: the complexity guard rejected a textbook Bengali availability question ("Doctor Sen,
    when will he sit in his chamber") because the word for "will sit" contains the word for "all" as a substring.
    Bengali writes without internal word boundaries, so short function words appear inside longer unrelated words
    constantly. Every cue in a complexity list is a standalone word, so matching them as whole words is both
    correct and strictly safer -- for every language."""
    words = set(text.split())
    return any((cue in words) if " " not in cue else (cue in text) for cue in table.complexity)


def serve_rate_gap(
    by_lang: dict[str, dict[str, int]], margin: float = LANGUAGE_GAP_MARGIN, min_turns: int = MIN_TURNS_FOR_GAP
) -> dict:
    """Serve rate per language and the widest gap between two languages with enough turns to compare. A gap wider
    than `margin` is a defect in the cue data of the lower one (KCD-095)."""
    rates = {}
    for lang, c in by_lang.items():
        turns = c["served"] + c["abstained"]
        rates[lang] = {
            "served": c["served"],
            "abstained": c["abstained"],
            "turns": turns,
            "serve_rate": round(c["served"] / turns, 3) if turns else 0.0,
        }
    eligible = {lang: r["serve_rate"] for lang, r in rates.items() if r["turns"] >= min_turns}
    gap, widest = None, None
    if len(eligible) >= 2:
        top, bottom = max(eligible, key=lambda k: eligible[k]), min(eligible, key=lambda k: eligible[k])
        gap, widest = round(eligible[top] - eligible[bottom], 3), [top, bottom]
    return {
        "by_language": rates,
        "margin": margin,
        "min_turns": min_turns,
        "gap": gap,
        "widest": widest,
        "defect": gap is not None and gap > margin,
    }


class FastPath:
    def __init__(self, catalogue: Catalogue, today: datetime.date | None = None):
        self.catalogue = catalogue
        self._today = today
        self.stats = {"served": 0, "abstained": 0}
        self.by_lang: dict[str, dict[str, int]] = defaultdict(lambda: {"served": 0, "abstained": 0})

    def _abstain(self, reason: str, intent: str = "unknown", lang: str = "") -> None:
        """KCD-457: abstention is a first-class, EXPECTED result here (see
        module docstring), but "why" still needs to be visible -- a shift
        in the reason distribution usually means an upstream change
        (catalogue growth, a new phrasing pattern), not that this module
        got worse. Recorded even though the caller ends up going to the
        LLM anyway, which is the whole point: this file's own decision is
        the thing being measured, not just the turn's eventual outcome."""
        self.stats["abstained"] += 1
        self.by_lang[lang or "?"]["abstained"] += 1
        abstentions.record(reason, intent, lang)

    def _serve(self, intent: str, lang: str) -> None:
        """KCD-460: serve rate published PER INTENT AND PER LANGUAGE --
        "a routine question is answered instantly, and their serve rate is
        published per intent and language" is this story's own wording."""
        self.stats["served"] += 1
        self.by_lang[lang]["served"] += 1
        fast_path_served.record(intent, "served", lang)

    def _named_doctor(self, text: str, lang: str, strong: bool = True) -> tuple[str, str | None, float] | None:
        """sourav, 2026-10-16: (name, spoken form, score) of the ONE doctor the sentence names, or None when none is named or several
        doctors fit the words equally (a shared surname: the clinic API asks "which one?"). The same lookup the availability branch
        uses, taken out so the booking, schedule and FAQ decisions all ask it the same way. It reads the clinic's own catalogue, so
        a doctor added to the database is recognised with no code change."""
        name, form, score = self.catalogue.match(text, "doctorfull", lang, 0.9)  # the whole name, if it was said
        whole = bool(name)
        if not name:
            name, form, score = self.catalogue.match(text, "doctor", lang, COMMIT_FLOOR)
        if not (name and score >= COMMIT_FLOOR):
            return None
        if self.catalogue.doctor_owner_count(lang, form) > 1:
            return None
        # A short surname can be an ordinary word ("कर" = "do" is also Dr. Kar; "Das" ~ "days"). For the NEW doctor-word logic the sentence must
        # also say "doctor / Dr / डॉ / ডাঃ", or give the whole name, or the spoken form must be a long, distinctive name.
        # Fuzzy hits are noise ("कल्याण" scored 0.8 against "आपका पता क्या है" and "chand" 0.75 against "can"): without a doctor word beside it the
        # form must be said essentially exactly, as a word of the sentence.
        if strong and not (whole or _SAYS_DOCTOR.search(f" {text} ")):
            said = (form or name or "")
            if not (score >= 0.95 and said not in _COMMON_WORD_ALIASES and any(tok == said or tok.startswith(said) for tok in text.split())):
                return None
        if not whole and not self.catalogue.given_name_ok(text, name, form):
            return None
        return name, form, score

    @staticmethod
    def _doctor_form_is_said(text: str, table: cues.CueTable, form: str | None) -> bool:
        """sourav, 2026-10-16: did the caller actually SAY the doctor form the catalogue matched? A form of four letters or fewer must be
        present as a word of the sentence (Bengali's fuzzy match had read an unknown name "ঝঝাইক্স" as the short surname "সেন"), and a form
        that is also an ordinary word ("कर" = do, "কর" = do) needs "doctor / डॉ / ডাঃ" beside it. Failure = the turn goes to the model and
        the clinic API, which says plainly that the doctor could not be identified."""
        f = form or ""
        if not f:
            return True
        if len(f) <= 4:
            # A short surname must be said, not merely contained: Bengali "বসেন" (sits) ENDS in "সেন", so "ডাঃ ঝঝাইক্স কবে বসেন" had read as
            # Dr. Sen. Words that merely end in the form are set aside; some other word must be the form, begin with it (case endings:
            # সেনের, सेन को), or be a close misspelling of it (a recogniser's garble of the name).
            others = [t for t in text.split() if not (t.endswith(f) and t != f)]
            if not any(t == f or t.startswith(f) or difflib.SequenceMatcher(None, f, t).ratio() >= 0.75 for t in others):
                return False
        return not (f in _COMMON_WORD_ALIASES and not _SAYS_DOCTOR.search(f" {text} "))

    @staticmethod
    def _doctor_schedule_words(text: str, table: cues.CueTable) -> bool:
        """The extra words that make a sentence naming a doctor a SCHEDULE question: come / which day / next / Monday / evening, or
        "can I see ..." (see + a question frame). A wish ("I want to see", "dikhana hai") is not one -- that is a booking."""
        return table.any(text, table.doctor_schedule) or (
            table.any(text, table.doctor_see) and table.any(text, table.doctor_question)
        )

    def _is_schedule_question(self, text: str, table: cues.CueTable, doc) -> bool:
        """A sentence that holds a booking WORD and a doctor: is it still just a schedule question? Yes when there is no booking VERB and
        it asks (a time / availability word or a question frame), or when the booking verb is only "before I book"."""
        if doc is None:
            return False
        asks = table.any(text, table.avail) or table.any(text, table.doctor_question) or "?" in text
        if not table.any(text, table.book_verbs):
            return asks
        return bool(_BEFORE_BOOKING.search(f" {text} ")) and asks

    def doctor_booking_hint(self, transcript: str, lang: str = "bn") -> "FastPathResult | None":
        """sourav, 2026-10-16: a booking request that names ONE doctor ("I want to book an appointment with Dr Sen", "Dr Sen ka appointment
        book karna hai", "ami Dr Sen ke dekhabo", "mujhe Dr Sen ko dikhana hai") as a book_appointment result with that doctor (and the day,
        when the sentence names one the fast path can read). NOT used for ordinary turns -- the model still extracts booking details -- but as the
        fallback when the model fails or does not understand, so a doctor booking never dies on a model error. Never a schedule question
        (those are resolve()'s) and never a guess: no named doctor, no hint."""
        table = cues.table_for(lang)
        text = _normalize(transcript)
        if table is None or not text:
            return None
        doc = self._named_doctor(text, lang)
        if doc is None:
            return None
        if self._is_schedule_question(text, table, doc):
            return None
        wants_booking = table.any(text, table.book_verbs) or table.any(text, table.book) or table.any(text, cues.BOOK_NOUNS_ALL)
        wants_to_see = (
            table.any(text, table.doctor_see)
            and table.any(text, table.doctor_want)
            and not table.any(text, table.doctor_question)
            and not table.any(text, table.avail)
        )
        if not (wants_booking or wants_to_see):
            return None
        name, form, score = doc
        date_iso, confident = self._resolve_date(text, table)
        if not date_iso:  # "kal" / "parso" next to a booking word can only be the future: nobody books yesterday
            today = self._today or datetime.date.today()
            if _KAL.search(text):
                date_iso, confident = (today + datetime.timedelta(days=1)).isoformat(), True
            elif _PARSO.search(text):
                date_iso, confident = (today + datetime.timedelta(days=2)).isoformat(), True
        return FastPathResult(
            "book_appointment", _empty_slots(doctor_name=form or name, date=date_iso if confident else None), score, matched_form=form
        )

    def _resolve_date(self, text: str, table: cues.CueTable) -> tuple[str | None, bool]:
        """-> (iso_date_or_None, is_confident). Not confident means the
        utterance contains date-ish language this module will not try to
        parse, so the whole turn must go to the LLM."""
        today = self._today or datetime.date.today()
        # sourav, 2026-10-14: Hindi "कल" / "kal" means yesterday OR tomorrow; the TENSE of the sentence says which ("कल बैठेंगे" =
        # tomorrow). A future-tense word next to it is read as tomorrow; anything else is still left to the model (never guessed).
        if (
            (_KAL.search(text) or _PARSO.search(text))
            and (_FUTURE_WORD.search(text) or (_AVAIL_NOW.search(f" {text} ") and not _PAST_WORD.search(f" {text} ")))
            and not re.search(r"\d", text)
        ):
            return (today + datetime.timedelta(days=2 if _PARSO.search(text) else 1)).isoformat(), True
        # sourav, 2026-10-16: "on Monday" / "सोमवार को" / "সোমবার" -> the next such day (today counts), when it is the ONLY date word in the
        # sentence. "next Monday", "Monday 20th", "Monday or Tuesday", "tomorrow Monday" still go to the model, as before.
        if _THIS_TIME_OF_DAY.search(f" {text} "):
            rest = _THIS_TIME_OF_DAY.sub(" ", f" {text} ").strip()
            if not re.search(r"\d", rest) and not table.any(rest, table.date_words) and not any(table.contains(rest, k) for k in table.relative_days):
                return today.isoformat(), True
        softened = " ".join(_SOFT_DATE.sub(" ", f" {text} ").split())
        if softened != text:
            return self._resolve_date(softened, table)
        weekday = self._only_weekday(text, table)
        if weekday is not None:
            return (today + datetime.timedelta(days=(weekday - today.weekday()) % 7)).isoformat(), True
        if table.strict_dates and (re.search(r"\d", text) or table.any(text, table.date_words)):
            return None, False
        for word, offset in table.relative_days.items():
            if table.contains(text, word):
                return (today + datetime.timedelta(days=offset)).isoformat(), True
        # Any digit or weekday name means a date we are not handling here.
        if re.search(r"\d", text) or table.any(text, table.date_words):
            return None, False
        return None, True

    @staticmethod
    def _only_weekday(text: str, table: cues.CueTable) -> int | None:
        """The weekday (Monday = 0) when `text` names exactly ONE weekday and nothing else date-like (no digit, no other date word, no
        relative day); otherwise None."""
        found = {d for w, d in cues.WEEKDAYS.items() if table.contains(text, cues.normalise_cue(w))}
        if len(found) != 1:
            return None
        rest = f" {text} "
        for w in cues.WEEKDAYS:
            w = cues.normalise_cue(w)
            rest = rest.replace(f" {w} ", " ") if w.isascii() or table.match != "substring" else rest.replace(w, " ")
        rest = " ".join(rest.split())
        if re.search(r"\d", rest) or table.any(rest, table.date_words) or any(table.contains(rest, k) for k in table.relative_days):
            return None
        return found.pop()

    @staticmethod
    def _without_cues(text: str, table: cues.CueTable) -> str:
        """`text` with the rate and preparation cue phrases taken out: what is left is what NAMES something."""
        remaining = f" {text} "
        cues_longest_first = sorted(table.rate + table.prep, key=len, reverse=True)
        while True:  # taking one cue out can leave two words that form another ("how price much")
            before = remaining
            for cue in cues_longest_first:
                if table.match == "substring":
                    remaining = remaining.replace(cue, " ")
                else:
                    while f" {cue} " in remaining:
                        remaining = remaining.replace(f" {cue} ", " ")
            remaining = f" {' '.join(remaining.split())} "
            if remaining == before:
                break
        return " ".join(remaining.split())

    def _ambiguous_test(self, text: str, lang: str, table: cues.CueTable, score: float) -> bool:
        """Two different tests fit the words that NAME a test almost equally well. Judged on the text without the cue
        phrases: "খালি পেটে" is a cue for preparation, not part of the name of the test called "খালি পেটে সুগার", and
        letting it count made an ordinary "<test> ... খালি পেটে ..." look ambiguous."""
        if score >= 1.0:
            return False  # said in full: exactly one test's name
        body = self._without_cues(text, table)
        first, _form, first_score = self.catalogue.match(body, "test", lang, COMMIT_FLOOR)
        if first is None or first_score >= 1.0:
            return False
        other, _form2, other_score = self.catalogue.match(body, "test", lang, COMMIT_FLOOR, skip_name=first)
        return other is not None and first_score - other_score < AMBIGUITY_MARGIN

    @staticmethod
    def _drop_window(text: str, form: str) -> str:
        """sourav, 2026-10-14: `text` with the words that best match the catalogue spoken form `form` taken out.
        Why: the fast path needs to ask "what is LEFT after the test I found?" -- for a second test (multi-test
        sentence) and for cue words that are really part of the test's own name ("fasting" in "fasting blood sugar").
        Flow: slide a window of len(form)+-1 words over the text, drop the one with the best character similarity."""
        words = text.split()
        span = len(form.split())
        best_ratio, best_i, best_w = 0.0, None, 0
        for width in {max(1, span - 1), span, span + 1}:
            for i in range(max(1, len(words) - width + 1)):
                r = difflib.SequenceMatcher(None, form, " ".join(words[i : i + width])).ratio()
                if r > best_ratio:
                    best_ratio, best_i, best_w = r, i, width
        if best_i is None:
            return text
        return " ".join(words[:best_i] + words[best_i + best_w :])

    def _names_second_test(self, text: str, lang: str, table: cues.CueTable, name: str, form: str | None) -> bool:
        """sourav, 2026-10-14 (live defect, found by tests/test_lab_price_multilingual.py): "price of CBC uric acid" /
        "CBC, uric acid price" has TWO tests in it, but the best match (CBC) was served alone and the other test was
        silently dropped. True when, after the cue words and the first test's own words are taken out, a DIFFERENT
        test still matches at the commit floor. The only failure mode is an abstain (the turn goes to the model, which
        returns every name), never a wrong answer."""
        body = self._without_cues(text, table)
        left = self._drop_window(body, form) if form else body
        if not left.strip():
            return False
        other, _f, other_score = self.catalogue.match(left, "test", lang, COMMIT_FLOOR, skip_name=name)
        return other is not None and other_score >= COMMIT_FLOOR

    def named_tests(self, transcript: str, lang: str = "bn", limit: int = 6) -> list[str]:
        """sourav, 2026-10-16: EVERY catalogue test the sentence names, canonical names in the order said ([] when none).
        Why: "lipid profile and uric acid total koto hobe" needs both tests; the fast path only ever resolved one, so the sentence
        went to the model and, when the model was slow or failed on the pod, the caller heard "a little problem". Flow: match the best
        test, take its words out of the sentence (_drop_window), match again, until nothing is left that names a test. Names only --
        the prices still come from the clinic API."""
        table = cues.table_for(lang)
        text = _normalize(transcript)
        out: list[str] = []
        if table is None or not text:
            return out
        for _ in range(limit):
            name, form, score = self.catalogue.match(text, "test", lang, COMMIT_FLOOR)
            if not (name and form and score >= COMMIT_FLOOR):
                break
            if name not in out:
                out.append(name)
            text = self._drop_window(text, form)
            if not text.strip():
                break
        return out

    def looks_multi_test(self, transcript: str, lang: str = "bn") -> bool:
        """sourav, 2026-10-14: public yes/no -- does this sentence name more than one catalogue test? Used by
        main.py so such a sentence is never answered from the saved-answer cache (which can only hold ONE test's
        answer and was serving "Uric Acid" for a three-test question)."""
        table = cues.table_for(lang)
        text = _normalize(transcript)
        if table is None or not text:
            return False
        name, form, score = self.catalogue.match(text, "test", lang, COMMIT_FLOOR)
        return bool(name and score >= COMMIT_FLOOR and self._names_second_test(text, lang, table, name, form))

    def _without_test_name_cues(self, text: str, lang: str, table: cues.CueTable) -> tuple[bool, bool, bool] | None:
        """sourav, 2026-10-14: (wants_rate, wants_avail, wants_prep) re-read on the text WITHOUT the test's own name,
        or None when no test is named. Why: a test whose NAME contains a cue word ("Blood Sugar FASTING", "খালি পেটে
        সুগার", "खाली पेट शुगर") looked like a rate question AND a preparation question at once, so it always
        abstained to the model. The name is not a cue; the rest of the sentence decides."""
        name, form, score = self.catalogue.match(text, "test", lang, COMMIT_FLOOR)
        if not (name and form and score >= COMMIT_FLOOR):
            return None
        rest = self._drop_window(text, form)
        return table.any(rest, table.rate), table.any(rest, table.avail), table.any(rest, table.prep)

    def _names_nothing(self, text: str, table: cues.CueTable) -> bool:
        """True when, once the rate/preparation cue phrases are taken out, everything left is a little function word or
        a pointing word ("the same", "that", "test"): the caller asked about the topic, they did not name a test. One
        word this does not recognise -- possibly a test name -- makes it False, and the turn goes to the model."""
        known = set(table.function_words)
        return all(t in known or is_reference_only(t) for t in self._without_cues(text, table).split())

    def _faq_despite_complexity(self, text: str, lang: str, table: cues.CueTable, ignore_avail: bool = False) -> FastPathResult | None:
        """The clinic FAQ topic a sentence names although it holds a complexity word (or, with `ignore_avail`, a "when" word that made it look like a
        doctor-timing question), or None. Never when the sentence has a test, a doctor, a price / preparation cue or a booking word in it: those are
        not clinic questions and keep going to the model as before."""
        cues_to_clear = (table.rate, table.prep, table.book, table.book_verbs) if ignore_avail else (table.rate, table.avail, table.prep, table.book, table.book_verbs)
        if any(table.any(text, cue) for cue in cues_to_clear):
            return None
        if _SAYS_DOCTOR.search(f" {text} "):
            return None
        if self._named_doctor(text, lang) is not None:
            return None
        test, _form, test_score = self.catalogue.match(text, "test", lang, COMMIT_FLOOR)
        if test and test_score >= COMMIT_FLOOR:
            return None
        topic, form, score = self.catalogue.match(text, "faq", lang, FAQ_COMPLEX_FLOOR)
        if not topic or score < FAQ_COMPLEX_FLOOR:
            return None
        self._serve("clinic_faq", lang)
        logger.info("fast path: clinic_faq %r (%.2f) [%s] despite a complexity cue", topic, score, lang)
        return FastPathResult("clinic_faq", _empty_slots(faq_topic=topic), score, matched_form=form)

    def resolve(self, transcript: str, lang: str = "bn", topic_test: str | None = None) -> FastPathResult | None:
        """Returns None whenever it is not confident. None is the normal,
        expected outcome for anything non-routine -- the caller falls back
        to the semantic cache and then the LLM. A language with no cue table (including "unknown") always
        returns None: a table for one language never fires on another.

        `topic_test`: the test the last few turns were about (agent/enquiry_followup.recent_unique). A price or
        preparation question that names NO test -- nothing left in it but cue and function words, e.g. "do I need
        to fast?" straight after a price -- is answered about that test, with no model call. The reply always names the
        test, so a wrong assumption is heard and corrected. Without a topic, or with any word that could be a name,
        this abstains exactly as before."""
        table = cues.table_for(lang)
        if table is None:
            self._abstain("unknown_language", lang=str(lang))
            return None
        text = _normalize(transcript)
        if not text:
            self._abstain("empty_transcript", lang=lang)
            return None

        # Booking is never handled here: open-ended extraction with PII in
        # it. Check first, before any cue that might also appear in it.
        doc = None  # sourav, 2026-10-16: the ONE doctor this sentence names (looked up at most once per turn)
        if table.any(text, table.book) or table.any(text, table.book_verbs):
            # sourav, 2026-10-16: "when is Dr Sen's next available appointment" / "Dr Sen ka next available slot kab hai" use the NOUN
            # appointment / slot but ask for the doctor's schedule; they are answered from the clinic's real schedule. A sentence with
            # a booking VERB ("book", "बुक करना", "বুক করতে") is still a booking and still goes to the booking flow, exactly as before.
            doc = self._named_doctor(text, lang)
            if not self._is_schedule_question(text, table, doc):
                self._abstain("booking_excluded", lang=lang)
                return None

        if _complexity(text, table):
            # owner, 2026-10-26: "गूगल पे फोन पे सब हो जायेगा" ("Google Pay, PhonePe, all of it will work?") has the word "सब" (all), a complexity cue, so the whole
            # sentence went to the model, which answered "जी, बताइए". A sentence with NO test, doctor, price, timing or booking in it that still matches one clinic
            # FAQ topic nearly word for word is a plain clinic question; it is answered, at a stricter floor than the ordinary FAQ one.
            hit = self._faq_despite_complexity(text, lang, table)
            if hit is not None:
                return hit
            self._abstain("complexity_cue", lang=lang)
            return None

        wants_rate = table.any(text, table.rate)
        wants_avail = table.any(text, table.avail)
        wants_prep = table.any(text, table.prep)

        # sourav, 2026-10-14: "what is the total for CBC" / "सीबीसी का टोटल कितना होगा" has no rate cue word, so it fell to the
        # FAQ table and was answered with the clinic's address or opening hours (found by tests/test_lab_price_multilingual.py).
        # A total word next to a NAMED test is a price question about that test; with nothing named it stays as it was.
        if not (wants_rate or wants_avail or wants_prep) and _cart_tally.wants_total_or_remove(text)[0]:
            _n, _f, _s = self.catalogue.match(text, "test", lang, COMMIT_FLOOR)
            if _n and _s >= COMMIT_FLOOR:
                wants_rate = True

        # More than one cue set firing means an utterance asking about more
        # than one thing (or genuinely ambiguous between them -- "what must I
        # observe" alone can read as prep, but combined with a rate/avail cue
        # it is not this module's call to make). Let the model decide.
        if sum((wants_rate, wants_avail, wants_prep)) > 1 and (wants_rate or wants_prep) and not wants_avail:  # sourav: only a price-vs-prep clash is re-read; a doctor-timing cue always abstains as before
            reread = self._without_test_name_cues(text, lang, table)
            if reread is not None and sum(reread) <= 1:
                wants_rate, wants_avail, wants_prep = reread
        if sum((wants_rate, wants_avail, wants_prep)) > 1:
            self._abstain("ambiguous_multi_cue", lang=lang)
            return None

        # sourav, 2026-10-16: with a doctor named and no price / preparation cue, the doctor words (come, which day, Monday, next, "can I see
        # ... tomorrow") make it a schedule question. They are used ONLY here, so no other sentence is read differently.
        if not (wants_rate or wants_prep):
            if doc is None:
                doc = self._named_doctor(text, lang)
            if doc is not None and not wants_avail and self._doctor_schedule_words(text, table):
                wants_avail = True

        if wants_rate or wants_prep:
            # A question with no test named in it -- only the cue and function words -- has no entity to look for. Matching
            # it anyway let the cue word itself ("fasting", "फास्टिंग") pick the test called "... Fasting" (measured on
            # the Hindi table: "do I need to fast" was answered for a blood-sugar test). It is about the recent topic if
            # there is one, and otherwise it is left to the model, which asks which test.
            intent = "test_rate" if wants_rate else "test_prep"
            if self._names_nothing(text, table):
                if topic_test:
                    self._serve(intent, lang)
                    logger.info("fast path: %s %r from the topic [%s] for %r", intent, topic_test, lang, transcript)
                    return FastPathResult(
                        intent, _empty_slots(test_name=topic_test), COMMIT_FLOOR, matched_form="topic"
                    )
                self._abstain("names_no_entity", intent, lang)
                return None

        if wants_rate:
            name, form, score = self.catalogue.match(text, "test", lang, COMMIT_FLOOR)
            if name and score >= COMMIT_FLOOR and self._ambiguous_test(text, lang, table, score):
                self._abstain("ambiguous_test", "test_rate", lang)
                return None
            if name and score >= COMMIT_FLOOR and self._names_second_test(text, lang, table, name, form):
                self._abstain("multi_test", "test_rate", lang)
                return None
            if name and score >= COMMIT_FLOOR:
                self._serve("test_rate", lang)
                logger.info("fast path: test_rate %r (%.2f) [%s] from %r", name, score, lang, transcript)
                return FastPathResult("test_rate", _empty_slots(test_name=form or name), score, matched_form=form)
            self._abstain("below_commit_floor", "test_rate", lang)
            return None

        if wants_prep:
            name, form, score = self.catalogue.match(text, "test", lang, COMMIT_FLOOR)
            if name and score >= COMMIT_FLOOR and self._ambiguous_test(text, lang, table, score):
                self._abstain("ambiguous_test", "test_prep", lang)
                return None
            if name and score >= COMMIT_FLOOR and self._names_second_test(text, lang, table, name, form):
                self._abstain("multi_test", "test_prep", lang)
                return None
            if name and score >= COMMIT_FLOOR:
                self._serve("test_prep", lang)
                logger.info("fast path: test_prep %r (%.2f) [%s] from %r", name, score, lang, transcript)
                return FastPathResult("test_prep", _empty_slots(test_name=form or name), score, matched_form=form)
            self._abstain("below_commit_floor", "test_prep", lang)
            return None

        if wants_avail:
            name, form, score = self.catalogue.match(text, "doctorfull", lang, 0.9)  # the whole name, if it was said
            if not name:
                name, form, score = self.catalogue.match(text, "doctor", lang, COMMIT_FLOOR)
            if name and score >= COMMIT_FLOOR and not self._doctor_form_is_said(text, table, form):
                # sourav, 2026-10-16: an unknown doctor must not be answered as a similar-looking one, and an ordinary word must not be a doctor.
                self._abstain("doctor_form_not_said", "doctor_availability", lang)
                return None
            if name and score >= COMMIT_FLOOR and self.catalogue.doctor_owner_count(lang, form) > 1:
                # "Dr. A. Sen" when Ashok Sen and Abhishek Sen both sit here: the model's turn reaches the clinic API,
                # which answers "which one -- Ashok Sen or Abhishek Sen?" (by whole name, in the caller's script).
                self._abstain("ambiguous_doctor", "doctor_availability", lang)
                return None
            if name and score >= COMMIT_FLOOR:
                date_iso, confident = self._resolve_date(text, table)
                if not confident:
                    self._abstain("date_not_confident", "doctor_availability", lang)
                    return None
                self._serve("doctor_availability", lang)
                logger.info(
                    "fast path: doctor_availability %r (%.2f) date=%s [%s] from %r",
                    name,
                    score,
                    date_iso,
                    lang,
                    transcript,
                )
                return FastPathResult(
                    "doctor_availability",
                    _empty_slots(doctor_name=form or name, date=date_iso),
                    score,
                    matched_form=form,
                )
            if not table.faq_after_failed_avail:
                # owner, 2026-10-26: "কখন আসবো কটা থেকে খুলে" ("when shall I come, from what time does it open") holds the "when" word of a doctor-timing question
                # but names no doctor and says no doctor word: when it also fits one clinic FAQ topic nearly word for word, that is the answer.
                hit = self._faq_despite_complexity(text, lang, table, ignore_avail=True)
                if hit is not None:
                    return hit
                self._abstain("below_commit_floor", "doctor_availability", lang)
                return None
            # sourav, 2026-10-16: but a sentence that says "doctor" without naming one ("when is the doctor available in two days") is
            # still a DOCTOR question; falling through to the FAQ table answered it with "home collection available" (the word "available").
            # It goes to the model, which asks which doctor.
            if _SAYS_DOCTOR.search(f" {text} "):
                self._abstain("doctor_not_named", "doctor_availability", lang)
                return None
            # no doctor named: "what time do you open" is a clinic question, not a doctor's schedule

        # sourav, 2026-10-16: a sentence that names a doctor ("show my reports to Dr Sen", "I want to meet Dr Sen") is about that doctor,
        # not a clinic FAQ: the word "reports" or "clinic" in it used to answer it with the clinic's address or hours. It goes to the model,
        # which starts the booking (and agent/fast_path.doctor_booking_hint is the fallback if the model fails).
        if doc is not None:
            self._abstain("doctor_named_not_faq", lang=lang)
            return None

        # No rate/prep/availability cue at all -- try the FAQ topic table
        # before falling through to greeting/thanks/abstain. Deliberately
        # LAST among the "answerable" branches: every FAQ keyword phrase
        # is generic clinic language with no test/doctor cue word in it by
        # construction, so this cannot silently steal a
        # rate/prep/availability question -- those already returned above.
        faq_topic, faq_form, faq_score = self.catalogue.match(text, "faq", lang, FAQ_COMMIT_FLOOR)
        if faq_topic and faq_score >= FAQ_COMMIT_FLOOR:
            self._serve("clinic_faq", lang)
            logger.info("fast path: clinic_faq %r (%.2f) [%s] from %r", faq_topic, faq_score, lang, transcript)
            return FastPathResult("clinic_faq", _empty_slots(faq_topic=faq_topic), faq_score, matched_form=faq_form)

        # Pure greeting or thanks, with no entity and no question in it.
        if table.greeting_reply and table.any(text, table.greeting) and len(text.split()) <= 4:
            self._serve("smalltalk", lang)
            return FastPathResult("smalltalk", _empty_slots(), 1.0, direct_reply_bn=table.greeting_reply)
        if table.thanks_reply and table.any(text, table.thanks) and len(text.split()) <= 4:
            self._serve("smalltalk", lang)
            return FastPathResult("smalltalk", _empty_slots(), 1.0, direct_reply_bn=table.thanks_reply)

        # OWNER'S INSTRUCTION, 2026-10-10: an abstain was previously invisible after the fact -- _abstain only
        # ever recorded a bucketed REASON, never the transcript, so a turn that missed every cue and fell to the
        # LLM could not be debugged without reproducing it live while watching logs. Logging the near-miss (the
        # best FAQ candidate and its score, even though it did not reach the floor) costs nothing -- this branch
        # is already the slow path -- and turns "why did this one go to the LLM" from a live-repro exercise into
        # a grep.
        logger.info(
            "fast path abstain (no_cue_matched) [%s]: %r -- nearest faq candidate %r at %.2f (floor %.2f)",
            lang,
            text,
            faq_topic,
            faq_score,
            FAQ_COMMIT_FLOOR,
        )
        self._abstain("no_cue_matched", lang=lang)
        return None

    def snapshot(self) -> dict:
        total = self.stats["served"] + self.stats["abstained"]
        return {
            **self.stats,
            "catalogue_rows": len(self.catalogue),
            "serve_rate": round(self.stats["served"] / total, 3) if total else 0.0,
            "language_gap": serve_rate_gap(dict(self.by_lang)),
            "abstention_reasons": abstentions.snapshot(),
            "served_by_intent": fast_path_served.snapshot(),
        }
