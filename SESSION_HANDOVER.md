# SESSION HANDOVER — uncommitted work, 2026-09-23

**For:** a fresh Claude session (or engineer) picking up exactly where this
session left off. **Not** a replacement for `HANDOVER.md` — that file is the
pod-verified, ground-truth doc per `CLAUDE.md` §2 authority order. This file
is narrower and younger: it exists because `HANDOVER.md` (last written
2026-09-08/09-15) is now stale against everything below, and none of it has
been verified on a live pod, so it must not be blended into that file as if
it were pod-verified fact.

**Read `CLAUDE.md` first, then this file, then `HANDOVER.md` for the
baseline.** Where this file and `HANDOVER.md` disagree about what the code
can do today (multilingual support, barge-in, "1 built · 39 partial · 113
absent"), trust this file — `HANDOVER.md`'s "It cannot" list in particular is
now out of date and should be refreshed by whoever next has pod access.

---

## 1. Repo state right now

```
branch: epic-e26-booking-reschedule-cancel
remotes: origin -> sayantanIFAI/ivr.git, voicebot -> sayantanIFAI/voicebot.git
last commit: d67eaea "Multi-question/follow-up conversation handling,
             answer-quality fixes, and CodeRabbit security/correctness fixes"
working tree: 26 modified files + 15 new untracked files, NONE committed
```

Everything described in §2–§4 below is **uncommitted, local-only**. Run
`git status --short` before doing anything destructive. Nothing here has
been pushed since `d67eaea`.

Full local test suite (excludes `tests/test_smoke.py`, which needs live pod
services and is not runnable off-pod — see `scripts/gate.sh`):

```bash
python -m pytest tests/ --ignore=tests/test_smoke.py -q
```

Last run: **463 passed, 0 failed.** Re-run this yourself before trusting it —
don't take a prior session's word for a number that changes with every edit.

---

## 2. What this session actually built (uncommitted)

Three epics from the backlog — **Call Intelligence**, **Channels and Written
Confirmations**, **Conversation Latency as the Caller Feels It** — plus a
batch of fixes from a second CodeRabbit review pass on PR #1. All new
`agent/` modules are pure-Python, unit-tested, and import nothing from
`main.py` (the one-import rule holds). Everything touching `main.py` was
verified with `python -m py_compile main.py` and regenerated into
`main_pcm.py` via `python tools/make_pcm_variant.py` — **never edit
`main_pcm.py` directly**, it is generated.

### New files (untracked)
| File | Story | What it does |
|---|---|---|
| `agent/call_state.py` | KCD-061 | Per-call `CallState` (language, confidence, channel quality, senior/caller-state signals) |
| `agent/code_switch.py` | KCD-065 | Text-based script-mixture detection (bn/hi/en bucket) |
| `agent/channel_quality.py` | KCD-075 | FFT-based clean/narrowband/noisy/crosstalk classification |
| `agent/detector_budget.py` | KCD-076 | `asyncio.wait_for`-based time-budget wrapper for any detector |
| `agent/latency_metrics.py` | KCD-469 | Per-language turn-latency p50/p95 |
| `agent/clause_split.py` | KCD-462 | Splits a long reply into clauses at sentence punctuation |
| `agent/filler.py` | KCD-461 | Pure async race: speak a filler if a slow stage exceeds a threshold, suppressed if the real result arrives first |
| `tests/test_*.py` (7 files) | — | Unit tests for the above, all passing, no network/GPU |

### Modified files, grouped by what changed
- **`main.py` / `main_pcm.py`** (267 lines each, same diff by construction):
  `CallState` wiring into `_dispatch_turn`; channel-quality classification
  budgeted at 0.08s; code-switch/channel-quality/fast-path-served counters;
  `/api/health` now reports `ready`/`startup_duration_s`; `/api/stats`
  extended with `insufficient_information`, `code_switch_buckets`,
  `channel_quality_buckets`, `detector_budget`, `turn_latency_by_language`,
  and (this session) `barge_in_interrupts`; `_speak()` rewritten to stream
  a reply clause-by-clause (KCD-462) instead of one TTS call per reply;
  `_await_with_filler()` now wraps the **whole** cache-then-LLM sequence in
  `_resolve_intent`, not just the LLM call (KCD-459 — a slow embedding
  lookup used to burn into the filler threshold silently); `_handle_control`
  gained an `"interrupt"` control-message branch (KCD-464, see §3).
- **`agent/fast_path.py`, `agent/outcome_metrics.py`, `agent/phrases.py`**:
  `_serve()`/`fast_path_served` per-intent counter (KCD-460);
  `barge_in_interrupts` counter added; `please_wait` filler phrase reworded
  per explicit instruction (bn "আচ্ছা, ঠিক আছে", hi "ठीक है", en "Let me
  check, please." — the English wording was a judgment call, the request
  said "let me please" verbatim, which read as a stray fragment).
- **`static/index.html`**: manual "Stop, let me talk" interrupt button
  (KCD-464, see §3) plus the clause-streaming-aware `pendingClips` counter
  that was already correct before this session (verified, not rebuilt).
- **`agent/llm.py`, `agent/phonetic_match.py`, `clinic-api/*.py`,
  `tests/test_*.py`**: the second-batch CodeRabbit fixes — see §4, all
  10 findings there were independently verified against current code
  before being applied (one was disproven-then-reproven: the "NFC
  decomposes flap consonants" claim looked backwards until checked with
  `unicodedata.normalize` directly — it's correct, Bengali/Devanagari flap
  letters are in Unicode's NFC composition exclusion list).
- **Doctor-side "ask correct questions back"** (`clinic-api/main.py`,
  `agent/reply_templates.py`, `agent/reply_templates_i18n.py`): tests
  already had ambiguous-name clarification (KCD-446); doctors didn't.
  Added `_find_doctor_candidates`/`_doctor_suggestions` mirroring the
  test-side logic, so two similarly-spelled doctors now trigger a "which
  one did you mean" reply instead of `_find_doctor` silently picking one —
  the "Doctor Nobody" bug class `CLAUDE.md` §1 already warns about.

---

## 3. KCD-464 — read this before assuming barge-in works

`main.py`'s own module docstring (§ "HALF-DUPLEX GATE") explains the mic is
muted client-side for the **entire** duration the agent is speaking, and
there is no acoustic echo canceller — so the server structurally cannot
detect a caller talking over the agent. That constraint was **not** removed
this session and cannot be removed without real AEC infrastructure (a much
larger change, already flagged as such in that same docstring).

What was added instead: a visible **manual** interrupt button
(`static/index.html`) that, when tapped, immediately stops local playback,
unmutes the mic, and sends `{"type": "interrupt"}` over the control
WebSocket; `main.py`'s `_handle_control` treats it exactly like
`playback_done` (releases the gate early) and records it in the new
`barge_in_interrupts` counter. **This is a UI affordance, not acoustic
barge-in.** Don't describe it as barge-in to anyone without this caveat —
it was built and presented to the user with this distinction explicit.

## 4. Second CodeRabbit review batch — all 10 findings, verified and fixed

Every finding below was checked against current code (not taken on faith)
before fixing; none were false positives, all 10 were applied:

1. `agent/llm.py` — `secondary_slots.faq_topic` wasn't validated against
   `FAQ_TOPICS` (the primary slots' version already was) — an invented
   topic could 404 through `_tools.get_faq` into a wrong `tool_failure`
   reply instead of an ordinary re-ask.
2. `agent/phonetic_match.py` + `clinic-api/phonetic_match.py` (deliberately
   duplicated, kept in sync by hand — see that file's own docstring) — flap
   consonants (ড়/ঢ়/ड़/ढ़) silently misclassified after NFC normalization
   because Unicode excludes them from NFC composition. Fixed in both
   scripts, both files, with 2 new regression tests built from explicit
   codepoints (typed glyphs would silently hide the exact bug being
   tested — this bit once already while drafting the fix, see `git diff`
   history on `tests/test_phonetic_match.py` if curious).
3. `clinic-api/booking_migrate.py` — no PostgreSQL branch existed for the
   partial-unique-index migration; an existing Postgres database would
   either crash on SQLite-only `PRAGMA` calls or (with a naive dialect
   guard alone) silently keep the old blanket constraint forever. Added a
   real Postgres branch (`DROP CONSTRAINT IF EXISTS` + `CREATE UNIQUE
   INDEX ... WHERE`).
4. `clinic-api/booking_service.py` — `_release_expired_hold` was a
   read-then-delete-by-primary-key with no DB-side re-check, a real TOCTOU
   race under SQLite's legacy transaction mode. Converted to a single
   filtered `DELETE`.
5. `clinic-api/enquiry_migrate.py` — `BOOLEAN ... DEFAULT 1/0` rejected by
   PostgreSQL; changed to `TRUE`/`FALSE`.
6. `clinic-api/models.py` — the partial unique index only had
   `sqlite_where`; added `postgresql_where` too, for a **brand-new**
   Postgres database (an existing one still needs #3's migration branch).
7. **Eight** test fixtures were missing `enquiry_migrate` (and in one case
   `booking_migrate`) from their per-test module-eviction list, letting a
   stale migration module leak across tests and target the wrong SQLite
   engine. CodeRabbit named 4 files; the same pattern was found by
   inspection in 4 more (`test_booking_concurrency.py`,
   `test_booking_authorization.py`, `test_booking_service.py`,
   `test_clinic_matching_i18n.py`) and fixed for consistency, not just the
   ones named.
8. `tests/test_booking_service.py` + `tests/test_booking_concurrency.py` —
   `_next_weekday`-style helpers started their search from *today*, which
   could select today's date after today's chamber hours had already
   passed, silently emptying `available_slots()`. Changed to start
   tomorrow — same fix pattern already applied earlier this session to a
   different helper (`_next_weekday_with_schedule`).
9. `tests/test_digit_fidelity.py` — `PHONE_CORPUS` was built from an
   unseeded `random.randint`, so a failing generated case couldn't be
   reproduced. Seeded with `random.Random(445)`.
10. `tests/test_doctor_matching_e29.py` — the "badly garbled surname" test
    queried "Roy" exactly, which matches `_find_doctor`'s first (exact
    substring) branch and never reached fuzzy/phonetic matching at all —
    a false-confidence test that couldn't have caught a regression in the
    code path it claimed to test. Verified CodeRabbit's own suggested
    replacement independently (`difflib` ratio 0.588 for "Nukharji" vs
    "Mukherjee", confirmed below `FUZZY_SURNAME_FLOOR=0.60` and above
    `PHONETIC_ASSISTED_FLOOR=0.45`, confirmed same phonetic key) before
    applying it.

---

## 5. Open loops — pick one of these next

1. **Nothing is committed.** The user has not yet said whether to commit
   and push this batch. Ask before doing so — don't assume.
2. **The very first CodeRabbit review this session ever triggered** (on an
   earlier commit, via `@coderabbitai full review`) was never checked/
   reported back to the user — that's a loop from earlier in the day that
   is still technically open, separate from the second review batch in §4
   (which WAS fully handled).
3. **KCD-466 (queue position at peak)** — the user explicitly said "do not
   do anything" here. `agent/admission.py`/`docs/adr/0001` deliberately
   never queues a call (refused call → human, immediately). Do not build a
   queue without the user re-opening this and asking for a new ADR.
4. **Everything touching `main.py`/`main_pcm.py` this session is compiled
   and regenerated but not run against live ASR/TTS/Ollama/clinic-api** —
   no pod exists right now (see `HANDOVER.md` §12 pattern: pods are
   ephemeral, get a fresh one from the RunPod console, don't trust any
   hardcoded IP). Treat every latency/behavior claim above as "proven by
   local unit tests", not "proven live", until someone runs
   `tests/test_smoke.py` on a real pod.
5. **`HANDOVER.md` itself is stale** — its "It cannot: ... Speak anything
   but Bengali... Be interrupted" section predates the multilingual
   routing (`agent/lid.py`, `agent/asr_router.py`) and this session's
   interrupt affordance. Whoever next has pod access should refresh it —
   don't hand-wave a rewrite without re-measuring the numbers in its
   "Measured numbers" table, that table's whole value is that it's real.

## 6. Gotchas specific to this working tree

- **`gh` CLI** is installed but not on `PATH` in either Bash or PowerShell
  on this machine — invoke it at
  `C:\Program Files\GitHub CLI\gh.exe` by full path.
- **Editing `agent/phonetic_match.py` or `clinic-api/phonetic_match.py`
  with the `Edit` tool's exact-string match is fragile** — mixed
  Bengali/Devanagari/Latin content in one file occasionally fails an
  exact match for reasons that aren't visually obvious (normalization
  form of a typed/pasted glyph is not guaranteed stable). When it happens,
  drop to a small Python script that reads/replaces/writes the file with
  explicit `\uXXXX` escapes and verifies the byte content afterward
  (`[hex(ord(c)) for c in s]`) rather than fighting the editor on a retry.
- **`main.py` cannot be imported or run in this dev environment** —
  torchaudio isn't installed here. The verification discipline for any
  `main.py` change is `py_compile` + `tools/make_pcm_variant.py`
  regeneration + stating it's "pending live verification", never claiming
  a `main.py` change is proven by a passing local test suite alone.
- **`tests/test_smoke.py` needs live services** (`localhost:8080` clinic-api,
  a running Ollama, a running voice-agent process) and will show ~20+
  failures off-pod. That's expected — `scripts/gate.sh` itself only runs
  it when `VOICE_AGENT_ON_POD=1` is set. Don't mistake those failures for
  regressions from anything in this session's diff.

---

## 7. Update -- later the same day (read this before trusting section 2)

Sections 1-6 describe the state at commit `02d7ae1`. Everything below was
built AFTER it. Same rule as before: proven by local unit tests, not on a pod.

### Done, tested, and wired into the call path
| Story | What | Where |
|---|---|---|
| KCD-483..488 (E32) | 483/484/485 verified already satisfied; **486** post-commit write verification + hold/escalate; **487** fault-injection test; **488** versioned `CancellationPolicy` table. Also fixed `classify_yes_no("not sure") -> "yes"`. | `clinic-api/booking_service.py`, `models.py`, `agent/booking_flow.py` |
| KCD-162 / 164 | Prosody logic extracted to `agent/prosody.py` and made testable; `.` sentence splitting, punctuation-aware pauses, every parameter per-request tunable. Audio cache pre-warm fixed to warm the SAME clauses `_speak` sends (my KCD-462 change had silently broken it). | `agent/prosody.py`, `tts_server.py`, `agent/tts.py` |
| KCD-157 | Phone/ID grouping. **Fixed a real bug: `speed=0.8` was being sent as Coqui `length_scale`, so prices/phones were spoken ~35% FASTER, not slower.** Server now treats `speed` as a rate. | `agent/figures.py`, `agent/prosody.resolve_length_scale` |
| KCD-149 / 155 / 084 / 087 | Appendix C speech policy, acknowledgement templates, senior-voice detector + persistence (`Patient.senior_mode`), language policy. | `agent/speech_policy.py`, `acknowledgement.py`, `senior_voice.py`, `language_policy.py` |
| KCD-159 | Explicit pronunciation path (lexicon + acronym spelling; unknown words still block, KCD-455). | `agent/pronunciation.py` |
| Audio robustness | Cross-talk (two-pitch) detection, Wiener-style noise filter + level gain, jumbled-transcript detection, and **empathetic re-ask (twice) before any hand-off**. | `agent/audio_quality.py`, `agent/reask_policy.py` |

### NOT done -- do not report these as complete
- **KCD-321** kill switch (per-intent/department, non-engineer UI, clean finish of in-flight calls)
- **KCD-322** graceful drain on SIGTERM
- **KCD-326** per-call resource bounds
- **KCD-151** per-language verbalisation table registry + table-driven tests
- **KCD-160 / 167** voice identity document + automated audio-fingerprint drift check
- **KCD-161** first-clause streaming synthesis on the TTS server
- **KCD-163** cancellable synthesis/playback (note: after a manual interrupt the server can still send already-rendered clauses)
- **KCD-131 / 132** External-gated (real hospital/lab systems)
- **KCD-150** External-gated (native-reviewer sign-off); KCD-159's 200-term sign-off also pending. ~70 lexicon entries exist, all unreviewed.

### Things a reviewer must know
- Every audio threshold (`audio_quality.py`, `senior_voice.py`) is REASONED and validated on SYNTHETIC signals only. The cross-talk THRESHOLD (0.05) is REASONED: it was placed inside the gap between two scores measured on SYNTHETIC signals only (single voices <= 0.012, two-talker mixes >= 0.13). The scores are synthetic-measured; the threshold is not calibrated on real audio, and real handset audio has not been validated. Recalibrate on real calls before trusting.
- `senior_voice.py` is a heuristic prior, not an age classifier. It needs evidence over >= 2 clips, or an explicit request/stated age.
- One existing test assertion was changed on purpose: `tests/test_speech_norm.py` pinned the *ungrouped* phone reading, which KCD-157 deliberately changes.
- `/api/v1/patients/senior` is unauthenticated like every other clinic-api endpoint (Epic E14 gap); it exposes only a delivery-mode boolean per phone number.
- `main.py` changes are compiled + PCM-regenerated + checked for undefined names; they are not exercised end-to-end.
