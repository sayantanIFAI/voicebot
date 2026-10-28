# External review: verified against the code, 2026-10-28

Every claim below was checked in the repository or on the pod, not taken from the review. Status words: **CONFIRMED** (reproduced or read in the code), **PARTLY** (true with a qualification), **ALREADY DONE**, **NOT YET CHECKED**.

## Already fixed in this round (deployed, commit a54895e / a6ff105)

| Item | Result |
|---|---|
| Negated language requests ("do not speak in Hindi", "हिंदी में मत बोलिए", "হিন্দিতে বলবেন না") returned the rejected language | FIXED: sentences are split into clauses; a negated clause is never a request; an affirmative clause beside it wins. 8 tests. |
| Idempotency not atomic | FIXED: the key is claimed first (unique constraint), a twin gets 409 while the first runs, a failed write frees the key, a stale claim is taken over. 3 concurrency tests. |
| The agent never sent a key on booking writes | FIXED: found while checking. The pod database held only 30 idempotency records (payments, WhatsApp). hold / confirm / reschedule / cancel / add-test carried no key at all, so the guard protected none of them. Every write now sends one key per operation and reuses it on a retry. |
| `lang_explicit` set before a place-name false positive was rejected | FIXED (P2). |
| Fallback apologies missing | FIXED: the files were gitignored, so a TTS or model failure was silent (pod log: "Fallback audio ... missing" x7). Nine clips now committed. |

## Priority order I recommend

1. **P0 ownership check on cancel / reschedule** (below). A stranger who knows or guesses a confirmation ID, or states someone's phone number, can change or hear another patient's booking.
2. **Negation in `asks_to_book`** and the same for cancel/reschedule detection (small, local).
3. **Semantic cache**: stop caching transactional intents / slots (small).
4. **WebSocket control validation + buffer and call-duration limits** (small, mechanical).
5. **Three language fields** (medium; mostly renaming and one rule).
6. **State vocabulary** (not requested / not available / unknown / unverified / cancelled / failed): audit of reply templates (medium).
7. **Transactional outbox** when a real SMS / payment provider is chosen.

## P0: reschedule and cancellation accept a confirmation ID alone. CONFIRMED

- `clinic-api/main.py`: `cancel_booking(req)` and `reschedule_booking(req)` take only `confirmation_id` (+ new slot / `confirm_charge`). `booking_service.cancel_appointment` looks the row up by `confirmation_id` and `status="confirmed"` and cancels it. There is no patient, phone, proxy or session check at the API.
- The documented design (booking_service.lookup_bookings) calls the 8-hex confirmation ID "the bearer token for that one row". That is 32 bits, spoken aloud, sent by SMS and printed; it is not a credential.
- Agent side (`main.py`, cancel/reschedule flow): the confirmation ID comes from the caller's own words, or from `lookup_bookings(phone=<number the caller said>)`. Nothing requires `session.identity.is_verified`; that flag is only used to detect a changed voice. The caller-phone default in `lookup_bookings` is the phone that was searched, so "I am that number" is accepted.
- The only brake is the deterministic yes before the write and the cancellation-charge confirmation. Neither proves who is asking.
- **Fix (do this first):** (1) API: cancel/reschedule require `caller_phone` plus a proof tier from `authorize_disclosure` (self / delegated proxy / DOB-confirmed) and return `reason: not_authorized` otherwise; (2) agent: run the existing identity-verification flow (`session.security`, used for history) before any cancel or reschedule, and treat a failed check as a hand-off, never a retry loop; (3) rate-limit failed lookups per call and per number; (4) log denied attempts in `audit_log`. Caller ID does not exist in this stack, so a spoken phone number is a claim, not an identity: the second factor (DOB / security question, or an OTP to the number on file once SMS is real) is the real control.

## Test scenario table (what the code does today)

| Scenario | Today | Verdict |
|---|---|---|
| "Please do not speak in Hindi" | no language selected | FIXED |
| "Speak Bengali, not Hindi" | Bengali | FIXED |
| "My address is English Bazar" | no switch (place-name rule), and the flag is now set only after acceptance | OK |
| "Don't book an appointment" | `booking_carryover.asks_to_book` returns **True** (it ignores negation); also "বুক করবেন না", "मत बुक करो". A booking flow can start; it still cannot write without a spoken yes. | GAP, small fix |
| "Don't cancel my appointment" | `classify_yes_no` gives None for the bare sentence and `no` for "no don't cancel" / "ক্যান্সেল করবেন না"; cancel needs a yes before the write | OK for the write; intent detection is model-led |
| "Book CBC tomorrow" vs "today" | shared semantic cache can return the other's cached slots (see cache section) | GAP |
| Unverified caller gives another patient's phone | disclosure uses the stated phone as the caller (no caller ID) | GAP (P0 fix) |
| Valid confirmation ID of somebody else | read: bearer-token model; write: no check | GAP (P0) |
| Malformed JSON (`null`, `[]`), bad sample rate | see WebSocket | GAP |
| Large / continuous PCM | see buffer | GAP |
| Two concurrent identical booking requests | idempotency now exact-once per key; slot holds and unique slot rows already exist for the business effect | FIXED / OK |
| TTS text in wrong language | `_speak` logs the mismatch and still synthesises; scripts the voice cannot say are blocked (unspeakable spans), which silently drops the line | PARTLY, see below |
| DB write succeeds, response times out | now retried with the SAME key: a finished write returns its stored answer, a running one returns 409 and waits; still unknown after that = failure message, never a made-up confirmation | FIXED |

## P1: semantic cache. CONFIRMED (design risk)

`agent/semantic_cache.py` is one process-wide store shared by every call. Guards: PII slots (`phone`, `patient_name`) make an entry L1-only; entity slots (`test_name`, `doctor_name`, `faq_topic`, `symptom_description`) are checked by character overlap; `smalltalk` / `unclear` are L1-only. NOT guarded: `date`, `time`, a negation, `confirmation_id`, `relationship`. Cosine for "Book CBC tomorrow" vs "Book CBC today" is high (the file itself measures 0.78 to 0.82 for rewordings) and the cached value is the model's intent + slots, so the other day's slots can be returned. Fix: only non-transactional, entity-only intents are L2-eligible; any entry whose slots contain a date, time, id, count or relationship, or whose intent is a book / cancel / reschedule, is exact-text only; add a negation token check on the L2 path. Never trade this for hit rate.

## P1: WebSocket control validation and authentication

- **Validation. CONFIRMED.** `_handle_control`: `json.loads(raw)` then `msg.get(...)` crashes on `null` / `[]` / a number (AttributeError); `int(msg.get("sampleRate") or SAMPLE_RATE)` raises on a string and accepts 0 or 10**9. The outer handler logs "session crashed" and ends that call, so one bad client cannot take others down, but it is an unhandled failure, not a controlled rejection. Fix: require an object, whitelist control types, bound the rate to {8000, 16000, 22050, 24000, 44100, 48000}, reject with a close code.
- **Authentication. PARTLY.** The agent socket itself checks nothing; protection is the nginx gateway (HTTP basic auth on the public port 8888; `/twilio/` is deliberately open and must verify the provider signature). The agent listens on 0.0.0.0:8101 inside the pod. Fix: bind it to 127.0.0.1 behind the gateway, keep the signature check on the Twilio route, and add a per-session token for any non-browser client.

## P1: audio buffer. CONFIRMED

`PCMBuffer` appends to a `bytearray` with no cap (about 32 KB per second at 16 kHz). The idle timeout counts audio frames as activity, so a continuous stream never idles. `ADMISSION_MAX_CALLS` limits the number of calls, not their length or memory. Fix: a maximum call duration (for example 30 minutes), a maximum bytes per call, drop-and-close above both, and a per-message size limit on the websocket.

## P1: `_speak` and wrong-language text. PARTLY

`language_mismatch(lang, text)` is only logged. The unspeakable-spans check does block, but blocking produces silence for that sentence (seen on the pod: "ঠিক আছে, Dr. Abhijit Sarkar।" and "আমি নামটা শুনেছি Abu Ali" were dropped, the caller heard nothing). Fix: on mismatch or blocked spans, say the same sentence with the name omitted or in the language's own script, or a same-language fallback line, never silence. The doctor-name echo case is already fixed.

## P1: `action_gate`. PARTLY (the review is out of date)

The module docstring still says nothing calls it, but `main.py` does call `action_gate.decide(...)` before factual answers (REPEAT and READ_BACK are acted on). What it does not cover is authorization: its rule list has no "who is asking" step. Fix: add an AUTHORIZE step (verified identity and relationship to the record) ahead of CONFIRM for privileged reads and every write, and update the docstring.

## Adaptive wake delays. ALREADY SUPPORTED

`_turn_poll_loop` sleeps `session.take_wake_delay()`; `_next_wake_delay(result)` shortens it when the detector says a commit is near, with `POLL_INTERVAL_S` 0.5 s idle and `ACTIVE_POLL_INTERVAL_S` 0.2 s (env tunable). Not yet done: measuring the scheduling overhead before lowering the cadence further. Plan: run `tools/bench_transport.py` at 10, 20, 30 and 36 simultaneous calls at 0.5 / 0.2 / 0.1 s and record event-loop lag and CPU; do not go below the point where lag rises.

## Latency opportunities (do not trade correctness)

Measured on the pod (698 turns): total p50 907 ms, p95 2182 ms, p99 5365 ms; the intent model is the largest stage (p50 777 ms, p95 1874 ms, p99 4613 ms); language ID returned "not decisive" on 365 turns against 341 decisive ones, so all engines run on more than half the turns. In order: remove the model call for clearly deterministic requests (fast path already exists); skip the second recogniser when the first agrees; stream endpointing (ADR 0002 item 1). Caller authorization and cache correctness are never reduced to save time.

## Three language fields

Today: `session.lang` (reply language, locked on the first decisive turn), a local `detected_lang` per turn, `session.lang_explicit`, `session.lang_lock_weak`, and the TTS engine chosen from `lang`. They overlap. Target: `detected_input_language` (per turn, never persisted as a preference), `preferred_reply_language` (changes ONLY on an accepted affirmative request, now polarity-safe), `tts_output_language` (the voice actually used after the mismatch check). A detected difference alone never moves the preference; the existing weak-lock recovery and the English-lock rule remain the only automatic exceptions and should be expressed on `preferred_reply_language`.

## State vocabulary

The reply templates mostly distinguish found / not found / unavailable, but "clinic API down" and "not requested" are not separate answers: a failed lookup becomes the generic tool_failure line. Define one enum (NOT_REQUESTED, NOT_AVAILABLE, UNKNOWN, UNVERIFIED, CANCELLED, FAILED, AMBIGUOUS) returned by the tool client, with a fixed spoken line each; an ambiguous entity always asks, an unreachable API says it cannot be verified, a timed-out write is reconciled by its key. The model never decides identity, authorization, facts or a transaction: it only proposes the intent and slots that deterministic code then checks.

## Test setups that were stale (not product bugs)

- `test_an_emergency_at_low_confidence_or_mid_booking_still_fires`: the test used a bare `object()` as the booking; `_booking_open` now reads `.stage`.
- `test_a_written_surname_or_title_and_surname_resolves_to_that_doctor`: the doctor lookup returns the full name now; the expected "Dr. A. Sen" was out of date.
- `tests/test_smoke.py::TestClinicAPI`: needs a live clinic API at localhost.
- `tests/test_lab_price_multilingual.py`: the stand-in tool had no `route_department`.

## Implemented later the same day (commit after 60f5a08)

| Item | What was done |
|---|---|
| P0 ownership check | `booking_service.check_change_authorized`: cancel and reschedule need the confirmation id AND a phone number that belongs to the booking (booked-from, contact, patient, or a verified proxy; last ten digits compared). Refusals (`phone_required`, `not_authorized`, `too_many_attempts`) are written to `audit_log`; five refusals per booking or per number lock changes for 15 minutes (`CHANGE_MAX_FAILS`, `CHANGE_LOCK_MINUTES`, `CHANGE_REQUIRES_PHONE=0` turns it off). The agent asks for the number (`phone` is now a required slot of cancel / reschedule; a booking found by phone already has it), allows three wrong numbers, then hands the caller to the counter. A spoken phone number is still a claim, not an identity: the second factor (date of birth, or a code to the number on file once SMS is real) remains the next step. |
| `asks_to_book` negation | Clause-wise: a clause that names a booking and refuses it is not a request; another clause still can be. "বুক তো করিনি" stays a request. |
| Semantic cache | Transactional intents and entries with a date, time, id, count, relationship or contact number are exact-text only; an L2 hit between a negated and an un-negated wording is rejected. |
| WebSocket | `agent/ws_guard.py`: control frames must be a JSON object with a known type, bounded size; sample rate from a whitelist (else close 1003); per-message, per-call-bytes and per-call-duration limits (close 1009 / 1008). |
| Three language fields | `detected_input_language`, `tts_output_language` and a mismatch counter are recorded beside the reply language; no behaviour change. |
| State vocabulary | NOT done (audit of every reply template; separate piece of work). |
| Transactional outbox | NOT done (needs a real SMS / payment provider). |
