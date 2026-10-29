# ADR 0005 — Capacity round 2026-10-29: clinic API reads and workers, intent-model slots, a cheaper echo canceller

**Status:** Accepted, deployed on the pilot pod (git `main`). Extends [ADR 0002](0002-concurrent-calls-architecture.md); the process map there still holds, with the changes below.
**Date:** 2026-10-29
**Driver:** a booking died at the phone number ("এই মুহূর্তে দেখতে পারছি না") when several calls were running, and the 24-to-45 call ramp showed the single-worker clinic API, the intent model's tail and the echo canceller's CPU as the three limits.

## 1. What was found (measured on the pod, not assumed)

| Finding | Number |
|---|---|
| Every SQLite query costs about 24 ms: the database file is on the network volume (MooseFS) | one availability lookup made 3 or more queries: 76 ms |
| The agents' availability preload asked ~550 lookups every 20 s | ~27 requests a second, most of the clinic API's capacity (~45 requests a second in total) |
| The duplicate-booking check had a 4 s client timeout and one failure ended the booking | a booking died at the phone number |
| Each SQLite write commit on the network volume | about 150 ms; writes cap near 7 a second whatever the number of processes |
| Intent model, 1 request | median 0.52 s; at 8 parallel 1.3 s (writing the answer is the cost) |
| Intent model tail: a request landing on an Ollama slot that had not cached the shared 2,600-token prompt | 5.7 s of that 6.8 s |
| Echo canceller | 6.4 % of one core per speaking call; 22 % of that was the delay search; FFTs in double precision |

## 2. Decisions

### 2.1 Clinic API (`clinic-api/`)

* **Weekly schedule read once.** `_schedule_for_weekday` now serves from a cache of the whole `doctor_schedule` table (one query, at most 20 s old, dropped at once by any commit in the process). One lookup: 76 ms to 27 ms. A change made by another process is seen within the 20 s.
* **Bulk availability.** `GET /api/v1/doctors/availability/bulk?dates=a,b,` returns every doctor (short and whole name) for the dates, by calling the single-lookup function itself, so there is one rule. The agents' preload uses it once a minute (`preload_availability_bulk`); the single lookups remain the fallback. Idle clinic requests per 30 s: 738 to 12.
* **Reads are retried once.** `agent/tools_client._ReadRetryingClient` asks a timed-out or refused GET once more; the duplicate-booking check, if still unanswered, is skipped and the booking goes on (the hold and the confirm still check the slot).
* **Several worker processes** (`CLINIC_WORKERS`, default 3 in `deploy/start_all.sh`; the live pod's `/workspace/bin/_run_clinic.sh` carries the same flag; `.1worker.bak` beside it is the rollback). Two things had to change for that to be safe:
  * Start-up work (create tables, alter columns, seed, backfill) runs under a file lock on the local disk (`/tmp/clinic_api_startup.lock`), and the workers of one launch share a marker file, so only the first worker does it (`_run_startup_once_per_launch`).
  * `Idempotency-Key` claims are marked with the pid of the process working on them (`~running:<pid>`). Before, only the process's own memory said "still running", so a retry landing on another worker would have run a booking twice. Now a claim whose process is alive answers 409; a claim whose process is dead is taken over.
* **Tests that guard this:** `tests/test_bulk_availability_and_schedule_cache.py`, `tests/test_idempotency_cross_process.py`, `tests/test_conflict_check_retry.py`. On the pod, 16 to 24 clients sending the same write with the same key created exactly one row in every round, at 1 and at 3 workers (`/tmp/idem_race.py`, a throwaway script, not in the repo).
* **Limit that remains:** SQLite on the network volume. Reads overlap across workers; writes serialise (~7 commits a second). Moving the database to a server database (`DATABASE_URL`) is the real answer if call volume grows.

### 2.2 Intent model

* The "fast" prompt (language and date at the end, so all languages share one cached start) has been the default since 2026-10-25.
* New: `main._warm_intent_slots` sends one request per Ollama slot (`OLLAMA_NUM_PARALLEL`, 8) at start-up, so no caller meets a slot that has not read the shared prompt. Slowest request at 4 parallel: 6.8 s to 1.1 s. The median at 8 parallel (1.3 s) is the GPU and the 10-core quota, not the cache.

### 2.3 Echo canceller (`agent/echo_cancel.py`, `agent/full_duplex.py`)

* About 60 % cheaper with the same cancelling: single precision, no per-block array copies, the delay search made four times rarer once the delay has been confirmed three times. 6.4 % to 2.5 % of a core per speaking call; ERLE on the synthetic scene unchanged (23.3 dB). Constraining the gradient on one partition per block was tried and rejected: it saved little and made the barge-in detector fire three times for one interruption.
* **Switch `AEC_MODE`** (`deploy/env.sh`): `full` (default; the canceller runs on every call, for browser testing) or `phone_light` (a phone-line call skips the canceller's maths; barge-in detection stays; browser calls keep the full canceller). The owner will choose the production setting after a real SIP caller has been tested. Nothing is switched off today.
* Talking over the AI, phone-line setting, measured on the pod with a replayed caller: playback stops 0.91 s after the caller starts (old code 0.86 s), and the caller's words are recognised.

## 3. Ramp with all three changes

The other session's ramp (`rampload.py`: 10 calls, then 5 more every 10 s for 120 s, up to 65 calls at once, each call plays g1, d1 to d5 in order and speaks the next only after the AI has answered; duplex path on, `AEC_MODE=full`; call limits and load shedding switched off for the test). Same recordings, same pod, compared with the same ramp run earlier the same day:

| | Earlier ON round | This round (all changes) |
|---|---|---|
| Calls completed / turn time-outs | 41 / 24 | 50 / 14 |
| Median, end of caller speech to first AI audio (all turns) | 12.9 s | 11.4 s |
| By time into the ramp: 0-20 s / 20-40 s / 40-60 s | 3.6 / 5.6 / 12.2 s | 3.3 / 5.4 / 8.2 s |
| Per stage with more than 30 active calls: intent p50 / p95 | 2.6 / 10.8 s | 2.1 / 5.7 s |
| reply p50 / p95 | 1.8 / 9.8 s | 1.3 / 4.2 s |
| whole turn in the agent, p50 / p95 | 7.9 / 23.8 s | 7.6 / 13.4 s |
| asr p50 (CPU-starved) | 2.6 s | 2.8 s |
| Peak ingress backlog / event-loop lag / agent CPU mean | 2.4 s / 0.65 s / 38 % | 3.5 s / 1.0 s / 42 % |
| Admission p95 | 32.7 s | 15.1 s |
| Calls refused because the clinic API failed its health probe (the earlier OFF round) | 20 | 0 |

With 15 or fewer active calls a whole turn takes about 1.2 s (p95 2.0 s). With 16 to 30 active calls, 1.6 s (p95 7.6 s). The design point is 30 calls; above it (the ramp reaches 65) the machine is out of CPU (load 15 on a 10-core quota): speech recognition rises to 2.8 s and everything waits. What improved is the intent and reply tails (the clinic API and the slot warm-up); what did not is the CPU ceiling. The mean agent CPU is not lower because more calls finished their turns in this run; the echo canceller's saving (about 3.9 points of a core per speaking call) is real but small next to speech recognition on a starved CPU. The earlier round was run before some of the day's other changes (for example the audio consumer batching), so the improvement is not all from this ADR.

Raw results: `round_new_aecon_*` (calls, samples, timing) were produced on the pod; the other session's `analyse_round.py` prints the tables above.

## 4. What was not done

* Server-side database for the clinic API (see 2.1, last bullet).
* Browser calls with real speaker echo ignore interruptions until the canceller has converged (by design, `MIN_TRUSTED_ERLE_DB`); phone calls do not use that rule.
* Six tests that failed before this round still fail (`tests/test_orchestrator_duplex.py` x5, `tests/test_barge_in.py::test_nothing_is_acted_on_when_the_agent_is_silent`); the reasons are test set-up and a counter, not behaviour.
