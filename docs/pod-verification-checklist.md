# Checks to run when the pod is live

Everything below was accepted as "do it when the pod is up". None of it can be measured off-pod; each
item says what to run, what to look at, and what result means what. The first run on a live pod is recorded in "Results" at the end.

## 1. English recogniser: one decoder or two (extra confirmation turn?)

```bash
curl -s -X POST localhost:8003/transcribe -H "Content-Type: application/json" -d '{"wav_path":"/workspace/sample_en.wav"}'
```
Read `decoder_used` in the reply.
- `rnnt` (with a `decoder_agreement` between 0 and 1): both decoders ran. Nothing changes.
- `fastconformer` or `ctc_fallback`: only one decoder produced the text, so every English factual question
  (test price, doctor, preparation, booking lookup) is read back once before it is looked up
  ("Do you mean CBC?"). Decide: accept the extra turn, or give the English service a second decoder.

Then make one English call: "what is the price of a CBC test". If the agent asks "Do you mean ...?" first,
you are on the single-decoder path.

## 2. TTS: the one-at-a-time-per-language ceiling

Run the load test (KCD-236) and log, per language, at 10, 20 and 30 simultaneous calls: time waiting for
the lock, time rendering, p95 time to first audio. Set a number it must stay under (for example p95 wait
under 100 ms at 30 calls). If it is exceeded, run one TTS worker per language or per GPU; do not remove
the lock (the test in `tests/test_tts_concurrency.py` fails without it).

## 3. Service token and the kill switch

1. `ls -l /workspace/.clinic_api_token` exists (mode 600); `CLINIC_API_REQUIRE_TOKEN=1` in `deploy/env.sh`.
2. `curl -i localhost:8080/api/v1/tests/search?name=CBC` must return **401**; with
   `-H "Authorization: Bearer $(cat /workspace/.clinic_api_token)"` it must return **200**.
3. The clinic API startup log must NOT contain "served WITHOUT authentication".
4. The internal ports (8080, 8002, 8003) must not be reachable from outside the pod; only 8100/8101 are.
5. Kill switch: `touch /workspace/.ai_bypass`; the next call must be handed to a person at once
   (`/api/admission` shows `closed_reason: bypass`); remove the file and confirm calls are admitted again.
6. `lookup_booking` by a stated phone number is accepted for the pilot with the kill switch as the
   mitigation. To turn that off later, route booking lookups through the security questions (the same
   `SecurityCheck` flow as history).

## 4. Voice (KCD-511)

```bash
python tools/audition_voices.py --url http://localhost:8002 --out voices/
```
Play the files and choose. Set `TTS_SPEAKER=<name>` in `deploy/env.sh`, restart the TTS service. A person
has to judge how old a voice sounds; nothing can measure it.

## 5. Recognition and audio (needs recordings)

- Endpointing (KCD-047): at least 200 labelled narrowband calls, `python tools/calibrate_endpointing.py`.
  The adaptive pause (`ADAPTIVE_PAUSE`) is reasoned; this replaces its constants with measured ones.
- Attention and noise (KCD-053/055/057): `python tools/conditioning_eval.py CLIPS_DIR --engine en=http://localhost:8003`,
  once with `NEAR_END_ATTENTION=on` and once `off`, per language, at the four levels and noise profiles.
  Keep it on only for a condition where the word error rate drops.
- Echo and barge-in (KCD-051/052): `python tools/echo_eval.py --pairs RECORDINGS_DIR`.

## 6. Messages and sample data on the pod

```bash
python tools/check_messages.py --url http://localhost:8080 --token-file /workspace/.clinic_api_token
python tools/demo_patient_history.py --url http://localhost:8080 --token-file /workspace/.clinic_api_token
```
The demo shows the sample patients through the security questions and history. To stop the sample rows
being created on a database that will hold real patients, set `CLINIC_SEED_SAMPLE_PATIENTS=0` in
`deploy/env.sh` before the first start.

---

## Results: first run on a live pod (2026-09-25)

Pod `pfup5m0iyql2l9` (RunPod EU-RO-1, RTX PRO 4500 Blackwell, 32 GB, 28 vCPU, 62 GB), running the code at
`voicebot/main` 8b67ccf plus the fixes listed under "Found on the pod". Run by Claude over SSH on the direct TCP
port. Synthetic input only; nothing here used a real caller's audio.

| # | Item | Result |
|---|---|---|
| 1 | English recogniser, one decoder or two | `decoder_used = rnnt`, `decoder_agreement = 0.8` on a synthetic English clip, three runs, identical text. Both decoders ran, so English factual questions do **not** take the extra confirmation turn. One clean synthetic clip; a real accented caller may differ, so re-check on recordings. |
| 2 | TTS per-language lock | Confirmed serialised: a single synthesis is about 25-30 ms; ten requests at the same instant give p95 about 0.5-0.6 s, twenty about 0.9-1.0 s, thirty about 1.4-1.5 s, in all three languages, with no errors. That is a **same-instant burst of one short sentence**, the worst case, not 30 calls' steady state. **Not yet decided:** the number this must stay under (the item asks for one). Reasoned only: 28 calls each needing a sentence every 5-10 s is 3-6 syntheses per second against a capacity of roughly 20-30 per second per language, so the lock is not the first limit; a real load test (KCD-236) with multi-clause replies is what settles it. |
| 3.1 | Token file exists | Exists (41 bytes). **Mode is 666, not 600**: the `/workspace` network volume ignores `chmod` (a test file also stayed 666), the same limitation as `chown`. The pod has one user, root, but the token is not confidential from anything else that can read the volume. Mitigation open: keep it off the shared volume if the volume is ever shared. |
| 3.2 | 401 without, 200 with the token | 401 and 200 as required. |
| 3.3 | No "served WITHOUT authentication" in the startup log | Zero occurrences. |
| 3.4 | Internal ports not reachable from outside | 8080, 8002, 8003 and 11434 bind 127.0.0.1 only. 8100 and 8101 bind 0.0.0.0, but RunPod's HTTP proxy routes only 8888 on this pod (the other ports return 404 from outside); the public URL is `deploy/tcp_forward.py 8888 8101`. |
| 3.5 | Kill switch | `touch /workspace/.ai_bypass` gave `open:false, closed_reason:"bypass"`; removing it reopened admission. |
| 3.6 | `lookup_booking` by stated phone, kill switch as the mitigation | Kill switch verified (3.5). The lookup-by-stated-phone acceptance is unchanged; no separate test was run. |
| 4 | Voice audition | Files written to `/workspace/voices/` (bn, hi, en; female and male). **A person still has to listen and choose**; nothing was judged. |
| 5 | Recordings (endpointing, attention/noise, echo) | **Not run**: needs recordings. |
| 6 | Messages and sample data | `check_messages.py`: 9 of 9 pass. `demo_patient_history.py`: all four sample patients pass the security questions, the refused-before-verification read, the wrong-answer attempt, the history answers in three languages and the "cannot find" wording. **The sample patients are now in the pod's persistent `clinic.db`** (`CLINIC_SEED_SAMPLE_PATIENTS` defaults to 1 on an empty registry); set it to 0 before this database ever holds real patients. |
| - | `tests/test_smoke.py` against the live services | 32 passed (after the token change to the test and the fallback audio below). |

### Found on the pod, and what was done

1. **The fast path was silently OFF on `main`.** `main.py` fetched `/api/v1/catalogue` without the service token,
   got a 401, took the error body for a catalogue and logged "fast path ready over 0 catalogue rows"; being
   non-`None` it was never retried, so every turn paid a model call. Fixed: the token is sent, a non-success
   reply or a body that is not a catalogue is refused (so it degrades to "no fast path yet" and is retried).
   Verified on the pod: "fast path ready over 74 catalogue rows".
2. **The first caller after a restart would hit the model's cold start.** Only `bge-m3` was resident. The
   startup warm-up existed but called the extractor with the per-turn 12 s deadline (KCD-465), so on a 47-74 s cold
   load it timed out, the client disconnected and Ollama abandoned the load (both entrypoints logged "intent model
   warmup failed ... 12.0s deadline"). Fixed: a separate `INTENT_WARMUP_DEADLINE_S` (default 300 s) used by
   `_warm_intent_model()`, with a test. Warmed by hand on the pod meanwhile; **the fixed code is verified by test, not
   yet by a cold restart of the pod.**
3. **The fallback audio did not exist** (`static/fallback_audio/*.wav` is git-ignored and recorded by hand per
   `setup_addon.sh`). They are what a caller hears when TTS itself is down. Synthesised once from the live TTS in
   Bengali (the three lines in `setup_addon.sh`) and in Hindi and English; the Hindi and English wording is
   a draft for native review.
4. **`speakable()` rejected every Bengali reply ending in a danda** (U+0964 is inside the Devanagari block it
   counted as the wrong script). Fixed in `agent/lang_select.py` with a test.
5. **`/speakers` listed a stray `"` speaker** from the Bengali checkpoint's own table. Filtered.
6. **`tests/test_smoke.py` sent no token** to a clinic API that now requires one. A marked, deliberate change: the
   four calls carry the token; no assertion was changed.
7. Packaging note: `git archive` on Windows converts line endings (`core.autocrlf=true`) and breaks the shell
   scripts on the pod; use `git -c core.autocrlf=false archive`.

### Still to do on the pod
A cold-restart check that the warm-up now leaves the 7B model resident (item 2); real recordings for item 5; the ear-choice for item 4;
the load-test number for item 2; a real caller's call through the public URL.

