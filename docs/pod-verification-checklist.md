# Checks to run when the pod is live

Everything below was accepted as "do it when the pod is up". None of it can be measured off-pod; each
item says what to run, what to look at, and what result means what. Nothing here has been run.

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
