# ADR 0002 — Many calls at once: separate recogniser service, per-call voice detection, scaling knobs

**Status:** Accepted, deployed on the pilot pod (git `main`). Extends [ADR 0001](0001-pilot-single-l4-architecture.md); it does not replace it (ADR 0001 still describes the routing of Bengali / Hindi / English and the truth boundary).
**Date:** 2026-10-26
**Driver:** a load test of 24 simultaneous calls (2026-10-25) showed callers waiting 12 to 36 s for an answer while the GPU sat about 95 % idle. The bottleneck was software: one Python process did everything for every call.

## 1. Decision in one paragraph

Take the heavy, serial work out of the one process that holds every call's websocket, and give each piece its own process (or several), so that 24 to 36 callers are spread over independent interpreters instead of queueing behind one lock:
Bengali / Hindi speech recognition and language identification become their own service with several worker processes; voice-activity detection runs per call on a streaming model; echo / barge-in arithmetic is skipped while the agent is silent; speech synthesis runs in three worker processes; Ollama serves eight intent requests in parallel. A cap on simultaneous calls (admission control) protects the rest.

## 2. Process map

```text
   browser / Twilio  --HTTPS-->  nginx gateway :8888 (basic auth; /twilio/ is open, signed by Twilio)
                                      |
                                      v
                         main_pcm.py  :8101   (one process; also main.py :8100 = old WebM path)
                          |  per call: websocket, PCM buffer, streaming Silero VAD,
                          |  turn-taking, booking logic, replies
          +---------------+---------------+------------------+---------------------+
          v               v               v                  v                     v
   indic_asr_server   english_asr_    tts_server.py     Ollama :11434        clinic-api :8080
   :8004 (6 workers)  server :8003    :8002 (3 workers) qwen2.5:7b (intent)  SQLite, the truth
   bn + hi ASR +      (own venv,      bn / hi / en      bge-m3 (cache)         for prices, doctors,
   language ID        mainline NeMo)  FastPitch+HiFiGAN  8 parallel requests    slots, bookings
```

All service-to-service calls are plain HTTP on `127.0.0.1`; only the gateway port is public. The two agents are one code base: `main.py` is the source, **`main_pcm.py` is generated** from it by `python tools/make_pcm_variant.py` (never edit `main_pcm.py` by hand; transport changes for the PCM variant go in the generator; the reasoning half below `_resolve_intent` must stay byte-identical, which the generator checks).

## 3. What changed and why

| Change | Where | Why (measured 2026-10-25) |
|---|---|---|
| Bengali / Hindi recognition as a separate multi-process service | `indic_asr_server.py` (`uvicorn --workers 6`), client `agent/asr_router.py` (`HTTPASREngine`) | the decoder runs one clip at a time in Python; in-process, 24 callers finishing together waited 12-28 s in one queue. Each worker owns its own interpreter lock |
| Utterance batching inside a recogniser | `agent/asr.py` (`ASR_MAX_BATCH`, `ASR_BATCH_WAIT_S`) | clips that arrive within 15 ms are decoded together |
| Language ID served by the same workers | `agent/lid.py` (`RemoteLID`), `POST /lid` on :8004 | language ID in the agent process was a second queue |
| Streaming per-call VAD | `agent/vad_stream.py` | one voice-activity model per call, fed 20 ms at a time, a pool of `VAD_POOL` models; the old design re-scanned the buffer |
| No echo / barge-in arithmetic while the agent is silent; the arithmetic in a worker thread | `agent/echo_cancel.py`, `agent/full_duplex.py` | the echo canceller was burning the event loop on every frame even when there was nothing to cancel |
| Three TTS workers, a lock per language | `tts_server.py` (`TTS_WORKERS`) | one synthesiser per language is single-file |
| English "WhatsApp" spliced from a recorded clip | `agent/whatsapp_tts.py`, `assets/whatsapp_en.wav` | the synthesiser mispronounced the word; the owner supplied the clip. Bengali / Hindi say it in English pronunciation |
| Ollama: 8 parallel requests, 6144 tokens of context each, compact "fast" prompt | `deploy/env.sh` (`OLLAMA_NUM_PARALLEL`, `OLLAMA_NUM_CTX`), `agent/llm.py` | the intent model is only asked on turns the deterministic paths cannot settle |
| Admission control, cap 36, slot freed at hang-up | `agent/admission.py`, `ADMISSION_MAX_CALLS` | beyond the cap a caller is told so (or sent to a person) instead of every call degrading |
| Watchdog relaunches a dead process | `deploy/gateway/watchdog.sh` | OOM / crash recovery without a human |

Deterministic code still decides everything factual (dates, digits, yes/no, prices, doctors, slots); the model only turns a sentence into an intent and slots. That is unchanged from ADR 0001 and is the truth boundary.

## 4. Results of the load tests (documentation, not a guarantee)

Method in section 7. Calls were four recorded Bengali / Hindi sentences played in real time, 24 or 30 at once, each hung up after the first answer.

| Run | Before (one process) | After |
|---|---|---|
| 24 calls, median wait from end of the caller's speech to the agent's answer | about 36 s | about 8.6 to 14 s |
| Server-side turn time (recogniser + intent + reply), p95 | not measured | about 2 to 3 s |
| 30 calls | - | **9 of 30 callers were not heard** (see section 5) |

A single call, with the stack otherwise idle, is much faster; the numbers above are about the worst moment, everybody finishing a sentence together.

## 5. Known problems that are still open

These are written down so nobody thinks they are solved. The code for them is to be changed later by the project owner.

1. **End-of-speech delay under load.** The silence timer and the end-of-sentence detection are measured on the **wall clock**, but under 30 simultaneous calls the audio the agent has *received* lags behind real time: the audio-ingestion loop of the one agent process falls behind (in the not-heard calls the second "resynced to" interval was only 2.3 to 2.7 s against 10 to 12 s in the calls that were heard). The caller's speech is still queued while the 5 s silence timer fires.
2. **Not-heard calls.** The same backlog: the agent says "hello? can you hear me?" to a caller whose words are still in the queue, and the sentence is then processed late (about 14 s). Candidate directions, untried: measure silence in *audio time* (frames received) instead of wall time; move websocket ingestion out of the booking / reply process (one ingestion process per N calls); run two agent processes behind the gateway, each with its own cap.

## 6. Scaling knobs (what to turn first)

* More simultaneous calls on the same machine: raise `INDIC_ASR_WORKERS`, `TTS_WORKERS`, `OLLAMA_NUM_PARALLEL`, then `ADMISSION_MAX_CALLS`, in that order, and watch GPU memory (`nvidia-smi`): each recogniser worker holds the Bengali model, the Hindi model and the language-ID model.
* A bigger GPU (for example 48 GB) mostly buys room for more workers; the GPU was idle in the test, so it does not by itself fix a queue in Python.
* More than one agent process: the admission cap is per process (`main_pcm.py` and `main.py` each count their own calls), so two agents behind the gateway need the gateway to balance between them. Not built.
* All variables and defaults: `docs/DEPLOY_FROM_SCRATCH.md`, section 8.

## 7. How to load-test without putting measuring code in the repo

Write the client **outside** the deployed tree (your laptop, or `/tmp` on the pod) and delete it afterwards.

1. Convert a few recorded sentences to raw 16 kHz mono signed 16-bit PCM (`ffmpeg -i in.m4a -ar 16000 -ac 1 -f s16le out.pcm`).
2. For each simulated call open `ws://127.0.0.1:8101/ws/audio` and send `{"type":"hello","sampleRate":16000,"format":"pcm_s16le","channels":1}`.
3. Wait for the greeting (binary WAV messages plus a text message with `sender: "AI"`); send `{"type":"playback_done"}` when each clip has finished playing (the agent waits for it).
4. Send the recording as 20 ms binary frames (640 bytes) **in real time**, then silence frames, until the answer arrives; stamp every event with the time since that call started.
5. Start all calls from one barrier; record end-of-speech to first AI text and first AI audio per call, then hang up.
6. Read the agent log (`/workspace/logs/pcm_app.log`) for the server-side view. Do not add timers or profilers to the agent: a short-lived sampler was used for the 2026-10-25 investigation and then removed.

## 8. Consequences

* Eight processes to keep alive instead of three; the watchdog and `deploy/status.sh` exist for that.
* Restarting the agents drops live calls. `deploy/safe_restart_agents.sh` waits for a quiet moment; see the deploy guide for its `FORCE=1` caveat.
* GPU memory is now the first limit when workers are added.
