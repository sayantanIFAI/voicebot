# Deploying the voice agent from scratch

For a junior engineer who has an empty machine and this repository. Follow it top to bottom. Every command here was taken from the scripts in `deploy/` or from the running pilot pod;
anything that has **not** been tried is marked **UNTESTED** so nobody trusts it by accident.

The architecture this builds is in [`docs/adr/0002-concurrent-calls-architecture.md`](adr/0002-concurrent-calls-architecture.md). Read its section 2 (the process map) first: it shows what each port is.

## 0. What you need

| Item | Reference value (the RunPod pilot pod) | Why |
|---|---|---|
| OS | Ubuntu 24.04, Python 3.12.3 | the lock files in `deploy/locks/` were frozen on it |
| GPU | one NVIDIA GPU, 24 GB or more (pilot: RTX PRO 4500 Blackwell, 32 GB, CUDA 12.8 build of torch) | recognisers, TTS and language ID are all on the one GPU |
| CPU / RAM | 48 cores / 251 GB (pilot) | six recogniser processes, three TTS processes, Ollama |
| Disk | a **persistent** volume of at least 200 GB mounted at `/workspace` | three virtual environments, ~6 GB of model checkpoints, Ollama models, the database |
| Network | outbound internet for the first install; one inbound HTTP port (pilot: 8888) | model downloads; the browser/phone entrance |
| Accounts | a Hugging Face account that has accepted the AI4Bharat IndicConformer licences, and a token | the Bengali and Hindi checkpoints are gated |

**Everything lives under `/workspace`.** On RunPod the container disk (`/`, `/root`, `/usr/local`) is wiped on every restart and only the network volume survives, so no script here installs anything outside `/workspace`.
If your machine is not RunPod you may use any directory, but the scripts hard-code `/workspace` (for example `deploy/env.sh`, `deploy/start_all.sh`), so the simplest way is to make `/workspace` exist: `sudo mkdir -p /workspace && sudo chown $USER /workspace`.

## 1. System packages

```bash
apt-get update
apt-get install -y git curl psmisc nginx openssl zstd build-essential ffmpeg libsndfile1 \
                   python3.12 python3.12-venv python3-pip
```

`psmisc` gives `fuser` (used by `start_all.sh` to free a port), `nginx` and `openssl` are used by the password gateway, `zstd` by the Ollama installer, `ffmpeg` only by the old WebM agent on :8100.

## 2. Get the code

```bash
cd /workspace
git clone https://github.com/sayantanIFAI/voicebot.git kolkata-care-voice-agent
```

The directory name matters: `deploy/env.sh` sets `PYTHONPATH=/workspace/kolkata-care-voice-agent:` and every `_run_*.sh` script `cd`s into it. Push and pull only `sayantanIFAI/voicebot` (branch `main`).

## 3. Three Python environments (they cannot share one)

The Bengali/Hindi recogniser needs the AI4Bharat fork of NeMo, the English recogniser needs mainline NeMo, and the TTS needs a newer `transformers`. Their pins conflict, so each service group has its own environment. The exact versions are the lock files in `deploy/locks/` (taken with `pip freeze` from the working pod, 2026-10-26).

```bash
cd /workspace
python3.12 -m venv venv           # agents (main.py, main_pcm.py), clinic-api, indic_asr_server.py
python3.12 -m venv venv-en-nemo   # english_asr_server.py
python3.12 -m venv tts_venv       # tts_server.py

# torch is a CUDA-12.8 build: install it from the PyTorch index first, then the rest of the lock file
/workspace/venv/bin/pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
/workspace/venv/bin/pip install -r kolkata-care-voice-agent/deploy/locks/requirements-main.lock.txt

/workspace/venv-en-nemo/bin/pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
/workspace/venv-en-nemo/bin/pip install -r kolkata-care-voice-agent/deploy/locks/requirements-english-asr.lock.txt

/workspace/tts_venv/bin/pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
/workspace/tts_venv/bin/pip install -r kolkata-care-voice-agent/deploy/locks/requirements-tts.lock.txt
```

Notes:
* The lock files record `torch==2.11.0+cu128`; pip may reject the `+cu128` suffix from PyPI. If a line fails, install that package from the PyTorch index as above and re-run with `--no-deps` for the rest.
* `requirements-main.lock.txt` contains `nemo_toolkit @ git+https://github.com/AI4Bharat/NeMo.git@8dce88c...` (branch `nemo-v2`): it is built from source, so `git` and `build-essential` must be present and the first install is slow.
* The older `requirements.txt` in the repo root is a historic stub (torch 2.1.0) and is **not** what runs. Use the lock files.
* A from-scratch rebuild from these lock files was **not** re-run end to end when they were written; they are the record of a working pod. If a pin cannot be satisfied, fix that pin and update the lock file in git.

## 4. Models

All of them go under `/workspace`, in the exact places `deploy/env.sh` points to.

| Model | Where it goes | How to get it |
|---|---|---|
| IndicConformer Bengali (`indicconformer_stt_bn_hybrid_rnnt_large.nemo`) | `/workspace/.cache/huggingface/hub/indicconformer_bn/snap/` | Hugging Face `ai4bharat/indicconformer_stt_bn_hybrid_ctc_rnnt_large` (gated; needs `/workspace/.hf_token`) |
| IndicConformer Hindi (`indicconformer_stt_hi_hybrid_rnnt_large.nemo`) | `/workspace/.cache/huggingface/hub/indicconformer_hi/snap/` | `ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large` (same naming as the Bengali one) |
| English FastConformer (`stt_en_fastconformer_hybrid_large_streaming_multi.nemo`) | `/workspace/.cache/huggingface/hub/fastconformer_en/snap/` | NVIDIA NGC / Hugging Face `nvidia/stt_en_fastconformer_hybrid_large_streaming_multi` |
| Language ID (SpeechBrain VoxLingua107 ECAPA, ~86 MB) | `/workspace/lid_model/` (files `classifier.ckpt`, `embedding_model.ckpt`, `hyperparams.yaml`, `label_encoder.ckpt`) | Hugging Face `speechbrain/lang-id-voxlingua107-ecapa` |
| Indic-TTS FastPitch + HiFi-GAN, Bengali, Hindi, English | `/workspace/tts_checkpoints/{bn,hi,en}/{fastpitch,hifigan}/` (each holds `best_model.pth`, `config.json`; FastPitch also `speakers.pth`) | the `v1-checkpoints-release` of `github.com/AI4Bharat/Indic-TTS` (one zip per language, coqui-tts format). The pod also keeps the unzipped originals in `/workspace/tts_checkpoints/models/v1/` |
| Silero VAD | downloaded by `torch.hub` into `/workspace/.cache/torch/hub/snakers4_silero-vad_master` on first use (or clone `snakers4/silero-vad` to `/workspace/silero-vad`, which `SILERO_VAD_REPO` points to) | automatic on first start with internet |
| Ollama + models | binary in `/workspace/bin`, weights in `/workspace/.ollama/models` | `bash deploy/install_ollama.sh`, then `ollama pull qwen2.5:7b` and `ollama pull bge-m3` (`qwen2.5:3b` is also kept on the pilot pod but the agent uses the 7b) |

Put the Hugging Face token in a file, never in git: `echo hf_xxx > /workspace/.hf_token && chmod 600 /workspace/.hf_token` (`env.sh` reads it).

Ollama needs its server running to pull:

```bash
bash /workspace/kolkata-care-voice-agent/deploy/install_ollama.sh
source /workspace/kolkata-care-voice-agent/deploy/env.sh
setsid /workspace/bin/ollama serve > /workspace/logs/ollama.log 2>&1 < /dev/null &
sleep 5 && /workspace/bin/ollama pull qwen2.5:7b && /workspace/bin/ollama pull bge-m3
```

## 5. Secrets and runtime files (all outside git)

| File | Purpose |
|---|---|
| `/workspace/.hf_token` | Hugging Face token, only needed to download gated checkpoints |
| `/workspace/.clinic_api_token` | if present, every `/api/v1/*` call must carry it and the agent sends it (`export CLINIC_API_REQUIRE_TOKEN=1` makes the API refuse everything without one) |
| `/workspace/.admission_token` | enables the HTTP kill switch `POST /api/admission/bypass`; without it the endpoint is off |
| `/workspace/gateway/CREDENTIALS.txt`, `htpasswd` | created by `gateway.sh` on first start: login id `tester` and a random password. Delete both files to rotate |
| Twilio: `TWILIO_AUTH_TOKEN`, `TWILIO_PUBLIC_BASE_URL`, optional `TWILIO_ALLOWED_CALLERS` | phone calls only, see `docs/twilio-telephony.md` |

The clinic database is SQLite at `/workspace/clinic.db` (created and seeded on the first start of clinic-api). It is covered by `*.db` in `.gitignore`.

## 6. Install the gateway and start everything

```bash
mkdir -p /workspace/gateway /workspace/logs
cd /workspace/kolkata-care-voice-agent
cp deploy/gateway/*.sh deploy/gateway/nginx.conf /workspace/gateway/
cp deploy/restart_voicebot.sh /workspace/restart_voicebot.sh
chmod +x /workspace/gateway/*.sh /workspace/restart_voicebot.sh
bash /workspace/restart_voicebot.sh
```

`restart_voicebot.sh` is **the** command after any stop, restart or recreate. It archives the logs, snapshots the database, runs `deploy/bootstrap_pod.sh` (Ollama binary check, then `deploy/start_all.sh`), starts the nginx gateway on :8888 and the watchdog, and prints the public URL and login.
The models take 1 to 3 minutes to load (the agents about 3 minutes). Watch with:

```bash
bash /workspace/kolkata-care-voice-agent/deploy/status.sh
```

Success looks like seven `UP` lines (ollama, tts, clinic-api, voice-agent, voice-pcm, english-asr, indic-asr). On RunPod the public URL is `https://<RUNPOD_POD_ID>-8888.proxy.runpod.net/`.

## 7. Ports and processes

| Port | Process | Venv | Public? | Health URL |
|---|---|---|---|---|
| 8888 | nginx gateway (basic auth) -> the PCM agent | system | **yes** | `/` |
| 8101 | `main_pcm.py` (browser/phone raw-PCM agent, the one to use) | `venv` | via gateway | `/api/health` |
| 8100 | `main.py` (older WebM agent, needs ffmpeg) | `venv` | no | `/api/health` |
| 8004 | `indic_asr_server.py` (Bengali + Hindi recogniser + language ID, 6 workers) | `venv` | no | `/health` |
| 8003 | `english_asr_server.py` | `venv-en-nemo` | no | `/health` |
| 8002 | `tts_server.py` (3 workers) | `tts_venv` | no | `/health` |
| 8080 | clinic API (`clinic-api/main.py`, SQLite) | `venv` | no | `/api/health` |
| 11434 | Ollama | - | no | `/api/tags` |

Mind the health URLs: the agents and the clinic API answer on `/api/health`, the speech services on `/health`. Using the wrong one returns 404 and looks like an outage.

## 8. The settings you may change

All are environment variables read from `deploy/env.sh` (defaults shown) or set in the `_run_*.sh` files that `start_all.sh` writes. After a change: `bash deploy/start_all.sh` only starts what is **down**, so restart the affected service (below).

| Variable | Default | Effect |
|---|---|---|
| `ADMISSION_MAX_CALLS` | 36 | calls accepted at once, per agent process; extra callers are told to wait or sent to a person. Raising it needs the worker counts below raised too |
| `INDIC_ASR_WORKERS` | 6 in `env.sh` (4 if unset) | Bengali/Hindi recogniser processes. Each loads both models plus language ID on the GPU |
| `INDIC_ASR_THREADS` | 4 | CPU threads per recogniser process (`OMP_NUM_THREADS`) |
| `INDIC_ASR_URL` | `http://localhost:8004` | unset it to load the recognisers inside the agent instead (single-call development) |
| `ASR_MAX_BATCH` / `ASR_BATCH_WAIT_S` | 16 / 0.015 | how many waiting utterances one recogniser decodes together |
| `TTS_WORKERS` | 3 | speech-synthesis processes |
| `VAD_POOL` | 6 | speech-detection models per agent process |
| `OLLAMA_NUM_PARALLEL` / `OLLAMA_NUM_CTX` | 8 / 6144 | simultaneous intent requests / context per request. The shared cache is `parallel x ctx`, so raise one only by lowering the other or adding GPU memory |
| `OLLAMA_KEEP_ALIVE` | -1 | keep models loaded (otherwise the first caller after 5 idle minutes waits ~47 s) |
| `AEC_BARGE_IN` | 1 (set in `_run_pcm.sh`) | the caller can talk over the agent |
| `VOICE_AGENT_LANGUAGES` | `bn,hi,en` | `bn` alone is the rollback to Bengali-only |
| `CALL_RECORDING_NOTICE` | 0 | `1` makes the greeting say that the call is being recorded. Set it whenever calls are recorded (the pilot) |
| `PRONUNCIATION_ALLOW_DRAFT` | 1 | lets the not-yet-reviewed pronunciation lexicon be spoken; switch off after the native-listener review, before a real launch |
| `SILENCE_PROMPT_S`, `SILENCE_CLOSE_S`, `SILENCE_HELLOS` | 5, 5, 3 | when the agent says "hello?" to a silent caller and when it hangs up |
| `ENDPOINT_CERTAIN_S`, `ENDPOINT_COMPLETE_S`, `ENDPOINT_INCOMPLETE_S` | 0.45, 1.2, 3.5 | how long a pause must last before the agent takes the caller's sentence as finished |
| `ADMISSION_BYPASS_FILE` | `/workspace/.ai_bypass` | `touch` it: every NEW call goes to a person, no deploy needed |

## 9. Restarting safely

* **After a pod restart:** `bash /workspace/restart_voicebot.sh`.
* **A code change on a running pod:**
  ```bash
  cd /workspace/kolkata-care-voice-agent && git pull origin main
  bash deploy/start_all.sh                       # starts anything that is down
  fuser -k 8080/tcp                              # clinic-api: needed for keyword / database changes, the watchdog relaunches it
  FORCE=1 bash deploy/safe_restart_agents.sh     # agents; the watchdog relaunches them
  ```
  Without `FORCE=1`, `safe_restart_agents.sh` waits up to 240 s for a quiet moment and **refuses** to restart while it thinks a call is active. It counts "call started" minus "call ended" in the log, so stale lines can make it refuse for ever: use `FORCE=1`, then **check that the agent's PID changed**: `ss -ltnp | grep 8101`.
* **Never** kill by command-line pattern over ssh (`pkill -f uvicorn`): the pattern also matches your own ssh command and kills it. Kill by port or PID (`fuser -k 8101/tcp`).
* **The watchdog** (`deploy/gateway/watchdog.sh`) relaunches a dead service by checking its process, every 30 s, and snapshots the database every 6 h. It hands **its own environment** to the services it relaunches: if you change an environment variable, restart the watchdog too, otherwise a relaunched agent keeps the old value (this once kept `ADMISSION_MAX_CALLS=28` alive after `env.sh` said 36).
* Long ssh commands drop. Start long jobs detached (`setsid nohup ... &`) and poll their log.

## 10. Checks after a deploy

1. `bash deploy/status.sh` shows seven `UP` lines.
2. `curl -s localhost:8101/api/health` returns `"ready":true`.
3. Open the public URL, log in, place a test call in each of Bengali, Hindi and English.
4. `tail -f /workspace/logs/pcm_app.log` shows `call started`, the transcripts and `call ended`.

## 11. Making a copy of the system on a different machine

1. Do sections 0 to 6 on the new machine. The only data that is not in git is: the model files (section 4), the secrets (section 5) and `/workspace/clinic.db`.
2. To keep the existing appointments, copy a snapshot from `/workspace/backups/` to the new `/workspace/clinic.db` **before** the first start. Without one, clinic-api creates and seeds a fresh database.
3. `restart_voicebot.sh` prints the new login.

## 12. Google Colab (UNTESTED)

Nobody has run this stack on Colab. What follows is the list of things that differ, so you know what to change; expect to debug.

* Colab gives a 15 GB (T4) GPU and no persistent disk. The full pilot configuration (6 recogniser workers, 3 TTS workers, Qwen2.5-7B) does not fit. Start with `INDIC_ASR_WORKERS=1`, `TTS_WORKERS=1`, `OLLAMA_NUM_PARALLEL=2`, `ADMISSION_MAX_CALLS=3`, and, to save memory, `qwen2.5:3b` (change the constant `OLLAMA_MODEL` in `agent/llm.py`; the intent prompt was tuned on the 7b, so the 3b's answers are **unmeasured**).
* Persistence: mount Google Drive and make `/workspace` a link to a folder on it (`ln -s /content/drive/MyDrive/workspace /workspace`); keep the venvs and models there, or re-install each session (section 3 takes a long time).
* There is no systemd and no inbound port. `setsid ... &` works for background processes. To reach the agent from a browser, run a tunnel to port 8101 (for example `cloudflared tunnel --url http://localhost:8101`); skip nginx and the watchdog, or run the watchdog by hand.
* Browser microphone access needs HTTPS: the tunnel provides it.
* Colab disconnects idle sessions; calls are lost when it does.

## 13. Do not add measuring code to the deployed tree

Load tests, profilers, samplers and report scripts are written and run **outside** this repository (the load-test method is in the ADR, section 7). Nothing that measures or reports belongs in the files the pod runs: it costs CPU on every call and can leak call data. The `/debug/profile` endpoint and `agent/sampler.py` that were used for the 2026-10-25 investigation have been removed for that reason.
