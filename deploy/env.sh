# Environment for every service in the stack. Source before launching.
#
# Everything referenced here MUST live under /workspace. RunPod wipes the
# container's overlay filesystem ("/" and "/root") on every restart and
# keeps only the network volume, so anything installed to /usr/local/bin
# or stored in /var/lib is gone the next time the pod boots. That is not a
# hypothetical: the ollama binary and the entire Postgres installation
# have each been lost to it three times.

export HF_HOME=/workspace/.cache/huggingface
export TORCH_HOME=/workspace/.cache/torch
export HF_HUB_ENABLE_HF_TRANSFER=0
export PIP_CACHE_DIR=/workspace/.cache/pip

export OLLAMA_MODELS=/workspace/.ollama/models
# -1 keeps models resident indefinitely. Without it Ollama unloads after
# five idle minutes and the next caller pays a 47s cold start mid-call.
export OLLAMA_KEEP_ALIVE=-1

export PYTHONPATH=/workspace/kolkata-care-voice-agent:

# ASR checkpoints -- pilot single-L4 architecture (docs/adr/0001), three
# routed models. Bengali and Hindi are the AI4Bharat NeMo-fork checkpoints
# and load IN-PROCESS in the main venv (same fork, no conflict between
# the two languages -- only English needs mainline NeMo, hence its own
# venv and its own process below). Explicit paths, not the glob
# agent/asr.py's _resolve_nemo_file() used when there was only one
# checkpoint on the box -- with two "*indicconformer*" directories now
# coexisting under HF_HOME, an unqualified glob is ambiguous.
export VOICE_AGENT_NEMO_FILE_BN=/workspace/.cache/huggingface/hub/indicconformer_bn/snap/indicconformer_stt_bn_hybrid_rnnt_large.nemo
export VOICE_AGENT_NEMO_FILE_HI=/workspace/.cache/huggingface/hub/indicconformer_hi/snap/indicconformer_stt_hi_hybrid_rnnt_large.nemo

# English ASR runs as its OWN process in /workspace/venv-en-nemo (mainline
# NVIDIA NeMo -- cannot share a Python environment with the AI4Bharat fork
# above; see agent/asr.py's module docstring for exactly why). Reached
# over HTTP by agent/asr_router.py's English engine adapter.
export VOICE_AGENT_NEMO_FILE_EN=/workspace/.cache/huggingface/hub/fastconformer_en/snap/stt_en_fastconformer_hybrid_large_streaming_multi.nemo
export ENGLISH_ASR_URL=http://localhost:8003

# HF token for the gated IndicConformer bn/hi checkpoints -- read from a
# file, not committed anywhere, never printed. Only needed to (re-)download
# the checkpoints; not needed to load an already-downloaded .nemo file.
if [ -f /workspace/.hf_token ]; then
    export HF_TOKEN
    HF_TOKEN=$(cat /workspace/.hf_token)
fi

# TTS -- all three languages are AI4Bharat Indic-TTS checkpoints (FastPitch
# + HiFi-GAN, coqui-tts format), including English: AI4Bharat's own
# v1-checkpoints-release ships en.zip in the same format as bn/hi, so one
# coqui-tts process serves all three languages -- no second TTS stack
# needed. See tts_server.py's TTS_CHECKPOINTS_ROOT.
export TTS_CHECKPOINTS_ROOT=/workspace/tts_checkpoints

# DATABASE_URL is deliberately NOT set: clinic-api/db.py then defaults to
# sqlite:////workspace/clinic.db, which persists across restarts. Postgres
# physically cannot run on this volume -- see that file's docstring. Set
# this only when pointing at a real external Postgres.
export CLINIC_API_BASE=http://localhost:8080
export TTS_URL=http://localhost:8002/synthesize
export SILERO_VAD_REPO=/workspace/silero-vad

# /workspace/bin first: that is where the persistent ollama binary lives
# (deploy/install_ollama.sh -- NOT the official installer, which ignores
# OLLAMA_INSTALL_DIR and writes to /usr/local, on the ephemeral overlay).
export PATH=/workspace/bin:/workspace/venv/bin:${PATH:-}
