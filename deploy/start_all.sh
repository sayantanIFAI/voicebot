#!/bin/bash
# Bring the whole stack up after a pod restart. Idempotent: run it as many
# times as you like.
#
# This exists because RunPod wipes the container overlay on every restart,
# so "the pod is back" and "the service is back" are different events. The
# manual rebuild that used to sit between them was the single largest time
# sink in this project.
#
#   bash /workspace/kolkata-care-voice-agent/deploy/start_all.sh
#
# Every service is launched with setsid + </dev/null so it is fully
# detached from the ssh session. A plain background job dies with the ssh
# channel, which has silently left services down after a deploy more than
# once -- and one memorable time, a pkill pattern matched the launching
# ssh command's own command line and killed the thing doing the launching.
set -u

REPO=/workspace/kolkata-care-voice-agent
source "$REPO/deploy/env.sh"
mkdir -p /workspace/logs /workspace/bin

start() {  # start <name> <port> <logfile> <command...>
    local name=$1 port=$2 log=$3; shift 3
    if curl -sf -m 3 "http://localhost:$port/api/health" >/dev/null 2>&1 \
    || curl -sf -m 3 "http://localhost:$port/health" >/dev/null 2>&1; then
        echo "  $name already up on :$port"
        return
    fi
    # Kill by PORT, never by command-line pattern: a pattern broad enough
    # to match the service is also broad enough to match this script.
    fuser -k "$port/tcp" >/dev/null 2>&1
    sleep 1
    setsid "$@" > "$log" 2>&1 < /dev/null &
    echo "  $name starting on :$port (log: $log)"
}

echo "== ollama =="
if ! pgrep -x ollama >/dev/null; then
    if [ -x /workspace/bin/ollama ]; then
        setsid /workspace/bin/ollama serve > /workspace/logs/ollama.log 2>&1 < /dev/null &
        echo "  ollama starting"
        sleep 5
    else
        echo "  !! /workspace/bin/ollama missing -- run deploy/install_ollama.sh"
    fi
else
    echo "  ollama already running"
fi

echo "== services =="
# tts_server.py now loads bn+hi+en (docs/adr/0001) -- one process, same
# port as before, but back in its OWN venv (/workspace/tts_venv), not the
# main venv. Tried sharing the main venv first; concretely broke, not
# hypothetically: coqui-tts needs transformers>=4.57 + huggingface_hub
# >=0.34, while the AI4Bharat NeMo fork's hf_io_mixin.py imports
# huggingface_hub's ModelFilter, removed in huggingface_hub 0.24 -- there
# is no single version pair that satisfies both in one environment. This
# is exactly the class of conflict the original tts_venv split (and
# docs/adr/0001's "isolate by service, not shared environment" guidance)
# already anticipated.
cat > /workspace/bin/_run_tts.sh <<'EOF'
#!/bin/bash
source /workspace/kolkata-care-voice-agent/deploy/env.sh
cd /workspace/kolkata-care-voice-agent
# several worker processes: one synthesizer per language is single-file (a lock), so 24 callers needing a reply at once queue behind it (load test 2026-10-25)
exec /workspace/tts_venv/bin/python3 -m uvicorn tts_server:app --host ${INTERNAL_BIND_HOST:-127.0.0.1} --port 8002 --workers ${TTS_WORKERS:-3}
EOF
cat > /workspace/bin/_run_clinic.sh <<'EOF'
#!/bin/bash
source /workspace/kolkata-care-voice-agent/deploy/env.sh
cd /workspace/kolkata-care-voice-agent/clinic-api
exec /workspace/venv/bin/python3 -m uvicorn main:app --host ${INTERNAL_BIND_HOST:-127.0.0.1} --port 8080
EOF
cat > /workspace/bin/_run_main.sh <<'EOF'
#!/bin/bash
source /workspace/kolkata-care-voice-agent/deploy/env.sh
cd /workspace/kolkata-care-voice-agent
exec /workspace/venv/bin/python3 -m uvicorn main:app --host 0.0.0.0 --port 8100
EOF
cat > /workspace/bin/_run_pcm.sh <<'EOF'
#!/bin/bash
source /workspace/kolkata-care-voice-agent/deploy/env.sh
cd /workspace/kolkata-care-voice-agent
# Barge-in: the caller can talk over the agent and be heard (agent/echo_cancel.py + agent/barge_in.py, KCD-051/052). In
# production callers interrupt constantly. Thresholds are validated on synthetic echo only -- watch for the agent cutting
# itself off on speakerphone, and set AEC_BARGE_IN=0 here to return to the half-duplex gate if it does.
export AEC_BARGE_IN=${AEC_BARGE_IN:-1}
exec /workspace/venv/bin/python3 -m uvicorn main_pcm:app --host 0.0.0.0 --port 8101
EOF
# English ASR -- its OWN venv (mainline NeMo, cannot share an environment
# with the AI4Bharat fork the main venv uses for bn/hi -- see
# english_asr_server.py's module docstring). Not yet called by the
# orchestrator (main.py/main_pcm.py still only ever route to Bengali --
# that wiring, plus agent/lid.py and the two *_router.py modules, is the
# next concrete step, not done as of docs/adr/0001). Started anyway so it
# can be smoke-tested on its own.
cat > /workspace/bin/_run_english_asr.sh <<'EOF'
#!/bin/bash
source /workspace/kolkata-care-voice-agent/deploy/env.sh
cd /workspace/kolkata-care-voice-agent
exec /workspace/venv-en-nemo/bin/python3 -m uvicorn english_asr_server:app --host ${INTERNAL_BIND_HOST:-127.0.0.1} --port 8003
EOF
cat > /workspace/bin/_run_indic_asr.sh <<'EOF'
#!/bin/bash
source /workspace/kolkata-care-voice-agent/deploy/env.sh
cd /workspace/kolkata-care-voice-agent
# several processes: each loads the Bengali and Hindi models and has its own interpreter lock, so 24 callers finishing together are not one queue (load test 2026-10-25)
# the CPU threads each process may use are limited: 6 recogniser processes on 48 cores, each trying to use all 48, only fight each other
export OMP_NUM_THREADS=${INDIC_ASR_THREADS:-4} MKL_NUM_THREADS=${INDIC_ASR_THREADS:-4}
exec /workspace/venv/bin/python3 -m uvicorn indic_asr_server:app --host ${INTERNAL_BIND_HOST:-127.0.0.1} --port 8004 --workers ${INDIC_ASR_WORKERS:-4}
EOF
chmod +x /workspace/bin/_run_*.sh

start tts         8002 /workspace/logs/tts_server.log   /workspace/bin/_run_tts.sh
start indic-asr   8004 /workspace/logs/indic_asr.log    /workspace/bin/_run_indic_asr.sh
start clinic-api  8080 /workspace/logs/clinic_api.log   /workspace/bin/_run_clinic.sh
# The agents load their language-ID client ONCE at startup. Started before the recogniser service answered, they came up "Bengali only" with no language ID and recognised nothing
# (2026-10-28, twice). So they wait here, up to WAIT_FOR_ASR_S, until the Bengali/Hindi recogniser and the clinic API answer.
wait_for_http() {
    local url=$1 name=$2 waited=0
    until curl -fs -m 3 "$url" >/dev/null 2>&1; do
        [ "$waited" -ge "${WAIT_FOR_ASR_S:-420}" ] && { echo "  WARNING: $name did not answer within ${WAIT_FOR_ASR_S:-420}s -- starting the agents anyway"; return 1; }
        sleep 5; waited=$((waited + 5))
    done
    echo "  $name answers ($waited s)"
}
wait_for_http http://127.0.0.1:8004/health indic-asr
wait_for_http http://127.0.0.1:8080/api/health clinic-api
# Admission control counts calls PER PROCESS, so two media entrypoints share no cap (an external
# review flagged this). ONLY_PCM=1 starts just the PCM entrypoint, which is the one to run when
# real capacity limits matter; the WebM one is the browser-prototype path.
if [ "${ONLY_PCM:-0}" != "1" ]; then
start voice-agent 8100 /workspace/logs/main_app.log     /workspace/bin/_run_main.sh
fi
start voice-pcm   8101 /workspace/logs/pcm_app.log      /workspace/bin/_run_pcm.sh
start english-asr 8003 /workspace/logs/english_asr.log  /workspace/bin/_run_english_asr.sh

echo
echo "Models load for 1-3 minutes. Watch readiness with:"
echo "  bash $REPO/deploy/status.sh"
