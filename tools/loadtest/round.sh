#!/bin/bash
LT=$(cd "$(dirname "$0")" && pwd)   # the folder this script lives in (tools/loadtest)
# usage: round.sh LABEL AEC(0|1)   -- restarts the live PCM agent with test settings, warms it, runs the 2-minute ramp, collects everything under /tmp/round_LABEL_*
LABEL=$1; AEC=$2
PY=/workspace/venv/bin/python3
log() { echo "[$(date -u +%H:%M:%S)] $*" >> /tmp/rounds.log; }
P=$(ss -ltnpH 'sport = :8101' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
log "round $LABEL aec=$AEC: stopping agent pid=$P"
kill "$P"; sleep 3
env AEC_BARGE_IN=$AEC ADMISSION_MAX_CALLS=150 ADMISSION_SHED_P95_S=999 INGRESS_SHED_MS=999999 INGRESS_SHED_LAG_MS=999999 setsid nohup /workspace/bin/_run_pcm.sh >> /workspace/logs/pcm_app.log 2>&1 < /dev/null &
sleep 5
NP=$(ss -ltnpH 'sport = :8101' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
for i in $(seq 1 120); do curl -s -m 3 localhost:8101/api/health 2>/dev/null | grep -q '"ready":true' && break; sleep 4; done
NP=$(ss -ltnpH 'sport = :8101' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
GOT=$(tr '\0' '\n' < /proc/$NP/environ | grep -E '^(AEC_BARGE_IN|ADMISSION_MAX_CALLS|ADMISSION_SHED_P95_S)=' | tr '\n' ' ')
log "agent ready pid=$NP env: $GOT"
case "$GOT" in *"AEC_BARGE_IN=$AEC"*"ADMISSION_MAX_CALLS=150"*|*"ADMISSION_MAX_CALLS=150"*"AEC_BARGE_IN=$AEC"*) ;; *) log "ENV MISMATCH -- aborting round"; echo MISMATCH > /tmp/round_${LABEL}_status; exit 1;; esac
# warm-up: a few short calls so models, caches and TTS are hot
$PY $LT/rampload.py /tmp/round_${LABEL}_warm.json 1 6 0 10 g1,d1 >> /tmp/rounds.log 2>&1
sleep 15
S_IST=$(TZ=Asia/Kolkata date +%H:%M:%S)
$PY $LT/sampler.py /tmp/round_${LABEL}_samples.jsonl &
SP=$!
log "round $LABEL ramp starts $S_IST"
$PY $LT/rampload.py /tmp/round_${LABEL}_calls.json 120 10 5 10 >> /tmp/rounds.log 2>&1
kill $SP
E_IST=$(TZ=Asia/Kolkata date +%H:%M:%S)
echo "$S_IST $E_IST" > /tmp/round_${LABEL}_window.txt
grep "turn timing" /workspace/logs/pcm_app.log | awk -v s="$S_IST" -v e="$E_IST" '{split($2,a,","); if (a[1] >= s && a[1] <= e) print}' > /tmp/round_${LABEL}_timing.log
log "round $LABEL finished $S_IST - $E_IST ($(wc -l < /tmp/round_${LABEL}_timing.log) timing lines)"
echo DONE > /tmp/round_${LABEL}_status
