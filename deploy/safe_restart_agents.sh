#!/bin/bash
# Restart the two voice agents WITHOUT cutting a live call (a restart on 2026-10-17 ended a caller's call mid-sentence).
# Waits (up to WAIT_S, default 240 s) until no call is active, then restarts; the watchdog relaunches the processes.
#   bash deploy/safe_restart_agents.sh            wait for a quiet moment, then restart
#   FORCE=1 bash deploy/safe_restart_agents.sh    restart now
LOG=/workspace/logs/pcm_app.log
WAIT_S=${WAIT_S:-240}
active() {
    # calls started minus calls ended since the agent last reported "startup complete"
    awk '/startup complete/ {s=0; e=0} /call started/ {s++} /call ended/ {e++} END {print s-e}' "$LOG" 2>/dev/null
}
if [ "${FORCE:-0}" != "1" ]; then
    waited=0
    while [ "$(active)" -gt 0 ] 2>/dev/null && [ "$waited" -lt "$WAIT_S" ]; do
        [ "$waited" -eq 0 ] && echo "a call is in progress -- waiting for it to end (up to ${WAIT_S}s)"
        sleep 5
        waited=$((waited + 5))
    done
    if [ "$(active)" -gt 0 ] 2>/dev/null; then echo "still a call in progress after ${WAIT_S}s -- NOT restarting (FORCE=1 to override)"; exit 2; fi
fi
for p in 8101 8100; do
    pid=$(ss -ltnpH "sport = :$p" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    [ -n "$pid" ] && kill "$pid"
done
echo "agents restarting (the watchdog relaunches them)"
