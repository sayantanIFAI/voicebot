#!/bin/bash
# Switch the live clinic API from the SQLite file to PostgreSQL. Takes about 3 minutes; the clinic API is down for part of it (run it when no call is in progress).
#   1. the watchdog is stopped (it would relaunch the old API in the middle),
#   2. the clinic API is stopped, so nothing writes while the data is copied,
#   3. a final consistent snapshot of the SQLite file is taken (and kept in /workspace/backups),
#   4. the snapshot is copied into the 'clinic' database; every table's row count must match,
#   5. only then /workspace/.use_postgres is created and the API started on PostgreSQL; on ANY problem it starts on SQLite again (nothing in the SQLite file was changed),
#   6. the watchdog is started again.
# Roll back later by deleting /workspace/.use_postgres and restarting the clinic API: it goes back to the SQLite file (which then lacks what was booked since the switch).
set -u
REPO=/workspace/kolkata-care-voice-agent
PY=/workspace/venv/bin/python3
LOG=/workspace/logs/cutover_postgres.log
say() { echo "[$(date +%T)] $*" | tee -a "$LOG"; }
source "$REPO/deploy/env.sh"
say "cutover starts"
bash "$REPO/deploy/postgres/setup_postgres.sh" >> "$LOG" 2>&1 || { say "postgres is not ready: nothing changed"; exit 1; }
PW=$(cat /workspace/.pg_clinic_password)

say "stopping the watchdog and the clinic API"
[ -s /tmp/vb-watchdog.pid ] && kill "$(cat /tmp/vb-watchdog.pid)" 2>/dev/null; rm -f /tmp/vb-watchdog.pid
CP=$(ss -ltnpH 'sport = :8080' | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
[ -n "$CP" ] && kill "$CP" 2>/dev/null
for i in $(seq 1 30); do ss -ltnH 'sport = :8080' | grep -q . || break; sleep 1; done
for p in $(pgrep -f "uvicorn main:app --host 127.0.0.1 --port 8080"); do kill "$p" 2>/dev/null; done; sleep 2

start_api() { setsid nohup /workspace/bin/_run_clinic.sh >> /workspace/logs/clinic_api.log 2>&1 < /dev/null & }
wait_api() { for i in $(seq 1 90); do curl -s -m 4 localhost:8080/api/health | grep -q doctors && return 0; sleep 3; done; return 1; }
fallback() { say "FALLING BACK to SQLite: $1"; rm -f /workspace/.use_postgres; start_api; wait_api && say "clinic API is back on SQLite"; setsid nohup bash /workspace/gateway/watchdog.sh >> /workspace/logs/watchdog.log 2>&1 < /dev/null & exit 1; }

SNAP=/workspace/backups/clinic-cutover-$(date +%Y%m%d-%H%M%S).db
$PY - "$SNAP" <<'PY' || fallback "snapshot failed"
import sqlite3, sys
src = sqlite3.connect("file:/workspace/clinic.db?mode=ro", uri=True); dst = sqlite3.connect(sys.argv[1]); src.backup(dst); dst.close(); src.close()
PY
say "final snapshot: $SNAP"
OUT=$($PY "$REPO/tools/migrate_sqlite_to_postgres.py" --sqlite "$SNAP" --target "postgresql+psycopg2://clinic:$PW@127.0.0.1:5432/clinic" --replace 2>&1) || fallback "the data copy reported a problem"
echo "$OUT" >> "$LOG"
echo "$OUT" | tail -1 | grep -q "every table matches" || fallback "row counts differ"
say "data copied, every table matches"

touch /workspace/.use_postgres
start_api
wait_api || fallback "the clinic API did not come up on PostgreSQL"
say "clinic API is up on PostgreSQL: $(curl -s -m 5 localhost:8080/api/health)"
setsid nohup bash /workspace/gateway/watchdog.sh >> /workspace/logs/watchdog.log 2>&1 < /dev/null &
say "watchdog restarted; cutover finished"
