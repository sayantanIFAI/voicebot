#!/bin/bash
# Relaunch any voice-agent service whose PROCESS has died (OOM, crash). Checks the process, never the health URL:
# a service that is still loading its models has no open port yet, and relaunching it would load the models twice.
# Logs are appended (>>), never truncated. Also snapshots the DB every 6 h. Run via restart_voicebot.sh.
PIDF=/tmp/vb-watchdog.pid
if [ -s $PIDF ] && kill -0 "$(cat $PIDF)" 2>/dev/null; then echo "watchdog already running"; exit 0; fi
echo $$ > $PIDF
source /workspace/kolkata-care-voice-agent/deploy/env.sh
L=/workspace/logs
relaunch() {  # relaunch <name> <pgrep-pattern> <command...>
    local name=$1 pat=$2; shift 2
    if ! pgrep -f -- "$pat" >/dev/null; then
        echo "$(date -Is) watchdog: $name is not running -- relaunching" >> $L/watchdog.log
        setsid nohup "$@" >> "$L/$name.log" 2>&1 < /dev/null &
    fi
}
sleep 180   # grace: let the first start finish loading
last_backup=0
last_archive=""
while true; do
    relaunch ollama     "ollama serve"                                       /workspace/bin/ollama serve
    relaunch tts_server "uvicorn tts_server"                                 /workspace/bin/_run_tts.sh
    relaunch clinic_api "uvicorn main:app --host 127.0.0.1 --port 8080"      /workspace/bin/_run_clinic.sh
    relaunch main_app   "uvicorn main:app --host 0.0.0.0 --port 8100"        /workspace/bin/_run_main.sh
    relaunch pcm_app    "uvicorn main_pcm:app"                               /workspace/bin/_run_pcm.sh
    relaunch english_asr "uvicorn english_asr_server"                        /workspace/bin/_run_english_asr.sh
    relaunch indic_asr   "uvicorn indic_asr_server"                          /workspace/bin/_run_indic_asr.sh
    if ! { [ -s /tmp/voicebot-gw.pid ] && kill -0 "$(cat /tmp/voicebot-gw.pid)" 2>/dev/null; }; then
        echo "$(date -Is) watchdog: gateway down -- restarting" >> $L/watchdog.log
        bash /workspace/gateway/gateway.sh start >> $L/watchdog.log 2>&1
    fi
    now=$(date +%s)
    if [ -f /workspace/.use_postgres ]; then
        pg_lsclusters 2>/dev/null | grep -q online || { echo "$(date -Is) watchdog: postgresql is not running -- starting" >> $L/watchdog.log; pg_ctlcluster "$(ls /usr/lib/postgresql | sort -V | tail -1)" main start >> $L/watchdog.log 2>&1; }
    fi
    if [ $((now - last_backup)) -ge 21600 ]; then
        bash /workspace/gateway/backup_db.sh >> $L/watchdog.log 2>&1
        [ -f /workspace/.use_postgres ] && bash /workspace/kolkata-care-voice-agent/deploy/postgres/backup_postgres.sh >> $L/watchdog.log 2>&1
        last_backup=$now
    fi
    # 01:00 clinic time, once a day: load yesterday's call journals into the database, write the CSV, compress and purge (tools/call_log_archiver.py; one process, one call at a time)
    day=$(TZ=Asia/Kolkata date +%F); hour=$(TZ=Asia/Kolkata date +%H)
    if [ "$hour" = "01" ] && [ "$last_archive" != "$day" ]; then
        last_archive=$day
        echo "$(date -Is) watchdog: nightly call-log archive starts" >> $L/watchdog.log
        ( cd /workspace/kolkata-care-voice-agent && /workspace/venv/bin/python3 tools/call_log_archiver.py >> $L/call_log_archiver.log 2>&1 ) &
    fi
    sleep 30
done
