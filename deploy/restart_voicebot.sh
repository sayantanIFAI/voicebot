#!/bin/bash
# THE one command after any pod stop/restart/recreate (run it over SSH; it detaches everything it starts):
#     bash /workspace/restart_voicebot.sh
# Nothing here deletes anything: logs are COPIED to logs/archive/ before the service scripts overwrite them,
# the DB is snapshotted to /workspace/backups/, and every model/venv/db already lives on /workspace.
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p /workspace/logs/archive/$TS /workspace/backups
for f in ollama tts_server clinic_api main_app pcm_app english_asr; do
    [ -s /workspace/logs/$f.log ] && cp -p /workspace/logs/$f.log /workspace/logs/archive/$TS/ 2>/dev/null
done
bash /workspace/gateway/backup_db.sh
bash /workspace/kolkata-care-voice-agent/deploy/bootstrap_pod.sh
bash /workspace/gateway/gateway.sh start
setsid nohup bash /workspace/gateway/watchdog.sh >> /workspace/logs/watchdog.log 2>&1 < /dev/null &
echo
echo "Public URL:  https://$(tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p')-8888.proxy.runpod.net/"
cat /workspace/gateway/CREDENTIALS.txt
