#!/bin/bash
# A base backup (usable on its own) plus a plain dump of the clinic database, into /workspace/pg_base. Keeps the newest 8 of each and removes archived log older than the oldest kept base backup.
# Run every 6 h by deploy/gateway/watchdog.sh, and by hand before a risky change.
set -uo pipefail
BASE=/workspace/pg_base; ARCH=/workspace/pg_archive
VER=$(ls /usr/lib/postgresql | sort -V | tail -1)
pg_lsclusters | grep "^$VER *main" | grep -q online || { echo "postgres is not running -- no backup"; exit 0; }
TS=$(date +%Y%m%d-%H%M%S)
OUT="$BASE/base-$TS"
mkdir -p "$OUT"; chmod 777 "$OUT"
# -X fetch puts the log needed to make the copy consistent INSIDE the backup, so it does not depend on the archive being complete
su postgres -c "pg_basebackup -D '$OUT' -Ft -z -X fetch -c fast" && echo "base backup: $OUT" || { echo "base backup FAILED"; rm -rf "$OUT"; exit 1; }
su postgres -c "pg_dump -Fc clinic -f '$BASE/clinic-$TS.dump'" && echo "dump: $BASE/clinic-$TS.dump" || echo "dump failed"
# keep the newest 8
ls -1dt "$BASE"/base-* 2>/dev/null | tail -n +9 | xargs -r rm -rf
ls -1t "$BASE"/clinic-*.dump 2>/dev/null | tail -n +9 | xargs -r rm -f
# archived log that no kept base backup needs any more
OLDEST=$(ls -1dt "$BASE"/base-* | tail -1)
[ -n "$OLDEST" ] && find "$ARCH" -type f ! -newer "$OLDEST" -delete 2>/dev/null
exit 0
