#!/bin/bash
# After a pod restart the local disk is empty: rebuild the cluster from the newest base backup in /workspace/pg_base and replay the archived log in /workspace/pg_archive.
# Does nothing when a cluster with data already exists (so it is safe to call at every boot). deploy/bootstrap_pod.sh calls it before setup_postgres.sh.
set -euo pipefail
BASE=/workspace/pg_base; ARCH=/workspace/pg_archive
command -v pg_ctlcluster >/dev/null 2>&1 || { echo "postgresql is not installed: run setup_postgres.sh"; exit 0; }
VER=$(ls /usr/lib/postgresql | sort -V | tail -1)
DATA=/var/lib/postgresql/$VER/main
if [ -s "$DATA/PG_VERSION" ]; then echo "a cluster already exists in $DATA: nothing to restore"; exit 0; fi
LATEST=$(ls -1dt "$BASE"/base-* 2>/dev/null | head -1 || true)
[ -n "$LATEST" ] || { echo "no base backup in $BASE: setup_postgres.sh will create an empty cluster"; exit 0; }
echo "restoring from $LATEST"
pg_lsclusters | grep -q "^$VER *main" && pg_dropcluster --stop "$VER" main || true
pg_createcluster "$VER" main --start-conf=manual
rm -rf "${DATA:?}"/*
tar -xzf "$LATEST/base.tar.gz" -C "$DATA"
[ -f "$LATEST/pg_wal.tar.gz" ] && { mkdir -p "$DATA/pg_wal"; tar -xzf "$LATEST/pg_wal.tar.gz" -C "$DATA/pg_wal"; }
chown -R postgres:postgres "$DATA"; chmod 700 "$DATA"
touch "$DATA/recovery.signal"
cat >> "/etc/postgresql/$VER/main/conf.d/restore.conf" <<EOF
restore_command = 'cp $ARCH/%f %p'
EOF
pg_ctlcluster "$VER" main start
for i in $(seq 1 120); do su postgres -c "psql -qAt -c 'select pg_is_in_recovery()'" 2>/dev/null | grep -q '^f$' && break; sleep 1; done
rm -f "/etc/postgresql/$VER/main/conf.d/restore.conf"
echo "restored; recovery finished"
