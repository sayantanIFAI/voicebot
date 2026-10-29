#!/bin/bash
# PostgreSQL for the clinic API (free, PostgreSQL licence: commercial use allowed). Idempotent: run it again any time.
#
# WHERE THE DATA LIVES. The pod's /workspace is a network volume that cannot hold a Postgres data directory (every file shows as root:root and Postgres refuses it, see clinic-api/db.py) and the
# pod's local disk is wiped when the whole pod restarts. So: the cluster runs on the LOCAL disk (fast, real fsync), and everything needed to rebuild it is kept on /workspace:
#   /workspace/pg_archive   every finished write-ahead-log segment, copied here continuously (and at least every 60 s: archive_timeout)
#   /workspace/pg_base      base backups (backup_postgres.sh, every 6 h from the watchdog) and a plain pg_dump beside each
# restore_postgres.sh rebuilds the cluster from the newest base backup plus the archived log; at most the last ~60 seconds of commits can be missing after a pod restart.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
if ! command -v pg_ctlcluster >/dev/null 2>&1; then
    apt-get update -y >/dev/null
    apt-get install -y postgresql postgresql-client >/dev/null
fi
VER=$(ls /usr/lib/postgresql | sort -V | tail -1)
ARCH=/workspace/pg_archive; BASE=/workspace/pg_base
mkdir -p "$ARCH" "$BASE"; chmod 777 "$ARCH" "$BASE"   # the postgres user must be able to write here; the volume ignores chown
PWFILE=/workspace/.pg_clinic_password
if [ ! -s "$PWFILE" ]; then ( umask 077; head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 24 > "$PWFILE" ); fi
PW=$(cat "$PWFILE")

pg_lsclusters | grep -q "^$VER *main" || pg_createcluster "$VER" main
CONF=/etc/postgresql/$VER/main/conf.d
mkdir -p "$CONF"
cat > "$CONF/clinic.conf" <<EOF
listen_addresses = '127.0.0.1'
port = 5432
max_connections = 100
shared_buffers = 512MB
wal_level = replica
archive_mode = on
archive_command = 'test ! -f $ARCH/%f && cp %p $ARCH/%f'
archive_timeout = 60
synchronous_commit = on
checkpoint_timeout = 10min
log_min_duration_statement = 500
EOF
pg_lsclusters | grep "^$VER *main" | grep -q online || pg_ctlcluster "$VER" main start
for i in $(seq 1 30); do su postgres -c "psql -qAt -c 'select 1'" >/dev/null 2>&1 && break; sleep 1; done
su postgres -c "psql -qAt" <<EOF >/dev/null
DO \$\$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'clinic') THEN CREATE ROLE clinic LOGIN PASSWORD '$PW'; END IF;
END \$\$;
ALTER ROLE clinic PASSWORD '$PW';
EOF
su postgres -c "psql -qAt -c \"select 1 from pg_database where datname='clinic'\"" | grep -q 1 || su postgres -c "createdb -O clinic clinic"
echo "postgresql $VER is up on 127.0.0.1:5432, database 'clinic', role 'clinic' (password in $PWFILE)"
echo "DATABASE_URL=postgresql+psycopg2://clinic:<password>@127.0.0.1:5432/clinic"
