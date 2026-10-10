#!/bin/bash
# Consistent snapshot of the clinic database into /workspace/backups. Never deletes an older snapshot.
DB=/workspace/clinic.db
[ -f "$DB" ] || { echo "no $DB yet"; exit 0; }
OUT=/workspace/backups/clinic-$(date +%Y%m%d-%H%M%S).db
/workspace/venv/bin/python3 - "$DB" "$OUT" <<'PY'
import sqlite3, sys
src = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
dst = sqlite3.connect(sys.argv[2])
src.backup(dst); dst.close(); src.close()
print("backup written:", sys.argv[2])
PY
