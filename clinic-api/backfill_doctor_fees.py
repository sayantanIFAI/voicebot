"""One-time, non-destructive backfill for Doctor.consultation_fee_inr on an ALREADY-RUNNING clinic database.

Owner, 2026-10-27: placeholder visit fees (no real fee schedule exists yet) for every doctor currently sitting at
the model's bare default (500), so "how much is Dr X's visit" gives a varied answer instead of the same number for
everyone. seed.py's FEE_LADDER (500-1500 INR) now assigns this automatically for a FRESH database, but seed.py's
own seed() drops and recreates every table -- running it against the live pod would destroy real bookings, patient
history and everything else accumulated there. This script only UPDATEs the one column, only on rows that still
hold the bare default, and touches nothing else.

    python backfill_doctor_fees.py            # apply
    python backfill_doctor_fees.py --dry-run   # show what would change, without writing anything
"""

from __future__ import annotations

import sys

from db import SessionLocal
from models import Doctor
from seed import FEE_LADDER

DEFAULT_FEE = 500  # models.Doctor.consultation_fee_inr's own column default


def backfill(dry_run: bool = False) -> int:
    db = SessionLocal()
    try:
        doctors = db.query(Doctor).order_by(Doctor.id).all()
        changed = 0
        for i, doc in enumerate(doctors):
            if doc.consultation_fee_inr != DEFAULT_FEE:
                continue  # already has a real (or previously-backfilled) fee: never overwritten
            new_fee = FEE_LADDER[i % len(FEE_LADDER)]
            print(f"  {'[dry-run] ' if dry_run else ''}{doc.name} (id={doc.id}): {doc.consultation_fee_inr} -> {new_fee}")
            if not dry_run:
                doc.consultation_fee_inr = new_fee
            changed += 1
        if not dry_run and changed:
            db.commit()
        return changed
    finally:
        db.close()


if __name__ == "__main__":
    n = backfill(dry_run="--dry-run" in sys.argv)
    print(f"{n} doctor(s) {'would be' if '--dry-run' in sys.argv else ''} updated.")
