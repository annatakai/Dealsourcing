"""Export eligible companies + a fresh sent-log for the cloud-routine bridge.

The cloud routine that does daily research/scoring/emailing runs in an
isolated sandbox with no access to this machine's SQLite DB, so this
script produces the two files it reads instead, which get uploaded to a
private Google Drive folder ("Genesia Dealsourcing") - never committed to
this (public) repo alongside real company/founder data.

Usage:
    python export_for_routine.py [--out-dir DIR]

Re-run any time after `python run_ingest.py ...` to refresh candidates.csv
(safe to overwrite on Drive). sent_log.csv is only written here if it
doesn't already exist locally - once the routine is live, Drive's copy is
the source of truth for what's already been sent, and this script must
never clobber it.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from dealsourcing.db import connect

CANDIDATE_COLUMNS = [
    "id", "name", "category", "business_description", "representative_name",
    "website", "speeda_funding_series", "total_funding_million_yen",
    "region", "founded_date",
]

SENT_LOG_COLUMNS = [
    "company_id", "sent_date", "category", "total_score", "grade", "email_status",
]


def export_candidates(out_path: Path) -> int:
    with connect() as conn:
        rows = conn.execute(
            f"""
            SELECT {", ".join(CANDIDATE_COLUMNS)}
            FROM companies
            WHERE is_eligible = 1
            ORDER BY id ASC
            """
        ).fetchall()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANDIDATE_COLUMNS)
        for row in rows:
            writer.writerow([row[col] for col in CANDIDATE_COLUMNS])
    return len(rows)


def ensure_sent_log(out_path: Path) -> bool:
    if out_path.exists():
        return False
    with out_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(SENT_LOG_COLUMNS)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/export", help="Directory to write candidates.csv / sent_log.csv into")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = out_dir / "candidates.csv"
    sent_log_path = out_dir / "sent_log.csv"

    n = export_candidates(candidates_path)
    created = ensure_sent_log(sent_log_path)

    print(f"Wrote {n} eligible candidate(s) to {candidates_path}")
    if created:
        print(f"Created empty {sent_log_path} (first run)")
    else:
        print(f"{sent_log_path} already exists locally - left untouched. "
              f"Drive's copy is the source of truth once the routine is live.")


if __name__ == "__main__":
    main()
