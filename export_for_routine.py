"""Export eligible companies + a fresh sent-log for the cloud-routine bridge.

The cloud routine that does daily research/scoring/emailing runs in an
isolated sandbox with no access to this machine's SQLite DB, so this
script produces the two files it reads instead, which get uploaded to a
private Google Drive folder ("Genesia Dealsourcing") - never committed to
this (public) repo alongside real company/founder data.

Usage:
    python export_for_routine.py [--out-dir DIR] [--limit N] [--min-id N]

Uploading to Drive goes through a chat tool call, so the whole eligible
set (thousands of rows) is too large to move in one go - `--limit` caps
how many rows this run writes (ordered by id) and `--min-id` skips ahead
past a batch that's already been uploaded, so refreshing Drive is a
series of small, cheap top-ups instead of one huge one. The routine only
processes one company/day, so a batch of a few dozen-hundred lasts weeks.

sent_log.csv is only written here if it doesn't already exist locally -
once the routine is live, Drive's copy is the source of truth for what's
already been sent, and this script must never clobber it.
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


def export_candidates(out_path: Path, min_id: int = 0, limit: int | None = None) -> tuple[int, int, int]:
    query = f"""
        SELECT {", ".join(CANDIDATE_COLUMNS)}
        FROM companies
        WHERE is_eligible = 1 AND id > ?
        ORDER BY id ASC
    """
    if limit is not None:
        query += " LIMIT ?"
        params = (min_id, limit)
    else:
        params = (min_id,)

    with connect() as conn:
        rows = conn.execute(query, params).fetchall()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(CANDIDATE_COLUMNS)
        for row in rows:
            writer.writerow([row[col] for col in CANDIDATE_COLUMNS])

    first_id = rows[0]["id"] if rows else None
    last_id = rows[-1]["id"] if rows else None
    return len(rows), first_id, last_id


def ensure_sent_log(out_path: Path) -> bool:
    if out_path.exists():
        return False
    with out_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(SENT_LOG_COLUMNS)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/export", help="Directory to write candidates.csv / sent_log.csv into")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to write, ordered by id (default: all)")
    parser.add_argument("--min-id", type=int, default=0, help="Only include companies with id > this (default: 0)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates_path = out_dir / "candidates.csv"
    sent_log_path = out_dir / "sent_log.csv"

    n, first_id, last_id = export_candidates(candidates_path, min_id=args.min_id, limit=args.limit)
    created = ensure_sent_log(sent_log_path)

    if n:
        print(f"Wrote {n} eligible candidate(s) (id {first_id}-{last_id}) to {candidates_path}")
    else:
        print(f"Wrote 0 eligible candidates to {candidates_path} (no rows with id > {args.min_id})")
    if created:
        print(f"Created empty {sent_log_path} (first run)")
    else:
        print(f"{sent_log_path} already exists locally - left untouched. "
              f"Drive's copy is the source of truth once the routine is live.")


if __name__ == "__main__":
    main()
