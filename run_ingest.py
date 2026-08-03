#!/usr/bin/env python3
"""One-off / re-run step: import INITIAL xlsx exports and (re)classify everything.

Usage:
    python run_ingest.py data/initial_exports/*.xlsx

Run this once against today's INITIAL export(s), and again any time you add
more exports (e.g. new searches) - ingestion is idempotent (upsert by
company name) and classification is recomputed for every row each time.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dealsourcing.classify import classify_all
from dealsourcing.ingest import ingest_files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xlsx_files", nargs="+", type=Path, help="INITIAL epcompanies*.xlsx export(s)")
    args = parser.parse_args()

    missing = [p for p in args.xlsx_files if not p.exists()]
    if missing:
        print(f"File(s) not found: {missing}", file=sys.stderr)
        return 1

    stats = ingest_files(args.xlsx_files)
    print(f"Ingested {stats.files_processed} file(s), {stats.rows_seen} row(s) seen, "
          f"{stats.unique_companies_upserted} unique companies in DB.")

    summary = classify_all()
    print(f"Eligible: {summary['eligible']}, ineligible: {summary['ineligible']}")
    for category, count in sorted(summary["by_category"].items(), key=lambda kv: -kv[1]):
        print(f"  {category}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
