#!/usr/bin/env python3
"""The builder-sourcing job: scan all signal sources, score every builder
with new signals, and immediately email any that cross
BUILDER_SCORE_THRESHOLD.

    シグナル収集 → スコアリング → 有望なら即メール

This is a separate, event-driven pipeline from run_daily.py's INITIAL-
export company sourcing - it has no "one per day" cap and no relation to
the companies/sent_log tables. See README.md for the source list and
what's not configured yet.

Idempotent-ish: re-running is always safe (a builder already alerted is
never re-alerted; a builder with no new signals since their last score is
skipped), but unlike run_daily.py this isn't meant to run once and stop -
schedule it to run periodically (e.g. hourly via cron) so new signals get
picked up promptly.
"""
from __future__ import annotations

from dealsourcing.db import init_db
from dealsourcing.builders.pipeline import run


def main() -> int:
    init_db()
    summary = run()
    print(
        f"Done. signals_ingested={summary.get('signals_ingested', 0)} "
        f"builders_touched={summary.get('builders_touched', 0)} "
        f"scored={summary.get('scored', 0)} alerted={summary.get('alerted', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
