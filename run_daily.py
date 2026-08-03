#!/usr/bin/env python3
"""The daily job: pick 1 company, research, structure, score, email, log.

    毎日1社を選択 → Webリサーチ → ローカルOllamaで構造化 → 採点 → メール → SQLiteに記録

Idempotent: if a company has already been sent today, this exits
immediately without doing anything (safe to re-run, e.g. after a cron
misfire). Intended to be invoked once per day by cron/systemd/a scheduled
GitHub Action - see README.md for scheduling options.
"""
from __future__ import annotations

import sys

from dealsourcing.db import connect
from dealsourcing.email_sender import send_daily_email
from dealsourcing.ollama_client import score_founder, structure_founder_background
from dealsourcing.research import ManualResearchNotFoundError, get_researcher
from dealsourcing.scheduler import already_sent_today, pick_next_company, record_sent


def main() -> int:
    with connect() as conn:
        existing = already_sent_today(conn)
    if existing is not None:
        print(f"Already sent today ({existing['sent_date']}): company_id={existing['company_id']}. Nothing to do.")
        return 0

    company = pick_next_company()
    if company is None:
        print("No eligible, unsent companies remain. Run run_ingest.py against more INITIAL exports.")
        return 0

    print(f"Today's pick: {company['name']} (id={company['id']}, category={company['category']})")

    researcher = get_researcher()
    try:
        raw_research = researcher.research(
            company_name=company["name"],
            founder_name=company["representative_name"],
            website=company["website"],
            company_id=company["id"],
        )
    except ManualResearchNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    structured = structure_founder_background(
        company_name=company["name"],
        founder_name=company["representative_name"],
        raw_research=raw_research,
    )
    scoring = score_founder(structured)

    email_status = send_daily_email(company, company["category"], scoring)

    record_sent(
        company_id=company["id"],
        scoring=scoring,
        research_raw_text=raw_research,
        email_status=email_status,
        category=company["category"],
    )
    print(f"Done. total_score={scoring.get('total_score')} grade={scoring.get('grade')} email_status={email_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
