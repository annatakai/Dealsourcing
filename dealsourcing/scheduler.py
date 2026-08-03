"""Daily selection + idempotent send-tracking ("毎日1社を選択" / "送信済みとしてSQLiteに記録").

Selection order is deliberately simple and reproducible: the lowest `id`
among eligible companies that have never appeared in sent_log. `id` order
follows ingestion order, which is stable across re-runs of run_ingest.py
(existing rows keep their id; only genuinely new companies get new,
higher ids). Swap this for a category round-robin or a score-based order
later if you want a different daily sequence - this function is the only
place that decision is made.

sent_log.company_id and sent_log.sent_date both carry UNIQUE constraints
(see db.py), so a company can only ever be sent once, and at most one
company can be sent on a given calendar date - re-running the daily job
on a date that already has an entry is a safe no-op.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

from dealsourcing.db import connect

def _today_iso() -> str:
    return datetime.date.today().isoformat()


def already_sent_today(conn: sqlite3.Connection, sent_date: str | None = None) -> sqlite3.Row | None:
    sent_date = sent_date or _today_iso()
    return conn.execute(
        "SELECT * FROM sent_log WHERE sent_date = ?", (sent_date,)
    ).fetchone()


def pick_next_company(db_path: Path | None = None) -> sqlite3.Row | None:
    """Return the next eligible, never-sent company, or None if exhausted."""
    with connect(db_path) as conn:
        return conn.execute(
            """
            SELECT c.* FROM companies c
            WHERE c.is_eligible = 1
              AND c.id NOT IN (SELECT company_id FROM sent_log)
            ORDER BY c.id ASC
            LIMIT 1
            """
        ).fetchone()


def record_sent(
    company_id: int,
    scoring: dict,
    research_raw_text: str,
    email_status: str,
    category: str | None = None,
    sent_date: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Persist a sent_log row. `scoring` is the dict produced by dealsourcing.ollama_client.score_founder."""
    sent_date = sent_date or _today_iso()
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO sent_log (
                company_id, sent_date, category,
                founder_company_score, founder_program_score,
                founder_education_score, founder_funding_score,
                total_score, grade, scoring_raw_json, research_raw_text,
                email_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                company_id,
                sent_date,
                category,
                scoring.get("company_score"),
                scoring.get("program_score"),
                scoring.get("education_score"),
                scoring.get("funding_score"),
                scoring.get("total_score"),
                scoring.get("grade"),
                json.dumps(scoring, ensure_ascii=False),
                research_raw_text,
                email_status,
            ),
        )
