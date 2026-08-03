"""SQLite schema and connection helpers.

Two tables:
  companies  - one row per unique company name (post-dedupe), holds the
               normalized INITIAL export fields plus derived eligibility/
               category columns.
  sent_log   - one row per day a company was emailed out; the presence of
               a row here is what makes "one company per day, never repeat"
               idempotent even if the daily job is re-run.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from dealsourcing.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS companies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    business_description TEXT,
    tags TEXT,
    company_type TEXT,
    region TEXT,
    founded_date TEXT,
    representative_name TEXT,
    employee_count INTEGER,
    prefecture TEXT,
    phone TEXT,
    industry TEXT,
    origin TEXT,
    university_or_institution TEXT,
    website TEXT,
    corporate_number TEXT,
    speeda_funding_series TEXT,
    crunchbase_round_type TEXT,
    latest_round_date TEXT,
    total_funding_million_yen INTEGER,
    funding_calc_date TEXT,
    shareholder_status TEXT,
    research_status TEXT,
    is_unlisted INTEGER NOT NULL DEFAULT 0,
    is_eligible INTEGER NOT NULL DEFAULT 0,
    category TEXT,
    source_file TEXT,
    imported_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_companies_eligible_category
    ON companies (is_eligible, category);

CREATE TABLE IF NOT EXISTS sent_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company_id INTEGER NOT NULL REFERENCES companies(id),
    sent_date TEXT NOT NULL,
    category TEXT,
    founder_company_score INTEGER,
    founder_program_score INTEGER,
    founder_education_score INTEGER,
    founder_funding_score INTEGER,
    total_score INTEGER,
    grade TEXT,
    scoring_raw_json TEXT,
    research_raw_text TEXT,
    email_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (company_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_sent_log_one_per_day
    ON sent_log (sent_date);
"""


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | None = None) -> None:
    conn = get_connection(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect(db_path: Path | None = None):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
