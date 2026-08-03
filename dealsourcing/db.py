"""SQLite schema and connection helpers.

Company-sourcing pipeline (INITIAL exports, one email/day) tables:
  companies  - one row per unique company name (post-dedupe), holds the
               normalized INITIAL export fields plus derived eligibility/
               category columns.
  sent_log   - one row per day a company was emailed out; the presence of
               a row here is what makes "one company per day, never repeat"
               idempotent even if the daily job is re-run.

Builder-sourcing pipeline (pre-founding signal detection, event-driven
alerts - see dealsourcing/builders/) tables:
  builders                  - one row per unique individual detected across
                               sources (GitHub, Qiita, connpass, accelerator
                               alumni pages), deduped by github/qiita
                               username or normalized name.
  builder_signals            - append-only raw signal log (many rows per
                               builder over time); this is what re-scoring
                               triggers off of.
  builder_scoring_log        - one row per builder ever scored/alerted.
                               UNIQUE(builder_id) means each builder is
                               alerted at most once, ever - there is no
                               "one per day" cap here, unlike sent_log,
                               because alerts fire as soon as a builder
                               crosses the threshold, not on a daily cadence.
  accelerator_alumni_history - past accelerator cohort selectees, scraped
                               from each program's public alumni page. Used
                               both as an immediate discovery source and as
                               labeled training data for the lookalike
                               scorer (dealsourcing/builders/lookalike.py).
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

CREATE TABLE IF NOT EXISTS builders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    name_normalized TEXT NOT NULL,
    github_username TEXT,
    qiita_username TEXT,
    current_affiliation TEXT,
    affiliation_type TEXT,
    profile_urls TEXT,
    status TEXT NOT NULL DEFAULT 'new',
    last_scored_at TEXT,
    last_scored_signal_id INTEGER NOT NULL DEFAULT 0,
    discovered_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_builders_github_username
    ON builders (github_username) WHERE github_username IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_builders_qiita_username
    ON builders (qiita_username) WHERE qiita_username IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_builders_name_normalized
    ON builders (name_normalized);

CREATE TABLE IF NOT EXISTS builder_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    builder_id INTEGER NOT NULL REFERENCES builders(id),
    source TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    signal_text TEXT,
    signal_url TEXT,
    detected_at TEXT NOT NULL DEFAULT (datetime('now')),
    raw_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_builder_signals_builder_id
    ON builder_signals (builder_id);

CREATE TABLE IF NOT EXISTS builder_scoring_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    builder_id INTEGER NOT NULL REFERENCES builders(id),
    scored_date TEXT NOT NULL,
    background_score INTEGER,
    lookalike_score INTEGER,
    signal_strength_score INTEGER,
    total_score INTEGER,
    grade TEXT,
    estimated_founding_window TEXT,
    comment TEXT,
    scoring_raw_json TEXT,
    email_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (builder_id)
);

CREATE TABLE IF NOT EXISTS accelerator_alumni_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    program_name TEXT NOT NULL,
    cohort_label TEXT,
    person_name TEXT NOT NULL,
    affiliation_at_selection TEXT,
    one_line_pitch TEXT,
    source_url TEXT,
    scraped_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (program_name, cohort_label, person_name)
);
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
