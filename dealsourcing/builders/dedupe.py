"""Match a raw Signal to an existing builders row, or create a new one,
and append it to builder_signals.

Matching precedence: github_username > qiita_username > normalized name.
Name-only matching is a best-effort heuristic - there's no shared unique
ID across sources for a person known only by name, so two different
people who share a common Japanese name could get merged. Acceptable for
a lead-scoring tool where a human reviews every alert before contacting
anyone; revisit if false-merges show up in practice.

Also writes "program_selected" signals into accelerator_alumni_history -
that table doubles as training data for the lookalike scorer
(dealsourcing/builders/lookalike.py), so every selection signal we detect
also grows the sample we compare future candidates against.
"""
from __future__ import annotations

import json
import re
import sqlite3

from dealsourcing.db import connect
from dealsourcing.builders.sources.base import Signal


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", "", name).lower()


def find_or_create_builder(conn: sqlite3.Connection, signal: Signal) -> int:
    row = None
    if signal.github_username:
        row = conn.execute(
            "SELECT id FROM builders WHERE github_username = ?", (signal.github_username,)
        ).fetchone()
    if row is None and signal.qiita_username:
        row = conn.execute(
            "SELECT id FROM builders WHERE qiita_username = ?", (signal.qiita_username,)
        ).fetchone()
    if row is None and signal.name:
        row = conn.execute(
            "SELECT id FROM builders WHERE name_normalized = ?", (normalize_name(signal.name),)
        ).fetchone()

    if row is not None:
        builder_id = row["id"]
        conn.execute(
            """
            UPDATE builders SET
                github_username = COALESCE(github_username, ?),
                qiita_username = COALESCE(qiita_username, ?),
                current_affiliation = COALESCE(current_affiliation, ?),
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (signal.github_username, signal.qiita_username, signal.affiliation, builder_id),
        )
        return builder_id

    cur = conn.execute(
        """
        INSERT INTO builders (name, name_normalized, github_username, qiita_username, current_affiliation)
        VALUES (?, ?, ?, ?, ?)
        """,
        (signal.name, normalize_name(signal.name), signal.github_username, signal.qiita_username, signal.affiliation),
    )
    return cur.lastrowid


def record_signal(conn: sqlite3.Connection, builder_id: int, signal: Signal) -> None:
    conn.execute(
        """
        INSERT INTO builder_signals (builder_id, source, signal_type, signal_text, signal_url, raw_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            builder_id,
            signal.source,
            signal.signal_type,
            signal.signal_text,
            signal.signal_url,
            json.dumps(signal.raw, ensure_ascii=False, default=str) if signal.raw else None,
        ),
    )


def _maybe_record_alumni_history(conn: sqlite3.Connection, signal: Signal) -> None:
    if signal.signal_type != "program_selected" or not signal.raw:
        return
    program_name = signal.raw.get("program_name")
    if not program_name:
        return
    conn.execute(
        """
        INSERT OR IGNORE INTO accelerator_alumni_history
            (program_name, cohort_label, person_name, affiliation_at_selection, one_line_pitch, source_url)
        VALUES (?, NULL, ?, ?, ?, ?)
        """,
        (program_name, signal.name, signal.affiliation, signal.signal_text, signal.signal_url),
    )


def ingest_signals(signals: list[Signal], db_path=None) -> dict:
    """Upsert every signal into builders/builder_signals (and
    accelerator_alumni_history where applicable). Returns a summary dict."""
    signals_ingested = 0
    builders_touched: set[int] = set()

    with connect(db_path) as conn:
        for signal in signals:
            if not signal.name:
                continue
            builder_id = find_or_create_builder(conn, signal)
            record_signal(conn, builder_id, signal)
            _maybe_record_alumni_history(conn, signal)
            signals_ingested += 1
            builders_touched.add(builder_id)

    return {"signals_ingested": signals_ingested, "builders_touched": len(builders_touched)}
