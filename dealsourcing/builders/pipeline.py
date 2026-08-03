"""Orchestrates one run of the builder-sourcing pipeline:

    全ソースをスキャン → シグナルをdedupe/記録
    → 新シグナルのあるビルダーを再採点
    → 閾値以上かつ未通知ならメールで即通知 → builder_scoring_logに記録

Unlike run_daily.py's company pipeline, this is NOT capped at one email
per day - alerts are event-driven, since the point is "有望なら即メール"
(contact the moment someone looks promising), not a fixed daily cadence.
A builder can be re-scored multiple times as new signals arrive (their
score can only go up as more evidence accumulates), but is never alerted
more than once - see _already_alerted() below.

Intended to be run periodically (e.g. hourly/daily via cron) - see
run_builder_scan.py.
"""
from __future__ import annotations

import datetime
import json
import sqlite3

from dealsourcing.config import settings
from dealsourcing.db import connect
from dealsourcing.builders.dedupe import ingest_signals
from dealsourcing.builders.notifier import send_builder_alert
from dealsourcing.builders.scoring import score_builder
from dealsourcing.builders.sources import ALL_SOURCES


def scan_all_sources() -> dict:
    all_signals = []
    for module in ALL_SOURCES:
        try:
            signals = module.get_signals()
        except Exception as exc:
            print(f"[pipeline] source {module.__name__} failed: {exc}")
            continue
        print(f"[pipeline] {module.__name__}: {len(signals)} signal(s)")
        all_signals.extend(signals)

    return ingest_signals(all_signals)


def _builders_with_new_signals(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    # Compares against builder_signals.id (monotonic, autoincrement)
    # rather than a timestamp - timestamps from SQLite's datetime('now')
    # only have 1-second resolution, so a builder scored and re-signaled
    # within the same second would be silently skipped by a ">" timestamp
    # comparison. last_scored_signal_id has no such tie.
    return conn.execute(
        """
        SELECT b.* FROM builders b
        WHERE EXISTS (
            SELECT 1 FROM builder_signals s
            WHERE s.builder_id = b.id
              AND s.id > b.last_scored_signal_id
        )
        """
    ).fetchall()


def _max_signal_id(conn: sqlite3.Connection, builder_id: int) -> int:
    row = conn.execute(
        "SELECT MAX(id) AS max_id FROM builder_signals WHERE builder_id = ?", (builder_id,)
    ).fetchone()
    return row["max_id"] or 0


def _signals_for(conn: sqlite3.Connection, builder_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM builder_signals WHERE builder_id = ? ORDER BY detected_at", (builder_id,)
    ).fetchall()


def _prior_email_status(conn: sqlite3.Connection, builder_id: int) -> str | None:
    row = conn.execute(
        "SELECT email_status FROM builder_scoring_log WHERE builder_id = ?", (builder_id,)
    ).fetchone()
    return row["email_status"] if row else None


def score_and_notify(db_path=None) -> dict:
    """Re-score every builder with unscored new signals; alert immediately
    on anyone crossing settings.builder_score_threshold for the first time."""
    summary = {"scored": 0, "alerted": 0}

    with connect(db_path) as conn:
        candidates = _builders_with_new_signals(conn)

    for builder in candidates:
        with connect(db_path) as conn:
            signals = _signals_for(conn, builder["id"])
            prior_status = _prior_email_status(conn, builder["id"])
        already_alerted = prior_status in ("sent", "dry_run")

        scoring = score_builder(builder, signals, db_path=db_path)
        summary["scored"] += 1
        total_score = scoring.get("total_score") or 0

        email_status = prior_status if already_alerted else "not_sent"
        if not already_alerted and total_score >= settings.builder_score_threshold:
            email_status = send_builder_alert(builder, scoring, signals)
            summary["alerted"] += 1
            print(f"[pipeline] ALERT sent for {builder['name']} (score={total_score})")
        else:
            print(
                f"[pipeline] {builder['name']} scored {total_score} "
                f"(below threshold {settings.builder_score_threshold}), no alert"
            )

        with connect(db_path) as conn:
            max_signal_id = _max_signal_id(conn, builder["id"])
            conn.execute(
                """
                UPDATE builders SET
                    last_scored_at = datetime('now'),
                    last_scored_signal_id = ?,
                    status = ?,
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (max_signal_id, "sent" if email_status in ("sent", "dry_run") else "scored", builder["id"]),
            )
            conn.execute(
                """
                INSERT INTO builder_scoring_log (
                    builder_id, scored_date, background_score, lookalike_score,
                    signal_strength_score, total_score, grade,
                    estimated_founding_window, comment, scoring_raw_json, email_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(builder_id) DO UPDATE SET
                    scored_date = excluded.scored_date,
                    background_score = excluded.background_score,
                    lookalike_score = excluded.lookalike_score,
                    signal_strength_score = excluded.signal_strength_score,
                    total_score = excluded.total_score,
                    grade = excluded.grade,
                    estimated_founding_window = excluded.estimated_founding_window,
                    comment = excluded.comment,
                    scoring_raw_json = excluded.scoring_raw_json,
                    email_status = excluded.email_status
                """,
                (
                    builder["id"],
                    datetime.date.today().isoformat(),
                    scoring.get("company_score"),
                    scoring.get("lookalike_score"),
                    scoring.get("signal_score"),
                    total_score,
                    scoring.get("grade"),
                    scoring.get("estimated_founding_window"),
                    scoring.get("comment"),
                    json.dumps(scoring, ensure_ascii=False, default=str),
                    email_status,
                ),
            )

    return summary


def run(db_path=None) -> dict:
    scan_summary = scan_all_sources()
    score_summary = score_and_notify(db_path=db_path)
    return {**scan_summary, **score_summary}
