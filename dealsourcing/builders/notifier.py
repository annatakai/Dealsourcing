"""Builds and sends the immediate builder alert email. Reuses
dealsourcing.email_sender.send_email (the generic SendGrid helper shared
with the company-sourcing pipeline) - this pipeline just builds a
different subject/body and points at settings.builder_email_to instead of
settings.email_to.
"""
from __future__ import annotations

import sqlite3

from dealsourcing.config import settings
from dealsourcing.email_sender import send_email


def build_subject(builder: sqlite3.Row, scoring: dict) -> str:
    grade = scoring.get("grade") or "?"
    return f"[Builder Alert] {builder['name']}（評価{grade}）- 起業シグナル検知"


def build_body(builder: sqlite3.Row, scoring: dict, signals: list[sqlite3.Row]) -> str:
    def score_or_na(value) -> str:
        return str(value) if value is not None else "不明"

    signal_lines = "\n".join(
        f"- [{s['source']}] {s['signal_text']} ({s['signal_url'] or 'URL不明'})" for s in signals
    )

    return (
        f"氏名：{builder['name']}\n"
        f"現在の所属：{builder['current_affiliation'] or '不明'}\n"
        f"GitHub：{builder['github_username'] or '-'}\n"
        f"Qiita：{builder['qiita_username'] or '-'}\n"
        f"\n"
        f"出身企業・現職：{score_or_na(scoring.get('company_score'))} / 30\n"
        f"起業家プログラムとの関わり：{score_or_na(scoring.get('program_score'))} / 25\n"
        f"学歴：{score_or_na(scoring.get('education_score'))} / 25\n"
        f"起業シグナルの強さ：{score_or_na(scoring.get('signal_score'))} / 20\n"
        f"\n"
        f"合計スコア：{score_or_na(scoring.get('total_score'))} / 100\n"
        f"評価：{scoring.get('grade') or '不明'}\n"
        f"起業予測時期：{scoring.get('estimated_founding_window') or '不明'}\n"
        f"\n"
        f"検知シグナル：\n{signal_lines}\n"
        f"\n"
        f"コメント：{scoring.get('comment') or '-'}\n"
        f"アクセラレーター採択者との類似度：{score_or_na(scoring.get('lookalike_score'))} / 100"
        f"（{scoring.get('lookalike_reason') or '-'}）\n"
    )


def send_builder_alert(builder: sqlite3.Row, scoring: dict, signals: list[sqlite3.Row]) -> str:
    subject = build_subject(builder, scoring)
    body = build_body(builder, scoring, signals)
    return send_email(settings.builder_email_to, subject, body)
