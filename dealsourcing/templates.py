"""Builds the exact daily email body the user specified."""
from __future__ import annotations

import sqlite3


def _one_line_business(description: str | None) -> str:
    if not description:
        return "不明"
    first_line = description.strip().splitlines()[0].strip()
    return first_line


def _stage_and_funding(company: sqlite3.Row) -> str:
    stage = company["speeda_funding_series"] or "不明"
    funding = company["total_funding_million_yen"]
    funding_str = f"累計{funding}百万円" if funding is not None else "累計調達額不明"
    return f"{stage}・{funding_str}"


def build_subject(company: sqlite3.Row, category: str, scoring: dict) -> str:
    grade = scoring.get("grade") or "?"
    return f"[Dealsourcing] {company['name']}（{category}）- 評価{grade}"


def build_body(company: sqlite3.Row, category: str, scoring: dict) -> str:
    def score_or_na(value) -> str:
        return str(value) if value is not None else "不明"

    return (
        f"企業名：{company['name']}\n"
        f"カテゴリー：{category}\n"
        f"創設者名：{company['representative_name'] or '不明'}\n"
        f"事業内容（一行）：{_one_line_business(company['business_description'])}\n"
        f"Webサイト：{company['website'] or '不明'}\n"
        f"ステージ・累計調達額：{_stage_and_funding(company)}\n"
        f"\n"
        f"出身企業：{score_or_na(scoring.get('company_score'))} / 30\n"
        f"起業家プログラム：{score_or_na(scoring.get('program_score'))} / 25\n"
        f"学歴：{score_or_na(scoring.get('education_score'))} / 25\n"
        f"VC資金調達歴：{score_or_na(scoring.get('funding_score'))} / 20\n"
        f"\n"
        f"合計スコア：{score_or_na(scoring.get('total_score'))} / 100\n"
        f"評価：{scoring.get('grade') or '不明'}\n"
    )
