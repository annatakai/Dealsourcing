"""Turns a builder's accumulated signals into a founder-likelihood score:

  1. Aggregate builder_signals into a single text profile.
  2. Score it against config/builder_scoring_prompt.txt via Ollama (reuses
     dealsourcing.ollama_client.call_ollama - same "never invent facts"
     LLM-rubric pattern as the company pipeline's score_founder()).
  3. Blend in the accelerator lookalike score (lookalike.py) as a bonus -
     it's a fuzzy prior, not one of the rubric's 4 hard axes.

Parsing logic mirrors ollama_client.py's _extract_sections /
_extract_total_and_grade / _extract_comment (same section-header-splitting
approach), extended with an estimated-founding-window field.
"""
from __future__ import annotations

import re
import sqlite3

from dealsourcing.config import load_builder_scoring_prompt
from dealsourcing.ollama_client import call_ollama
from dealsourcing.builders.lookalike import score_lookalike

_SECTION_HEADERS = [
    ("company_score", "company_reason", "出身企業・現職"),
    ("program_score", "program_reason", "起業家プログラムとの関わり"),
    ("education_score", "education_reason", "学歴"),
    ("signal_score", "signal_reason", "起業シグナルの強さ"),
]


def _parse_int(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def _extract_sections(response_text: str) -> dict:
    result = {}
    boundaries = []
    for score_key, reason_key, header in _SECTION_HEADERS:
        m = re.search(re.escape(header), response_text)
        boundaries.append((score_key, reason_key, m.start() if m else None))

    total_match = re.search(r"合計スコア", response_text)
    end_of_sections = total_match.start() if total_match else len(response_text)

    positions = [b[2] for b in boundaries if b[2] is not None] + [end_of_sections]
    for score_key, reason_key, start in boundaries:
        if start is None:
            result[score_key] = None
            result[reason_key] = None
            continue
        next_positions = [p for p in positions if p > start]
        end = min(next_positions) if next_positions else end_of_sections
        section_text = response_text[start:end]

        score_match = re.search(r"スコア[:：]\s*([^\n]+)", section_text)
        reason_match = re.search(r"理由[:：]\s*([^\n]+)", section_text)
        result[score_key] = _parse_int(score_match.group(1)) if score_match else None
        result[reason_key] = reason_match.group(1).strip() if reason_match else None

    return result


def _extract_total_and_grade(response_text: str) -> tuple[int | None, str | None]:
    total_match = re.search(r"合計スコア[:：]?\s*(\d+)", response_text)
    total = int(total_match.group(1)) if total_match else None

    grade = None
    eval_match = re.search(r"評価[^\nSABCD]{0,10}([SABCD])", response_text)
    if eval_match:
        grade = eval_match.group(1)
    return total, grade


def _extract_estimated_window(response_text: str) -> str | None:
    m = re.search(r"起業予測時期\s*\n\s*[*\-]?\s*([^\n]+)", response_text)
    return m.group(1).strip() if m else None


def _extract_comment(response_text: str) -> str | None:
    paragraphs = [p.strip() for p in response_text.split("\n\n") if p.strip()]
    for p in reversed(paragraphs):
        if p.startswith(("*", "-")):
            continue
        if "点：" in p or "点満点" in p or "起業予測時期" in p:
            continue
        if len(p) >= 40:
            return p
    return None


def build_profile_text(builder: sqlite3.Row, signals: list[sqlite3.Row]) -> str:
    lines = [
        f"氏名：{builder['name']}",
        f"現在の所属：{builder['current_affiliation'] or '不明'}",
        "検知シグナル：",
    ]
    for s in signals:
        lines.append(f"- [{s['source']}/{s['signal_type']}] {s['signal_text']} ({s['signal_url'] or 'URL不明'})")
    return "\n".join(lines)


def score_builder(builder: sqlite3.Row, signals: list[sqlite3.Row], db_path=None) -> dict:
    profile_text = build_profile_text(builder, signals)

    prompt = load_builder_scoring_prompt() + "\n\n--- ビルダー情報 ---\n" + profile_text
    response_text = call_ollama(prompt)

    sections = _extract_sections(response_text)
    total_score, grade = _extract_total_and_grade(response_text)
    estimated_window = _extract_estimated_window(response_text)
    comment = _extract_comment(response_text)
    lookalike = score_lookalike(profile_text, db_path=db_path)

    return {
        **sections,
        "total_score": total_score,
        "grade": grade,
        "estimated_founding_window": estimated_window,
        "comment": comment,
        "lookalike_score": lookalike.get("lookalike_score"),
        "lookalike_reason": lookalike.get("reason"),
        "raw_response": response_text,
    }
