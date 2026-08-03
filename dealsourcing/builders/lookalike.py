"""Accelerator "lookalike" scorer: how similar is a builder candidate's
profile to past accelerator selectees (accelerator_alumni_history table -
populated by dealsourcing/builders/dedupe.py as "program_selected" signals
come in via websearch_source.py)?

This repo has no ML/embeddings library (openpyxl/PyYAML/requests/pytest
only, see requirements.txt) - instead this reuses the same "ask the local
LLM" pattern as ollama_client.py's rubric scoring: give it a sample of past
selectees' profiles plus the candidate's known background, ask for a
0-100 similarity score and a one-line reason. Treat the result as a fuzzy
prior that nudges the founder-likelihood score, not a calibrated
probability - the training set (at most a few hundred rows, scraped from
public announcement text) is far too small and noisy for anything more
rigorous. This is the honest, buildable version of "predict who gets
selected before the announcement": pattern-match against who was selected
before, not a true time-series prediction.
"""
from __future__ import annotations

import re
import sqlite3

from dealsourcing.db import connect
from dealsourcing.ollama_client import call_ollama

PROMPT_TEMPLATE = """あなたはベンチャーキャピタルのアナリストです。
以下は過去にアクセラレータープログラムに採択された人物の経歴サンプルです。

--- 過去の採択者サンプル ---
{alumni_sample}
--- ここまで ---

これから評価する候補者の経歴は以下の通りです。

--- 候補者 ---
{candidate_profile}
--- ここまで ---

候補者の経歴が、上記の過去採択者の傾向にどれくらい似ているかを0〜100のスコアで採点してください。
推測で経歴を補わず、与えられた情報のみで判断してください。
以下の形式で出力してください。
類似度スコア：○○
理由：（一文で簡潔に）"""


def _sample_alumni(conn: sqlite3.Connection, limit: int = 30) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT program_name, affiliation_at_selection, one_line_pitch
        FROM accelerator_alumni_history
        ORDER BY RANDOM() LIMIT ?
        """,
        (limit,),
    ).fetchall()


def score_lookalike(candidate_profile: str, db_path=None) -> dict:
    with connect(db_path) as conn:
        alumni = _sample_alumni(conn)

    if not alumni:
        return {"lookalike_score": None, "reason": "過去の採択者データがまだありません"}

    alumni_sample = "\n".join(
        f"- {row['program_name']} / {row['affiliation_at_selection'] or '不明'} / {row['one_line_pitch'] or ''}"
        for row in alumni
    )
    prompt = PROMPT_TEMPLATE.format(alumni_sample=alumni_sample, candidate_profile=candidate_profile)
    response_text = call_ollama(prompt)

    score_match = re.search(r"類似度スコア[:：]\s*(\d+)", response_text)
    reason_match = re.search(r"理由[:：]\s*([^\n]+)", response_text)
    return {
        "lookalike_score": int(score_match.group(1)) if score_match else None,
        "reason": reason_match.group(1).strip() if reason_match else None,
        "raw_response": response_text,
    }
