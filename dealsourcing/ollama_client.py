"""Local Ollama calls: "ローカルOllamaで経歴を構造化" → "指定の100点ルールで採点".

Two LLM calls per company:
  1. structure_founder_background() turns the raw research blob (search
     snippets, whatever) into a clean bullet list of founder facts across
     the 4 scoring axes (career, programs, education, VC funding), never
     inventing anything not present in the source text.
  2. score_founder() feeds that structured text into the exact scoring
     rubric from config/scoring_prompt.txt and parses the model's reply
     into the fields sent_log needs.

Requires an Ollama server reachable at settings.ollama_host (default
http://localhost:11434) with settings.ollama_model pulled - this only
works from wherever Ollama is actually running, i.e. your own machine.
"""
from __future__ import annotations

import re

import requests

from dealsourcing.config import load_scoring_prompt, settings

STRUCTURE_PROMPT_TEMPLATE = """あなたはベンチャーキャピタルのリサーチアシスタントです。
以下は「{company_name}」という会社と、その創業者「{founder_name}」についてのWeb検索結果です。

この情報から、創業者の経歴に関する事実のみを、以下の4つの観点に整理してください。
必ず与えられた情報のみを使用し、推測は行わないでください。該当する情報が見つからない場合は
「不明」と明記してください。

1. 出身企業（これまで勤務した企業・役職・期間がわかれば記載）
2. 参加した起業家プログラム・アクセラレーター
3. 最終学歴（大学・大学院名、学部があれば記載）
4. VCまたはCVCからの資金調達歴の有無

--- Web検索結果 ---
{raw_research}
--- ここまで ---

上記4項目について、箇条書きで簡潔にまとめてください。"""


def call_ollama(prompt: str) -> str:
    """Generic single-shot Ollama call, reused by both the company- and
    builder-sourcing pipelines - anywhere that needs "structure this text"
    or "score this against a rubric" without inventing its own HTTP call.
    """
    resp = requests.post(
        f"{settings.ollama_host}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=settings.ollama_timeout_seconds,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def structure_founder_background(company_name: str, founder_name: str | None, raw_research: str) -> str:
    prompt = STRUCTURE_PROMPT_TEMPLATE.format(
        company_name=company_name,
        founder_name=founder_name or "不明",
        raw_research=raw_research,
    )
    return call_ollama(prompt)


_SECTION_HEADERS = [
    ("company_score", "company_reason", "出身企業"),
    ("program_score", "program_reason", "起業家プログラム"),
    ("education_score", "education_reason", "学歴"),
    ("funding_score", "funding_reason", "VC資金調達歴"),
]


def _parse_int(text: str | None) -> int | None:
    if not text:
        return None
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def _extract_sections(response_text: str) -> dict:
    result = {}
    # Split the response at each of the 4 known section headers, in order,
    # so each section's body is everything up to the next header (or the
    # 合計スコア line for the last one).
    boundaries = []
    for score_key, reason_key, header in _SECTION_HEADERS:
        m = re.search(re.escape(header), response_text)
        boundaries.append((score_key, reason_key, m.start() if m else None))

    total_match = re.search(r"合計スコア", response_text)
    end_of_sections = total_match.start() if total_match else len(response_text)

    positions = [b[2] for b in boundaries if b[2] is not None] + [end_of_sections]
    for i, (score_key, reason_key, start) in enumerate(boundaries):
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


def _extract_comment(response_text: str) -> str | None:
    # The free-form closing comment is whatever substantial paragraph comes
    # after the grade legend / total score block - take the last non-bullet
    # paragraph of reasonable length as a best-effort heuristic.
    paragraphs = [p.strip() for p in response_text.split("\n\n") if p.strip()]
    for p in reversed(paragraphs):
        if p.startswith(("*", "-", "①", "②", "③", "④")):
            continue
        if "点：" in p or "点満点" in p:
            continue
        if len(p) >= 40:
            return p
    return None


def score_founder(structured_background: str) -> dict:
    prompt = load_scoring_prompt() + "\n\n--- 創業者情報 ---\n" + structured_background
    response_text = call_ollama(prompt)

    sections = _extract_sections(response_text)
    total_score, grade = _extract_total_and_grade(response_text)
    comment = _extract_comment(response_text)

    return {
        **sections,
        "total_score": total_score,
        "grade": grade,
        "comment": comment,
        "raw_response": response_text,
    }
