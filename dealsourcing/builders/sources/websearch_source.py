"""Web-search signal source - the broadest-net source, and the only one
that reaches consultant/business-background builders (GitHub/Qiita skew
heavily toward engineers). Runs two query sets through SerpAPI/Google CSE
(dealsourcing/websearch.py, shared with the company pipeline's
research.py):

  1. config/builder_keywords.yaml's general_founder_prep_queries - direct
     language ("退職しました", "起業します") plus site:-scoped queries for
     Wantedly and researchmap, neither of which has a usable public search
     API (see README) but both of which have public, Google-indexed
     profile pages.
  2. One query per config/accelerator_programs.yaml entry, since alumni
     pages turned out too inconsistent to hand-parse (1stRound has a
     dedicated page; Incubate Camp/Onlab announcements are scattered
     across PR TIMES/TechCrunch Japan/thebridge.jp articles; MUFG mixes
     HTML and PDF across cohorts) - searching for the announcement instead
     sidesteps needing a bespoke parser per program.

Raw search snippets are unstructured, so each query's results are handed
to the local Ollama model to pull out person names/affiliations - the same
"structure messy text, never invent facts" approach ollama_client.py
already uses for founder research in the company pipeline.
"""
from __future__ import annotations

import json
import re

from dealsourcing.config import load_accelerator_programs, load_builder_keywords, settings
from dealsourcing.ollama_client import call_ollama
from dealsourcing.websearch import search
from dealsourcing.builders.sources.base import Signal

EXTRACTION_PROMPT_TEMPLATE = """以下はWeb検索結果です。この中から、起業準備中と思われる人物を抽出してください。
必ず与えられた情報のみを使用し、推測は行わないでください。該当する人物がいない場合は空のリストを返してください。

--- 検索結果 ---
{raw_results}
--- ここまで ---

見つかった人物ごとに、以下のJSON配列の形式で出力してください。他のテキストは含めないでください。
[
  {{"name": "氏名", "affiliation": "所属企業(不明ならnull)", "context": "検知した文脈を一文で", "source_url": "根拠となったURL"}}
]"""


def _extract_people(raw_results: str) -> list[dict]:
    prompt = EXTRACTION_PROMPT_TEMPLATE.format(raw_results=raw_results)
    response_text = call_ollama(prompt)
    match = re.search(r"\[.*\]", response_text, re.DOTALL)
    if not match:
        return []
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return []


def _run_query(query: str, signal_type: str, extra_raw: dict | None = None) -> list[Signal]:
    try:
        results = search(query)
    except Exception as exc:
        print(f"[websearch_source] search failed for {query!r}: {exc}")
        return []
    if not results:
        return []

    raw_blocks = [f"- {r['title']}\n  {r['snippet']}\n  ({r['link']})" for r in results if r.get("snippet")]
    if not raw_blocks:
        return []

    people = _extract_people("\n".join(raw_blocks))
    signals = []
    for person in people:
        if not person.get("name"):
            continue
        raw = {"query": query, "extracted": person}
        if extra_raw:
            raw.update(extra_raw)
        signals.append(
            Signal(
                source="websearch",
                signal_type=signal_type,
                name=person["name"],
                signal_text=person.get("context") or query,
                signal_url=person.get("source_url"),
                affiliation=person.get("affiliation"),
                raw=raw,
            )
        )
    return signals


def get_signals() -> list[Signal]:
    if not settings.search_api_key:
        print("[websearch_source] SEARCH_API_KEY not set - skipping.")
        return []

    signals: list[Signal] = []
    keywords = load_builder_keywords()

    for query in keywords.get("general_founder_prep_queries", []):
        signals.extend(_run_query(query, "founder_statement"))

    for program in load_accelerator_programs():
        query = f"{program['name']} 採択 発表"
        signals.extend(_run_query(query, "program_selected", extra_raw={"program_name": program["name"]}))

    return signals
