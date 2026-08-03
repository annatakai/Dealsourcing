"""Qiita signal source: search articles matching
config/builder_keywords.yaml's qiita_keywords (resignation/side-project
language) and surface the authors as candidates.

Public read access (https://qiita.com/api/v2/items) works without
QIITA_TOKEN, just at a much lower rate limit (60 req/hour vs 1000/hour
with a token).
"""
from __future__ import annotations

import requests

from dealsourcing.config import load_builder_keywords, settings
from dealsourcing.builders.sources.base import Signal

API_ROOT = "https://qiita.com/api/v2"


def _headers() -> dict:
    headers = {}
    if settings.qiita_token:
        headers["Authorization"] = f"Bearer {settings.qiita_token}"
    return headers


def _search_items(keyword: str, per_page: int = 20) -> list[dict]:
    resp = requests.get(
        f"{API_ROOT}/items",
        params={"query": keyword, "per_page": per_page},
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def get_signals() -> list[Signal]:
    signals: list[Signal] = []
    for keyword in load_builder_keywords().get("qiita_keywords", []):
        try:
            items = _search_items(keyword)
        except requests.RequestException as exc:
            print(f"[qiita_source] search failed for {keyword!r}: {exc}")
            continue

        for item in items:
            user = item.get("user") or {}
            username = user.get("id")
            if not username:
                continue
            signals.append(
                Signal(
                    source="qiita",
                    signal_type="article_published",
                    name=user.get("name") or username,
                    signal_text=f"「{item.get('title')}」を投稿 (検索キーワード: {keyword})",
                    signal_url=item.get("url"),
                    qiita_username=username,
                    affiliation=user.get("organization") or None,
                    raw=item,
                )
            )
    return signals
