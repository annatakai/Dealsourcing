"""Generic web search: SerpAPI or Google CSE, one query in, a list of
{title, snippet, link} out. Shared by the company-sourcing pipeline's
research.py (targeted company/founder queries) and the builder-sourcing
pipeline's builders/sources/websearch_source.py (keyword-driven candidate
discovery) - extracted here so neither module duplicates the HTTP call.
"""
from __future__ import annotations

import requests

from dealsourcing.config import settings

SERPAPI_ENDPOINT = "https://serpapi.com/search"
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


def search(
    query: str,
    max_results: int | None = None,
    provider: str | None = None,
    api_key: str | None = None,
) -> list[dict]:
    provider = provider or settings.search_api_provider
    api_key = api_key or settings.search_api_key
    max_results = max_results or settings.research_max_results
    if not api_key:
        raise ValueError("No search API key configured (SEARCH_API_KEY)")

    if provider == "serpapi":
        resp = requests.get(
            SERPAPI_ENDPOINT,
            params={"q": query, "api_key": api_key, "num": max_results, "hl": "ja", "gl": "jp"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {"title": r.get("title"), "snippet": r.get("snippet"), "link": r.get("link")}
            for r in data.get("organic_results", [])[:max_results]
        ]
    elif provider == "google_cse":
        resp = requests.get(
            GOOGLE_CSE_ENDPOINT,
            params={"q": query, "key": api_key, "cx": settings.google_cse_id, "num": max_results},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {"title": r.get("title"), "snippet": r.get("snippet"), "link": r.get("link")}
            for r in data.get("items", [])[:max_results]
        ]
    else:
        raise ValueError(f"Unknown SEARCH_API_PROVIDER: {provider}")
