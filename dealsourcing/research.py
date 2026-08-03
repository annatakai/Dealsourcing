"""Web research step: "会社名・代表者名を起点にWebリサーチ".

Produces one raw text blob (search snippets + source URLs) per company,
which ollama_client.py then structures into the founder-background fields
the scoring rubric needs. Kept deliberately dumb (no LLM calls here) so
this module can be swapped independently of how structuring/scoring works.

Two implementations are provided:

  SearchApiResearcher - issues a handful of targeted queries via a search
      API (SerpAPI or Google Programmable Search / CSE) and concatenates
      the result snippets. This is the default when SEARCH_API_KEY is set.

  ManualResearcher - fallback used when no search API key is configured.
      It looks for data/manual_research/<company_id>.txt and expects you
      (or whoever runs the job) to have pasted research findings there
      ahead of time; it raises a clear, actionable error if the file is
      missing rather than silently sending an empty/garbage email.

No API key has been provided yet for this project, so until SEARCH_API_KEY
(and SEARCH_API_PROVIDER) are set in the environment, the pipeline runs in
ManualResearcher mode.
"""
from __future__ import annotations

import abc
from pathlib import Path

import requests

from dealsourcing.config import settings

SERPAPI_ENDPOINT = "https://serpapi.com/search"
GOOGLE_CSE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


class Researcher(abc.ABC):
    @abc.abstractmethod
    def research(
        self,
        company_name: str,
        founder_name: str | None,
        website: str | None,
        company_id: int | None = None,
    ) -> str:
        """Return a raw text blob of findings for the given company/founder."""


class ManualResearchNotFoundError(RuntimeError):
    pass


class ManualResearcher(Researcher):
    """Reads pre-collected research from data/manual_research/<company_id>.txt."""

    def __init__(self, research_dir: Path | None = None):
        self.research_dir = research_dir or (Path(__file__).resolve().parent.parent / "data" / "manual_research")

    def path_for(self, company_id: int) -> Path:
        return self.research_dir / f"{company_id}.txt"

    def research(
        self,
        company_name: str,
        founder_name: str | None,
        website: str | None,
        company_id: int | None = None,
    ) -> str:
        if company_id is None:
            raise ValueError("ManualResearcher requires company_id")
        path = self.path_for(company_id)
        if not path.exists():
            self.research_dir.mkdir(parents=True, exist_ok=True)
            raise ManualResearchNotFoundError(
                f"No SEARCH_API_KEY is configured, and no manual research file exists yet.\n"
                f"Create {path} with whatever you find researching:\n"
                f"  company: {company_name}\n"
                f"  founder: {founder_name}\n"
                f"  website: {website}\n"
                f"then re-run the daily job."
            )
        return path.read_text(encoding="utf-8")


class SearchApiResearcher(Researcher):
    """Issues a few targeted queries via SerpAPI or Google CSE and joins snippets."""

    def __init__(self, provider: str | None = None, api_key: str | None = None, max_results: int | None = None):
        self.provider = provider or settings.search_api_provider
        self.api_key = api_key or settings.search_api_key
        self.max_results = max_results or settings.research_max_results
        if not self.api_key:
            raise ValueError("SearchApiResearcher requires SEARCH_API_KEY to be set")

    def _queries(self, company_name: str, founder_name: str | None) -> list[str]:
        queries = [f"{company_name} 会社概要"]
        if founder_name:
            queries.append(f"{founder_name} {company_name} 経歴")
            queries.append(f"{founder_name} 創業者 経歴 出身")
        queries.append(f"{company_name} 資金調達")
        return queries

    def _search_one(self, query: str) -> list[dict]:
        if self.provider == "serpapi":
            resp = requests.get(
                SERPAPI_ENDPOINT,
                params={"q": query, "api_key": self.api_key, "num": self.max_results, "hl": "ja", "gl": "jp"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {"title": r.get("title"), "snippet": r.get("snippet"), "link": r.get("link")}
                for r in data.get("organic_results", [])[: self.max_results]
            ]
        elif self.provider == "google_cse":
            resp = requests.get(
                GOOGLE_CSE_ENDPOINT,
                params={"q": query, "key": self.api_key, "cx": settings.google_cse_id, "num": self.max_results},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return [
                {"title": r.get("title"), "snippet": r.get("snippet"), "link": r.get("link")}
                for r in data.get("items", [])[: self.max_results]
            ]
        else:
            raise ValueError(f"Unknown SEARCH_API_PROVIDER: {self.provider}")

    def research(
        self,
        company_name: str,
        founder_name: str | None,
        website: str | None,
        company_id: int | None = None,
    ) -> str:
        blocks = []
        if website:
            blocks.append(f"公式サイト: {website}")
        for query in self._queries(company_name, founder_name):
            try:
                results = self._search_one(query)
            except requests.RequestException as exc:
                blocks.append(f"[検索エラー] クエリ「{query}」: {exc}")
                continue
            block_lines = [f"### 検索クエリ: {query}"]
            for r in results:
                block_lines.append(f"- {r['title']}\n  {r['snippet']}\n  ({r['link']})")
            blocks.append("\n".join(block_lines))
        return "\n\n".join(blocks)


def get_researcher() -> Researcher:
    if settings.search_api_key:
        return SearchApiResearcher()
    return ManualResearcher()
