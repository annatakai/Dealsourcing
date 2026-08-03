"""Shared shape every signal source returns, so dedupe.py/pipeline.py don't
need to know which source a signal came from."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Signal:
    source: str  # "github" / "qiita" / "connpass" / "websearch"
    signal_type: str  # e.g. "repo_created" / "article_published" / "event_organized" / "founder_statement" / "program_selected"
    name: str | None
    signal_text: str
    signal_url: str | None = None
    github_username: str | None = None
    qiita_username: str | None = None
    affiliation: str | None = None
    raw: dict | None = None
