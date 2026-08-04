"""Central configuration: environment variables + config/categories.yaml.

Everything that differs between deployments (API keys, thresholds, the
email recipient) lives here so the rest of the codebase never reads
os.environ directly.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "dealsourcing.db"


def _env(name: str, default: str | None = None) -> str | None:
    val = os.environ.get(name)
    return val if val not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val not in (None, "") else default


@dataclass
class Settings:
    # --- storage ---
    db_path: Path = field(default_factory=lambda: Path(_env("DEALSOURCING_DB_PATH", str(DEFAULT_DB_PATH))))

    # --- eligibility thresholds ---
    max_total_funding_million_yen: int = field(
        default_factory=lambda: _env_int("MAX_TOTAL_FUNDING_MILLION_YEN", 200)
    )
    unlisted_type_value: str = field(default_factory=lambda: _env("UNLISTED_TYPE_VALUE", "未公開企業"))

    # --- email ---
    email_to: str = field(default_factory=lambda: _env("EMAIL_TO", "takai@genesiaventures.com"))
    email_from: str | None = field(default_factory=lambda: _env("EMAIL_FROM"))
    sendgrid_api_key: str | None = field(default_factory=lambda: _env("SENDGRID_API_KEY"))

    # --- web research ---
    # Defaults to google_cse (free tier: 100 queries/day) rather than
    # serpapi (paid) - see .env.example for setup and the query-budget note
    # in README.md's builder-sourcing section.
    search_api_provider: str = field(default_factory=lambda: _env("SEARCH_API_PROVIDER", "google_cse"))
    search_api_key: str | None = field(default_factory=lambda: _env("SEARCH_API_KEY"))
    google_cse_id: str | None = field(default_factory=lambda: _env("GOOGLE_CSE_ID"))
    research_max_results: int = field(default_factory=lambda: _env_int("RESEARCH_MAX_RESULTS", 5))

    # --- ollama ---
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1"))
    ollama_timeout_seconds: int = field(default_factory=lambda: _env_int("OLLAMA_TIMEOUT_SECONDS", 180))

    # --- misc ---
    dry_run: bool = field(default_factory=lambda: _env("DEALSOURCING_DRY_RUN", "0") == "1")

    # --- builder-sourcing pipeline (separate from the above) ---
    github_token: str | None = field(default_factory=lambda: _env("GITHUB_TOKEN"))
    qiita_token: str | None = field(default_factory=lambda: _env("QIITA_TOKEN"))
    builder_db_path: Path = field(
        default_factory=lambda: Path(_env("BUILDER_DB_PATH", str(DEFAULT_DB_PATH)))
    )
    builder_score_threshold: int = field(
        default_factory=lambda: _env_int("BUILDER_SCORE_THRESHOLD", 70)
    )
    builder_email_to: str = field(
        default_factory=lambda: _env("BUILDER_EMAIL_TO", _env("EMAIL_TO", "takai@genesiaventures.com"))
    )


def load_categories_config() -> dict:
    path = CONFIG_DIR / "categories.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_scoring_prompt() -> str:
    path = CONFIG_DIR / "scoring_prompt.txt"
    return path.read_text(encoding="utf-8")


def load_target_companies() -> list[dict]:
    path = CONFIG_DIR / "target_companies.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["companies"]


def load_accelerator_programs() -> list[dict]:
    path = CONFIG_DIR / "accelerator_programs.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["programs"]


def load_builder_keywords() -> dict:
    path = CONFIG_DIR / "builder_keywords.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_builder_scoring_prompt() -> str:
    path = CONFIG_DIR / "builder_scoring_prompt.txt"
    return path.read_text(encoding="utf-8")


settings = Settings()
