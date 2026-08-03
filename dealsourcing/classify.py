"""Eligibility filtering + category classification.

Matches the pipeline steps:
    未上場・累計調達額2億円以下を判定
    ↓
    カテゴリーに分類（分類数・ルールはconfig/categories.yaml次第）

Eligibility = unlisted (company_type == settings.unlisted_type_value)
              AND total_funding_million_yen <= settings.max_total_funding_million_yen
              (a NULL/unknown total funding is treated as eligible - "no
              disclosed funding" is not evidence of exceeding the cap).

Category assignment is driven entirely by config/categories.yaml so the
rules can be retuned without touching code.
"""
from __future__ import annotations

from dealsourcing.config import load_categories_config, settings
from dealsourcing.db import connect


def is_eligible(company_type: str | None, total_funding_million_yen: int | None) -> bool:
    if company_type != settings.unlisted_type_value:
        return False
    if total_funding_million_yen is None:
        return True
    return total_funding_million_yen <= settings.max_total_funding_million_yen


def _split_tags(tags: str | None) -> set[str]:
    if not tags:
        return set()
    return {t.strip() for t in tags.split(",") if t.strip()}


def classify_category(tags: str | None, industry: str | None, rules: list[dict], other_label: str = "other") -> str:
    tag_set = _split_tags(tags)
    for rule in rules:
        if tag_set & set(rule.get("tags", [])):
            return rule["name"]
        if industry and industry in rule.get("industries", []):
            return rule["name"]
    return other_label


def classify_all(db_path=None) -> dict:
    """Recompute is_eligible + category for every row in `companies`.

    Returns a summary dict: {"eligible": N, "ineligible": N, "by_category": {...}}.
    """
    config = load_categories_config()
    rules = config["categories"]
    other_label = config.get("other_category", "other")

    summary = {"eligible": 0, "ineligible": 0, "by_category": {}}

    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT id, company_type, total_funding_million_yen, tags, industry FROM companies"
        ).fetchall()

        for row in rows:
            eligible = is_eligible(row["company_type"], row["total_funding_million_yen"])
            category = classify_category(row["tags"], row["industry"], rules, other_label) if eligible else None
            if not eligible:
                category = None

            conn.execute(
                "UPDATE companies SET is_unlisted = ?, is_eligible = ?, category = ? WHERE id = ?",
                (
                    1 if row["company_type"] == settings.unlisted_type_value else 0,
                    1 if eligible else 0,
                    category,
                    row["id"],
                ),
            )

            if eligible:
                summary["eligible"] += 1
                key = category or other_label
                summary["by_category"][key] = summary["by_category"].get(key, 0) + 1
            else:
                summary["ineligible"] += 1

    return summary
