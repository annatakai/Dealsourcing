"""GitHub signal source: search users whose profile lists one of
config/target_companies.yaml's companies, then check recent public
activity for a "building something new" signal (new repo creation in the
last 30 days).

Requires GITHUB_TOKEN for a usable rate limit - unauthenticated search is
capped at 10 requests/min, which isn't enough to cover the target company
list plus per-user activity checks. Skipped entirely if unset (see
get_signals()), same fallback philosophy as research.py's ManualResearcher
for the company pipeline: degrade visibly, don't fail silently.

Known limitation (flagged up front, not a bug to fix later): GitHub's
`company:` search qualifier only matches free-text in the user's profile,
which plenty of engineers never fill in, and misses non-engineer builders
entirely (consultants, business-side founders) by construction - that's
why config/builder_keywords.yaml's general_founder_prep_queries and
websearch_source.py exist as a separate, broader-net source.
"""
from __future__ import annotations

import datetime

import requests

from dealsourcing.config import load_target_companies, settings
from dealsourcing.builders.sources.base import Signal

API_ROOT = "https://api.github.com"
ACTIVITY_WINDOW_DAYS = 30


def _headers() -> dict:
    headers = {"Accept": "application/vnd.github+json"}
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"
    return headers


def _search_users_by_company(company_name: str, per_page: int = 10) -> list[dict]:
    resp = requests.get(
        f"{API_ROOT}/search/users",
        params={"q": f'company:"{company_name}"', "per_page": per_page},
        headers=_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json().get("items", [])


def _recent_repo_creations(username: str) -> list[dict]:
    resp = requests.get(f"{API_ROOT}/users/{username}/events/public", headers=_headers(), timeout=30)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()

    cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=ACTIVITY_WINDOW_DAYS)
    events = []
    for event in resp.json():
        if event.get("type") != "CreateEvent" or event.get("payload", {}).get("ref_type") != "repository":
            continue
        created_at = event.get("created_at")
        if not created_at:
            continue
        if datetime.datetime.fromisoformat(created_at.replace("Z", "+00:00")) >= cutoff:
            events.append(event)
    return events


def get_signals() -> list[Signal]:
    if not settings.github_token:
        print("[github_source] GITHUB_TOKEN not set - skipping (unauthenticated rate limit is too low to be useful).")
        return []

    signals: list[Signal] = []
    for company in load_target_companies():
        # Search the primary name plus every alias (e.g. "アクセンチュア" /
        # "Accenture") - GitHub's company: qualifier is an exact substring
        # match against whatever the user typed in their profile, so a
        # company listed only in Japanese would miss profiles that filled
        # in the English name, and vice versa.
        seen_usernames: set[str] = set()
        for name_variant in [company["name"], *company.get("aliases", [])]:
            try:
                users = _search_users_by_company(name_variant)
            except requests.RequestException as exc:
                print(f"[github_source] search failed for {name_variant!r}: {exc}")
                continue

            for user in users:
                username = user.get("login")
                if not username or username in seen_usernames:
                    continue
                seen_usernames.add(username)

                try:
                    repo_events = _recent_repo_creations(username)
                except requests.RequestException as exc:
                    print(f"[github_source] activity check failed for {username!r}: {exc}")
                    continue

                if not repo_events:
                    continue

                signals.append(
                    Signal(
                        source="github",
                        signal_type="repo_created",
                        name=username,
                        signal_text=f"直近{ACTIVITY_WINDOW_DAYS}日で新規リポジトリ{len(repo_events)}件作成 (プロフィール所属: {company['name']})",
                        signal_url=user.get("html_url"),
                        github_username=username,
                        affiliation=company["name"],
                        raw={"user": user, "repo_events": repo_events},
                    )
                )
    return signals
