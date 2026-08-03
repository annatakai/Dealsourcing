"""connpass signal source: search events matching
config/builder_keywords.yaml's connpass_keywords and surface organizers as
candidates.

connpass's public API (https://connpass.com/api/v1/) is open, no key
required. It only exposes the event organizer (owner_nickname /
owner_display_name), not the participant list - connpass doesn't expose
attendee lists via the API for privacy reasons - so this only surfaces
people organizing startup-adjacent events, not attendees.
"""
from __future__ import annotations

import requests

from dealsourcing.config import load_builder_keywords
from dealsourcing.builders.sources.base import Signal

API_ROOT = "https://connpass.com/api/v1/event/"


def _search_events(keyword: str, count: int = 20) -> list[dict]:
    resp = requests.get(API_ROOT, params={"keyword": keyword, "count": count}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("events", [])


def get_signals() -> list[Signal]:
    signals: list[Signal] = []
    for keyword in load_builder_keywords().get("connpass_keywords", []):
        try:
            events = _search_events(keyword)
        except requests.RequestException as exc:
            print(f"[connpass_source] search failed for {keyword!r}: {exc}")
            continue

        for event in events:
            owner = event.get("owner_display_name") or event.get("owner_nickname")
            if not owner:
                continue
            signals.append(
                Signal(
                    source="connpass",
                    signal_type="event_organized",
                    name=owner,
                    signal_text=f"「{event.get('title')}」を主催 (検索キーワード: {keyword})",
                    signal_url=event.get("event_url"),
                    raw=event,
                )
            )
    return signals
