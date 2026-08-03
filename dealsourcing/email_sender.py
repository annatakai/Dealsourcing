"""Sends the daily email via SendGrid ("takai@genesiaventures.comへメール").

Requires SENDGRID_API_KEY and EMAIL_FROM (a sender identity verified in
your SendGrid account) to be set. Recipient defaults to
takai@genesiaventures.com but is configurable via EMAIL_TO.

Set DEALSOURCING_DRY_RUN=1 to print the email instead of sending it -
useful for testing the rest of the pipeline without a real SendGrid
account.
"""
from __future__ import annotations

import sqlite3

import requests

from dealsourcing.config import settings
from dealsourcing.templates import build_body, build_subject

SENDGRID_ENDPOINT = "https://api.sendgrid.com/v3/mail/send"


class EmailConfigError(RuntimeError):
    pass


def send_daily_email(company: sqlite3.Row, category: str, scoring: dict) -> str:
    """Returns the email_status string to store in sent_log ('sent', 'dry_run', or raises)."""
    subject = build_subject(company, category, scoring)
    body = build_body(company, category, scoring)

    if settings.dry_run:
        print("=== DRY RUN: would send email ===")
        print(f"To: {settings.email_to}")
        print(f"Subject: {subject}")
        print(body)
        print("==================================")
        return "dry_run"

    if not settings.sendgrid_api_key:
        raise EmailConfigError("SENDGRID_API_KEY is not set. Set it, or set DEALSOURCING_DRY_RUN=1 to test without sending.")
    if not settings.email_from:
        raise EmailConfigError("EMAIL_FROM is not set to a SendGrid-verified sender identity.")

    payload = {
        "personalizations": [{"to": [{"email": settings.email_to}]}],
        "from": {"email": settings.email_from},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }
    resp = requests.post(
        SENDGRID_ENDPOINT,
        headers={"Authorization": f"Bearer {settings.sendgrid_api_key}"},
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    return "sent"
