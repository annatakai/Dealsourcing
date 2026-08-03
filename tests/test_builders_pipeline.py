import pytest

from dealsourcing.db import connect, init_db
from dealsourcing.builders import pipeline
from dealsourcing.builders.dedupe import ingest_signals
from dealsourcing.builders.sources.base import Signal


def _seed_builder(db_path, name="Taro Yamada", github_username="taro"):
    signal = Signal(source="github", signal_type="repo_created", name=name,
                     signal_text="new repo", github_username=github_username)
    ingest_signals([signal], db_path=db_path)
    with connect(db_path) as conn:
        return conn.execute(
            "SELECT id FROM builders WHERE github_username = ?", (github_username,)
        ).fetchone()["id"]


def test_score_and_notify_sends_alert_above_threshold(db_path, monkeypatch):
    init_db(db_path)
    _seed_builder(db_path)

    monkeypatch.setattr(
        pipeline,
        "score_builder",
        lambda builder, signals, db_path=None: {
            "total_score": 85,
            "grade": "A",
            "company_score": 30,
            "program_score": 25,
            "education_score": 25,
            "signal_score": 5,
            "lookalike_score": None,
            "estimated_founding_window": "3〜6ヶ月以内",
            "comment": "test",
        },
    )
    sent_to = []

    def fake_send(builder, scoring, signals):
        sent_to.append(builder["name"])
        return "sent"

    monkeypatch.setattr(pipeline, "send_builder_alert", fake_send)
    monkeypatch.setattr(pipeline.settings, "builder_score_threshold", 70)

    summary = pipeline.score_and_notify(db_path=db_path)
    assert summary == {"scored": 1, "alerted": 1}
    assert sent_to == ["Taro Yamada"]

    with connect(db_path) as conn:
        log = conn.execute("SELECT * FROM builder_scoring_log").fetchall()
        assert len(log) == 1
        assert log[0]["email_status"] == "sent"
        assert log[0]["total_score"] == 85


def test_score_and_notify_skips_below_threshold(db_path, monkeypatch):
    init_db(db_path)
    _seed_builder(db_path)

    monkeypatch.setattr(
        pipeline, "score_builder", lambda builder, signals, db_path=None: {"total_score": 40, "grade": "D"}
    )
    monkeypatch.setattr(
        pipeline, "send_builder_alert", lambda *a, **kw: pytest.fail("should not send below threshold")
    )
    monkeypatch.setattr(pipeline.settings, "builder_score_threshold", 70)

    summary = pipeline.score_and_notify(db_path=db_path)
    assert summary == {"scored": 1, "alerted": 0}

    with connect(db_path) as conn:
        log = conn.execute("SELECT * FROM builder_scoring_log").fetchall()
        assert log[0]["email_status"] == "not_sent"


def test_already_alerted_builder_is_never_resent(db_path, monkeypatch):
    init_db(db_path)
    _seed_builder(db_path)

    call_count = {"n": 0}

    def fake_send(builder, scoring, signals):
        call_count["n"] += 1
        return "sent"

    monkeypatch.setattr(
        pipeline, "score_builder", lambda builder, signals, db_path=None: {"total_score": 90, "grade": "S"}
    )
    monkeypatch.setattr(pipeline, "send_builder_alert", fake_send)
    monkeypatch.setattr(pipeline.settings, "builder_score_threshold", 70)

    pipeline.score_and_notify(db_path=db_path)
    assert call_count["n"] == 1

    # A new signal arrives for the same builder - re-scoring should pick
    # it up (score could change), but the alert must never fire twice.
    new_signal = Signal(source="github", signal_type="repo_created", name="Taro Yamada",
                         signal_text="another repo", github_username="taro")
    ingest_signals([new_signal], db_path=db_path)

    summary = pipeline.score_and_notify(db_path=db_path)
    assert summary["scored"] == 1  # did re-score
    assert call_count["n"] == 1  # but did not re-send
