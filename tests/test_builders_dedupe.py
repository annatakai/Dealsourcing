from dealsourcing.db import connect, init_db
from dealsourcing.builders.dedupe import ingest_signals
from dealsourcing.builders.sources.base import Signal


def test_ingest_creates_new_builder_and_records_signal(db_path):
    init_db(db_path)
    signal = Signal(
        source="github",
        signal_type="repo_created",
        name="testuser",
        signal_text="new repo",
        signal_url="https://github.com/testuser",
        github_username="testuser",
        affiliation="Mercari",
    )
    summary = ingest_signals([signal], db_path=db_path)
    assert summary == {"signals_ingested": 1, "builders_touched": 1}

    with connect(db_path) as conn:
        builder = conn.execute(
            "SELECT * FROM builders WHERE github_username = ?", ("testuser",)
        ).fetchone()
        assert builder is not None
        assert builder["current_affiliation"] == "Mercari"
        signals = conn.execute(
            "SELECT * FROM builder_signals WHERE builder_id = ?", (builder["id"],)
        ).fetchall()
        assert len(signals) == 1


def test_ingest_merges_by_github_username(db_path):
    init_db(db_path)
    s1 = Signal(source="github", signal_type="repo_created", name="Taro Yamada",
                signal_text="repo 1", github_username="taro123")
    s2 = Signal(source="github", signal_type="repo_created", name="Taro Yamada",
                signal_text="repo 2", github_username="taro123")
    ingest_signals([s1], db_path=db_path)
    ingest_signals([s2], db_path=db_path)

    with connect(db_path) as conn:
        builders = conn.execute(
            "SELECT * FROM builders WHERE github_username = ?", ("taro123",)
        ).fetchall()
        assert len(builders) == 1
        signals = conn.execute(
            "SELECT * FROM builder_signals WHERE builder_id = ?", (builders[0]["id"],)
        ).fetchall()
        assert len(signals) == 2


def test_ingest_merges_by_normalized_name_when_no_username(db_path):
    init_db(db_path)
    s1 = Signal(source="connpass", signal_type="event_organized", name="山田 太郎", signal_text="event 1")
    s2 = Signal(source="websearch", signal_type="founder_statement", name="山田太郎", signal_text="quit job")
    ingest_signals([s1], db_path=db_path)
    ingest_signals([s2], db_path=db_path)

    with connect(db_path) as conn:
        builders = conn.execute("SELECT * FROM builders").fetchall()
        assert len(builders) == 1


def test_program_selected_signal_recorded_as_alumni_history(db_path):
    init_db(db_path)
    signal = Signal(
        source="websearch",
        signal_type="program_selected",
        name="花子 佐藤",
        signal_text="1stRoundに採択",
        signal_url="https://example.com",
        affiliation="スタートアップX",
        raw={"program_name": "東大IPC 1stRound"},
    )
    ingest_signals([signal], db_path=db_path)

    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM accelerator_alumni_history").fetchall()
        assert len(rows) == 1
        assert rows[0]["program_name"] == "東大IPC 1stRound"
        assert rows[0]["person_name"] == "花子 佐藤"


def test_non_program_selected_signal_not_recorded_as_alumni_history(db_path):
    init_db(db_path)
    signal = Signal(source="qiita", signal_type="article_published", name="Someone", signal_text="post")
    ingest_signals([signal], db_path=db_path)

    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM accelerator_alumni_history").fetchall()
        assert len(rows) == 0
