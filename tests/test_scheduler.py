from dealsourcing.classify import classify_all
from dealsourcing.db import connect
from dealsourcing.ingest import ingest_files
from dealsourcing.scheduler import already_sent_today, pick_next_company, record_sent


def _seed(sample_xlsx_factory, row_builder, db_path, names):
    path = sample_xlsx_factory("a.xlsx", [row_builder(n) for n in names])
    ingest_files([path], db_path=db_path)
    classify_all(db_path=db_path)


def test_picks_lowest_id_unsent_eligible_company(sample_xlsx_factory, row_builder, db_path):
    _seed(sample_xlsx_factory, row_builder, db_path, ["A", "B", "C"])

    first_pick = pick_next_company(db_path=db_path)
    assert first_pick["name"] == "A"

    record_sent(
        company_id=first_pick["id"],
        scoring={"total_score": 80, "grade": "A"},
        research_raw_text="dummy",
        email_status="sent",
        category=first_pick["category"],
        sent_date="2026-08-01",
        db_path=db_path,
    )

    second_pick = pick_next_company(db_path=db_path)
    assert second_pick["name"] == "B"


def test_already_sent_today_short_circuits(sample_xlsx_factory, row_builder, db_path):
    _seed(sample_xlsx_factory, row_builder, db_path, ["A"])
    company = pick_next_company(db_path=db_path)

    with connect(db_path) as conn:
        assert already_sent_today(conn, sent_date="2026-08-02") is None

    record_sent(
        company_id=company["id"],
        scoring={"total_score": 80, "grade": "A"},
        research_raw_text="dummy",
        email_status="sent",
        category=company["category"],
        sent_date="2026-08-02",
        db_path=db_path,
    )

    with connect(db_path) as conn:
        assert already_sent_today(conn, sent_date="2026-08-02") is not None
