from dealsourcing.db import connect
from dealsourcing.ingest import ingest_files


def test_dedupes_across_files_by_name(sample_xlsx_factory, row_builder, db_path):
    file_a = sample_xlsx_factory("a.xlsx", [
        row_builder("株式会社Alpha", website="https://alpha.example.com"),
        row_builder("株式会社Beta"),
    ])
    file_b = sample_xlsx_factory("b.xlsx", [
        row_builder("株式会社Alpha", website=None),  # overlaps with file_a
        row_builder("株式会社Gamma"),
    ])

    stats = ingest_files([file_a, file_b], db_path=db_path)

    assert stats.files_processed == 2
    assert stats.rows_seen == 4
    assert stats.unique_companies_upserted == 3

    with connect(db_path) as conn:
        names = {r["name"] for r in conn.execute("SELECT name FROM companies").fetchall()}
    assert names == {"株式会社Alpha", "株式会社Beta", "株式会社Gamma"}


def test_later_file_does_not_null_out_earlier_data(sample_xlsx_factory, row_builder, db_path):
    file_a = sample_xlsx_factory("a.xlsx", [
        row_builder("株式会社Alpha", website="https://alpha.example.com"),
    ])
    file_b = sample_xlsx_factory("b.xlsx", [
        row_builder("株式会社Alpha", website=None),
    ])

    ingest_files([file_a, file_b], db_path=db_path)

    with connect(db_path) as conn:
        row = conn.execute("SELECT website FROM companies WHERE name = ?", ("株式会社Alpha",)).fetchone()
    assert row["website"] == "https://alpha.example.com"


def test_blank_name_rows_are_skipped(sample_xlsx_factory, row_builder, db_path):
    path = sample_xlsx_factory("a.xlsx", [
        row_builder("株式会社Alpha"),
        row_builder(None),
        row_builder(""),
    ])

    stats = ingest_files([path], db_path=db_path)

    assert stats.rows_seen == 1
    assert stats.unique_companies_upserted == 1
