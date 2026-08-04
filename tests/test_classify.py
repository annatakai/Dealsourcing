from dealsourcing.classify import classify_all
from dealsourcing.ingest import ingest_files


def test_listed_company_is_ineligible(sample_xlsx_factory, row_builder, db_path):
    path = sample_xlsx_factory("a.xlsx", [
        row_builder("株式会社Listed", company_type="上場企業"),
        row_builder("株式会社Unlisted", company_type="未公開企業"),
    ])
    ingest_files([path], db_path=db_path)
    summary = classify_all(db_path=db_path)

    assert summary["eligible"] == 1
    assert summary["ineligible"] == 1


def test_funding_over_cap_is_ineligible(sample_xlsx_factory, row_builder, db_path):
    path = sample_xlsx_factory("a.xlsx", [
        row_builder("株式会社UnderCap", total_funding=200),
        row_builder("株式会社OverCap", total_funding=201),
    ])
    ingest_files([path], db_path=db_path)
    summary = classify_all(db_path=db_path)

    assert summary["eligible"] == 1
    assert summary["ineligible"] == 1


def test_series_a_and_later_is_ineligible(sample_xlsx_factory, row_builder, db_path):
    path = sample_xlsx_factory("a.xlsx", [
        row_builder("株式会社Seed", speeda_series="シード"),
        row_builder("株式会社NoStage", speeda_series=None),
        row_builder("株式会社SeriesA", speeda_series="シリーズA"),
        row_builder("株式会社SeriesB", speeda_series="シリーズB"),
    ])
    ingest_files([path], db_path=db_path)
    summary = classify_all(db_path=db_path)

    assert summary["eligible"] == 2  # Seed + NoStage (unset stage is not evidence of Series A+)
    assert summary["ineligible"] == 2


def test_category_assignment_priority(sample_xlsx_factory, row_builder, db_path):
    path = sample_xlsx_factory("a.xlsx", [
        row_builder("EdTech株式会社", tags="EdTech, モバイルアプリ"),
        row_builder("MedTech株式会社", tags="MedTech"),
        row_builder("Consumer株式会社", tags="BtoC, オンラインサービス"),
        row_builder("Other株式会社", tags="コンサルティング"),
    ])
    ingest_files([path], db_path=db_path)
    classify_all(db_path=db_path)

    from dealsourcing.db import connect
    with connect(db_path) as conn:
        rows = {r["name"]: r["category"] for r in conn.execute("SELECT name, category FROM companies").fetchall()}

    assert rows["EdTech株式会社"] == "EdTech"
    # This taxonomy has no dedicated MedTech/Consumer buckets - both verticals
    # fold into "Tech" (MedTech tag) / "Other" (no vertical tag, just BtoC).
    assert rows["MedTech株式会社"] == "Tech"
    assert rows["Consumer株式会社"] == "Other"
    assert rows["Other株式会社"] == "Other"


def test_medtech_tag_is_not_misclassified_as_edtech(sample_xlsx_factory, row_builder, db_path):
    # Regression check: "MedTech" contains the substring "edtech" (case-
    # insensitively), so a naive substring match on "EdTech" would
    # wrongly classify MedTech companies as EdTech. Exact tag-token
    # matching must prevent that - MedTech should land in "Tech" (this
    # taxonomy's catch-all for the old MedTech/HealthTech/FinTech verticals),
    # never "EdTech".
    path = sample_xlsx_factory("a.xlsx", [row_builder("株式会社Med", tags="MedTech")])
    ingest_files([path], db_path=db_path)
    classify_all(db_path=db_path)

    from dealsourcing.db import connect
    with connect(db_path) as conn:
        row = conn.execute("SELECT category FROM companies WHERE name = ?", ("株式会社Med",)).fetchone()
    assert row["category"] == "Tech"
