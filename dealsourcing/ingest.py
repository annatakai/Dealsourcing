"""Ingest one or more INITIAL ("epcompanies*.xlsx") exports into SQLite.

INITIAL's UI caps a single search export at ~1000 rows, so sourcing the
full universe usually means running several searches and exporting several
files. Those files overlap heavily (same company can appear in multiple
searches) - this module merges them and dedupes by company name, matching
the "INITIAL Excelを一括取込 → 会社名で重複排除" step of the pipeline.

Column headers are matched by name (not position) so a reordered or
slightly different export still ingests correctly; an export missing a
column we expect just yields None for that field rather than crashing.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path

import openpyxl

from dealsourcing.db import connect, init_db

SHEET_NAME = "企業リスト"

# INITIAL header (Japanese) -> normalized column name in the `companies` table.
HEADER_MAP = {
    "名前": "name",
    "事業内容": "business_description",
    "タグ": "tags",
    "タイプ": "company_type",
    "国・地域": "region",
    "設立年月日": "founded_date",
    "代表者名": "representative_name",
    "従業員数": "employee_count",
    "都道府県": "prefecture",
    "電話番号": "phone",
    "業種": "industry",
    "起源": "origin",
    "大学・研究機関": "university_or_institution",
    "ウェブサイト": "website",
    "法人番号": "corporate_number",
    "スピーダ調達シリーズ": "speeda_funding_series",
    "Crunchbase最新ラウンドタイプ": "crunchbase_round_type",
    "最新ラウンド調達日": "latest_round_date",
    "総調達額（百万円）": "total_funding_million_yen",
    "総調達額算出日": "funding_calc_date",
    "株主状況": "shareholder_status",
    "調査状況": "research_status",
}

TEXT_COLUMNS = {
    "business_description", "tags", "company_type", "region", "founded_date",
    "representative_name", "prefecture", "phone", "industry", "origin",
    "university_or_institution", "website", "corporate_number",
    "speeda_funding_series", "crunchbase_round_type", "latest_round_date",
    "funding_calc_date", "shareholder_status", "research_status",
}


def _normalize_value(col: str, value):
    if value is None:
        return None
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if col in TEXT_COLUMNS:
        return str(value).strip()
    if col == "employee_count":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if col == "total_funding_million_yen":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return value


def _read_rows(path: Path):
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    if SHEET_NAME not in wb.sheetnames:
        raise ValueError(f"{path.name}: expected sheet '{SHEET_NAME}', found {wb.sheetnames}")
    ws = wb[SHEET_NAME]
    rows = ws.iter_rows(min_row=1, values_only=True)
    header = next(rows)
    header_idx = {h: i for i, h in enumerate(header) if h in HEADER_MAP}
    missing = set(HEADER_MAP) - set(header_idx)
    if missing:
        raise ValueError(f"{path.name}: missing expected columns: {sorted(missing)}")

    for raw_row in rows:
        name_cell = raw_row[header_idx["名前"]]
        if name_cell is None or str(name_cell).strip() == "":
            continue
        record = {}
        for jp_header, col in HEADER_MAP.items():
            record[col] = _normalize_value(col, raw_row[header_idx[jp_header]])
        yield record
    wb.close()


@dataclass
class IngestStats:
    files_processed: int = 0
    rows_seen: int = 0
    unique_companies_upserted: int = 0


UPSERT_SQL = """
INSERT INTO companies (
    name, business_description, tags, company_type, region, founded_date,
    representative_name, employee_count, prefecture, phone, industry,
    origin, university_or_institution, website, corporate_number,
    speeda_funding_series, crunchbase_round_type, latest_round_date,
    total_funding_million_yen, funding_calc_date, shareholder_status,
    research_status, source_file
) VALUES (
    :name, :business_description, :tags, :company_type, :region, :founded_date,
    :representative_name, :employee_count, :prefecture, :phone, :industry,
    :origin, :university_or_institution, :website, :corporate_number,
    :speeda_funding_series, :crunchbase_round_type, :latest_round_date,
    :total_funding_million_yen, :funding_calc_date, :shareholder_status,
    :research_status, :source_file
)
ON CONFLICT(name) DO UPDATE SET
    business_description = COALESCE(excluded.business_description, companies.business_description),
    tags = COALESCE(excluded.tags, companies.tags),
    company_type = COALESCE(excluded.company_type, companies.company_type),
    region = COALESCE(excluded.region, companies.region),
    founded_date = COALESCE(excluded.founded_date, companies.founded_date),
    representative_name = COALESCE(excluded.representative_name, companies.representative_name),
    employee_count = COALESCE(excluded.employee_count, companies.employee_count),
    prefecture = COALESCE(excluded.prefecture, companies.prefecture),
    phone = COALESCE(excluded.phone, companies.phone),
    industry = COALESCE(excluded.industry, companies.industry),
    origin = COALESCE(excluded.origin, companies.origin),
    university_or_institution = COALESCE(excluded.university_or_institution, companies.university_or_institution),
    website = COALESCE(excluded.website, companies.website),
    corporate_number = COALESCE(excluded.corporate_number, companies.corporate_number),
    speeda_funding_series = COALESCE(excluded.speeda_funding_series, companies.speeda_funding_series),
    crunchbase_round_type = COALESCE(excluded.crunchbase_round_type, companies.crunchbase_round_type),
    latest_round_date = COALESCE(excluded.latest_round_date, companies.latest_round_date),
    total_funding_million_yen = COALESCE(excluded.total_funding_million_yen, companies.total_funding_million_yen),
    funding_calc_date = COALESCE(excluded.funding_calc_date, companies.funding_calc_date),
    shareholder_status = COALESCE(excluded.shareholder_status, companies.shareholder_status),
    research_status = COALESCE(excluded.research_status, companies.research_status)
"""


def ingest_files(paths: list[Path], db_path: Path | None = None) -> IngestStats:
    init_db(db_path)
    stats = IngestStats()

    with connect(db_path) as conn:
        for path in paths:
            path = Path(path)
            stats.files_processed += 1
            for record in _read_rows(path):
                stats.rows_seen += 1
                record["source_file"] = path.name
                conn.execute(UPSERT_SQL, record)

        row = conn.execute("SELECT COUNT(*) AS c FROM companies").fetchone()
        stats.unique_companies_upserted = row["c"]

    return stats
