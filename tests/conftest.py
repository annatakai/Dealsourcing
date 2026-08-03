import datetime

import openpyxl
import pytest

HEADER = [
    "名前", "事業内容", "タグ", "タイプ", "国・地域", "設立年月日", "代表者名",
    "従業員数", "都道府県", "電話番号", "業種", "起源", "大学・研究機関",
    "ウェブサイト", "法人番号", "スピーダ調達シリーズ", "Crunchbase最新ラウンドタイプ",
    "最新ラウンド調達日", "総調達額（百万円）", "総調達額算出日", "株主状況", "調査状況",
]


def _row(
    name,
    tags="",
    company_type="未公開企業",
    industry=None,
    total_funding=0,
    representative_name="山田　太郎",
    website="https://example.com",
    speeda_series="シード",
):
    return [
        name, "テスト事業内容の一行目。\n二行目。", tags, company_type, "日本",
        datetime.datetime(2020, 1, 1), representative_name, 1, "東京都", None,
        industry, "", "", website, 1234567890123, speeda_series, "", None,
        total_funding, "2024/01/01", "VC不明", "調査継続",
    ]


def make_xlsx(path, rows):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "企業リスト"
    ws.append(HEADER)
    for row in rows:
        ws.append(row)
    wb.create_sheet("必ずお読みください")
    wb.save(path)


@pytest.fixture
def sample_xlsx_factory(tmp_path):
    def _make(filename, rows):
        path = tmp_path / filename
        make_xlsx(path, rows)
        return path

    return _make


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def row_builder():
    return _row
