# Dealsourcing

Daily startup-sourcing automation built around INITIAL exports:

```
INITIAL Excelを一括取込
  → 会社名で重複排除
  → 未上場・累計調達額2億円以下を判定
  → 8カテゴリー + その他 に分類
  → 毎日1社を選択
  → 会社名・代表者名を起点にWebリサーチ
  → ローカルOllamaで経歴を構造化
  → 指定の100点ルールで採点
  → メール送信 (デフォルト宛先: takai@genesiaventures.com)
  → 送信済みとしてSQLiteに記録
  → 翌日は次の会社
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in real values, then export them into your shell
```

### 1. Import INITIAL exports

Drop your `epcompanies*.xlsx` exports anywhere (e.g. `data/initial_exports/`)
and run:

```bash
python run_ingest.py data/initial_exports/*.xlsx
```

This is safe to re-run any time you add more exports: companies are
deduped by name (`INSERT ... ON CONFLICT(name) DO UPDATE`, keeping
whichever file has the most complete data for each field), and every row
is reclassified (eligibility + category) on every run. It prints a
breakdown like:

```
Ingested 5 file(s), 3235 row(s) seen, 2044 unique companies in DB.
Eligible: 2044, ineligible: 0
  other: 728
  Consumer: 499
  DeepTech: 310
  MedTech: 139
  HealthTech: 127
  FinTech: 58
  AgriTech: 56
  Media: 76
  EdTech: 51
```

Eligibility = `タイプ == 未公開企業` (unlisted) AND `総調達額（百万円） <= 200`
(2億円). Category assignment is config-driven — see `config/categories.yaml`
— matching INITIAL's own `タグ` column (it already tags companies with
`EdTech`/`FinTech`/`HealthTech`/`MedTech`/`AgriTech` directly) plus `業種`
(industry) as a fallback. Retune it there, no code changes needed.

### 2. Run the daily job

```bash
python run_daily.py
```

Each run:
1. Exits immediately (no-op) if a company was already sent today — the
   job is safe to run more than once a day or retry after a crash.
2. Picks the lowest-`id` eligible company that has never been sent
   (`dealsourcing/scheduler.py` — swap the ordering there if you want
   category round-robin or score-based ordering instead of FIFO).
3. Researches the company + founder (`dealsourcing/research.py`).
4. Structures the findings and scores them with your exact rubric via
   local Ollama (`dealsourcing/ollama_client.py`,
   `config/scoring_prompt.txt`).
5. Emails the result (`dealsourcing/email_sender.py`,
   `dealsourcing/templates.py`).
6. Records the send in SQLite (`sent_log` table) so the company is never
   picked again and at most one email goes out per calendar date.

Schedule it with cron, e.g. 9am daily:

```
0 9 * * * cd /path/to/Dealsourcing && /path/to/venv/bin/python run_daily.py >> logs/daily.log 2>&1
```

or a systemd timer, or a scheduled GitHub Actions workflow — whichever
fits your infra. The important constraint: **Ollama must be reachable
from wherever this runs** (default `http://localhost:11434`), so if you
run this on a server without a local Ollama install, either run Ollama
there too or point `OLLAMA_HOST` at an Ollama-compatible endpoint you
control.

## Things that need your input before this fully runs unattended

- **Web research (`SEARCH_API_KEY`)**: not yet configured. Without it,
  `run_daily.py` falls back to `ManualResearcher`, which expects you to
  paste research findings into `data/manual_research/<company_id>.txt`
  yourself before re-running — it tells you the exact path and company
  when it blocks. Set `SEARCH_API_KEY` (+ `SEARCH_API_PROVIDER=serpapi`
  or `google_cse`) in `.env` once you have a key, and it switches to
  automatic search.
- **Email (`SENDGRID_API_KEY`, `EMAIL_FROM`)**: not yet configured. Set
  `DEALSOURCING_DRY_RUN=1` to test the whole pipeline without a real
  SendGrid account — it prints the email instead of sending it.
- **Ollama**: assumes a local Ollama server with `OLLAMA_MODEL` (default
  `llama3.1`) already pulled. Nothing in this repo installs or runs
  Ollama itself.
- **Deployment target**: not yet decided (cron on your own machine vs.
  GitHub Actions vs. something else) — see the note above about Ollama
  reachability, which is the main constraint on that choice.

## Project layout

```
dealsourcing/
  config.py         env vars + config/categories.yaml loader
  db.py             SQLite schema (companies, sent_log)
  ingest.py         xlsx import + name-based dedupe
  classify.py       eligibility filter + category assignment
  scheduler.py      daily pick + idempotent sent-log
  research.py       pluggable web research (search API or manual file)
  ollama_client.py  structure findings + run the scoring rubric
  templates.py      builds the exact email body format
  email_sender.py   SendGrid send (or dry-run print)
config/
  categories.yaml       category matching rules (data, not code)
  scoring_prompt.txt    the exact 100-point founder scoring rubric
run_ingest.py       one-off/re-run: import xlsx + reclassify
run_daily.py        the daily job
tests/              unit tests against synthetic INITIAL-shaped fixtures
```

## Tests

```bash
python -m pytest tests/ -v
```

Covers dedupe-across-files, eligibility filtering, category assignment
(including the EdTech/MedTech substring-collision edge case), and the
daily-pick/idempotency logic — all against synthetic fixtures shaped
like real INITIAL exports, not your actual (private) data.
