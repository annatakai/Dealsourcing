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
  when it blocks. Set `SEARCH_API_KEY` (+ `GOOGLE_CSE_ID`) in `.env` once
  you have a key, and it switches to automatic search. Defaults to Google
  CSE (free tier: 100 queries/day) rather than SerpAPI (paid) - see
  `.env.example` for setup and the query-budget note below.
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
  config.py         env vars + config/*.yaml loaders (both pipelines)
  db.py             SQLite schema (companies/sent_log + builders/*)
  ingest.py         xlsx import + name-based dedupe
  classify.py       eligibility filter + category assignment
  scheduler.py      daily pick + idempotent sent-log
  research.py       pluggable web research (search API or manual file)
  ollama_client.py  structure findings + run the scoring rubric
  websearch.py      generic SerpAPI/Google CSE call (shared by research.py
                     and builders/sources/websearch_source.py)
  templates.py      builds the exact email body format
  email_sender.py   generic send_email() + the company-alert template
  builders/         the builder-sourcing pipeline (see below) - entirely
                     separate from the company pipeline above; shares only
                     db.py's connection helpers, ollama_client.call_ollama,
                     and email_sender.send_email
config/
  categories.yaml            category matching rules (data, not code)
  scoring_prompt.txt         the exact 100-point founder scoring rubric
  target_companies.yaml      builder pipeline: companies to search for on
                             GitHub/Qiita (mirrors scoring_prompt.txt's
                             30-point tier)
  accelerator_programs.yaml  builder pipeline: programs to search for
                             selection announcements of (mirrors
                             scoring_prompt.txt's 25-point tier)
  builder_keywords.yaml      builder pipeline: Qiita/connpass/websearch
                             search keywords
  builder_scoring_prompt.txt builder pipeline's 100-point rubric
run_ingest.py       one-off/re-run: import xlsx + reclassify
run_daily.py        the company-sourcing daily job (1 email/day)
run_builder_scan.py the builder-sourcing job (event-driven alerts)
tests/              unit tests against synthetic INITIAL-shaped fixtures
                     and against the builder pipeline's DB logic
```

## Builder-sourcing pipeline (`run_builder_scan.py`)

A second, separate pipeline from the INITIAL-export flow above. Instead of
scoring companies that already exist, it tries to catch people *before*
they've founded anything:

```
複数ソースをスキャン
  → GitHub: 対象企業の所属を名乗るユーザーの新規リポジトリ作成
  → Qiita: 退職/独立を示唆する記事の投稿者
  → connpass: スタートアップ関連イベントの主催者
  → Web検索(SerpAPI/Google CSE): 「退職しました」等の言明、
    Wantedly/researchmapの公開プロフィール、
    アクセラレーター採択発表の記事
  → シグナルを人物ごとにdedupe・蓄積
  → 新しいシグナルが増えるたびに100点ルールで再採点
    (config/builder_scoring_prompt.txt) + 過去のアクセラレーター
    採択者との類似度スコアを加味
  → 閾値(BUILDER_SCORE_THRESHOLD、デフォルト70点)を超えたら
    その場でメール送信、以後は二度と送らない
```

Run it periodically (hourly/daily via cron, same deployment constraints as
`run_daily.py` - Ollama must be reachable):

```bash
python run_builder_scan.py
```

Unlike `run_daily.py`, this is **not** capped at one email per day - it's
event-driven, since the point is to contact someone the moment they look
promising, not on a fixed schedule. A builder is re-scored every time new
signals arrive (so a below-threshold candidate can cross the bar later as
more evidence accumulates) but is never emailed twice.

### What this can and can't actually predict

Being upfront about the limits: this is signal detection + LLM scoring,
not a trained ML time-to-event model. There isn't enough labeled data
(who founded a company and when, with a history of prior signals) for
Genesia or anyone else to fit something more rigorous. Two consequences:

- **"起業予測時期" (estimated founding window)** is a rule-based bucket
  (1ヶ月以内 / 3〜6ヶ月 / 6ヶ月〜1年 / 不明) driven by which signal fired,
  not a calibrated date estimate.
- **The accelerator "lookalike" score** (`dealsourcing/builders/
  lookalike.py`) compares a candidate's profile against past selectees'
  *post-selection* profiles (that's all the public data gives us) via an
  LLM similarity judgment - a fuzzy prior that nudges the score, not a
  trained classifier (this repo has no ML library; see requirements.txt).

### Things that need your input before this fully runs unattended

- **GitHub (`GITHUB_TOKEN`)**: not yet configured. Without it,
  `github_source.py` is skipped entirely - unauthenticated search is
  capped at ~10 req/min, too low to cover the target company list. Also
  worth knowing going in: GitHub's `company:` search qualifier only
  matches people who filled in that profile field, which skews this
  source toward engineers and misses consultant/business-background
  builders by construction - that's what websearch_source.py is for.
- **Qiita (`QIITA_TOKEN`)**: optional. Works without it at 60 req/hour;
  set it for 1000/hour.
- **Web search (`SEARCH_API_KEY`)**: shared with the company pipeline's
  research.py. Without it, `websearch_source.py` is skipped - this is the
  source that covers accelerator-selection announcements, Wantedly, and
  researchmap (neither has a usable public search API - Wantedly's public
  API is for embedding company stories, not profile search; researchmap's
  real V2 API requires a formal institutional application - so both are
  covered via `site:`-scoped search queries instead, see
  config/builder_keywords.yaml). Defaults to **Google CSE**, which has a
  free tier (100 queries/day, no cost) - see
  https://developers.google.com/custom-search/v1/overview to create a key
  + search engine ID. Direct scraping of these sites without a search API
  was tried and rejected: PR TIMES, TechCrunch Japan, researchmap, and
  Wantedly all returned 403s to a plain HTTP fetch in testing (bot
  protection), so a search API is the only reliable free path to this
  coverage right now.
  **Query budget**: as shipped, one `run_builder_scan.py` run issues
  ~1 query per `config/accelerator_programs.yaml` entry plus
  `config/builder_keywords.yaml`'s `general_founder_prep_queries` (~18
  queries total). Running it hourly would use 432 queries/day, well over
  the free tier - schedule it every few hours (e.g. every 6 hours = 72
  queries/day) instead of hourly to stay within the free quota alongside
  the company pipeline's own usage.
- **connpass**: no key needed, works out of the box - but the API only
  exposes event organizers, not attendee lists (connpass doesn't expose
  those for privacy reasons).
- **Email (`BUILDER_EMAIL_TO`, `BUILDER_SCORE_THRESHOLD`)**: reuses
  `SENDGRID_API_KEY`/`EMAIL_FROM`/`DEALSOURCING_DRY_RUN` from the company
  pipeline; only the recipient and alert threshold are separately
  configurable.
- **Accelerator alumni pages**: intentionally *not* scraped per-program -
  real pages turned out too inconsistent to hand-parse reliably (some have
  a dedicated page, others are scattered across PR TIMES/TechCrunch Japan
  articles, others mix HTML and PDF across cohorts). Handled instead via
  websearch_source.py searching for each program's selection announcement.

## Tests

```bash
python -m pytest tests/ -v
```

Covers dedupe-across-files, eligibility filtering, category assignment
(including the EdTech/MedTech substring-collision edge case), and the
daily-pick/idempotency logic for the company pipeline — all against
synthetic fixtures shaped like real INITIAL exports, not your actual
(private) data. For the builder pipeline: signal dedupe/merge logic and
the score-then-alert-once-ever pipeline logic (both against a temp SQLite
DB, no real network/Ollama calls).
