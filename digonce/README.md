# Dig Once — explainable pipe / road / coordination scoring

A multi-agent scoring and budget-optimisation MVP for coordinating water-main replacement
with road maintenance.

* **Pipe Agent**: XGBoost G/A/R condition model → condition, likelihood, consequence,
  risk, urgency, economic value → 0–100 priority with a plain-text explanation.
* **Road Agent**: PCI (observed, aged forward or estimated) and criticality → priority
  and treatment.
* **Budget Agent**: pipe–road overlap, a Coordination Opportunity Score kept separate
  from estimated ¥ savings, and an OR-Tools MIP budget optimiser with joint bundles.

**Full method, citations, formulas, results, schema and limitations:
[`docs/FRAMEWORK.md`](docs/FRAMEWORK.md).**

## Quick start

```bash
cd digonce
pip install -r requirements.txt
pytest tests                                   # 29 tests
python scripts/run_pipeline.py --synthetic 20000 --out outputs/demo   # synthetic demo
```

## Your data

```bash
python scripts/inspect_csv.py pipes  ~/Desktop/Water_Mains_5473643541506873803.csv
python scripts/inspect_csv.py breaks ~/Desktop/Water_Main_Breaks_-8394745777553033727.csv
python scripts/inspect_csv.py roads  ~/Desktop/Roads.csv

python scripts/run_pipeline.py \
  --pipes ~/Desktop/Water_Mains_5473643541506873803.csv \
  --breaks ~/Desktop/Water_Main_Breaks_-8394745777553033727.csv \
  --roads ~/Desktop/Roads.csv \
  --column-map configs/column_map.example.yaml \
  --out outputs/city

# anonymised 103k condition dataset (no geometry -> no road pairing, by design)
python scripts/run_pipeline.py --pipes path/to/pipe_condition.csv --out outputs/anon
```

Add `--diameter-unit in` and/or `--length-unit ft` for US-unit exports.

## Important caveats

* Unit costs in `dig_once/config.py` are **placeholders**, not real prices. Fill in
  `configs/local_costs_template.yaml` and pass `--config`.
* `outputs/example_synthetic/` was produced from **synthetic** data. It shows the output
  format, not findings about real infrastructure.
* Scores are decision indices. None is a validated probability of failure.
