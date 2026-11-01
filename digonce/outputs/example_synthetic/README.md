# SYNTHETIC example outputs

Generated with:

```bash
python scripts/run_pipeline.py --synthetic 20000 --out outputs/example_synthetic --sensitivity-draws 50
```

All inputs come from `dig_once/synthetic.py` and all costs are placeholders. These files
show the output format and how to read it. They are **not** findings about any real
network. The large files (`pipe_scores_all.*` and `coordination_pairs.*`) were removed
to keep the repository small. Re-run the command above to regenerate them.

| File | Content |
|------|---------|
| `pipe_scores_sample500.csv/json` | 500-segment stratified prototype with all sub-scores and explanations |
| `example_top10_pipes.csv`, `example_top10_roads.csv`, `example_top10_coordination.csv` | Top 10 of each |
| `road_scores.csv/json` | All synthetic roads |
| `budget_selection.csv` | MIP-selected pipe / road / joint options |
| `validation_strategies.csv` | Strategy comparison at a 10% budget |
| `sensitivity_*.csv` | Weight (Monte Carlo and one-at-a-time) and cost sensitivity |
| `run_report.json` | Model metrics, baselines, data-quality counts, optimiser summary |
