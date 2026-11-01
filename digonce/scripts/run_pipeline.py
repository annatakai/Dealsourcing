"""End-to-end Dig Once pipeline.

Real data (run on the machine that holds the files):
    python scripts/run_pipeline.py --pipes data/pipe_condition.csv --out outputs/run1
    python scripts/run_pipeline.py --pipes ~/Desktop/Water_Mains_*.csv \
        --breaks ~/Desktop/Water_Main_Breaks_*.csv --roads ~/Desktop/Roads.csv \
        --column-map configs/column_map.yaml --out outputs/city

Synthetic demo (no real data needed; results are NOT findings):
    python scripts/run_pipeline.py --synthetic 20000 --out outputs/example
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dig_once import synthetic  # noqa: E402
from dig_once.config import cost_basis, load_config  # noqa: E402
from dig_once.coordination import calculate_coordination_opportunity, match_pipes_to_roads  # noqa: E402
from dig_once.costs import estimate_pipe_replacement_cost, estimate_road_repair_cost  # noqa: E402
from dig_once.export import export_scores, write_json  # noqa: E402
from dig_once.optimize import build_options, optimize_maintenance_budget  # noqa: E402
from dig_once.pipe import (CLASSES, PipeConditionModel, baseline_metrics, score_pipe_condition,  # noqa: E402
                           score_pipe_priority)
from dig_once.road import score_road_condition, score_road_priority  # noqa: E402
from dig_once.sampling import select_representative  # noqa: E402
from dig_once.schema import attach_breaks_to_pipes, clean_breaks, clean_pipes, clean_roads  # noqa: E402
from dig_once.validation import compare_strategies, cost_sensitivity, weight_sensitivity  # noqa: E402


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipes")
    ap.add_argument("--breaks")
    ap.add_argument("--roads")
    ap.add_argument("--config")
    ap.add_argument("--column-map", help="YAML: {pipes: {canonical: source}, roads: {...}, breaks: {...}}")
    ap.add_argument("--diameter-unit", default="auto", choices=["auto", "mm", "in"])
    ap.add_argument("--length-unit", default="m", choices=["m", "ft"])
    ap.add_argument("--synthetic", type=int, default=0, help="generate N synthetic pipes instead")
    ap.add_argument("--sample-size", type=int, default=500)
    ap.add_argument("--budget-fraction", type=float, default=None)
    ap.add_argument("--sensitivity-draws", type=int, default=50)
    ap.add_argument("--out", default="outputs/run")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cmap = {}
    if a.column_map:
        import yaml
        cmap = yaml.safe_load(open(a.column_map, encoding="utf-8")) or {}
    report: dict = {"data_source": "SYNTHETIC (not real infrastructure)" if a.synthetic else "user files",
                    "cost_basis": cost_basis(cfg), "reference_year": cfg["reference_year"]}

    # ------------------------------------------------------------ load
    if a.synthetic:
        raw_p = synthetic.make_pipes(a.synthetic, seed=cfg["random_seed"], ref_year=cfg["reference_year"])
        raw_r = synthetic.make_roads(400, seed=cfg["random_seed"] + 1, ref_year=cfg["reference_year"])
        raw_p = synthetic.place_on_roads(raw_p, raw_r, seed=cfg["random_seed"] + 2)
        raw_r = raw_r.drop(columns=["_x0", "_y"])
        raw_b = None
    else:
        raw_p = pd.read_csv(a.pipes, low_memory=False)
        raw_r = pd.read_csv(a.roads, low_memory=False) if a.roads else None
        raw_b = pd.read_csv(a.breaks, low_memory=False) if a.breaks else None
    pipes = clean_pipes(raw_p, cfg, cmap.get("pipes"), a.diameter_unit, a.length_unit)
    if raw_b is not None:
        pipes = attach_breaks_to_pipes(pipes, clean_breaks(raw_b, cfg, cmap.get("breaks")))
    if "status" in pipes:
        active = ~pipes["status"].astype(str).str.lower().isin(["abandoned", "removed", "inactive", "retired"])
        report["excluded_inactive"] = int((~active).sum())
        pipes = pipes[active]
    log(f"pipes: {len(pipes):,} rows; condition counts {pipes['condition'].value_counts(dropna=False).to_dict()}")
    report["n_pipes"] = len(pipes)
    report["dq_flag_counts"] = (pipes["dq_flags"].str.split(";").explode().replace("", np.nan).dropna()
                                .value_counts().to_dict())

    # ---------------------------------------------- 1-2 model + evaluation
    rated = pipes["condition"].isin(CLASSES)
    proba_oof = None
    proba_all = None
    if rated.sum() >= 200 and pipes.loc[rated, "condition"].nunique() == 3:
        model = PipeConditionModel(seed=cfg["random_seed"])
        log("cross-validating condition model (stratified 5-fold)")
        proba_oof, m_strat = model.cross_validate(pipes, 5)
        report["model_metrics_stratified_cv"] = m_strat
        if pipes["region"].nunique() >= 3:
            log("cross-validating condition model (grouped by region = spatial transfer)")
            _, m_group = model.cross_validate(pipes, 5, group_col="region")
            report["model_metrics_region_grouped_cv"] = m_group
        report["baselines"] = baseline_metrics(pipes, cfg["random_seed"])
        model.fit(pipes)
        proba_all = model.predict_proba(pipes)
        # rated pipes get OOF predictions (honest), unrated get full-model predictions
        proba_all.loc[proba_oof.index] = proba_oof
        shap = model.explain(pipes)
        report["model_global_importance_mean_abs_shap_red"] = shap.abs().mean().sort_values(ascending=False).to_dict()
        log(f"model red AUC={m_strat.get('red_roc_auc', float('nan')):.3f} "
            f"macroF1={m_strat['macro_f1']:.3f} QWK={m_strat['quadratic_weighted_kappa']:.3f}")
    else:
        report["model"] = "skipped: fewer than 200 rated pipes or missing classes; age proxy used"

    # ---------------------------------------------- 3 score, 5 cost, 6 rank
    cond = score_pipe_condition(pipes, cfg, proba_all)
    log("estimating replacement costs (Monte Carlo)")
    costs = estimate_pipe_replacement_cost(pipes, cfg)
    scores = score_pipe_priority(pipes, cfg, condition=cond, costs=costs, explanations=False)
    scores = scores.join(costs[["cost_p10", "cost_p50", "cost_p90", "cost_flags"]])
    base_cols = [c for c in ["material", "diameter_mm", "length_m", "age_years", "install_year", "region"] if c in pipes]
    full = pipes[base_cols].join(scores)
    full = full.sort_values("priority_rank")
    export_scores(full, out, "pipe_scores_all", {"data_source": report["data_source"], "cost_basis": cost_basis(cfg)})
    log(f"exported {len(full):,} pipe scores")

    # ---------------------------------------------- 4 prototype sample
    sample = select_representative(pipes, a.sample_size, seed=cfg["random_seed"])
    s_scores = score_pipe_priority(sample, cfg, condition=cond.loc[sample.index], costs=costs.loc[sample.index])
    s_scores = s_scores.join(costs.loc[sample.index, ["cost_p10", "cost_p50", "cost_p90"]])
    s_out = sample[base_cols + ["sample_weight", "stratum"]].join(s_scores).sort_values("priority_score", ascending=False)
    s_out["prototype_rank"] = np.arange(1, len(s_out) + 1)
    export_scores(s_out, out, "pipe_scores_sample500", {"data_source": report["data_source"],
                                                        "cost_basis": cost_basis(cfg), "sampling": "stratified"})
    s_out.head(10).to_csv(out / "example_top10_pipes.csv", index=False)

    # ---------------------------------------------- 8 validation
    if proba_oof is not None:
        log("comparing strategies (label-blind)")
        comp = compare_strategies(pipes, cfg, proba_oof, a.budget_fraction)
        comp.to_csv(out / "validation_strategies.csv", index=False)
        report["validation_strategies"] = comp.to_dict(orient="records")
        sens_idx = pipes.sample(min(len(pipes), 20000), random_state=0).index
        log(f"weight sensitivity on {len(sens_idx):,} pipes, {a.sensitivity_draws} draws")
        mc, oat = weight_sensitivity(pipes.loc[sens_idx], cfg, cond.loc[sens_idx], costs.loc[sens_idx],
                                     draws=a.sensitivity_draws, top_k=min(500, len(sens_idx) // 10))
        mc.to_csv(out / "sensitivity_weights_mc.csv")
        oat.to_csv(out / "sensitivity_weights_oneatatime.csv", index=False)
        report["sensitivity_weights_mc"] = mc.to_dict()
        budget = (a.budget_fraction or cfg["validation"]["budget_fraction"]) * costs.loc[sens_idx, "cost_point"].sum()
        cs = cost_sensitivity(pipes.loc[sens_idx], cfg, cond.loc[sens_idx], budget)
        cs.to_csv(out / "sensitivity_costs.csv", index=False)
        report["sensitivity_costs"] = cs.to_dict(orient="records")

    # ---------------------------------------------- roads + coordination
    if raw_r is not None:
        roads = clean_roads(raw_r, cfg, cmap.get("roads"), a.length_unit)
        rcond = score_road_condition(roads, cfg)
        rcost = estimate_road_repair_cost(roads.join(rcond[["pci_effective"]]), cfg)
        rscores = score_road_priority(roads, cfg, condition=rcond, costs=rcost)
        export_scores(roads[["street_name"]].join(rscores) if "street_name" in roads else rscores,
                      out, "road_scores", {"data_source": report["data_source"], "cost_basis": cost_basis(cfg)})
        rscores.sort_values("priority_score", ascending=False).head(10).to_csv(out / "example_top10_roads.csv", index=False)
        report["n_roads"] = len(roads)
        pairs = match_pipes_to_roads(pipes, roads)
        report["n_candidate_pairs"] = len(pairs)
        report["pair_match_method"] = pairs["match_method"].value_counts().to_dict() if len(pairs) else {}
        if len(pairs):
            full_scores = score_pipe_priority(pipes, cfg, condition=cond, costs=costs, explanations=False)
            full_scores = full_scores.assign(pipe_id=pipes["pipe_id"])
            coord = calculate_coordination_opportunity(pairs, pipes, full_scores, roads, rscores, cfg)
            coord = coord.sort_values("coordination_score", ascending=False)
            export_scores(coord, out, "coordination_pairs", {"cost_basis": cost_basis(cfg)})
            coord.head(10).to_csv(out / "example_top10_coordination.csv", index=False)
            report["coordination_gate_counts"] = coord["gate_reason"].replace("", "PASSED").value_counts().to_dict()
            # Budget Agent: optimise pipes on matched streets + roads + bundles
            in_scope = set(coord["pipe_id"])
            sub = pipes[pipes["pipe_id"].isin(in_scope)]
            opts = build_options(full_scores[full_scores["pipe_id"].isin(in_scope)], sub, rscores, roads, coord)
            budget = 0.25 * opts.loc[opts["kind"] != "joint", "cost"].sum()
            sel, summ = optimize_maintenance_budget(opts, budget, periods=1, time_limit_s=30)
            sel.to_csv(out / "budget_selection.csv", index=False)
            report["budget_optimisation"] = summ
            log(f"budget optimisation: {summ}")
        else:
            report["coordination"] = ("no pairs: tables lack shared geometry or street names "
                                      "(anonymised pipes are never matched to streets)")

    write_json(report, out / "run_report.json")
    log(f"done -> {out}")
    return report


if __name__ == "__main__":
    main()
