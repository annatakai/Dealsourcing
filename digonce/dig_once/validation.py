"""Validation: strategy comparison under a budget and sensitivity analysis.

Fairness rule: in ``label_blind`` mode every strategy that uses condition gets
the OUT-OF-FOLD *predicted* condition, and is scored against the OBSERVED
Green/Amber/Red labels. Otherwise any strategy that reads the observed label
trivially "finds" Red pipes. Metrics are about condition targeting and
simulated risk; none of them shows that real failures were prevented (that
requires observed future failure outcomes, see ``time_split_failure_check``).
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import load_config
from .costs import estimate_pipe_replacement_cost
from .optimize import optimize_maintenance_budget
from .pipe import score_pipe_condition, score_pipe_priority


def _select_by_rank(order: pd.Index, cost: pd.Series, budget: float) -> pd.Index:
    c = cost.loc[order].fillna(np.inf).to_numpy()
    take = np.zeros(len(c), dtype=bool)
    spent = 0.0
    for i, ci in enumerate(c):
        if spent + ci <= budget:
            take[i] = True
            spent += ci
    return order[take]


def selection_metrics(sel: pd.Index, pipes: pd.DataFrame, cost: pd.Series, risk_obs: pd.Series,
                      cfg: dict) -> dict:
    states = cfg["pipe"]["condition_state_scores"]
    red = pipes["condition"] == "R"
    km = pipes["length_m"].fillna(0) / 1000
    d_obs = pipes["condition"].map(states).fillna(0)
    s = pipes.index.isin(sel)
    spend = float(cost[s].sum())
    risk_km = risk_obs.fillna(0) * km
    return {
        "n_selected": int(s.sum()),
        "cost": spend,
        "km": float(km[s].sum()),
        "red_recall_count": float((red & s).sum() / max(1, red.sum())),
        "red_recall_length": float(km[red & s].sum() / max(1e-9, km[red].sum())),
        "red_precision": float((red & s).sum() / max(1, s.sum())),
        "condition_pt_km_per_100M": float((d_obs[s] * km[s]).sum() / spend * 1e8) if spend else 0.0,
        "simulated_risk_reduction_share": float(risk_km[s].sum() / max(1e-9, risk_km.sum())),
    }


def compare_strategies(pipes: pd.DataFrame, cfg: dict, proba_oof: pd.DataFrame,
                       budget_fraction: float | None = None, use_optimizer: bool = True,
                       seed: int = 0) -> pd.DataFrame:
    """Compare ranking strategies at a fixed budget (fraction of total replacement cost)."""
    budget_fraction = budget_fraction or cfg["validation"]["budget_fraction"]
    rated = pipes[pipes["condition"].isin(["G", "A", "R"])].copy()
    costs = estimate_pipe_replacement_cost(rated, cfg, n_draws=0)
    cost = costs["cost_point"]
    budget = budget_fraction * float(cost.sum())

    blind_cond = score_pipe_condition(rated, cfg, proba_oof.reindex(rated.index), use_observed=False)
    blind = score_pipe_priority(rated, cfg, condition=blind_cond, costs=costs, explanations=False)
    aware = score_pipe_priority(rated, cfg, condition=score_pipe_condition(rated, cfg, proba_oof.reindex(rated.index)),
                                costs=costs, explanations=False)
    risk_obs = aware["risk_score"]   # "truth-side" risk for the simulation metric
    km = rated["length_m"].fillna(0) / 1000

    rank = {}
    rank["random"] = rated.sample(frac=1, random_state=seed).index
    rank["oldest_first"] = rated["age_years"].fillna(-1).sort_values(ascending=False).index
    rank["worst_predicted_condition_first"] = blind_cond["condition_score"].sort_values(ascending=False).index
    if "n_breaks" in rated and rated["n_breaks"].notna().any():
        rank["break_history_first"] = blind["break_rate_hat"].fillna(0).sort_values(ascending=False).index
    rank["cost_effectiveness"] = (blind_cond["condition_score"] * km / cost).sort_values(ascending=False).index
    rank["proposed_priority"] = blind["priority_score"].sort_values(ascending=False).index
    # budget-aware: priority-weighted km per unit cost (greedy knapsack approximation)
    rank["proposed_priority_per_cost"] = (blind["priority_score"] * km / cost).sort_values(ascending=False).index

    rows = []
    for name, order in rank.items():
        sel = _select_by_rank(order, cost, budget)
        rows.append({"strategy": name, "mode": "label_blind", **selection_metrics(sel, rated, cost, risk_obs, cfg)})
    if use_optimizer:
        opts = pd.DataFrame({"option_id": rated["pipe_id"].values, "kind": "pipe",
                             "covers": [(p,) for p in rated["pipe_id"]], "cost": cost.values,
                             "benefit": (blind["priority_score"] * rated["length_m"].fillna(0) / 100).values})
        opts = opts[np.isfinite(opts["cost"])]
        chosen, summ = optimize_maintenance_budget(opts, budget, time_limit_s=20)
        sel = rated.index[rated["pipe_id"].isin(chosen["option_id"])]
        rows.append({"strategy": "budget_optimized_priority", "mode": "label_blind",
                     **selection_metrics(sel, rated, cost, risk_obs, cfg), "solver": summ["status"]})
    # reference only: strategies that read the observed label
    ref = {
        "worst_observed_condition_first(reference)":
            rated.assign(_s=rated["condition"].map({"R": 2, "A": 1, "G": 0}))
                 .sort_values(["_s", "age_years"], ascending=False).index,
        "proposed_priority_observed(reference)": aware["priority_score"].sort_values(ascending=False).index,
    }
    for name, order in ref.items():
        sel = _select_by_rank(order, cost, budget)
        rows.append({"strategy": name, "mode": "label_aware", **selection_metrics(sel, rated, cost, risk_obs, cfg)})
    out = pd.DataFrame(rows)
    out["budget"] = budget
    out["budget_fraction"] = budget_fraction
    return out


def _perturb(weights: dict, rng, p: float) -> dict:
    w = {k: v * rng.uniform(1 - p, 1 + p) for k, v in weights.items()}
    s = sum(w.values())
    return {k: v / s for k, v in w.items()}


def weight_sensitivity(pipes: pd.DataFrame, cfg: dict, condition: pd.DataFrame, costs: pd.DataFrame,
                       draws: int | None = None, top_k: int = 500, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monte-Carlo weight perturbation + one-at-a-time (tornado) sweeps.

    Returns (summary of random draws, one-at-a-time table). Stability metrics:
    Spearman rank correlation vs base, top-K overlap, Red share in top-K.
    """
    rng = np.random.default_rng(seed)
    draws = draws or cfg["validation"]["sensitivity_draws"]
    p = cfg["validation"]["weight_perturbation"]
    base = score_pipe_priority(pipes, cfg, condition=condition, costs=costs, explanations=False)["priority_score"]
    top_base = set(base.nlargest(top_k).index)
    red = pipes.get("condition") == "R"

    def stats(score):
        top = set(score.nlargest(top_k).index)
        return {"spearman": float(spearmanr(base, score).statistic),
                "topk_overlap": len(top & top_base) / top_k,
                "topk_red_share": float(red.loc[list(top)].mean()) if red is not None else np.nan}

    groups = [("pipe", "likelihood_weights"), ("pipe", "consequence_weights"), ("pipe", "priority_weights")]
    rows = []
    for _ in range(draws):
        c2 = copy.deepcopy(cfg)
        for sec, key in groups:
            c2[sec][key] = _perturb(cfg[sec][key], rng, p)
        rows.append(stats(score_pipe_priority(pipes, c2, condition=condition, costs=costs,
                                              explanations=False)["priority_score"]))
    mc = pd.DataFrame(rows)
    oat = []
    for sec, key in groups:
        for w in cfg[sec][key]:
            for f in (0.5, 1.5):
                c2 = copy.deepcopy(cfg)
                c2[sec][key][w] = cfg[sec][key][w] * f
                s = sum(c2[sec][key].values())
                c2[sec][key] = {k: v / s for k, v in c2[sec][key].items()}
                oat.append({"weight": f"{key}.{w}", "factor": f, **stats(
                    score_pipe_priority(pipes, c2, condition=condition, costs=costs,
                                        explanations=False)["priority_score"])})
    # The ordinal G/A/R -> 0/50/100 mapping is itself an assumption: sweep the Amber value.
    for a_val in (30.0, 70.0):
        states = {**cfg["pipe"]["condition_state_scores"], "A": a_val}
        c2 = copy.deepcopy(cfg)
        c2["pipe"]["condition_state_scores"] = states
        oat.append({"weight": "condition_state_scores.A", "factor": a_val / 50.0, **stats(
            score_pipe_priority(pipes, c2, condition=_remap_condition(condition, states),
                                costs=costs, explanations=False)["priority_score"])})
    return mc.describe().T, pd.DataFrame(oat)


def _remap_condition(condition: pd.DataFrame, states: dict) -> pd.DataFrame:
    """Recompute the condition score under a different G/A/R state mapping."""
    c = condition.copy()
    obs = c["condition_observed"].map(states).astype(float)
    if {"p_A", "p_R"}.issubset(c.columns):
        pred = states["A"] * c["p_A"] + states["R"] * c["p_R"]
    else:
        pred = c["condition_pred_score"]
    w = c["observation_weight"]
    is_obs = c["condition_source"].astype(str).str.startswith("observed")
    blended = (w * obs + (1 - w) * pred.fillna(obs)).where(w < 0.999, obs)
    c["condition_score"] = np.where(is_obs, blended, np.where(c["condition_source"] == "predicted", pred,
                                                               c["condition_score"]))
    return c


def cost_sensitivity(pipes: pd.DataFrame, cfg: dict, condition: pd.DataFrame, budget: float,
                     scenarios: dict | None = None) -> pd.DataFrame:
    """Re-select under alternative cost assumptions at a FIXED budget."""
    scenarios = scenarios or {
        "base": {},
        "all_unit_costs_x0.7": {"_scale": 0.7},
        "all_unit_costs_x1.5": {"_scale": 1.5},
        "pipe_cost_exponent_0.7": {"pipe_unit_cost_b": 0.7},
        "pipe_cost_exponent_1.2": {"pipe_unit_cost_b": 1.2},
        "restoration_x2": {"trench_restore_cost_per_m2_paved": cfg["costs"]["trench_restore_cost_per_m2_paved"] * 2},
    }
    scale_keys = ["pipe_unit_cost_a", "excavation_cost_per_m3", "trench_restore_cost_per_m2_paved",
                  "trench_restore_cost_per_m2_unpaved", "mobilisation_per_project"]
    base_sel = None
    rows = []
    for name, ov in scenarios.items():
        c2 = copy.deepcopy(cfg)
        for k, v in ov.items():
            if k == "_scale":
                for sk in scale_keys:
                    c2["costs"][sk] *= v
            else:
                c2["costs"][k] = v
        costs = estimate_pipe_replacement_cost(pipes, c2, n_draws=0)
        sc = score_pipe_priority(pipes, c2, condition=condition, costs=costs, explanations=False)
        sel = set(_select_by_rank(sc["priority_score"].sort_values(ascending=False).index, costs["cost_point"], budget))
        base_sel = base_sel if base_sel is not None else sel
        red = pipes["condition"] == "R"
        rows.append({"scenario": name, "n_selected": len(sel),
                     "jaccard_vs_base": len(sel & base_sel) / max(1, len(sel | base_sel)),
                     "red_selected": int(red.loc[list(sel)].sum()),
                     "total_network_cost": float(costs["cost_point"].sum())})
    return pd.DataFrame(rows)


def time_split_failure_check(pipes_before: pd.DataFrame, future_breaks: pd.Series, score_col: pd.Series,
                             top_fraction: float = 0.1) -> dict:
    """Only valid test of failure prevention potential: rank with data up to year T,
    then check the share of breaks observed AFTER T that fall in the top fraction."""
    k = max(1, int(len(score_col) * top_fraction))
    top = score_col.nlargest(k).index
    fb = future_breaks.reindex(score_col.index).fillna(0)
    return {"top_fraction": top_fraction, "share_future_breaks_captured": float(fb.loc[top].sum() / max(1, fb.sum())),
            "lift": float((fb.loc[top].sum() / max(1, fb.sum())) / top_fraction)}
