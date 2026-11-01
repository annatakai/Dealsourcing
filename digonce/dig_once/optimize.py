"""Budget Agent part 2: budget-constrained selection with OR-Tools (SCIP MIP).

Options: replace pipe i alone, treat road j alone, or a joint bundle k=(i,j)
at the bundle's joint cost. Each asset is covered by at most one chosen option
(and in at most one period). Maximise discounted benefit subject to a budget
per period.

    max  sum_{o,t} b_o * delta_t * x_{o,t}
    s.t. sum_o c_o * x_{o,t} <= B_t                 for every period t
         sum_t sum_{o covers a} x_{o,t} <= 1        for every asset a
         x binary,  delta_t = 1/(1+r)^t

Benefit b_o is supplied by the caller (default: priority_score * length/100,
i.e. "priority-weighted 100 m units"). It is a decision index, not money.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from ortools.linear_solver import pywraplp
except ImportError:  # pragma: no cover
    pywraplp = None


def build_options(pipe_scores: pd.DataFrame, pipes: pd.DataFrame, road_scores: pd.DataFrame | None = None,
                  roads: pd.DataFrame | None = None, coordination: pd.DataFrame | None = None,
                  road_benefit_weight: float = 1.0, min_cos: float = 1e-9) -> pd.DataFrame:
    """Assemble the option table (option_id, kind, covers, cost, benefit)."""
    rows = []
    plen = pipes.set_index("pipe_id")["length_m"].fillna(0)
    pb = {}
    for r in pipe_scores.itertuples():
        b = r.priority_score * plen.get(r.pipe_id, 0) / 100.0
        pb[r.pipe_id] = b
        rows.append(("pipe:" + r.pipe_id, "pipe", (f"p:{r.pipe_id}",), r.replacement_cost, b))
    rb = {}
    if road_scores is not None and roads is not None:
        rlen = roads.set_index("road_id")["length_m"].fillna(0)
        for r in road_scores.itertuples():
            b = road_benefit_weight * r.priority_score * rlen.get(r.road_id, 0) / 100.0
            rb[r.road_id] = b
            rows.append(("road:" + r.road_id, "road", (f"r:{r.road_id}",), r.repair_cost, b))
    if coordination is not None and len(coordination):
        for r in coordination[coordination["coordination_score"] > min_cos].itertuples():
            rows.append((f"joint:{r.pipe_id}|{r.road_id}", "joint", (f"p:{r.pipe_id}", f"r:{r.road_id}"),
                         r.joint_cost + r.timing_cost, pb.get(r.pipe_id, 0) + rb.get(r.road_id, 0)))
    opt = pd.DataFrame(rows, columns=["option_id", "kind", "covers", "cost", "benefit"])
    return opt[np.isfinite(opt["cost"]) & (opt["cost"] > 0)].reset_index(drop=True)


def optimize_maintenance_budget(options: pd.DataFrame, budget: float, periods: int = 1,
                                discount_rate: float = 0.04, time_limit_s: float = 30.0,
                                solver_name: str = "SCIP", mip_gap: float = 1e-4) -> tuple[pd.DataFrame, dict]:
    """Select options under a per-period ``budget`` (OR-Tools MIP). Returns (selected, summary).

    SCIP (bundled with OR-Tools) solves this knapsack-with-coverage MIP far
    faster than pure constraint programming because of its LP bound. Falls back
    to a benefit/cost greedy if OR-Tools is unavailable.
    """
    if options.empty:
        return options.assign(period=[]), {"status": "EMPTY", "total_cost": 0.0, "total_benefit": 0.0}
    if pywraplp is None:
        return greedy_budget(options, budget)
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        return greedy_budget(options, budget)
    solver.SetTimeLimit(int(time_limit_s * 1000))
    n = len(options)
    cost = options["cost"].to_numpy(dtype=float)
    ben = options["benefit"].to_numpy(dtype=float)
    x = {(o, t): solver.BoolVar(f"x{o}_{t}") for o in range(n) for t in range(periods)}
    scale = max(1.0, float(np.median(cost)))          # numerical conditioning
    for t in range(periods):
        solver.Add(solver.Sum([(cost[o] / scale) * x[o, t] for o in range(n)]) <= budget / scale)
    cover: dict[str, list[int]] = {}
    for o, cov in enumerate(options["covers"]):
        for a in cov:
            cover.setdefault(a, []).append(o)
    for a, opts in cover.items():
        if len(opts) * periods > 1:
            solver.Add(solver.Sum([x[o, t] for o in opts for t in range(periods)]) <= 1)
    disc = [1 / (1 + discount_rate) ** t for t in range(periods)]
    solver.Maximize(solver.Sum([ben[o] * disc[t] * x[o, t] for o in range(n) for t in range(periods)]))
    params = pywraplp.MPSolverParameters()
    params.SetDoubleParam(params.RELATIVE_MIP_GAP, mip_gap)
    status = solver.Solve(params)
    names = {pywraplp.Solver.OPTIMAL: "OPTIMAL", pywraplp.Solver.FEASIBLE: "FEASIBLE",
             pywraplp.Solver.INFEASIBLE: "INFEASIBLE", pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED"}
    ok = status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
    chosen = [(o, t) for (o, t), v in x.items() if ok and v.solution_value() > 0.5]
    sel = options.iloc[[o for o, _ in chosen]].copy()
    sel["period"] = [t for _, t in chosen]
    obj = solver.Objective().Value() if ok else 0.0
    bound = solver.Objective().BestBound() if ok else 0.0
    summary = {
        "status": names.get(status, str(status)), "solver": solver_name,
        "relative_gap": float(abs(bound - obj) / max(1e-9, abs(obj))) if ok else None,
        "budget_per_period": float(budget), "periods": periods,
        "total_cost": float(sel["cost"].sum()), "total_benefit": float(sel["benefit"].sum()),
        "n_selected": int(len(sel)), "n_joint": int((sel["kind"] == "joint").sum()),
        "n_pipe": int((sel["kind"] == "pipe").sum()), "n_road": int((sel["kind"] == "road").sum()),
    }
    if not ok:
        sel, g = greedy_budget(options, budget)
        summary.update({"fallback": "greedy", **{k: g[k] for k in ("total_cost", "total_benefit", "n_selected")}})
    return sel.sort_values(["period", "benefit"], ascending=[True, False]), summary


def greedy_budget(options: pd.DataFrame, budget: float, key: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Benefit/cost greedy (fallback and baseline)."""
    order = options.assign(_r=options["benefit"] / options["cost"] if key is None else options[key])
    order = order.sort_values("_r", ascending=False)
    used: set[str] = set()
    spent, picks = 0.0, []
    for idx, r in order.iterrows():
        if spent + r["cost"] > budget or any(a in used for a in r["covers"]):
            continue
        picks.append(idx)
        used.update(r["covers"])
        spent += r["cost"]
    sel = options.loc[picks].assign(period=0)
    return sel, {"status": "GREEDY", "total_cost": float(spent), "total_benefit": float(sel["benefit"].sum()),
                 "n_selected": len(sel), "n_joint": int((sel["kind"] == "joint").sum())}
