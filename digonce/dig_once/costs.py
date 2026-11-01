"""Transparent cost-estimation equations with Monte-Carlo uncertainty.

All unit costs come from ``cfg["costs"]``. The shipped defaults are DEMO
PLACEHOLDERS (see config.py); every output carries ``cost_basis``.

Pipe replacement (open cut), per segment of length L [m] and diameter D [mm]:

    W      = D/1000 + 2*side_clearance                      trench width [m]
    H      = cover + D/1000                                 trench depth [m]
    c_pipe = a * (D/100) ** b                               supply & lay [cur/m]
    c_exc  = W * H * c_m3                                   excavation [cur/m]
    c_res  = (W + 2*overcut) * c_m2(paved|unpaved)          surface restoration [cur/m]
    direct = L * (c_pipe + c_exc + c_res) + mobilisation
    total  = direct * (1 + f_traffic + f_overhead) * (1 + f_contingency)
           + reopen_penalty

Reopening a recently resurfaced road (pavement-cut damage):

    reopen_penalty = A_restore * c_resurface * life_loss * max(0, 1 - t/T)

    where t = years since resurfacing and T = resurfacing design life.

Road treatment: total = A * c_treatment * (1+f_traffic+f_overhead)*(1+f_cont)
Emergency break repair: c_break = base * (D/100) ** e   (+ social cost)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import cost_basis


def _arr(x, default=np.nan, n=None):
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float)
    a = np.asarray(x, dtype=float)
    if n is not None and a.ndim == 0:
        a = np.full(n, float(a))
    return a


def pipe_cost_components(diameter_mm, length_m, paved=True, cfg=None,
                         years_since_resurface=None, multipliers=None) -> dict[str, np.ndarray]:
    """Vectorised point estimate of each pipe cost component."""
    c = cfg["costs"]
    m = {"pipe_unit": 1.0, "excavation": 1.0, "restoration": 1.0, **(multipliers or {})}
    d = _arr(diameter_mm)
    n = d.shape[0] if d.ndim else None
    length = _arr(length_m, n=n)
    paved = np.broadcast_to(np.asarray(paved, dtype=bool), d.shape) if d.ndim else bool(paved)
    width = d / 1000.0 + 2 * c["trench_side_clearance_m"]
    depth = c["trench_cover_m"] + d / 1000.0
    c_pipe = c["pipe_unit_cost_a"] * (d / 100.0) ** c["pipe_unit_cost_b"] * m["pipe_unit"]
    c_exc = width * depth * c["excavation_cost_per_m3"] * m["excavation"]
    restore_rate = np.where(paved, c["trench_restore_cost_per_m2_paved"], c["trench_restore_cost_per_m2_unpaved"])
    restore_area = (width + 2 * c["trench_restore_overcut_m"]) * length
    c_res = restore_area * restore_rate * m["restoration"]
    markup = (1 + c["traffic_control_fraction"] + c["overhead_fraction"]) * (1 + c["contingency_fraction"])
    reopen = np.zeros_like(d)
    if years_since_resurface is not None:
        t = _arr(years_since_resurface, n=n)
        remaining = np.clip(1 - np.nan_to_num(t, nan=1e9) / c["resurface_design_life_years"], 0, 1)
        resurf_rate = c["road_treatment_cost_per_m2"]["mill_overlay"] * m.get("road_unit", 1.0)
        reopen = restore_area * resurf_rate * c["cut_life_loss_fraction"] * remaining * np.asarray(paved, float)
    return {
        "supply_lay": length * c_pipe * markup,
        "excavation": length * c_exc * markup,
        "restoration": c_res * markup,
        "mobilisation": np.full_like(d, c["mobilisation_per_project"]) * markup,
        "reopen_penalty": reopen,
        "restore_area_m2": restore_area,
    }


def estimate_pipe_replacement_cost(pipes: pd.DataFrame, cfg: dict, paved_col: str | None = None,
                                   n_draws: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Point estimate + P10/P50/P90 replacement cost per pipe segment.

    Missing diameter is imputed with the network median (flagged); missing
    length yields NaN cost (cannot be estimated, flagged).
    """
    rng = np.random.default_rng(cfg["random_seed"] if seed is None else seed)
    d = pipes["diameter_mm"].astype(float)
    d_flag = d.isna()
    d = d.fillna(d.median() if d.notna().any() else 150.0)
    length = pipes["length_m"].astype(float)
    paved = pipes[paved_col].fillna(True).astype(bool).to_numpy() if paved_col and paved_col in pipes else True
    ysr = pipes["years_since_resurface"] if "years_since_resurface" in pipes else None
    comp = pipe_cost_components(d, length, paved, cfg, ysr)
    keys = ["supply_lay", "excavation", "restoration", "mobilisation", "reopen_penalty"]
    point = sum(comp[k] for k in keys)
    out = pd.DataFrame(index=pipes.index)
    out["pipe_id"] = pipes["pipe_id"].values
    out["cost_point"] = point
    for k in keys:
        out[f"cost_{k}"] = comp[k]

    draws = cfg["costs"]["monte_carlo_draws"] if n_draws is None else n_draws
    if draws and draws > 0:
        unc = cfg["costs"]["uncertainty"]
        sims = np.empty((draws, len(pipes)))
        for i in range(draws):
            mult = {k: rng.triangular(unc[k][0], 1.0, unc[k][1]) for k in ["pipe_unit", "excavation", "restoration"]}
            cc = pipe_cost_components(d, length, paved, cfg, ysr, multipliers=mult)
            sims[i] = sum(cc[k] for k in keys)
        out["cost_p10"], out["cost_p50"], out["cost_p90"] = np.nanpercentile(sims, [10, 50, 90], axis=0)
    out["cost_per_m"] = out["cost_point"] / length
    out["cost_basis"] = cost_basis(cfg)
    out["cost_flags"] = np.where(d_flag, "diameter_imputed_median", "")
    out.loc[length.isna().to_numpy(), "cost_flags"] = "length_missing_cost_unavailable"
    return out


def emergency_repair_cost(diameter_mm, cfg: dict) -> np.ndarray:
    c = cfg["costs"]
    d = np.nan_to_num(_arr(diameter_mm), nan=150.0)
    return c["emergency_repair_cost_base"] * (d / 100.0) ** c["emergency_repair_diameter_exponent"] + c["social_cost_per_break"]


def select_treatment(pci, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (treatment name, PCI after, service life) from the PCI decision tree."""
    pci = np.nan_to_num(_arr(pci), nan=50.0)
    names = np.empty(pci.shape, dtype=object)
    after = np.zeros(pci.shape)
    life = np.zeros(pci.shape)
    assigned = np.zeros(pci.shape, dtype=bool)
    for min_pci, name, pci_after, svc in sorted(cfg["road"]["treatments"], key=lambda t: -t[0]):
        m = (pci >= min_pci) & ~assigned
        names[m], after[m], life[m] = name, pci_after, svc
        assigned |= m
    return names, after, life


def estimate_road_repair_cost(roads: pd.DataFrame, cfg: dict, pci_col: str = "pci_effective",
                              n_draws: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Treatment selection + cost per road segment (point and P10/P50/P90)."""
    rng = np.random.default_rng(cfg["random_seed"] if seed is None else seed)
    c = cfg["costs"]
    pci = roads[pci_col] if pci_col in roads else roads["pci"]
    treat, after, life = select_treatment(pci, cfg)
    width = roads["width_m"].astype(float)
    if "lanes" in roads:
        width = width.fillna(roads["lanes"] * cfg["road"]["default_lane_width_m"])
    w_flag = width.isna()
    width = width.fillna(cfg["road"]["default_width_m"])
    area = roads["length_m"].astype(float) * width
    unit = np.array([c["road_treatment_cost_per_m2"][t] for t in treat])
    markup = (1 + c["traffic_control_fraction"] + c["overhead_fraction"]) * (1 + c["contingency_fraction"])
    out = pd.DataFrame(index=roads.index)
    out["road_id"] = roads["road_id"].values
    out["treatment"] = treat
    out["pci_after"] = after
    out["treatment_life_years"] = life
    out["area_m2"] = area.values
    out["cost_point"] = area.values * unit * markup + c["mobilisation_per_project"] * markup
    draws = c["monte_carlo_draws"] if n_draws is None else n_draws
    if draws and draws > 0:
        lo, hi = c["uncertainty"]["road_unit"]
        mult = rng.triangular(lo, 1.0, hi, size=(draws, 1))
        sims = area.values[None, :] * unit[None, :] * markup * mult + c["mobilisation_per_project"] * markup
        out["cost_p10"], out["cost_p50"], out["cost_p90"] = np.nanpercentile(sims, [10, 50, 90], axis=0)
    out["cost_basis"] = cost_basis(cfg)
    out["cost_flags"] = np.where(w_flag, "width_imputed_default", "")
    return out


def fit_cost_function(diameter_mm, unit_cost_per_m) -> dict[str, float]:
    """Fit c = a * (D/100) ** b to local bid data by log-log least squares.

    Returns a, b, R^2 and n. Use the result to replace pipe_unit_cost_a/b.
    """
    d = _arr(diameter_mm)
    y = _arr(unit_cost_per_m)
    ok = (d > 0) & (y > 0) & np.isfinite(d) & np.isfinite(y)
    if ok.sum() < 3:
        raise ValueError("need at least 3 valid (diameter, cost) observations")
    x = np.log(d[ok] / 100.0)
    ly = np.log(y[ok])
    b, log_a = np.polyfit(x, ly, 1)
    pred = log_a + b * x
    ss_res = ((ly - pred) ** 2).sum()
    ss_tot = ((ly - ly.mean()) ** 2).sum()
    return {"a": float(np.exp(log_a)), "b": float(b), "r2": float(1 - ss_res / ss_tot) if ss_tot else 1.0,
            "n": int(ok.sum())}


def annuity_factor(rate: float, years: float) -> float:
    """Present value of 1/yr for ``years`` at ``rate``."""
    if rate == 0:
        return float(years)
    return (1 - (1 + rate) ** -years) / rate
