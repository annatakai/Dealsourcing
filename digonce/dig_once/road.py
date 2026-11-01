"""Road Agent: pavement condition and maintenance priority.

    Condition   PCI_eff = observed PCI (ASTM D6433 scale), aged forward if stale,
                          or estimated from years since resurfacing if missing,
                          minus a capped penalty for potholes reported AFTER the
                          survey (damage the PCI could not yet see).
                Dr      = 100 - PCI_eff                         (deficiency, 0-100)
    Criticality Cr      = weighted mean of traffic (log AADT), heavy vehicles
                          (log heavy AADT), functional class, emergency route and
                          closure disruption (detour), renormalised if missing.
    Risk        Rr      = sqrt(Dr * Cr)
    Economic    Er      = percentile of (PCI gain * service life * lane-km * traffic
                          weight) per unit cost      (relative index, not money)
    Priority    Pr      = w_R*Rr + w_E*Er

Climate and heavy traffic affect the deterioration RATE only when PCI has to be
estimated; with an observed, current PCI they are already reflected in it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import cost_basis
from .costs import estimate_road_repair_cost
from .utils import log1p_scale, weighted_nanmean, window_from_score


def fit_pci_deterioration(roads: pd.DataFrame, cfg: dict, min_obs: int = 30) -> dict[str, float]:
    """Estimate linear PCI loss/year per functional class from the data itself.

    Uses rows with PCI, pci_year and last_resurface_year: rate = (100 - PCI)/age.
    Returns {class: rate, 'default': rate}; falls back to the configured default.
    """
    default = cfg["road"]["default_pci_loss_per_year"]
    need = {"pci", "pci_year", "last_resurface_year"}
    if not need.issubset(roads.columns):
        return {"default": default, "_source": "config_default"}
    age = roads["pci_year"] - roads["last_resurface_year"]
    ok = roads["pci"].notna() & (age >= 1)
    if ok.sum() < min_obs:
        return {"default": default, "_source": "config_default"}
    rate = ((100 - roads.loc[ok, "pci"]) / age[ok]).clip(0, 20)
    out = {"default": float(rate.median()), "_source": "fitted_from_data"}
    cls = roads.loc[ok, "functional_class"].fillna("other")
    for c, grp in rate.groupby(cls):
        if len(grp) >= min_obs:
            out[str(c)] = float(grp.median())
    return out


def score_road_condition(roads: pd.DataFrame, cfg: dict, rates: dict | None = None) -> pd.DataFrame:
    rc = cfg["road"]
    ref = cfg["reference_year"]
    rates = rates or fit_pci_deterioration(roads, cfg)
    base_rate = roads["functional_class"].map({k: v for k, v in rates.items() if not k.startswith("_")}
                                              ).fillna(rates["default"])
    climate = roads["climate_index"].fillna(0).clip(0, 1) if "climate_index" in roads else 0
    rate = base_rate * (1 + (rc["climate_rate_multiplier_max"] - 1) * climate)

    out = pd.DataFrame(index=roads.index)
    pci = roads["pci"].astype(float)
    pci_age = (ref - roads["pci_year"]).clip(lower=0)
    stale = pci.notna() & (pci_age > rc["pci_stale_years"])
    aged = (pci - rate * pci_age).clip(0, 100)
    ysr = (ref - roads["last_resurface_year"]).clip(lower=0)
    est = (100 - rate * ysr).clip(0, 100)

    pci_eff = pci.copy()
    source = pd.Series("observed", index=roads.index, dtype=object)
    source[pci.notna() & roads["pci_year"].isna()] = "observed_date_unknown"
    pci_eff[stale] = aged[stale]
    source[stale] = "observed_aged_forward"
    m = pci.isna() & est.notna()
    pci_eff[m] = est[m]
    source[m] = "estimated_from_resurface_age"
    m = pci_eff.isna()
    median = pci_eff.median() if pci_eff.notna().any() else 60.0
    pci_eff[m] = median
    source[m] = "imputed_network_median"

    penalty = pd.Series(0.0, index=roads.index)
    if "pothole_count" in roads:
        per_km = roads["pothole_count"].fillna(0) / (roads["length_m"].fillna(100) / 1000).clip(lower=0.01)
        penalty = (per_km * rc["pothole_pci_penalty_per_count_per_km"]).clip(upper=rc["pothole_pci_penalty_cap"])
    pci_eff = (pci_eff - penalty).clip(0, 100)

    out["pci_observed"] = pci
    out["pci_effective"] = pci_eff
    out["pci_source"] = source
    out["pci_rate_per_year"] = rate
    out["pothole_penalty"] = penalty
    out["years_since_resurface"] = ysr
    out["condition_deficiency_score"] = 100 - pci_eff
    out["pci_rate_source"] = rates.get("_source", "config_default")
    return out


def criticality_score(roads: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rc = cfg["road"]
    comp = pd.DataFrame(index=roads.index)
    aadt = roads["aadt"].astype(float)
    imputed = aadt.isna()
    if aadt.notna().any():
        by_class = aadt.groupby(roads["functional_class"].fillna("other")).transform("median")
        aadt = aadt.fillna(by_class)
    out_flag = np.where(imputed & aadt.notna(), "aadt_imputed_class_median",
                        np.where(aadt.isna(), "aadt_missing_component_dropped", ""))
    ref = np.nanpercentile(aadt, 99) if aadt.notna().any() else 1.0
    comp["traffic"] = pd.Series(log1p_scale(aadt, ref), index=roads.index).where(aadt.notna())
    heavy = aadt * roads["heavy_pct"] / 100.0
    if heavy.notna().any():
        comp["heavy"] = pd.Series(log1p_scale(heavy, np.nanpercentile(heavy, 99)), index=roads.index
                                  ).where(heavy.notna())
    else:
        comp["heavy"] = np.nan
    comp["road_class"] = roads["functional_class"].map(rc["road_class_scores"])
    comp["emergency"] = roads["emergency_route"].map({True: 100.0, False: 0.0, 1.0: 100.0, 0.0: 0.0})
    comp["disruption"] = 100 * np.clip(roads["detour_km"] / rc["detour_ref_km"], 0, 1)
    score, coverage = weighted_nanmean(comp, rc["criticality_weights"])
    comp["criticality_score"] = score.fillna(50.0)
    comp["criticality_coverage"] = coverage
    comp["aadt_used"] = aadt
    comp["criticality_flag"] = out_flag
    comp.loc[score.isna(), "criticality_flag"] = "no_criticality_data_neutral_50"
    return comp


def score_road_priority(roads: pd.DataFrame, cfg: dict, condition: pd.DataFrame | None = None,
                        costs: pd.DataFrame | None = None, explanations: bool = True) -> pd.DataFrame:
    rc = cfg["road"]
    if condition is None:
        condition = score_road_condition(roads, cfg)
    if costs is None:
        costs = estimate_road_repair_cost(roads.join(condition[["pci_effective"]]), cfg, n_draws=0)
    out = pd.DataFrame(index=roads.index)
    out["road_id"] = roads["road_id"]
    out = out.join(condition)
    crit = criticality_score(roads, cfg)
    out = out.join(crit.add_prefix("crit_").rename(columns={
        "crit_criticality_score": "criticality_score", "crit_criticality_coverage": "criticality_coverage",
        "crit_criticality_flag": "criticality_flag", "crit_aadt_used": "aadt_used"}))
    out["risk_score"] = np.sqrt(out["condition_deficiency_score"] * out["criticality_score"])
    out["treatment"] = costs["treatment"].values
    out["repair_cost"] = costs["cost_point"].values
    out["cost_basis"] = cost_basis(cfg)
    gain = (costs["pci_after"].values - out["pci_effective"]).clip(lower=0)
    lanes = roads["lanes"].fillna(2) if "lanes" in roads else 2
    traffic_w = 1 + np.log1p(out["aadt_used"].fillna(out["aadt_used"].median() if out["aadt_used"].notna().any() else 0)) / 10
    benefit = gain * costs["treatment_life_years"].values * (roads["length_m"].fillna(0) / 1000) * lanes * traffic_w
    out["benefit_pci_lane_km_years"] = benefit
    out["economic_score"] = (100 * (benefit / out["repair_cost"]).rank(pct=True)).fillna(0)
    w = rc["priority_weights"]
    out["priority_risk_part"] = w["risk"] * out["risk_score"]
    out["priority_econ_part"] = w["economic"] * out["economic_score"]
    out["priority_score"] = (out["priority_risk_part"] + out["priority_econ_part"]).round(2)
    out["priority_rank"] = out["priority_score"].rank(ascending=False, method="first").astype(int)
    labels, years = window_from_score(out["condition_deficiency_score"], rc["action_windows"])
    out["action_window"] = labels
    out["target_year_offset"] = years
    out["dq_flags"] = roads.get("dq_flags", "")
    if explanations:
        expl = []
        for idx in out.index:
            r = out.loc[idx]
            parts = [f"{c[5:]} {r[c]:.0f}" for c in out.columns
                     if c.startswith("crit_") and isinstance(r[c], (float, int, np.floating)) and pd.notna(r[c])]
            expl.append(
                f"PCI_eff={r['pci_effective']:.0f} [{r['pci_source']}"
                + (f", pothole -{r['pothole_penalty']:.0f}" if r["pothole_penalty"] > 0 else "") + "]"
                + f" -> Dr={r['condition_deficiency_score']:.0f} | Cr={r['criticality_score']:.0f} ({', '.join(parts)}"
                + (f"; {r['criticality_flag']}" if r["criticality_flag"] else "") + ")"
                + f" | Risk={r['risk_score']:.0f} | treatment {r['treatment']} | Econ={r['economic_score']:.0f}"
                + f" | Priority={r['priority_score']:.1f} | window {r['action_window']}")
        out["explanation"] = expl
    return out
