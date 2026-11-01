"""Budget Agent part 1: pipe-road pairing and the Coordination Opportunity Score.

Two clearly separated outputs per pipe-road pair:

* ``net_savings``  - an ESTIMATED MONETARY quantity (currency), from explicit
  cost equations: shared trench restoration + shared mobilisation + shared
  traffic control + avoided pavement-cut damage + avoided user delay
  - cost of advancing the less urgent asset (or of delaying the urgent one).
* ``coordination_score`` (COS, 0-100) - a unitless DECISION INDEX:

      COS = 100 * o * tau * sqrt(n_min * s) * (1 - rho)       if gates pass, else 0

      o     = fraction of the pipe beneath the road (overlap)
      tau   = exp(-gap / timing_scale)    scheduling compatibility of target years
      n_min = min(P_pipe, P_road) / 100   the LESS needy asset drives the score
      s     = clip((net_savings / separate_cost) / savings_ratio_ref, 0, 1)
      rho   = (delay / max_delay) * (L_urgent / 100)  if the urgent asset is delayed

  Gates (COS = 0, with reason): no overlap; either asset priority below
  need_min_priority; either target year beyond the horizon; net_savings <= 0.
"""
from __future__ import annotations


import numpy as np
import pandas as pd

from .config import cost_basis
from .costs import pipe_cost_components


def _norm_street(s: pd.Series) -> pd.Series:
    s = s.astype("string").str.upper().str.strip()
    s = s.str.replace(r"[^A-Z0-9 ]", " ", regex=True).str.replace(r"\s+", " ", regex=True)
    for full, abbr in [("STREET", "ST"), ("AVENUE", "AVE"), ("ROAD", "RD"), ("DRIVE", "DR"),
                       ("BOULEVARD", "BLVD"), ("LANE", "LN"), ("COURT", "CT"), ("PLACE", "PL"),
                       ("NORTH", "N"), ("SOUTH", "S"), ("EAST", "E"), ("WEST", "W")]:
        s = s.str.replace(rf"\b{full}\b", abbr, regex=True)
    return s.str.strip()


def match_pipes_to_roads(pipes: pd.DataFrame, roads: pd.DataFrame, buffer_m: float = 3.0,
                         max_roads_per_name: int = 25) -> pd.DataFrame:
    """Find pipe-road pairs that geographically overlap.

    1. Geometry (preferred): both tables carry WKT line geometry in a projected
       CRS (metres). overlap = length(pipe ∩ buffer(road, width/2 + buffer)) / length(pipe).
    2. Street name (fallback, low confidence): same normalised street name.
       overlap_fraction is left NaN and marked ``match_method='street_name'``.
    Anonymised data without geometry or street names cannot be matched and
    returns an empty frame - never invent locations.
    """
    cols = ["pipe_id", "road_id", "overlap_fraction", "match_method"]
    if "geometry_wkt" in pipes and "geometry_wkt" in roads and pipes["geometry_wkt"].notna().any():
        try:
            from shapely import wkt
            from shapely.strtree import STRtree
        except ImportError:  # pragma: no cover
            wkt = None
        if wkt is not None:
            r = roads.dropna(subset=["geometry_wkt"])
            widths = r["width_m"].fillna(7.0) if "width_m" in r else pd.Series(7.0, index=r.index)
            polys = [wkt.loads(g).buffer(w / 2 + buffer_m) for g, w in zip(r["geometry_wkt"], widths)]
            tree = STRtree(polys)
            rows = []
            for pid, g in zip(pipes["pipe_id"], pipes["geometry_wkt"]):
                if pd.isna(g):
                    continue
                line = wkt.loads(g)
                if line.length <= 0:
                    continue
                for j in tree.query(line):
                    frac = line.intersection(polys[j]).length / line.length
                    if frac > 0.05:
                        rows.append((pid, r["road_id"].iloc[j], min(frac, 1.0), "geometry"))
            return pd.DataFrame(rows, columns=cols)
    if "street_name" in pipes and "street_name" in roads:
        p = pipes[["pipe_id"]].assign(key=_norm_street(pipes["street_name"])).dropna(subset=["key"])
        r = roads[["road_id"]].assign(key=_norm_street(roads["street_name"])).dropna(subset=["key"])
        counts = r["key"].value_counts()
        r = r[r["key"].map(counts) <= max_roads_per_name]
        pairs = p.merge(r, on="key")[["pipe_id", "road_id"]]
        pairs["overlap_fraction"] = np.nan
        pairs["match_method"] = "street_name"
        return pairs
    return pd.DataFrame(columns=cols)


def calculate_coordination_opportunity(pairs: pd.DataFrame, pipes: pd.DataFrame, pipe_scores: pd.DataFrame,
                                       roads: pd.DataFrame, road_scores: pd.DataFrame, cfg: dict,
                                       name_match_overlap: float = 0.5) -> pd.DataFrame:
    """Score every candidate pair. Inputs are indexed by any index; joined on ids."""
    if pairs.empty:
        return pairs.assign(coordination_score=[], net_savings=[])
    cc, c = cfg["coordination"], cfg["costs"]
    P = pipes.set_index("pipe_id")
    PS = pipe_scores.set_index("pipe_id")
    R = roads.set_index("road_id")
    RS = road_scores.set_index("road_id")
    df = pairs.copy().reset_index(drop=True)
    pid, rid = df["pipe_id"], df["road_id"]

    o = df["overlap_fraction"].astype(float)
    o_flag = o.isna()
    o = o.fillna(name_match_overlap).clip(0, 1)
    Lp = P.loc[pid, "length_m"].to_numpy(dtype=float)
    Lo = o.to_numpy() * Lp
    d = P.loc[pid, "diameter_mm"].fillna(P["diameter_mm"].median()).to_numpy(dtype=float)
    Pp = PS.loc[pid, "priority_score"].to_numpy(dtype=float)
    Pr = RS.loc[rid, "priority_score"].to_numpy(dtype=float)
    tp = PS.loc[pid, "target_year_offset"].to_numpy(dtype=float)
    tr = RS.loc[rid, "target_year_offset"].to_numpy(dtype=float)
    Lik_p = PS.loc[pid, "likelihood_score"].to_numpy(dtype=float)
    Dr = RS.loc[rid, "condition_deficiency_score"].to_numpy(dtype=float)
    C_pipe = PS.loc[pid, "replacement_cost"].to_numpy(dtype=float)
    C_road = RS.loc[rid, "repair_cost"].to_numpy(dtype=float)
    life_r = np.where(RS.loc[rid, "treatment"].to_numpy() == "preventive", 7.0, c["resurface_design_life_years"])
    aadt = RS.loc[rid, "aadt_used"].fillna(0).to_numpy(dtype=float)
    Lr = R.loc[rid, "length_m"].fillna(0).to_numpy(dtype=float)

    comp = pipe_cost_components(d, Lp, True, cfg)
    markup = (1 + c["traffic_control_fraction"] + c["overhead_fraction"]) * (1 + c["contingency_fraction"])
    save_restore = comp["restoration"] * o.to_numpy()
    save_mob = np.full(len(df), c["mobilisation_per_project"] * markup)
    direct_pipe_overlap = (comp["supply_lay"] + comp["excavation"]) / markup * o.to_numpy()
    direct_road = (C_road / markup)
    save_tc = c["traffic_control_fraction"] * np.minimum(direct_pipe_overlap, direct_road)
    gap = np.abs(tp - tr)
    # pavement-cut damage avoided: separate plan with road first, pipe later
    restore_area_overlap = comp["restore_area_m2"] * o.to_numpy()
    remaining = np.clip(1 - gap / c["resurface_design_life_years"], 0, 1)
    save_reopen = np.where(tr < tp, restore_area_overlap * c["road_treatment_cost_per_m2"]["mill_overlay"]
                           * c["cut_life_loss_fraction"] * remaining, 0.0)
    days_p = c["pipe_closure_days_per_100m"] * Lo / 100
    days_r = c["road_closure_days_per_100m"] * Lr / 100
    shared_days = c["joint_closure_overlap_fraction"] * np.minimum(days_p, days_r)
    save_delay = shared_days * aadt / 1000 * c["user_delay_cost_per_day_per_1000_aadt"]
    gross = save_restore + save_mob + save_tc + save_reopen + save_delay

    # timing: advance the less urgent asset, or delay the urgent one
    r_ = c["discount_rate"]
    pipe_first = tp <= tr
    adv_cost = np.where(pipe_first,
                        C_road * np.minimum(gap / life_r, 1.0),      # wasted remaining surface life
                        C_pipe * (1 - (1 + r_) ** -gap))             # time value of early capex
    exp_breaks = PS.loc[pid, "expected_breaks_per_year"].fillna(0).to_numpy(dtype=float) \
        if "expected_breaks_per_year" in PS else np.zeros(len(df))
    from .costs import emergency_repair_cost
    delay_cost = np.where(pipe_first, gap * exp_breaks * emergency_repair_cost(d, cfg), np.nan)  # road delay not monetised
    can_delay = gap <= cc["max_delay_years"]
    use_delay = pipe_first & can_delay & np.isfinite(delay_cost) & (delay_cost < adv_cost)
    timing_cost = np.where(use_delay, delay_cost, adv_cost)
    net = gross - timing_cost
    C_sep = C_pipe + C_road + np.where(tr < tp, save_reopen, 0)
    C_joint = C_sep - gross

    tau = np.exp(-gap / cc["timing_scale_years"])
    n_min = np.minimum(Pp, Pr) / 100
    s = np.clip((net / C_sep) / cc["savings_ratio_ref"], 0, 1)
    rho = np.where(use_delay, np.clip(gap / cc["max_delay_years"], 0, 1) * Lik_p / 100, 0.0)
    cos = 100 * o.to_numpy() * tau * np.sqrt(n_min * s) * (1 - rho)

    reason = np.full(len(df), "", dtype=object)
    gates = [
        (o.to_numpy() <= 0, "no_overlap"),
        (Pp < cc["need_min_priority"], "pipe_does_not_need_work"),
        (Pr < cc["need_min_priority"], "road_does_not_need_work"),
        (tp > cc["horizon_years"], "pipe_target_beyond_horizon"),
        (tr > cc["horizon_years"], "road_target_beyond_horizon"),
        (net <= 0, "no_net_savings"),
    ]
    for mask, text in gates:
        reason = np.where(mask & (reason == ""), text, reason)
    cos = np.where(reason == "", cos, 0.0)

    out = df.assign(
        overlap_used=o.to_numpy(), overlap_assumed=o_flag.to_numpy(), overlap_length_m=Lo,
        pipe_priority=Pp, road_priority=Pr, pipe_target_offset=tp, road_target_offset=tr, timing_gap_years=gap,
        joint_timing=np.where(gap == 0, "same_window", np.where(use_delay, "delay_pipe_to_road",
                              np.where(pipe_first, "advance_road", "advance_pipe"))),
        pipe_cost=C_pipe, road_cost=C_road, separate_cost=C_sep, joint_cost=C_joint,
        save_trench_restoration=save_restore, save_mobilisation=save_mob, save_traffic_control=save_tc,
        save_avoided_pavement_cut=save_reopen, save_user_delay=save_delay, gross_savings=gross,
        timing_cost=timing_cost, net_savings=net, net_savings_ratio=net / C_sep,
        tau=tau, need_min=n_min, savings_norm=s, delay_risk=rho, road_deficiency=Dr,
        coordination_score=np.round(cos, 2), gate_reason=reason, cost_basis=cost_basis(cfg),
    )
    out["explanation"] = [
        (f"GATED: {g}" if g else
         f"overlap {ov:.0%}{' (assumed)' if oa else ''}, needs P_pipe={pp:.0f}/P_road={pr:.0f}, "
         f"gap {gp:.0f}y ({jt}), gross savings {gs:,.0f} - timing cost {tc:,.0f} = net {ns:,.0f} "
         f"({nr:.0%} of separate cost) -> COS {cs:.1f}")
        for g, ov, oa, pp, pr, gp, jt, gs, tc, ns, nr, cs in zip(
            out["gate_reason"], out["overlap_used"], out["overlap_assumed"], out["pipe_priority"],
            out["road_priority"], out["timing_gap_years"], out["joint_timing"], out["gross_savings"],
            out["timing_cost"], out["net_savings"], out["net_savings_ratio"], out["coordination_score"])
    ]
    return out
