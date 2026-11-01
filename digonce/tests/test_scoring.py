import numpy as np
import pandas as pd
import pytest

from dig_once import synthetic
from dig_once.config import load_config, normalise_weights
from dig_once.coordination import calculate_coordination_opportunity, match_pipes_to_roads
from dig_once.costs import (estimate_pipe_replacement_cost, estimate_road_repair_cost, fit_cost_function,
                            pipe_cost_components)
from dig_once.optimize import build_options, greedy_budget, optimize_maintenance_budget
from dig_once.pipe import (PipeConditionModel, break_evidence, consequence_score, score_pipe_condition,
                           score_pipe_priority)
from dig_once.road import score_road_condition, score_road_priority
from dig_once.sampling import select_representative
from dig_once.schema import clean_pipes, clean_roads, normalise_condition


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def pipes(cfg):
    return clean_pipes(synthetic.make_pipes(3000, seed=3), cfg)


@pytest.fixture(scope="module")
def roads(cfg):
    return clean_roads(synthetic.make_roads(120, seed=4).drop(columns=["_x0", "_y"]), cfg)


# ------------------------------------------------------------------ cleaning
def test_condition_label_normalisation():
    s = pd.Series(["Red", " r ", "GREEN", "amber", "3", 2.0, None, "purple"])
    assert normalise_condition(s).tolist() == ["R", "R", "G", "A", "R", "A", None, None]


def test_inconsistent_and_unusual_pipe_records(cfg):
    raw = pd.DataFrame({
        "OBJECTID": [1, 1, 2, 3, 4],
        "YEARINSTALLED": [1960, 2090, 1800, 1975, "1980-05-01"],
        "MATERIAL": ["cast iron", "DIP", None, "pvc", "Transite"],
        "DIAMETER": [150, -5, 300, 0, 200],
        "Shape__Length": [100, 0, 50, 20, 5000.5 + 100],
        "RAG": ["Red", "green", "bogus", "A", "R"],
        "INSPECTION_YEAR": [2020, 2020, 2020, 1970, 2030],
    })
    df = clean_pipes(raw, cfg, diameter_unit="mm")
    assert df["pipe_id"].is_unique
    assert "duplicate_id_suffixed" in df["dq_flags"].iloc[1]
    assert np.isnan(df["age_years"].iloc[1]) and "install_year_in_future" in df["dq_flags"].iloc[1]
    assert np.isnan(df["age_years"].iloc[2])                      # 1800 implausible
    assert df["age_years"].iloc[4] == cfg["reference_year"] - 1980  # date string parsed
    assert np.isnan(df["diameter_mm"].iloc[1]) and np.isnan(df["diameter_mm"].iloc[3])
    assert np.isnan(df["length_m"].iloc[1])
    assert "length_unusually_long" in df["dq_flags"].iloc[4]
    assert df["material"].tolist() == ["CI", "DI", "UNKNOWN", "PVC", "AC"]
    assert df["condition"].iloc[2] is None and "condition_unrecognised" in df["dq_flags"].iloc[2]
    assert np.isnan(df["inspection_year"].iloc[3])                # before install
    assert np.isnan(df["inspection_year"].iloc[4])                # future


def test_diameter_inches_autodetected(cfg):
    df = clean_pipes(pd.DataFrame({"diameter": [6, 8, 12, 4], "length": [1, 1, 1, 1]}), cfg)
    assert df["diameter_mm"].round(1).tolist() == [152.4, 203.2, 304.8, 101.6]


def test_road_cleaning_out_of_range(cfg):
    df = clean_roads(pd.DataFrame({"PCI": [120, 55, -3], "AADT": [-1, 500, None], "truck_pct": [0.1, 0.05, 0.2],
                                   "LENGTH": [100, 0, 50]}), cfg)
    assert np.isnan(df["pci"].iloc[0]) and np.isnan(df["pci"].iloc[2])
    assert np.isnan(df["aadt"].iloc[0]) and np.isnan(df["length_m"].iloc[1])
    assert df["heavy_pct"].tolist() == [10.0, 5.0, 20.0]


def test_normalise_weights():
    w = normalise_weights({"a": 0.5, "b": 0.3, "c": 0.2}, {"a": True, "b": False, "c": True})
    assert w["b"] == 0 and abs(w["a"] + w["c"] - 1) < 1e-12


# ---------------------------------------------------------------- pipe scores
def test_condition_sources(cfg):
    df = pd.DataFrame({"condition": ["R", "G", None, None, "A"], "inspection_year": [2026, 2006, np.nan, np.nan, np.nan],
                       "age_years": [50, 50, 50, np.nan, 10], "material": ["CI"] * 5})
    proba = pd.DataFrame({"p_G": [0.1, 0.1, 0.2, np.nan, 0.5], "p_A": [0.3, 0.3, 0.3, np.nan, 0.3],
                          "p_R": [0.6, 0.6, 0.5, np.nan, 0.2]})
    out = score_pipe_condition(df, cfg, proba)
    assert out["condition_score"].iloc[0] == 100 and out["condition_source"].iloc[0] == "observed"
    # inspected 20 y ago, half-life 10 -> weight 0.25
    assert out["condition_source"].iloc[1] == "observed_aged_toward_prediction"
    assert out["condition_score"].iloc[1] == pytest.approx(0.25 * 0 + 0.75 * 75)
    assert out["condition_source"].iloc[2] == "predicted"
    assert out["condition_source"].iloc[3] == "unknown"
    assert out["condition_source"].iloc[4] == "observed_date_unknown"
    blind = score_pipe_condition(df, cfg, proba, use_observed=False)
    assert blind["condition_source"].iloc[0] == "predicted"


def test_age_proxy_without_model(cfg):
    df = pd.DataFrame({"age_years": [10, 40, 80], "material": ["DI", "DI", "DI"]})
    out = score_pipe_condition(df, cfg)
    assert (out["condition_source"] == "age_proxy").all()
    assert out["condition_score"].is_monotonic_increasing


def test_break_evidence_shrinks_small_exposure(cfg):
    df = pd.DataFrame({"n_breaks": [1, 10, 0, 0], "length_m": [10, 1000, 1000, 1000], "age_years": [50] * 4,
                       "break_obs_years": [10] * 4, "material": ["CI"] * 4, "diameter_mm": [150] * 4})
    b = break_evidence(df, cfg)
    raw = df["n_breaks"] / (df["length_m"] / 1000 * 10)
    assert raw.iloc[0] > raw.iloc[1]                               # raw: short pipe looks worst
    assert b["break_rate_hat"].iloc[0] < raw.iloc[0]                # shrunk toward cohort
    assert b["break_score"].between(0, 100).all()


def test_consequence_handles_missing(cfg):
    df = pd.DataFrame({"diameter_mm": [100, np.nan, 600], "customers_served": [np.nan, np.nan, 500],
                       "critical_facility": [np.nan, np.nan, True]})
    c = consequence_score(df, cfg)
    assert c["consequence_score"].iloc[1] == 50 and c["consequence_flag"].iloc[1]
    assert c["consequence_score"].iloc[2] > c["consequence_score"].iloc[0]


def test_priority_range_and_explanations(cfg, pipes):
    s = score_pipe_priority(pipes, cfg)
    assert s["priority_score"].between(0, 100).all()
    assert s["risk_score"].between(0, 100).all()
    assert s["explanation"].str.contains("Priority=").all()
    red, green = s[pipes["condition"] == "R"], s[pipes["condition"] == "G"]
    assert red["priority_score"].median() > green["priority_score"].median()
    # exactly additive decomposition
    assert np.allclose(s["priority_score"], s["priority_risk_part"] + s["priority_econ_part"], atol=0.01)


def test_no_double_counting_of_age(cfg):
    """Two pipes identical except age, same observed condition and breaks -> same priority."""
    df = clean_pipes(pd.DataFrame({"pipe_id": ["a", "b"], "install_year": [1950, 2000], "material": ["CI", "CI"],
                                   "diameter_mm": [150, 150], "length_m": [100, 100], "condition": ["A", "A"],
                                   "inspection_year": [2026, 2026]}), cfg)
    s = score_pipe_priority(df, cfg)
    assert s["priority_score"].iloc[0] == s["priority_score"].iloc[1]


def test_condition_model_and_explain(cfg, pipes):
    m = PipeConditionModel(n_estimators=60)
    proba, metrics = m.cross_validate(pipes, 3)
    assert np.allclose(proba.sum(axis=1), 1)
    assert metrics["red_roc_auc"] > 0.75
    m.fit(pipes)
    shap = m.explain(pipes.head(20))
    assert list(shap.columns) == ["age_years", "diameter_mm", "length_m", "material", "region"]
    # model must cope with unseen categories and missing numerics
    odd = pipes.head(3).copy()
    odd["material"] = "NEWMAT"
    odd["age_years"] = np.nan
    assert np.isfinite(m.predict_proba(odd).to_numpy()).all()


def test_model_requires_all_classes(cfg, pipes):
    with pytest.raises(ValueError):
        PipeConditionModel().fit(pipes[pipes["condition"] != "R"])


# --------------------------------------------------------------------- costs
def test_pipe_cost_monotonic_and_intervals(cfg):
    df = pd.DataFrame({"pipe_id": list("abc"), "diameter_mm": [100, 300, np.nan], "length_m": [100, 100, 100]})
    c = estimate_pipe_replacement_cost(df, cfg, n_draws=200)
    assert c["cost_point"].iloc[1] > c["cost_point"].iloc[0]
    assert (c["cost_p10"] <= c["cost_p50"]).all() and (c["cost_p50"] <= c["cost_p90"]).all()
    assert c["cost_flags"].iloc[2] == "diameter_imputed_median"
    assert c["cost_basis"].str.contains("PLACEHOLDER").all()


def test_reopen_penalty_only_for_recent_resurfacing(cfg):
    fresh = pipe_cost_components(np.array([150.0]), np.array([100.0]), True, cfg, years_since_resurface=[1])
    old = pipe_cost_components(np.array([150.0]), np.array([100.0]), True, cfg, years_since_resurface=[30])
    assert fresh["reopen_penalty"][0] > 0 and old["reopen_penalty"][0] == 0


def test_fit_cost_function_recovers_parameters():
    d = np.array([100, 150, 200, 300, 400, 600])
    fit = fit_cost_function(d, 50000 * (d / 100) ** 1.1)
    assert fit["a"] == pytest.approx(50000, rel=1e-6) and fit["b"] == pytest.approx(1.1, rel=1e-6)
    with pytest.raises(ValueError):
        fit_cost_function([100], [1])


def test_road_treatment_selection(cfg):
    df = pd.DataFrame({"road_id": list("abcd"), "pci_effective": [90, 60, 30, 10], "length_m": [100] * 4,
                       "width_m": [7, 7, np.nan, 7]})
    c = estimate_road_repair_cost(df, cfg, n_draws=0)
    assert c["treatment"].tolist() == ["preventive", "mill_overlay", "thick_overlay", "reconstruction"]
    assert c["cost_point"].is_monotonic_increasing
    assert c["cost_flags"].iloc[2] == "width_imputed_default"


# --------------------------------------------------------------------- roads
def test_road_condition_missing_and_stale(cfg):
    df = clean_roads(pd.DataFrame({"pci": [80, 80, np.nan, np.nan], "pci_year": [2026, 2016, np.nan, np.nan],
                                   "last_resurface_year": [2020, 2010, 2016, np.nan],
                                   "length_m": [100] * 4, "functional_class": ["local"] * 4}), cfg)
    c = score_road_condition(df, cfg)
    assert c["pci_source"].tolist() == ["observed", "observed_aged_forward", "estimated_from_resurface_age",
                                        "imputed_network_median"]
    assert c["pci_effective"].iloc[1] < 80
    assert c["pci_rate_source"].iloc[0] == "config_default"


def test_road_priority_missing_traffic(cfg, roads):
    s = score_road_priority(roads, cfg)
    assert s["priority_score"].between(0, 100).all()
    imputed = s["criticality_flag"] == "aadt_imputed_class_median"
    assert imputed.any()
    none = clean_roads(pd.DataFrame({"pci": [40, 90], "length_m": [100, 100]}), cfg)
    s2 = score_road_priority(none, cfg)
    assert (s2["criticality_score"] == 50).all()            # neutral when nothing is known
    assert s2["priority_score"].iloc[0] > s2["priority_score"].iloc[1]


# -------------------------------------------------------------- coordination
def _pair_inputs(cfg, pipe_cond, road_pci, aadt=8000, tr_offset=None):
    pipes = clean_pipes(pd.DataFrame({
        "pipe_id": ["p"], "install_year": [1960], "material": ["CI"], "diameter_mm": [200], "length_m": [150],
        "condition": [pipe_cond], "inspection_year": [2026], "n_breaks": [3 if pipe_cond == "R" else 0],
        "break_obs_years": [10], "customers_served": [200], "road_class_above": ["arterial"],
        "geometry_wkt": ["LINESTRING (0 1, 150 1)"]}), cfg)
    roads = clean_roads(pd.DataFrame({
        "road_id": ["r"], "pci": [road_pci], "pci_year": [2026], "aadt": [aadt], "functional_class": ["arterial"],
        "length_m": [200], "width_m": [10], "emergency_route": [True], "geometry_wkt": ["LINESTRING (0 0, 200 0)"]}), cfg)
    ps = score_pipe_priority(pipes, cfg)
    rs = score_road_priority(roads, cfg)
    pairs = match_pipes_to_roads(pipes, roads)
    return pairs, pipes, ps, roads, rs


def test_geometry_matching(cfg):
    pairs, *_ = _pair_inputs(cfg, "R", 30)
    assert len(pairs) == 1 and pairs["overlap_fraction"].iloc[0] == pytest.approx(1.0)
    assert pairs["match_method"].iloc[0] == "geometry"


def test_no_match_without_location(cfg, pipes, roads):
    anon = pipes.drop(columns=["street_name", "geometry_wkt"], errors="ignore")
    assert match_pipes_to_roads(anon, roads).empty


def test_coordination_both_need_work(cfg):
    pairs, pipes, ps, roads, rs = _pair_inputs(cfg, "R", 25)
    c = calculate_coordination_opportunity(pairs, pipes, ps, roads, rs, cfg)
    assert c["gate_reason"].iloc[0] == ""
    assert c["coordination_score"].iloc[0] > 0
    assert c["net_savings"].iloc[0] > 0
    assert c["joint_cost"].iloc[0] < c["separate_cost"].iloc[0]


def test_coordination_gated_when_road_is_fine(cfg):
    pairs, pipes, ps, roads, rs = _pair_inputs(cfg, "R", 98)
    c = calculate_coordination_opportunity(pairs, pipes, ps, roads, rs, cfg)
    assert c["coordination_score"].iloc[0] == 0
    assert c["gate_reason"].iloc[0] == "road_does_not_need_work"
    assert np.isfinite(c["net_savings"].iloc[0])     # savings still reported, just not prioritised


def test_coordination_gated_when_pipe_is_fine(cfg):
    pairs, pipes, ps, roads, rs = _pair_inputs(cfg, "G", 20)
    c = calculate_coordination_opportunity(pairs, pipes, ps, roads, rs, cfg)
    assert c["coordination_score"].iloc[0] == 0 and c["gate_reason"].iloc[0] == "pipe_does_not_need_work"


# ------------------------------------------------------------------ optimise
def test_optimizer_respects_budget_and_coverage():
    opts = pd.DataFrame({
        "option_id": ["pipe:a", "road:b", "joint:a|b", "pipe:c"], "kind": ["pipe", "road", "joint", "pipe"],
        "covers": [("p:a",), ("r:b",), ("p:a", "r:b"), ("p:c",)], "cost": [60, 50, 80, 40],
        "benefit": [10, 8, 18, 3]})
    sel, summ = optimize_maintenance_budget(opts, budget=120)
    assert summ["status"] == "OPTIMAL"
    assert sel["cost"].sum() <= 120
    assert set(sel["option_id"]) == {"joint:a|b", "pipe:c"}     # bundle beats separate (18 vs 10+8 at 110)
    g, _ = greedy_budget(opts, 120)
    assert g["cost"].sum() <= 120


def test_optimizer_multi_period():
    opts = pd.DataFrame({"option_id": ["a", "b"], "kind": ["pipe", "pipe"], "covers": [("a",), ("b",)],
                         "cost": [100, 100], "benefit": [5, 4]})
    sel, summ = optimize_maintenance_budget(opts, budget=100, periods=2)
    assert len(sel) == 2 and sorted(sel["period"]) == [0, 1]
    assert sel.set_index("option_id").loc["a", "period"] == 0     # higher benefit first (discounting)


def test_build_options_end_to_end(cfg):
    pairs, pipes, ps, roads, rs = _pair_inputs(cfg, "R", 25)
    coord = calculate_coordination_opportunity(pairs, pipes, ps, roads, rs, cfg)
    opts = build_options(ps, pipes, rs, roads, coord)
    assert set(opts["kind"]) == {"pipe", "road", "joint"}


# ------------------------------------------------------------------ sampling
def test_representative_sample(cfg, pipes):
    s = select_representative(pipes, 500, seed=1)
    assert len(s) == 500 and s.index.is_unique
    assert (s["condition"] == "R").sum() >= 25
    # weights re-estimate population size
    assert s["sample_weight"].sum() == pytest.approx(len(pipes), rel=0.1)
    s2 = select_representative(pipes, 500, seed=1)
    assert s.index.equals(s2.index)                            # reproducible


def test_unfilled_cost_template_is_rejected():
    with pytest.raises(ValueError, match="not filled in"):
        load_config("configs/local_costs_template.yaml")
