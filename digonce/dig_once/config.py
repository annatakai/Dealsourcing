"""Central configuration for the Dig Once scoring framework.

Every weight, threshold and unit cost lives here (or in a YAML override) so that
scores are reproducible and every assumption is auditable.

IMPORTANT - cost parameters
---------------------------
The unit costs in ``DEFAULT_CONFIG["costs"]`` are DEMO PLACEHOLDERS. They are
round numbers chosen only so the pipeline runs end to end. They are NOT
real-world unit costs and must be replaced with local bid tabulations,
completed-project costs or official cost functions (see docs/FRAMEWORK.md,
section 5) before any result is used for planning. Every cost output carries a
``cost_basis`` column that says "PLACEHOLDER" until ``costs.is_placeholder`` is
set to ``false`` in a YAML override.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

try:  # PyYAML is optional; defaults work without it.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "reference_year": 2026,
    "currency": "JPY",
    "random_seed": 42,
    # ------------------------------------------------------------------ PIPE
    "pipe": {
        # A. Condition: ordinal mapping of observed Green/Amber/Red to 0-100.
        "condition_state_scores": {"G": 0.0, "A": 50.0, "R": 100.0},
        # Observed ratings decay toward the model prediction as they age.
        # w_obs = 0.5 ** (years_since_inspection / half_life). Expert assumption.
        "observation_half_life_years": 10.0,
        # Fallback only when neither an observation nor a model exists.
        # Default = statutory accounting life of distribution mains in Japan
        # (40 years). This is an accounting life, NOT a physical service life;
        # override per material with locally observed service lives.
        "expected_service_life_years": {"default": 40.0},
        # B. Likelihood index: weights of condition evidence vs break evidence.
        "likelihood_weights": {"condition": 0.5, "breaks": 0.5},
        # Empirical-Bayes credibility: prior pseudo-exposure in km*years.
        "break_prior_exposure_km_years": 2.0,
        # Environmental multipliers are applied ONLY for factors that are not
        # already model features (see features_in_model). Expert judgment.
        "soil_corrosivity_multiplier_max": 1.20,   # at corrosivity = 1.0
        "pressure_multiplier_max": 1.15,           # at pressure >= pressure_high
        "pressure_low_m": 30.0,
        "pressure_high_m": 80.0,
        "features_in_model": ["age_years", "material", "diameter_mm", "length_m", "region"],
        # C. Consequence weights (renormalised over available components).
        "consequence_weights": {
            "diameter": 0.35, "customers": 0.25, "critical_facility": 0.25, "road_above": 0.15,
        },
        "diameter_min_mm": 50.0,
        "diameter_max_mm": 1000.0,
        "road_class_scores": {
            "arterial": 100.0, "collector": 60.0, "local": 30.0, "none": 10.0,
        },
        # Final priority = w_risk * Risk + w_econ * Economic.
        "priority_weights": {"risk": 0.8, "economic": 0.2},
        # D. Urgency -> action window thresholds on the likelihood index.
        "action_windows": [
            [75.0, "0-2y", 1.0], [50.0, "3-5y", 4.0], [25.0, "6-10y", 8.0], [0.0, ">10y", 15.0],
        ],
        "recent_break_years": 2,
        # Fraction of the expected break rate avoided by replacement (assumption).
        "replacement_break_reduction": 0.9,
    },
    # ------------------------------------------------------------------ ROAD
    "road": {
        "pci_stale_years": 3.0,
        # Linear PCI loss per year used ONLY to age-forward stale PCI or to
        # estimate missing PCI. Re-estimated from the data when possible
        # (fit_pci_deterioration). Placeholder default, flagged in output.
        "default_pci_loss_per_year": 2.5,
        "climate_rate_multiplier_max": 1.3,   # applied to the rate at climate_index=1
        "pothole_pci_penalty_per_count_per_km": 1.0,
        "pothole_pci_penalty_cap": 15.0,
        "criticality_weights": {
            "traffic": 0.30, "heavy": 0.15, "road_class": 0.20, "emergency": 0.20, "disruption": 0.15,
        },
        "road_class_scores": {
            "arterial": 100.0, "collector": 60.0, "local": 30.0, "other": 20.0,
        },
        "detour_ref_km": 5.0,
        "priority_weights": {"risk": 0.8, "economic": 0.2},
        # Treatment decision tree on PCI (illustrative defaults; calibrate).
        "treatments": [
            [70.0, "preventive", 85.0, 7.0],      # [min PCI, name, PCI after, service life]
            [50.0, "mill_overlay", 95.0, 12.0],
            [25.0, "thick_overlay", 98.0, 15.0],
            [0.0, "reconstruction", 100.0, 25.0],
        ],
        "action_windows": [
            [60.0, "0-2y", 1.0], [40.0, "3-5y", 4.0], [20.0, "6-10y", 8.0], [0.0, ">10y", 15.0],
        ],
        "default_width_m": 7.0,
        "default_lane_width_m": 3.5,
    },
    # ------------------------------------------------------------ COORDINATION
    "coordination": {
        "need_min_priority": 40.0,      # below this an asset "does not need work"
        "horizon_years": 10.0,
        "timing_scale_years": 3.0,      # tau = exp(-gap / scale)
        "max_delay_years": 5.0,
        "savings_ratio_ref": 0.30,      # net savings / separate cost that maps to 1.0
    },
    # ------------------------------------------------------------------ COSTS
    "costs": {
        "is_placeholder": True,
        "basis_label": "PLACEHOLDER - replace with local bid/project data",
        "discount_rate": 0.04,          # Japan MLIT public-works CBA guideline rate
        "analysis_horizon_years": 30,
        # Pipe supply+lay cost per metre: a * (D/100mm) ** b  (fit with fit_cost_function)
        "pipe_unit_cost_a": 60000.0,
        "pipe_unit_cost_b": 0.9,
        "trench_side_clearance_m": 0.3,
        "trench_cover_m": 1.2,
        "excavation_cost_per_m3": 8000.0,
        "trench_restore_overcut_m": 0.15,
        "trench_restore_cost_per_m2_paved": 12000.0,
        "trench_restore_cost_per_m2_unpaved": 3000.0,
        "traffic_control_fraction": 0.08,
        "overhead_fraction": 0.15,
        "contingency_fraction": 0.10,
        "mobilisation_per_project": 1500000.0,
        "emergency_repair_cost_base": 1500000.0,   # per break at 100 mm
        "emergency_repair_diameter_exponent": 0.5,
        "social_cost_per_break": 500000.0,
        # Road treatment cost per m2 of carriageway.
        "road_treatment_cost_per_m2": {
            "preventive": 1500.0, "mill_overlay": 5000.0,
            "thick_overlay": 8000.0, "reconstruction": 20000.0,
        },
        # Cutting a recently resurfaced road: fraction of remaining surface
        # life lost (literature reports reductions up to ~50%; see docs).
        "cut_life_loss_fraction": 0.30,
        "resurface_design_life_years": 12.0,
        # User delay (closure) cost per closure-day per 1000 AADT. Placeholder.
        "user_delay_cost_per_day_per_1000_aadt": 100000.0,
        "pipe_closure_days_per_100m": 10.0,
        "road_closure_days_per_100m": 3.0,
        "joint_closure_overlap_fraction": 0.6,  # shared days when done together
        # Triangular uncertainty multipliers (low, high) around each mode value.
        "uncertainty": {
            "pipe_unit": [0.7, 1.5], "excavation": [0.7, 1.6], "restoration": [0.7, 1.5],
            "road_unit": [0.7, 1.5], "emergency": [0.5, 2.0],
        },
        "monte_carlo_draws": 400,
    },
    # ------------------------------------------------------------- VALIDATION
    "validation": {
        "budget_fraction": 0.10,
        "sensitivity_draws": 200,
        "weight_perturbation": 0.5,   # +/-50% before renormalising
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    """Return DEFAULT_CONFIG deep-merged with an optional YAML file and dict."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read a config file")
        with open(path, encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    missing = [k for k, v in _flatten(cfg).items() if v is None]
    if missing:
        raise ValueError(f"config values not filled in: {', '.join(missing)}")
    return cfg


def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def cost_basis(cfg: dict) -> str:
    return cfg["costs"]["basis_label"] if cfg["costs"].get("is_placeholder", True) else "LOCAL"


def normalise_weights(weights: dict[str, float], available: dict[str, bool]) -> dict[str, float]:
    """Renormalise weights over the components that are available."""
    usable = {k: w for k, w in weights.items() if available.get(k, False) and w > 0}
    total = sum(usable.values())
    if total <= 0:
        return {k: 0.0 for k in weights}
    return {k: (usable.get(k, 0.0) / total) for k in weights}
