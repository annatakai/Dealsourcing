"""SYNTHETIC demo data generator.

Everything produced here is artificial and exists only to exercise and test
the pipeline when the real municipal files are not available. Distributions
are arbitrary, not calibrated to any city. Never report results computed on
synthetic data as findings about real infrastructure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MATERIALS = ["CI", "DI", "PVC", "PE", "AC", "ST"]
MAT_P = [0.18, 0.40, 0.18, 0.10, 0.09, 0.05]
MAT_HAZ = {"CI": 1.0, "DI": 0.35, "PVC": 0.45, "PE": 0.25, "AC": 0.8, "ST": 0.7}


def make_pipes(n: int = 5000, seed: int = 0, ref_year: int = 2026, with_extras: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mat = rng.choice(MATERIALS, n, p=MAT_P)
    install = np.where(np.isin(mat, ["PVC", "PE"]), rng.integers(1975, 2024, n), rng.integers(1930, 2015, n))
    age = ref_year - install
    diam = rng.choice([75, 100, 150, 200, 250, 300, 400, 500, 600, 800], n,
                      p=[0.08, 0.25, 0.25, 0.15, 0.07, 0.08, 0.05, 0.03, 0.02, 0.02])
    length = np.round(rng.lognormal(np.log(90), 0.6, n), 1)
    region = rng.choice([f"REG{i}" for i in range(8)], n)
    reg_eff = {f"REG{i}": v for i, v in enumerate(rng.normal(0, 0.3, 8))}
    latent = (np.array([MAT_HAZ[m] for m in mat]) * age / 40 - 0.15 * np.log(diam / 100)
              + np.array([reg_eff[r] for r in region]) + rng.normal(0, 0.45, n))
    cond = np.where(latent > 1.55, "R", np.where(latent > 0.9, "A", "G"))
    df = pd.DataFrame({
        "pipe_id": [f"SYN-P{i:06d}" for i in range(n)], "install_year": install, "material": mat,
        "diameter_mm": diam, "length_m": length, "region": region, "condition": cond,
        "inspection_year": np.maximum(install, rng.integers(2012, ref_year + 1, n)),
    })
    if with_extras:
        obs_years = 10
        rate = 0.05 * np.exp(1.3 * latent) * length / 1000 * obs_years
        df["n_breaks"] = rng.poisson(rate)
        df["break_obs_years"] = obs_years
        last = ref_year - rng.integers(0, obs_years, n)
        df["last_break_year"] = np.where(df["n_breaks"] > 0, last, np.nan)
        df["customers_served"] = np.round(rng.lognormal(np.log(40), 1.0, n) * (diam / 150))
        df["critical_facility"] = rng.random(n) < 0.03
        df["road_class_above"] = rng.choice(["arterial", "collector", "local", "none"], n, p=[0.15, 0.25, 0.5, 0.1])
    return df


def place_on_roads(pipes: pd.DataFrame, roads: pd.DataFrame, share_under_roads: float = 0.3,
                   seed: int = 2) -> pd.DataFrame:
    """Give SYNTHETIC pipes WKT geometry: a share run beneath synthetic roads
    (partly overlapping), the rest lie in a separate area with no roads."""
    rng = np.random.default_rng(seed)
    pipes = pipes.copy()
    n = len(pipes)
    under = rng.random(n) < share_under_roads
    ridx = rng.integers(0, len(roads), n)
    geoms, names, classes = [], [], []
    for i in range(n):
        L = float(pipes["length_m"].iloc[i])
        if under[i]:
            r = roads.iloc[ridx[i]]
            x0, y = r["_x0"], r["_y"]
            start = x0 + rng.uniform(-0.3, 0.9) * r["length_m"]
            geoms.append(f"LINESTRING ({start:.1f} {y + 1.5:.1f}, {start + L:.1f} {y + 1.5:.1f})")
            names.append(r["street_name"])
            classes.append(r["functional_class"])
        else:
            x, y = rng.uniform(0, 50000), rng.uniform(-60000, -10000)
            geoms.append(f"LINESTRING ({x:.1f} {y:.1f}, {x + L:.1f} {y:.1f})")
            names.append(None)
            classes.append("none")
    pipes["geometry_wkt"] = geoms
    pipes["street_name"] = names
    pipes["road_class_above"] = classes
    return pipes


def make_roads(n: int = 300, seed: int = 1, ref_year: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cls = rng.choice(["arterial", "collector", "local"], n, p=[0.2, 0.3, 0.5])
    resurf = rng.integers(1995, ref_year, n)
    rate = {"arterial": 3.0, "collector": 2.5, "local": 2.0}
    pci_year = np.minimum(ref_year, resurf + rng.integers(0, 15, n))
    pci = np.clip(100 - np.array([rate[c] for c in cls]) * (pci_year - resurf) + rng.normal(0, 7, n), 0, 100)
    aadt = np.round(np.where(cls == "arterial", rng.lognormal(np.log(15000), 0.5, n),
                     np.where(cls == "collector", rng.lognormal(np.log(5000), 0.5, n),
                              rng.lognormal(np.log(800), 0.6, n))))
    df = pd.DataFrame({
        "road_id": [f"SYN-R{i:05d}" for i in range(n)],
        "functional_class": cls, "pci": np.round(pci, 1), "pci_year": pci_year,
        "last_resurface_year": resurf, "aadt": aadt, "heavy_pct": np.round(rng.uniform(2, 15, n), 1),
        "length_m": np.round(rng.uniform(80, 400, n)), "width_m": np.where(cls == "local", 6.0, 10.0),
        "lanes": np.where(cls == "local", 2, 4), "emergency_route": rng.random(n) < 0.15,
        "detour_km": np.round(rng.uniform(0.2, 6, n), 1), "pothole_count": rng.poisson(1.0, n),
    })
    # synthetic grid geometry (projected metres): one road per row
    df["_y"] = np.arange(n) * 200.0
    df["_x0"] = 0.0
    df["geometry_wkt"] = [f"LINESTRING (0 {y:.1f}, {L:.1f} {y:.1f})" for y, L in zip(df["_y"], df["length_m"])]
    df["street_name"] = [f"SYN STREET {i}" for i in range(n)]
    # introduce realistic missingness
    df.loc[rng.random(n) < 0.15, "pci"] = np.nan
    df.loc[rng.random(n) < 0.25, "aadt"] = np.nan
    return df
