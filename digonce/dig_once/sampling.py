"""Representative prototype sample selection (stratified, reproducible)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _age_band(age: pd.Series) -> pd.Series:
    return pd.cut(age, [-1, 20, 40, 60, 80, 1000], labels=["0-20", "21-40", "41-60", "61-80", ">80"]
                  ).astype(str).replace("nan", "unknown")


def select_representative(df: pd.DataFrame, n: int = 500, strata: list[str] | None = None,
                          min_per_condition: int = 25, seed: int = 42) -> pd.DataFrame:
    """Proportional stratified sample with a floor per condition class.

    * strata default: region x material x condition x age band (collapsed to
      condition x age band if that yields more strata than n).
    * allocation: largest-remainder proportional allocation.
    * condition floor: classes with fewer than ``min_per_condition`` rows in the
      proportional sample are topped up (oversampled) so Red is visible.
    * ``sample_weight`` = population stratum size / sampled stratum size, so
      network totals can be re-estimated from the sample without bias.
    """
    rng = np.random.default_rng(seed)
    work = df.copy()
    work["_age_band"] = _age_band(work["age_years"])
    work["_cond"] = work["condition"].fillna("unrated") if "condition" in work else "unrated"
    strata = strata or ["region", "material", "_cond", "_age_band"]
    strata = [s for s in strata if s in work.columns]
    if work.groupby(strata, dropna=False).ngroups > n:
        strata = ["_cond", "_age_band", "material"]
        if work.groupby(strata, dropna=False).ngroups > n:
            strata = ["_cond", "_age_band"]
    work["_stratum"] = work[strata].astype(str).agg("|".join, axis=1)
    n = min(n, len(work))
    sizes = work["_stratum"].value_counts()
    quota = sizes / sizes.sum() * n
    alloc = np.floor(quota).astype(int)
    remainder = (quota - alloc).sort_values(ascending=False)
    for s in remainder.index[: n - alloc.sum()]:
        alloc[s] += 1
    picked = []
    for s, k in alloc.items():
        if k > 0:
            idx = work.index[work["_stratum"] == s]
            picked.extend(rng.choice(idx, size=min(k, len(idx)), replace=False))
    sample = work.loc[picked]
    # condition floor (oversampling, replaces random rows of the largest class)
    for cls in ["R", "A", "G"]:
        have = (sample["_cond"] == cls).sum()
        pool = work.index[(work["_cond"] == cls) & ~work.index.isin(sample.index)]
        need = min(max(0, min_per_condition - have), len(pool))
        if need:
            add = rng.choice(pool, size=need, replace=False)
            largest = sample["_cond"].value_counts().idxmax()
            drop = rng.choice(sample.index[sample["_cond"] == largest], size=need, replace=False)
            sample = pd.concat([sample.drop(drop), work.loc[add]])
    counts = sample["_stratum"].value_counts()
    sample["sample_weight"] = sample["_stratum"].map(sizes) / sample["_stratum"].map(counts)
    sample["stratum"] = sample["_stratum"]
    return sample.drop(columns=["_age_band", "_cond", "_stratum"])
