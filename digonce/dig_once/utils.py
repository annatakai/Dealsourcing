"""Small numeric helpers shared by the scoring modules."""
from __future__ import annotations

import numpy as np
import pandas as pd


def clip01(x):
    return np.clip(x, 0.0, 1.0)


def log_scale(x, lo: float, hi: float) -> np.ndarray:
    """Map x to [0, 100] on a log scale between lo and hi (clipped)."""
    x = np.asarray(x, dtype=float)
    lo = max(lo, 1e-12)
    hi = max(hi, lo * 1.0001)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (np.log(np.maximum(x, 1e-12)) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return 100.0 * clip01(v)


def log1p_scale(x, ref: float) -> np.ndarray:
    """100 * log(1+x)/log(1+ref), clipped; ref is typically a high percentile."""
    x = np.asarray(x, dtype=float)
    ref = max(float(ref), 1e-9)
    return 100.0 * clip01(np.log1p(np.maximum(x, 0)) / np.log1p(ref))


def weighted_nanmean(components: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.Series, pd.Series]:
    """Row-wise weighted mean that renormalises weights over non-missing values.

    Returns (score, share_of_weight_available). A row with no available
    component gets NaN.
    """
    cols = [c for c in weights if c in components.columns]
    w = np.array([weights[c] for c in cols], dtype=float)
    vals = components[cols].to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    wm = mask * w
    denom = wm.sum(axis=1)
    num = np.nansum(vals * w, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        score = np.where(denom > 0, num / denom, np.nan)
    coverage = denom / w.sum() if w.sum() > 0 else np.zeros(len(vals))
    return pd.Series(score, index=components.index), pd.Series(coverage, index=components.index)


def window_from_score(score, windows) -> tuple[np.ndarray, np.ndarray]:
    """Map a 0-100 score to (label, representative years) using descending thresholds."""
    score = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0)
    labels = np.empty(score.shape, dtype=object)
    years = np.zeros(score.shape)
    done = np.zeros(score.shape, dtype=bool)
    for thr, label, yrs in sorted(windows, key=lambda t: -t[0]):
        m = (score >= thr) & ~done
        labels[m], years[m] = label, yrs
        done |= m
    return labels, years


def top_factors(contribs: pd.DataFrame, k: int = 3) -> list[str]:
    """Format the k largest contributions per row as 'name (+x.x)'."""
    out = []
    arr = contribs.to_numpy(dtype=float)
    names = list(contribs.columns)
    for row in arr:
        order = np.argsort(-np.nan_to_num(np.abs(row), nan=-1))[:k]
        parts = [f"{names[i]} ({row[i]:+.1f})" for i in order if np.isfinite(row[i]) and abs(row[i]) > 0.05]
        out.append(", ".join(parts))
    return out
