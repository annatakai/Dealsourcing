"""CSV / JSON export of scores and explanations."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if pd.isna(o):
        return None
    return str(o)


def export_scores(df: pd.DataFrame, out_dir: str | Path, name: str, meta: dict | None = None) -> dict[str, Path]:
    """Write ``name.csv`` and ``name.json`` (records + metadata)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}.csv"
    json_path = out_dir / f"{name}.json"
    df.to_csv(csv_path, index=False)
    records = json.loads(df.to_json(orient="records", force_ascii=False))
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta or {}, "records": records}, fh, ensure_ascii=False, indent=1, default=_json_default)
    return {"csv": csv_path, "json": json_path}


def write_json(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=_json_default)
    return path
