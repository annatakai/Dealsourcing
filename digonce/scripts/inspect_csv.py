"""Print the columns of a CSV and how Dig Once would map them.

    python scripts/inspect_csv.py pipes  ~/Desktop/Water_Mains_*.csv
    python scripts/inspect_csv.py breaks ~/Desktop/Water_Main_Breaks_*.csv
    python scripts/inspect_csv.py roads  ~/Desktop/Roads.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dig_once.schema import BREAK_ALIASES, PIPE_ALIASES, ROAD_ALIASES, map_columns  # noqa: E402

ALIASES = {"pipes": PIPE_ALIASES, "roads": ROAD_ALIASES, "breaks": BREAK_ALIASES}


def main() -> None:
    kind, path = sys.argv[1], sys.argv[2]
    df = pd.read_csv(path, nrows=5000, low_memory=False)
    _, mapping = map_columns(df, ALIASES[kind])
    print(f"{path}: {len(df.columns)} columns (first 5000 rows read)\n")
    for col in df.columns:
        canon = next((k for k, v in mapping.items() if v == col), "")
        sample = df[col].dropna().astype(str).head(3).tolist()
        print(f"  {col:35s} -> {canon or '-':22s} null={df[col].isna().mean():5.1%}  e.g. {sample}")
    missing = [k for k in ALIASES[kind] if k not in mapping]
    print("\nCanonical fields NOT found (add to a column_map YAML if they exist under another name):")
    print("  " + ", ".join(missing))


if __name__ == "__main__":
    main()
