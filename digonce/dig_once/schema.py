"""Canonical schemas, column-alias mapping and data-quality cleaning.

Real municipal exports (e.g. ArcGIS Hub CSVs) use inconsistent column names
and units. ``map_columns`` renames known aliases to the canonical names used by
the scoring code, and the ``clean_*`` functions coerce types, repair or null
out impossible values, and record every intervention in a ``dq_flags`` column
(semicolon-separated) so nothing is silently changed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Canonical name -> list of aliases (compared case-insensitively, ignoring
# spaces/underscores). Extend this for your own exports or pass a column_map.
PIPE_ALIASES: dict[str, list[str]] = {
    "pipe_id": ["pipe_id", "id", "objectid", "facilityid", "assetid", "gisid", "mainid", "segment_id"],
    "install_year": ["install_year", "installyear", "yearinstalled", "year_installed", "yearinstall",
                     "installdate", "install_date", "dateinstalled", "constructionyear"],
    "age_years": ["age_years", "age", "pipe_age"],
    "material": ["material", "pipe_material", "matl", "mat", "pipematerial"],
    "diameter_mm": ["diameter_mm", "diameter", "diam", "pipe_size", "pipesize", "size", "nominal_diameter"],
    "length_m": ["length_m", "length", "shape_length", "shape__length", "pipe_length", "seglength"],
    "region": ["region", "zone", "district", "pressure_zone", "pressurezone", "area"],
    "condition": ["condition", "condition_rating", "rag", "rag_status", "cond", "condition_grade"],
    "inspection_year": ["inspection_year", "inspectionyear", "assessment_year", "survey_year", "inspdate"],
    "n_breaks": ["n_breaks", "breaks", "break_count", "num_breaks", "failures"],
    "break_obs_years": ["break_obs_years", "obs_years", "observation_years"],
    "last_break_year": ["last_break_year"],
    "soil_corrosivity": ["soil_corrosivity", "corrosivity", "soil_corr"],
    "pressure_m": ["pressure_m", "pressure", "static_pressure", "operating_pressure"],
    "customers_served": ["customers_served", "customers", "services", "service_count", "connections"],
    "critical_facility": ["critical_facility", "critical", "critical_customer", "hospital"],
    "road_class_above": ["road_class_above", "road_class", "street_class"],
    "street_name": ["street_name", "street", "streetname", "fullname", "stname", "location", "road_name"],
    "geometry_wkt": ["geometry_wkt", "wkt", "geometry", "the_geom", "shape"],
    "status": ["status", "lifecyclestatus", "lifecycle_status", "activeflag"],
}

ROAD_ALIASES: dict[str, list[str]] = {
    "road_id": ["road_id", "id", "objectid", "segment_id", "segmentid", "facilityid", "streetid"],
    "pci": ["pci", "pavement_condition_index", "oci", "condition_index", "pcr"],
    "pci_year": ["pci_year", "survey_year", "inspection_year", "pci_date", "surveydate"],
    "last_resurface_year": ["last_resurface_year", "resurface_year", "lastpaved", "last_paved",
                            "year_paved", "paveyear", "treatment_year", "construction_year"],
    "aadt": ["aadt", "adt", "traffic_volume", "aadt_count"],
    "heavy_pct": ["heavy_pct", "truck_pct", "pct_trucks", "hv_pct", "heavy_vehicle_pct"],
    "functional_class": ["functional_class", "funclass", "func_class", "fclass", "road_class",
                         "roadclass", "street_class", "classification", "hierarchy"],
    "pothole_count": ["pothole_count", "potholes", "pothole_reports"],
    "length_m": ["length_m", "length", "shape_length", "shape__length", "seg_length"],
    "width_m": ["width_m", "width", "pavement_width", "pavewidth"],
    "lanes": ["lanes", "num_lanes", "lane_count"],
    "emergency_route": ["emergency_route", "emergency", "evac_route", "priority_route"],
    "detour_km": ["detour_km", "detour_length", "detour"],
    "climate_index": ["climate_index", "freeze_thaw_index"],
    "street_name": ["street_name", "street", "streetname", "fullname", "stname", "name", "road_name"],
    "geometry_wkt": ["geometry_wkt", "wkt", "geometry", "the_geom", "shape"],
}

BREAK_ALIASES: dict[str, list[str]] = {
    "break_id": ["break_id", "objectid", "id", "workorder", "wo_number"],
    "pipe_id": ["pipe_id", "facilityid", "mainid", "main_id", "assetid", "pipeid", "main_facilityid"],
    "break_date": ["break_date", "breakdate", "date", "datefound", "failure_date", "reported_date",
                   "date_reported", "event_date"],
    "street_name": ["street_name", "street", "streetname", "location", "address"],
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def map_columns(df: pd.DataFrame, aliases: dict[str, list[str]],
                column_map: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename columns to canonical names. Returns (renamed df, mapping used).

    ``column_map`` (canonical -> source column) takes precedence over aliases.
    The first matching alias wins; unmatched columns are kept unchanged.
    """
    mapping: dict[str, str] = {}
    normalised = {_norm(c): c for c in df.columns}
    used: set[str] = set()
    for canon, src in (column_map or {}).items():
        if src in df.columns:
            mapping[src] = canon
            used.add(src)
    for canon, alias_list in aliases.items():
        if canon in mapping.values():
            continue
        for alias in alias_list:
            src = normalised.get(_norm(alias))
            if src is not None and src not in used:
                mapping[src] = canon
                used.add(src)
                break
    return df.rename(columns=mapping), {v: k for k, v in mapping.items()}


def _flag(flags: pd.Series, mask, text: str) -> pd.Series:
    mask = pd.Series(mask, index=flags.index).fillna(False).astype(bool)
    flags = flags.copy()
    flags[mask] = flags[mask].where(flags[mask] == "", flags[mask] + ";") + text
    return flags


def _to_year(series: pd.Series) -> pd.Series:
    """Accept years (1965), dates ('1965-04-01', '4/1/1965') or epoch-ms."""
    num = pd.to_numeric(series, errors="coerce")
    out = num.where((num >= 1800) & (num <= 2200))
    epoch_ms = num.where(num > 1e11)
    if epoch_ms.notna().any():
        out = out.fillna(pd.to_datetime(epoch_ms, unit="ms", errors="coerce").dt.year)
    rest = series.where(out.isna() & series.notna())
    if rest.notna().any():
        parsed = pd.to_datetime(rest.astype(str), errors="coerce", format="mixed")
        out = out.fillna(parsed.dt.year)
    return out.astype(float)


_CONDITION_MAP = {
    "g": "G", "green": "G", "good": "G", "1": "G", "low": "G",
    "a": "A", "amber": "A", "yellow": "A", "fair": "A", "2": "A", "medium": "A",
    "r": "R", "red": "R", "poor": "R", "3": "R", "high": "R", "bad": "R",
}


def normalise_condition(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    s = s.str.replace(r"\.0$", "", regex=True)
    mapped = s.map(_CONDITION_MAP)
    return pd.Series([v if isinstance(v, str) else None for v in mapped], index=series.index, dtype=object)


_MATERIAL_MAP = {
    "ci": "CI", "castiron": "CI", "cast iron": "CI", "cip": "CI",
    "di": "DI", "dip": "DI", "ductile": "DI", "ductileiron": "DI", "ductile iron": "DI",
    "pvc": "PVC", "upvc": "PVC", "hpvc": "PVC",
    "pe": "PE", "hdpe": "PE", "mdpe": "PE", "polyethylene": "PE",
    "ac": "AC", "asbestoscement": "AC", "asbestos cement": "AC", "transite": "AC",
    "st": "ST", "steel": "ST", "stl": "ST", "galv": "ST",
    "cu": "CU", "copper": "CU", "conc": "CONC", "concrete": "CONC", "pccp": "CONC",
}


def normalise_material(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    mapped = s.map(_MATERIAL_MAP)
    mapped = mapped.fillna(s.str.replace(" ", "").map(_MATERIAL_MAP))
    out = mapped.fillna(series.astype("string").str.strip().str.upper())
    return out.where(series.notna() & (series.astype("string").str.strip() != ""), "UNKNOWN")


_BOOL_TRUE = {"1", "y", "yes", "true", "t"}


def _to_bool(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    return s.isin(_BOOL_TRUE).where(series.notna(), np.nan)


def clean_pipes(df: pd.DataFrame, cfg: dict, column_map: dict | None = None,
                diameter_unit: str = "auto", length_unit: str = "m") -> pd.DataFrame:
    """Map, coerce and validate a pipe table. Adds ``dq_flags``."""
    df, _ = map_columns(df, PIPE_ALIASES, column_map)
    df = df.copy()
    ref_year = cfg["reference_year"]
    flags = pd.Series("", index=df.index, dtype=object)

    if "pipe_id" not in df.columns:
        df["pipe_id"] = [f"P{i:07d}" for i in range(len(df))]
        flags = _flag(flags, True, "id_generated")
    df["pipe_id"] = df["pipe_id"].astype(str)
    dup = df["pipe_id"].duplicated(keep="first")
    flags = _flag(flags, dup, "duplicate_id_suffixed")
    df.loc[dup, "pipe_id"] = df.loc[dup, "pipe_id"] + "_dup" + df.loc[dup].groupby("pipe_id").cumcount().add(1).astype(str)

    for col in ["install_year", "inspection_year", "last_break_year"]:
        if col in df.columns:
            df[col] = _to_year(df[col])
    if "age_years" in df.columns:
        df["age_years"] = pd.to_numeric(df["age_years"], errors="coerce")
    if "install_year" in df.columns:
        fut = df["install_year"] > ref_year
        flags = _flag(flags, fut, "install_year_in_future_nulled")
        df.loc[fut, "install_year"] = np.nan
        old = df["install_year"] < 1850
        flags = _flag(flags, old, "install_year_implausible_nulled")
        df.loc[old, "install_year"] = np.nan
        derived = ref_year - df["install_year"]
        if "age_years" in df.columns:
            mismatch = (df["age_years"] - derived).abs() > 2
            flags = _flag(flags, mismatch, "age_install_year_mismatch_used_install_year")
            df["age_years"] = derived.fillna(df["age_years"])
        else:
            df["age_years"] = derived
    if "age_years" not in df.columns:
        df["age_years"] = np.nan
    bad_age = (df["age_years"] < 0) | (df["age_years"] > 175)
    flags = _flag(flags, bad_age, "age_out_of_range_nulled")
    df.loc[bad_age, "age_years"] = np.nan
    flags = _flag(flags, df["age_years"].isna(), "age_missing")

    df["material"] = normalise_material(df["material"]) if "material" in df.columns else "UNKNOWN"
    flags = _flag(flags, df["material"] == "UNKNOWN", "material_missing")

    if "diameter_mm" in df.columns:
        d = pd.to_numeric(df["diameter_mm"], errors="coerce")
        unit = diameter_unit
        if unit == "auto":
            med = d[d > 0].median()
            unit = "in" if pd.notna(med) and med <= 48 else "mm"
        if unit == "in":
            d = d * 25.4
            flags = _flag(flags, d.notna(), "diameter_converted_from_inches")
        bad = (d <= 0) | (d > 4000)
        flags = _flag(flags, bad, "diameter_out_of_range_nulled")
        df["diameter_mm"] = d.where(~bad)
    else:
        df["diameter_mm"] = np.nan
    flags = _flag(flags, df["diameter_mm"].isna(), "diameter_missing")

    if "length_m" in df.columns:
        length = pd.to_numeric(df["length_m"], errors="coerce")
        if length_unit == "ft":
            length = length * 0.3048
        bad = (length <= 0)
        flags = _flag(flags, bad, "length_nonpositive_nulled")
        length = length.where(~bad)
        huge = length > 5000
        flags = _flag(flags, huge, "length_unusually_long")
        df["length_m"] = length
    else:
        df["length_m"] = np.nan
    flags = _flag(flags, df["length_m"].isna(), "length_missing")

    df["region"] = df["region"].astype("string").fillna("UNKNOWN") if "region" in df.columns else "UNKNOWN"

    if "condition" in df.columns:
        raw = df["condition"]
        df["condition"] = normalise_condition(raw)
        unrec = raw.notna() & (raw.astype("string").str.strip() != "") & df["condition"].isna()
        flags = _flag(flags, unrec, "condition_unrecognised_nulled")
    else:
        df["condition"] = None
    if "inspection_year" in df.columns and "install_year" in df.columns:
        before = df["inspection_year"] < df["install_year"]
        flags = _flag(flags, before, "inspection_before_install_nulled")
        df.loc[before, "inspection_year"] = np.nan
    if "inspection_year" in df.columns:
        fut = df["inspection_year"] > ref_year
        flags = _flag(flags, fut, "inspection_year_in_future_nulled")
        df.loc[fut, "inspection_year"] = np.nan

    for col, lo, hi in [("n_breaks", 0, 500), ("break_obs_years", 0, 100), ("customers_served", 0, 1e6),
                        ("pressure_m", 0, 300), ("soil_corrosivity", 0, 1)]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            bad = (v < lo) | (v > hi)
            flags = _flag(flags, bad, f"{col}_out_of_range_nulled")
            df[col] = v.where(~bad)
    if "critical_facility" in df.columns:
        df["critical_facility"] = _to_bool(df["critical_facility"])
    if "road_class_above" in df.columns:
        df["road_class_above"] = normalise_road_class(df["road_class_above"], none_label="none")

    df["dq_flags"] = flags
    return df


_ROAD_CLASS_MAP = {
    "arterial": "arterial", "principal arterial": "arterial", "minor arterial": "arterial",
    "major": "arterial", "highway": "arterial", "freeway": "arterial", "expressway": "arterial",
    "primary": "arterial", "trunk": "arterial", "national": "arterial", "prefectural": "arterial",
    "collector": "collector", "major collector": "collector", "minor collector": "collector",
    "secondary": "collector", "tertiary": "collector", "distributor": "collector",
    "local": "local", "residential": "local", "local street": "local", "alley": "local",
    "service": "local", "municipal": "local",
}


def normalise_road_class(series: pd.Series, none_label: str = "other") -> pd.Series:
    s = series.astype("string").str.strip().str.lower()
    out = s.map(_ROAD_CLASS_MAP)
    for key, val in [("arterial", "arterial"), ("collector", "collector"), ("local", "local"),
                     ("residential", "local")]:
        out = out.where(out.notna() | ~s.str.contains(key, na=False), val)
    out = out.where(s.notna(), None)
    return out.where(out.notna() | s.isna(), none_label)


def clean_roads(df: pd.DataFrame, cfg: dict, column_map: dict | None = None,
                length_unit: str = "m") -> pd.DataFrame:
    df, _ = map_columns(df, ROAD_ALIASES, column_map)
    df = df.copy()
    ref_year = cfg["reference_year"]
    flags = pd.Series("", index=df.index, dtype=object)
    if "road_id" not in df.columns:
        df["road_id"] = [f"R{i:07d}" for i in range(len(df))]
        flags = _flag(flags, True, "id_generated")
    df["road_id"] = df["road_id"].astype(str)
    dup = df["road_id"].duplicated(keep="first")
    flags = _flag(flags, dup, "duplicate_id_suffixed")
    df.loc[dup, "road_id"] = df.loc[dup, "road_id"] + "_dup" + df.loc[dup].groupby("road_id").cumcount().add(1).astype(str)

    for col in ["pci_year", "last_resurface_year"]:
        if col in df.columns:
            df[col] = _to_year(df[col])
            fut = df[col] > ref_year
            flags = _flag(flags, fut, f"{col}_in_future_nulled")
            df.loc[fut, col] = np.nan
        else:
            df[col] = np.nan
    if "pci" in df.columns:
        pci = pd.to_numeric(df["pci"], errors="coerce")
        bad = (pci < 0) | (pci > 100)
        flags = _flag(flags, bad, "pci_out_of_range_nulled")
        df["pci"] = pci.where(~bad)
    else:
        df["pci"] = np.nan
    flags = _flag(flags, df["pci"].isna(), "pci_missing")
    for col, lo, hi in [("aadt", 0, 500000), ("heavy_pct", 0, 100), ("pothole_count", 0, 1e4),
                        ("width_m", 2, 80), ("lanes", 1, 16), ("detour_km", 0, 200), ("climate_index", 0, 1)]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce")
            bad = (v < lo) | (v > hi)
            flags = _flag(flags, bad, f"{col}_out_of_range_nulled")
            df[col] = v.where(~bad)
        else:
            df[col] = np.nan
    flags = _flag(flags, df["aadt"].isna(), "aadt_missing")
    # heavy_pct given as fraction (0-1) instead of percent
    frac = df["heavy_pct"].notna() & (df["heavy_pct"].max() <= 1.0)
    if frac.any():
        df.loc[frac, "heavy_pct"] = df.loc[frac, "heavy_pct"] * 100
        flags = _flag(flags, frac, "heavy_pct_converted_from_fraction")
    if "length_m" in df.columns:
        length = pd.to_numeric(df["length_m"], errors="coerce")
        if length_unit == "ft":
            length = length * 0.3048
        bad = length <= 0
        flags = _flag(flags, bad, "length_nonpositive_nulled")
        df["length_m"] = length.where(~bad)
    else:
        df["length_m"] = np.nan
    flags = _flag(flags, df["length_m"].isna(), "length_missing")
    if "functional_class" in df.columns:
        df["functional_class"] = normalise_road_class(df["functional_class"])
    else:
        df["functional_class"] = None
    if "emergency_route" in df.columns:
        df["emergency_route"] = _to_bool(df["emergency_route"])
    else:
        df["emergency_route"] = np.nan
    if "pci_year" in df.columns and "last_resurface_year" in df.columns:
        incons = (df["pci"] < 60) & (df["pci_year"] - df["last_resurface_year"]).between(0, 1)
        flags = _flag(flags, incons, "low_pci_right_after_resurfacing_check")
    df["dq_flags"] = flags
    return df


def clean_breaks(df: pd.DataFrame, cfg: dict, column_map: dict | None = None) -> pd.DataFrame:
    df, _ = map_columns(df, BREAK_ALIASES, column_map)
    df = df.copy()
    if "break_date" in df.columns:
        df["break_year"] = _to_year(df["break_date"])
        df.loc[df["break_year"] > cfg["reference_year"], "break_year"] = np.nan
    else:
        df["break_year"] = np.nan
    if "pipe_id" in df.columns:
        df["pipe_id"] = df["pipe_id"].astype("string")
    return df


def attach_breaks_to_pipes(pipes: pd.DataFrame, breaks: pd.DataFrame) -> pd.DataFrame:
    """Aggregate break records onto pipes via a shared pipe_id.

    Only an explicit asset-ID link is used. Breaks without a pipe_id are NOT
    guessed onto pipes (that would need geometry; see coordination.match_*).
    """
    pipes = pipes.copy()
    if "pipe_id" not in breaks.columns or breaks["pipe_id"].isna().all():
        return pipes
    b = breaks.dropna(subset=["pipe_id"])
    years = b["break_year"].dropna()
    obs = float(years.max() - years.min() + 1) if len(years) else np.nan
    agg = b.groupby("pipe_id").agg(n_breaks=("pipe_id", "size"), last_break_year=("break_year", "max"))
    recent_cut = years.max() - 3 if len(years) else np.nan
    if len(years):
        agg["breaks_last3y"] = b[b["break_year"] > recent_cut].groupby("pipe_id").size()
    pipes = pipes.drop(columns=[c for c in ["n_breaks", "last_break_year", "breaks_last3y"] if c in pipes.columns])
    pipes = pipes.merge(agg, left_on="pipe_id", right_index=True, how="left")
    pipes["n_breaks"] = pipes["n_breaks"].fillna(0)
    pipes["breaks_last3y"] = pipes.get("breaks_last3y", pd.Series(0, index=pipes.index)).fillna(0)
    pipes["break_obs_years"] = obs
    return pipes
