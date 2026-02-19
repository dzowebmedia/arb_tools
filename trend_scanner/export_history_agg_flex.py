#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import re

CANONICAL_REQUIRED = ["date", "Keyword", "Device", "Geo", "RPC"]
CANONICAL_OPTIONAL = ["Clicks", "Niche"]
CANONICAL_ALL = CANONICAL_REQUIRED + CANONICAL_OPTIONAL

def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s

SYNONYMS = {
    "date": {"date", "day", "dt", "scan_date", "created_at", "timestamp", "time", "datetime"},
    "Keyword": {"keyword", "kw", "term", "query", "search_term", "searchquery", "phrase"},
    "Device": {"device", "dev", "platform", "os", "traffic_device", "user_device"},
    "Geo": {"geo", "country", "location", "region", "state", "dma", "geography"},
    "RPC": {"rpc", "rev_per_click", "revenue_per_click", "earnings_per_click", "epc"},
    "Clicks": {"clicks", "click", "clk", "total_clicks", "num_clicks", "click_count"},
    "Niche": {"niche", "vertical", "category", "cluster", "theme", "segment"},
}

def _infer_mapping(columns: list[str]) -> dict[str, str]:
    norm_cols = {_norm(c): c for c in columns}
    mapping: dict[str, str] = {}

    for canonical, opts in SYNONYMS.items():
        for opt in opts:
            if opt in norm_cols:
                mapping[norm_cols[opt]] = canonical
                break

    return mapping

def _parse_map(map_args: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in map_args:
        if "=" not in item:
            raise ValueError(f'Bad --map "{item}". Use ColumnName=CanonicalName (e.g. Term=Keyword).')
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def _apply_mapping(df: pd.DataFrame, explicit_map: dict[str, str] | None = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = _infer_mapping(list(df.columns))

    if explicit_map:
        for k, v in explicit_map.items():
            if k not in df.columns:
                raise ValueError(f'--map refers to missing column "{k}". Found: {list(df.columns)}')
            if v not in CANONICAL_ALL:
                raise ValueError(f'--map target must be one of {CANONICAL_ALL}. Got "{v}".')
            rename[k] = v

    return df.rename(columns=rename)

def _validate(df: pd.DataFrame, source: str) -> None:
    missing = [c for c in CANONICAL_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Could not find required fields in {source}: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Tip: pass --map 'YourCol=Keyword' etc. for anything unusual."
        )

def _mode_series(s: pd.Series):
    s = s.dropna()
    if s.empty:
        return np.nan
    vc = s.astype(str).value_counts()
    return vc.index[0] if not vc.empty else np.nan

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df["RPC"] = pd.to_numeric(df["RPC"], errors="coerce")

    for c in ["Keyword", "Geo", "Device"]:
        df[c] = df[c].astype(str).str.strip()

    if "Clicks" in df.columns:
        df["Clicks"] = pd.to_numeric(df["Clicks"], errors="coerce").fillna(0.0)
        df["__rpc_x_clicks"] = df["RPC"].fillna(0.0) * df["Clicks"]
    else:
        df["Clicks"] = np.nan
        df["__rpc_x_clicks"] = np.nan

    if "Niche" in df.columns:
        df["Niche"] = df["Niche"].astype(str).str.strip()
    else:
        df["Niche"] = np.nan

    return df

def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)
    has_clicks = df["Clicks"].notna().any()

    if has_clicks:
        out = g.agg(
            Total_Clicks=("Clicks", "sum"),
            RPC_x_Clicks=("__rpc_x_clicks", "sum"),
            Days_Seen=("date", lambda x: x.nunique()),
            Niche=("Niche", _mode_series),
        ).reset_index()

        out["Avg_RPC"] = np.where(out["Total_Clicks"] > 0, out["RPC_x_Clicks"] / out["Total_Clicks"], np.nan)
        out = out.drop(columns=["RPC_x_Clicks"])
    else:
        out = g.agg(
            Avg_RPC=("RPC", "mean"),
            Days_Seen=("date", lambda x: x.nunique()),
        ).reset_index()
        out["Total_Clicks"] = np.nan
        out["Niche"] = np.nan

    front = group_cols + ["Niche", "Avg_RPC", "Total_Clicks", "Days_Seen"]
    out = out[[c for c in front if c in out.columns] + [c for c in out.columns if c not in front]]
    return out

def _read_one(path: Path, explicit_map: dict[str, str] | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_parquet(path)
    df = _apply_mapping(df, explicit_map=explicit_map)
    _validate(df, str(path))
    df = _prep(df)
    df["__source_file"] = str(path)
    return df

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inputs", nargs="+", required=True, help="One or more parquet paths")
    p.add_argument("--outdir", default="exports", help="Output directory for CSVs")
    p.add_argument("--prefix", default="history", help="Filename prefix for outputs")
    p.add_argument("--map", action="append", default=[], help="Override mapping like: --map 'Term=Keyword'")
    args = p.parse_args()

    explicit_map = _parse_map(args.map) if args.map else None

    inputs = [Path(x).expanduser().resolve() for x in args.inputs]
    df = pd.concat([_read_one(pth, explicit_map) for pth in inputs], ignore_index=True)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    a = _aggregate(df, ["Keyword", "Geo"])
    b = _aggregate(df, ["Keyword", "Geo", "Device"])

    a_path = outdir / f"{args.prefix}_keyword_geo.csv"
    b_path = outdir / f"{args.prefix}_keyword_geo_device.csv"

    a.to_csv(a_path, index=False)
    b.to_csv(b_path, index=False)

    print(f"Loaded rows: {len(df):,}")
    print(f"Option A rows: {len(a):,} -> {a_path}")
    print(f"Option B rows: {len(b):,} -> {b_path}")

if __name__ == "__main__":
    main()
