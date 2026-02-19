#!/usr/bin/env python3
"""
Aggregate history.parquet from trend_scanner into 2 CSVs:

Option A (Keyword + Geo):
  - Total_Clicks = sum(Clicks)
  - Avg_RPC = sum(RPC * Clicks) / sum(Clicks)   (click-weighted)
  - Days_Seen = nunique(date)
  - Niche = mode (most common) within group

Option B (Keyword + Geo + Device):
  Same metrics, with Device in the grouping key.

Usage (your repo paths):
python export_history_agg.py \
  --in arb_tools/trend_scanner/app.py/history.parquet \
       arb_tools/trend_scanner/app_rpc_only.py/history.parquet \
  --outdir arb_tools/trend_scanner/exports
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


REQUIRED_COLS = ["date", "Keyword", "Device", "Geo", "RPC", "Clicks", "Niche"]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Case-insensitive mapping to expected names
    lower_map = {c.lower(): c for c in df.columns}
    rename = {}

    for want in REQUIRED_COLS:
        key = want.lower()
        if key in lower_map and lower_map[key] != want:
            rename[lower_map[key]] = want

    if rename:
        df = df.rename(columns=rename)

    return df


def _validate_schema(df: pd.DataFrame, source: str) -> None:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {source}: {missing}\n"
            f"Found columns: {list(df.columns)}"
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
    df["Clicks"] = pd.to_numeric(df["Clicks"], errors="coerce").fillna(0.0)
    df["RPC"] = pd.to_numeric(df["RPC"], errors="coerce")

    df["Keyword"] = df["Keyword"].astype(str).str.strip()
    df["Geo"] = df["Geo"].astype(str).str.strip()
    df["Device"] = df["Device"].astype(str).str.strip()
    df["Niche"] = df["Niche"].astype(str).str.strip()

    df["__rpc_x_clicks"] = df["RPC"].fillna(0.0) * df["Clicks"]

    return df


def _aggregate(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)

    out = g.agg(
        Total_Clicks=("Clicks", "sum"),
        RPC_x_Clicks=("__rpc_x_clicks", "sum"),
        Days_Seen=("date", lambda x: x.nunique()),
        Niche=("Niche", _mode_series),
    ).reset_index()

    out["Avg_RPC"] = np.where(
        out["Total_Clicks"] > 0,
        out["RPC_x_Clicks"] / out["Total_Clicks"],
        np.nan
    )

    out = out.drop(columns=["RPC_x_Clicks"])

    # Nicely order columns
    front = group_cols + ["Niche", "Avg_RPC", "Total_Clicks", "Days_Seen"]
    cols = [c for c in front if c in out.columns] + [c for c in out.columns if c not in front]
    out = out[cols]

    return out


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_parquet(path)
    df = _normalize_columns(df)
    _validate_schema(df, str(path))
    df = _prep(df)
    df["__source_file"] = str(path)
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in",
        dest="inputs",
        nargs="+",
        required=True,
        help="One or more history.parquet paths",
    )
    p.add_argument(
        "--outdir",
        default=".",
        help="Output directory for CSVs",
    )
    p.add_argument(
        "--prefix",
        default="history",
        help="Filename prefix for outputs",
    )
    args = p.parse_args()

    inputs = [Path(x).expanduser().resolve() for x in args.inputs]
    df = pd.concat([read_parquet(pth) for pth in inputs], ignore_index=True)

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