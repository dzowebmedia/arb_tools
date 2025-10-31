# app.py  — Arb LaunchOps — Daily Trend Scanner & Planner
import io
import os
import json
from datetime import date, timedelta, datetime
import pandas as pd
import numpy as np
import streamlit as st
import yaml

# ---- app.py schema utilities ----
import re
from difflib import SequenceMatcher
import pandas as pd

CANON = ["Keyword","Device","Geo","RPC","Clicks","Date","Rev","TQ"]

# ==== PATCH 1: broadened synonym table (same structure) ====
SYNONYMS = {
    "Keyword": {
        "exact": {
            "keyword","keywords","kw","kwd",
            "relatedkw","related_kw","relatedkeywords","realtedkw",
            "query","term","search term"
        },
        "regex": [
            r".*\bkw(s|d)?\b.*",
            r".*\bkey ?word(s)?\b.*",
            r".*related ?kw(s|d)?.*",
            r".*\bquery\b.*",
            r".*\bterm\b.*",
            r".*\bsearch ?term\b.*",
        ]
    },
    "Device": {
        "exact": {"device","platform","name","resolution"},
        "regex": [
            r".*\bdevice\b.*",
            r".*\bplatform\b.*",
            r".*\bname\b.*",
            r".*\bresolution\b.*",
        ]
    },
    "Geo": {
        "exact": {"geo","country","region","market","locale"},
        "regex": [
            r".*\bgeo\b.*",
            r".*\bcountry\b.*",
            r".*\bregion\b.*",
            r".*\bmarket\b.*",
            r".*\blocale\b.*",
        ]
    },
    "Rev": {
        "exact": {"rev","revenue","grossrev","netrev"},
        "regex": [
            r"^rev.*$",
            r".*revenue.*",
        ]
    },
    "Clicks": {
        "exact": {
            "clicks","rev clicks","y clicks","kwx clicks","kwx_clicks","y_clicks",
            "kwxclicks","yclicks"  # allow no-space variants after normalization
        },
        "regex": [
            r".*\bclicks?\b.*",
            r".*\brev.*clicks?\b.*",
            r".*\by.*clicks?\b.*",
            r".*\bkwx.*clicks?\b.*",
        ]
    },
    "RPC": {
        "exact": {
            "rpc","y rpc","y_rpc","yrpc","rev per click","revenue per click",
            "kwx rpc"  # “kwx rpc” occasionally appears
        },
        "regex": [
            r"^rpc$",
            r".*rpc.*",
            r".*rev.*per.*click.*",
            r".*\bkwx.*rpc.*",
        ]
    },
    "TQ": {
        "exact": {"tq","trafficquality","traffic_quality"},
        "regex": [
            r"^tq$",
            r".*traffic.*quality.*",
        ]
    },
    "Date": {
        "exact": {"date","dt","day"},
        "regex": [
            r"^date$",
            r"^dt$",
            r".*\bday\b.*",
        ]
    }
}

def _norm_header(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s_]+"," ", s)
    s = re.sub(r"[^a-z0-9 ]+","", s)
    s = re.sub(r"\s*\d+$","", s)
    return s

def _best_canon(raw: str) -> str | None:
    n = _norm_header(raw)
    for canon, rules in SYNONYMS.items():
        if n in rules.get("exact", set()):
            return canon
    for canon, rules in SYNONYMS.items():
        for rx in rules.get("regex", []):
            if re.fullmatch(rx, n):
                return canon
    best = (0.0, None)
    for canon, rules in SYNONYMS.items():
        for token in rules.get("exact", set()):
            score = SequenceMatcher(None, n, token).ratio()
            if score > best[0]:
                best = (score, canon)
    return best[1] if best[0] >= 0.80 else None

def harmonize_headers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    header_map = {}
    for c in df.columns:
        canon = _best_canon(str(c)) or c
        if canon in header_map:
            a = df[header_map[canon]].notna().sum()
            b = df[c].notna().sum()
            if b > a:
                header_map[canon] = c
        else:
            header_map[canon] = c
    rename = {}
    for canon, orig in header_map.items():
        if canon in CANON and canon != orig:
            rename[orig] = canon
    if rename:
        df = df.rename(columns=rename)
    for need in CANON:
        if need not in df.columns:
            df[need] = pd.NA
    return df

# ==== PATCH 2: numeric coercion (same behavior + safe RPC fallback support) ====
def _coerce_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    header_map = {c.lower(): c for c in out.columns}

    # RPC-like
    rpc_candidates = [
        orig for low, orig in header_map.items()
        if ("rpc" in low) or (low in ("y rpc","y_rpc","yrpc")) or ("rev per click" in low)
    ]
    for col in rpc_candidates:
        out[col] = _coerce_numeric(out[col]).fillna(0.0)

    # Clicks-like
    clicks_candidates = [orig for low, orig in header_map.items() if "click" in low]
    for col in clicks_candidates:
        out[col] = _coerce_numeric(out[col]).fillna(0).astype("Int64")

    # Revenue-like
    if "Rev" in out.columns:
        out["Rev"] = _coerce_numeric(out["Rev"]).fillna(0.0)

    return out

# ==== PATCH 3: ensure_core_columns (same signature & return) ====
def ensure_core_columns(
    df: pd.DataFrame,
    default_geo: str,
    device_filter: str | None = None,
):
    if not isinstance(df, pd.DataFrame) or df.empty:
        empty = pd.DataFrame(columns=["Keyword","Device","Geo","RPC","Clicks","Date","Rev","TQ"])
        return empty, {
            "kw":"Keyword","device":"Device","geo":"Geo","rpc":"RPC",
            "clicks":"Clicks","date":"Date","rev":"Rev","tq":"TQ"
        }

    out = harmonize_headers(df.copy())

    # Ensure required columns exist
    if "Keyword" not in out.columns:
        out["Keyword"] = ""
    if "Device" not in out.columns:
        out["Device"] = ""
    if "Geo" not in out.columns:
        out["Geo"] = default_geo
    if "RPC" not in out.columns:
        out["RPC"] = 0.0
    if "Clicks" not in out.columns:
        out["Clicks"] = 0
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    else:
        out["Date"] = pd.Timestamp.utcnow().normalize()

    # Optional device filter
    if device_filter:
        filt = str(device_filter).strip().lower()
        out = out[out["Device"].astype(str).str.lower() == filt]

    # Coerce numerics (RPC/Clicks/Rev)
    out = _coerce_numeric_cols(out)

    # ---- NEW: safe fallback if RPC is missing or effectively zero but Rev+Clicks exist
    try:
        rpc_all_zero_or_nan = pd.to_numeric(out["RPC"], errors="coerce").fillna(0).eq(0).all()
    except Exception:
        rpc_all_zero_or_nan = True

    if rpc_all_zero_or_nan and ("Rev" in out.columns) and ("Clicks" in out.columns):
        denom = out["Clicks"].replace(0, pd.NA)
        out["RPC"] = (out["Rev"] / denom).fillna(0.0)

    # Final tidy
    out["Keyword"] = out["Keyword"].astype(str).str.strip()
    out["Device"]  = out["Device"].astype(str).str.strip()
    out["Geo"]     = out["Geo"].astype(str).str.strip().str.upper()

    # Drop rows with empty Keyword
    out = out[out["Keyword"] != ""]

    cols_map = {
        "kw":"Keyword","device":"Device","geo":"Geo","rpc":"RPC",
        "clicks":"Clicks","date":"Date","rev":"Rev","tq":"TQ"
    }
    return out, cols_map
# ---- end schema utilities ----

# =====================
# Paths & folders
# =====================
BASE_DIR = os.path.dirname(__file__)
OUTPUTS = os.path.join(BASE_DIR, "outputs")
EXPANSIONS = os.path.join(BASE_DIR, "expansions")
os.makedirs(OUTPUTS, exist_ok=True)
os.makedirs(EXPANSIONS, exist_ok=True)

BLOCKLIST_PATH = os.path.join(OUTPUTS, "custom_blocklist.txt")
HISTORY_PATH = os.path.join(OUTPUTS, "history.parquet")
LAUNCH_LOG_PATH = os.path.join(OUTPUTS, "launched_log.parquet")

st.set_page_config(
    page_title="Arb LaunchOps — Daily Trend Scanner & Planner", layout="wide"
)

# --- Safe default so diagnostics don't crash on first load ---
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = pd.DataFrame()
raw = st.session_state["raw_df"]

# --- SESSION INIT (ensures keys exist across reruns) ---
for _k in ("dod", "wow", "sur", "today_roll", "dod_plan", "wow_plan", "sur_plan"):
    if _k not in st.session_state:
        st.session_state[_k] = pd.DataFrame()

# ---- Launch log (full app) ----
LAUNCH_LOG_PATH = os.path.join(OUTPUTS, "launch_log.parquet")

def load_launch_log() -> pd.DataFrame:
    if os.path.exists(LAUNCH_LOG_PATH):
        try:
            return pd.read_parquet(LAUNCH_LOG_PATH)
        except Exception:
            csv_fallback = LAUNCH_LOG_PATH.replace(".parquet", ".csv")
            if os.path.exists(csv_fallback):
                return pd.read_csv(csv_fallback, parse_dates=["launched_at"])
    # default empty schema
    return pd.DataFrame(columns=["Keyword", "Geo", "Device", "launched_at", "source"])

def save_launch_log(df: pd.DataFrame) -> None:
    try:
        df.to_parquet(LAUNCH_LOG_PATH, index=False)
    except Exception:
        df.to_csv(LAUNCH_LOG_PATH.replace(".parquet", ".csv"), index=False)

def append_launch_log(plan_df: pd.DataFrame, source: str = "daily_plan") -> None:
    """Append today's planned rows to launch log (dedupe keys are Keyword+Geo)."""
    if plan_df is None or plan_df.empty:
        return
    df = plan_df.copy()
    # ensure columns exist
    for c in ("Keyword", "Geo", "Device"):
        if c not in df.columns:
            df[c] = ""
    df = df[["Keyword", "Geo", "Device"]]
    df["launched_at"] = pd.Timestamp.now().normalize()
    df["source"] = source

    log = load_launch_log()
    log = pd.concat([log, df], ignore_index=True)
    save_launch_log(log)

def pick_topN_avoiding_recent(
    plan_pool: pd.DataFrame,
    N: int,
    prevent_relaunch: bool,
    prevent_days: int,
) -> pd.DataFrame:
    """
    Return first N rows from *sorted* plan_pool that have NOT appeared
    in launch_log within the last `prevent_days`, using (Keyword, Geo) keys.
    """
    if not isinstance(plan_pool, pd.DataFrame) or plan_pool.empty:
        return pd.DataFrame(columns=(plan_pool.columns if isinstance(plan_pool, pd.DataFrame) else []))
    if not prevent_relaunch:
        return plan_pool.head(int(N)).copy()

    launch_log = load_launch_log()
    if launch_log is None or launch_log.empty:
        return plan_pool.head(int(N)).copy()

    for col in ("Keyword", "Geo"):
        if col not in launch_log.columns:
            launch_log[col] = ""

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(prevent_days))
    if "launched_at" in launch_log.columns:
        launch_log["launched_at"] = pd.to_datetime(launch_log["launched_at"], errors="coerce")
        recent = launch_log[launch_log["launched_at"] >= cutoff]
    else:
        recent = launch_log

    recent_keys = set(zip(recent["Keyword"].astype(str), recent["Geo"].astype(str)))

    pp = plan_pool.copy()
    for col in ("Keyword", "Geo"):
        if col not in pp.columns:
            pp[col] = ""
    mask = pp.apply(lambda r: (str(r["Keyword"]), str(r["Geo"])) in recent_keys, axis=1)
    filtered = pp[~mask]
    return filtered.head(int(N)).copy()

# =====================
# Storage helpers
# =====================
def load_launch_log() -> pd.DataFrame:
    if os.path.exists(LAUNCH_LOG_PATH):
        try:
            return pd.read_parquet(LAUNCH_LOG_PATH)
        except Exception:
            csv_fallback = LAUNCH_LOG_PATH.replace(".parquet", ".csv")
            if os.path.exists(csv_fallback):
                return pd.read_csv(csv_fallback, parse_dates=["launched_at"])
    return pd.DataFrame(columns=["Keyword", "Niche", "Geo", "launched_at", "source"])


def save_launch_log(df: pd.DataFrame):
    try:
        df.to_parquet(LAUNCH_LOG_PATH, index=False)
    except Exception:
        df.to_csv(LAUNCH_LOG_PATH.replace(".parquet", ".csv"), index=False)


def filter_not_launched_recent(
    df: pd.DataFrame, log: pd.DataFrame, window_days: int
) -> pd.DataFrame:
    if df.empty or log.empty:
        return df
    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(window_days))
    recent = log[pd.to_datetime(log["launched_at"], errors="coerce") >= cutoff]
    if recent.empty:
        return df
    keyset = set(
        zip(
            recent["Keyword"].astype(str),
            recent["Niche"].astype(str),
            recent["Geo"].astype(str),
        )
    )
    return df[
        ~df.apply(
            lambda r: (str(r["Keyword"]), str(r["Niche"]), str(r["Geo"])) in keyset,
            axis=1,
        )
    ]


@st.cache_data(show_spinner=False)
def _load_history_cache(mtime: int) -> pd.DataFrame:
    if os.path.exists(HISTORY_PATH):
        try:
            return pd.read_parquet(HISTORY_PATH)
        except Exception:
            csv_fallback = HISTORY_PATH.replace(".parquet", ".csv")
            if os.path.exists(csv_fallback):
                return pd.read_csv(csv_fallback, parse_dates=["date"])
    return pd.DataFrame(columns=["date", "Keyword", "Niche", "Geo", "Clicks", "RPC"])


def load_history() -> pd.DataFrame:
    mtime = int(os.path.getmtime(HISTORY_PATH)) if os.path.exists(HISTORY_PATH) else 0
    return _load_history_cache(mtime)


def save_history(df_hist: pd.DataFrame):
    try:
        df_hist.to_parquet(HISTORY_PATH, index=False)
    except Exception:
        df_hist.to_csv(HISTORY_PATH.replace(".parquet", ".csv"), index=False)


def save_csv(df: pd.DataFrame, name: str) -> str:
    p = os.path.join(OUTPUTS, name)
    df.to_csv(p, index=False)
    return p


# =====================
# Daily plan helpers: blacklist + ranking
# =====================


def _read_yaml_list(path, key):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        items = data.get(key, [])
        return [str(x).strip() for x in items if str(x).strip()]
    except Exception:
        return []


# Sensible fallbacks if filters.yaml doesn't exist
DEFAULT_HEALTH = [
    "diabetes",
    "cancer",
    "asthma",
    "arthritis",
    "depression",
    "anxiety",
    "autism",
    "adhd",
    "hiv",
    "covid",
]
DEFAULT_MEDS = [
    "ozempic",
    "wegovy",
    "metformin",
    "statin",
    "lipitor",
    "xanax",
    "adderall",
    "viagra",
]
DEFAULT_BRANDS = [
    "amazon",
    "walmart",
    "facebook",
    "google",
    "apple",
    "microsoft",
    "netflix",
    "tesla",
    "verizon",
    "t-mobile",
    "att",
    "bankrate",
    "cnet",
    "the verge",
]


def load_custom_blocklist() -> list[str]:
    try:
        if os.path.exists(BLOCKLIST_PATH):
            with open(BLOCKLIST_PATH, "r", encoding="utf-8") as f:
                items = [ln.strip() for ln in f.read().splitlines() if ln.strip()]
            # de-dupe, keep alpha-ish order
            return sorted(set(items), key=str.lower)
    except Exception:
        pass
    return []


def save_custom_blocklist(items: list[str]):
    uniq = sorted(set([i.strip() for i in items if str(i).strip()]), key=str.lower)
    os.makedirs(OUTPUTS, exist_ok=True)
    with open(BLOCKLIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(uniq))


def add_block_terms(new_text: str) -> list[str]:
    terms = [t.strip() for t in re.split(r"[,\n]", str(new_text or "")) if t.strip()]
    merged = sorted(set(load_custom_blocklist() + terms), key=str.lower)
    save_custom_blocklist(merged)
    return merged


def remove_block_terms(to_remove: list[str]) -> list[str]:
    cur = load_custom_blocklist()
    drop = set([t.strip().lower() for t in (to_remove or [])])
    kept = [t for t in cur if t.lower() not in drop]
    save_custom_blocklist(kept)
    return kept


def compile_blacklist():
    health = (
        _read_yaml_list(os.path.join(BASE_DIR, "filters.yaml"), "health_terms")
        or DEFAULT_HEALTH
    )
    meds = (
        _read_yaml_list(os.path.join(BASE_DIR, "filters.yaml"), "medications")
        or DEFAULT_MEDS
    )
    brands = (
        _read_yaml_list(os.path.join(BASE_DIR, "filters.yaml"), "brands")
        or DEFAULT_BRANDS
    )
    # include your persisted custom list
    try:
        custom = load_custom_blocklist()
    except Exception:
        custom = []

    # normalize, dedupe, and drop empties
    words = sorted(
        {w.strip().lower() for w in (health + meds + brands + custom)} - {""}
    )
    if not words:
        return None

    # case-insensitive, word-boundary match (handles single words & most phrases)
    return re.compile(r"(?i)\b(" + "|".join(re.escape(w) for w in words) + r")\b")


def apply_blacklist(
    df: pd.DataFrame, patt, text_cols=("Keyword", "Niche")
) -> pd.DataFrame:
    if df is None or df.empty or patt is None:
        return df
    mask = pd.Series(False, index=df.index)
    for col in text_cols:
        if col in df.columns:
            mask = mask | df[col].astype(str).str.contains(patt, na=False)
    return df[~mask]


def rank_and_pick_top(
    dod: pd.DataFrame, wow: pd.DataFrame, sur: pd.DataFrame, top_n: int = 25
) -> pd.DataFrame:
    parts = []
    if isinstance(dod, pd.DataFrame) and not dod.empty:
        d = dod.copy()
        d["segment"] = "DoD"
        d["lift_metric"] = d.get("rpc_lift_pct", 0.0)
        parts.append(d)
    if isinstance(wow, pd.DataFrame) and not wow.empty:
        w = wow.copy()
        w["segment"] = "WoW"
        w["lift_metric"] = w.get("rpc_lift_pct", 0.0)
        parts.append(w)
    if isinstance(sur, pd.DataFrame) and not sur.empty:
        s = sur.copy()
        s["segment"] = "Surge"
        s["lift_metric"] = s.get("lift_vs_base_pct", 0.0)
        parts.append(s)
    if not parts:
        return pd.DataFrame()
    allw = pd.concat(parts, ignore_index=True)
    allw.sort_values(
        ["lift_metric", "RPC", "Clicks"], ascending=[False, False, False], inplace=True
    )
    return allw.head(int(top_n))


def pick_topN_avoiding_recent(
    plan_pool: pd.DataFrame, N: int, prevent_relaunch: bool, prevent_days: int
) -> pd.DataFrame:
    """
    From a *sorted* plan_pool, return the first N rows that have NOT appeared
    on a daily plan (logged) in the last `prevent_days`.

    Works whether your plan uses Niche or Device (uses whichever exists in BOTH
    plan_pool and the launch log).
    """
    if not isinstance(plan_pool, pd.DataFrame) or plan_pool.empty:
        return pd.DataFrame(columns=(plan_pool.columns if isinstance(plan_pool, pd.DataFrame) else []))

    if not prevent_relaunch:
        return plan_pool.head(int(N)).copy()

    log = load_launch_log()
    if log is None or log.empty:
        return plan_pool.head(int(N)).copy()

    cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=int(prevent_days))
    recent = log[pd.to_datetime(log["launched_at"], errors="coerce") >= cutoff]
    if recent.empty:
        return plan_pool.head(int(N)).copy()

    # Decide which middle key to use: prefer Niche if both have it; else Device.
    middle_key = None
    if "Niche" in plan_pool.columns and "Niche" in recent.columns:
        middle_key = "Niche"
    elif "Device" in plan_pool.columns and "Device" in recent.columns:
        middle_key = "Device"

    # Build the list of key columns present in BOTH frames.
    key_cols = ["Keyword"]
    if middle_key:
        key_cols.append(middle_key)
    if "Geo" in plan_pool.columns and "Geo" in recent.columns:
        key_cols.append("Geo")

    # If we somehow only have Keyword in common, still dedupe on that.
    recent_keys = set(
        map(tuple, recent[key_cols].astype(str).itertuples(index=False, name=None))
    )
    plan_keys = plan_pool[key_cols].astype(str).apply(tuple, axis=1)

    filtered = plan_pool[~plan_keys.isin(recent_keys)]
    return filtered.head(int(N)).copy()

# =====================
# Ingest & normalization
# =====================
@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(file_bytes))
    except Exception:
        return pd.read_excel(io.BytesIO(file_bytes))


def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["Clicks", "RPC", "CPC", "Ad_Clicks", "Revenue"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "Clicks" in out.columns:
        out["Clicks"] = out["Clicks"].fillna(0)
    if "RPC" in out.columns:
        out["RPC"] = out["RPC"].fillna(0.0)
    if "CPC" in out.columns:
        out["CPC"] = out["CPC"].fillna(0.0)
    if "Ad_Clicks" in out.columns:
        out["Ad_Clicks"] = out["Ad_Clicks"].fillna(0)
    if "Revenue" in out.columns:
        out["Revenue"] = out["Revenue"].fillna(0.0)
    return out

def ensure_core_columns_rpc(df: pd.DataFrame, default_geo: str, device_filter: str | None = None):
    """
    Normalize to: Keyword, Device, RPC, Geo
    Accepts: Keyword/kw/relatedkw/realtedkw; Device/name/platform; RPC/Y RPC; Geo/Country/Region
    """
    df = df.copy()
    header_map = {c.lower().strip(): c for c in df.columns}

    def _find_col_by_prefix(header_map: dict, *prefixes: str) -> str | None:
        for pref in prefixes:
            for low, orig in header_map.items():
                if low.startswith(pref):
                    return orig
        return None

    kw_col = (header_map.get("keyword") or header_map.get("kw")
              or header_map.get("relatedkw") or header_map.get("realtedkw")
              or _find_col_by_prefix(header_map, "keyword", "kw", "relatedkw", "realtedkw"))
    dev_col = (header_map.get("device") or header_map.get("name") or header_map.get("platform")
               or _find_col_by_prefix(header_map, "device", "name", "platform"))
    rpc_col = (header_map.get("y rpc") or header_map.get("rpc")
               or _find_col_by_prefix(header_map, "y rpc", "rpc"))
    geo_col = (header_map.get("geo") or header_map.get("country") or header_map.get("region")
               or _find_col_by_prefix(header_map, "geo", "country", "region"))

    if kw_col and kw_col != "Keyword":
        df.rename(columns={kw_col: "Keyword"}, inplace=True)
    if dev_col and dev_col != "Device":
        df.rename(columns={dev_col: "Device"}, inplace=True)
    if rpc_col and rpc_col != "RPC":
        df.rename(columns={rpc_col: "RPC"}, inplace=True)
    if geo_col and geo_col != "Geo":
        df.rename(columns={geo_col: "Geo"}, inplace=True)

    # Optional device filter
    if device_filter and "Device" in df.columns:
        df = df[df["Device"].astype(str).str.lower() == device_filter.strip().lower()]

    # Ensure required columns exist
    if "Keyword" not in df.columns:
        df["Keyword"] = ""
    if "Device" not in df.columns:
        df["Device"] = ""
    if "RPC" not in df.columns:
        df["RPC"] = 0.0
    if "Geo" not in df.columns:
        df["Geo"] = default_geo

    # **CRITICAL**: fill NaNs in groupby keys; NaN in any key drops the row in groupby
    df["Keyword"] = df["Keyword"].fillna("").astype(str).str.strip()
    df["Device"]  = df["Device"].fillna("").astype(str).str.strip()
    df["Geo"]     = df["Geo"].fillna(default_geo).astype(str).str.strip()

    # Coerce RPC to numeric
    if "RPC" in df.columns:
        df["RPC"] = df["RPC"].astype(str).str.replace(r"[^0-9.\-]", "", regex=True)
        df["RPC"] = pd.to_numeric(df["RPC"], errors="coerce").fillna(0.0)

    return df, {"rpc": "RPC", "geo": "Geo", "device": "Device", "kw": "Keyword"}

def rollup_today(df: pd.DataFrame, cols: dict, day: date) -> pd.DataFrame:
    """
    Build today's rollup robustly:
    - Ensures Keyword / Device-or-Niche / Geo / RPC exist (renames from 'cols' when needed)
    - Groups by the subset of those columns that actually exist
    - Averages RPC and (if present) sums Clicks
    """
    day_str = day.strftime("%Y-%m-%d")
    out = df.copy()

    # --- Normalize column names (use mapping in `cols` when available) ---
    # Keyword
    kw_src = "Keyword"
    if "Keyword" not in out.columns and cols and cols.get("kw") in out.columns:
        kw_src = cols["kw"]
    if kw_src in out.columns and kw_src != "Keyword":
        out.rename(columns={kw_src: "Keyword"}, inplace=True)

    # Device or Niche (prefer Device; fall back to Niche; else Name)
    dev_src = None
    if "Device" in out.columns:
        dev_src = "Device"
    elif "Niche" in out.columns:
        dev_src = "Niche"
    elif "Name" in out.columns:
        dev_src = "Name"
    if dev_src and dev_src != "Device":
        # standardize to Device for grouping
        out.rename(columns={dev_src: "Device"}, inplace=True)

    # Geo
    geo_src = "Geo"
    if "Geo" not in out.columns and cols and cols.get("geo") in out.columns:
        geo_src = cols["geo"]
    if geo_src in out.columns and geo_src != "Geo":
        out.rename(columns={geo_src: "Geo"}, inplace=True)
    if "Geo" not in out.columns:
        # final fallback if no geo present at all
        out["Geo"] = "US"

    # RPC
    rpc_src = "RPC"
    if "RPC" not in out.columns and cols and cols.get("rpc") in out.columns:
        rpc_src = cols["rpc"]
    if rpc_src in out.columns and rpc_src != "RPC":
        out.rename(columns={rpc_src: "RPC"}, inplace=True)
    if "RPC" not in out.columns:
        # last-ditch: try compute RPC if we have revenue & clicks columns
        if "Revenue" in out.columns and "Clicks" in out.columns:
            rev = pd.to_numeric(out["Revenue"], errors="coerce")
            clk = pd.to_numeric(out["Clicks"], errors="coerce").replace(0, pd.NA)
            out["RPC"] = (rev / clk).fillna(0.0)
        else:
            out["RPC"] = 0.0

    # Coerce numerics
    out["RPC"] = pd.to_numeric(out["RPC"], errors="coerce").fillna(0.0)
    if "Clicks" in out.columns:
        out["Clicks"] = pd.to_numeric(out["Clicks"], errors="coerce").fillna(0)

    # --- Build grouping keys based on what actually exists ---
    group_keys = [c for c in ["Keyword", "Device", "Geo"] if c in out.columns]
    if not group_keys:
        # If nothing to group by, return empty with expected columns
        res = pd.DataFrame(columns=["date", "Keyword", "Device", "Geo", "RPC"])
        return res

    # --- Aggregations ---
    agg_map = {"RPC": "mean"}
    if "Clicks" in out.columns:
        agg_map["Clicks"] = "sum"

    agg = out.groupby(group_keys, as_index=False).agg(agg_map)
    agg.insert(0, "date", day_str)
    return agg

# ---- Normalized join keys (for robust DoD/WoW matching) ----
def _add_norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalized join keys for robust matching across days/weeks."""
    out = df.copy()
    for col in ("Keyword", "Niche", "Geo"):
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].astype(str)

    # case/whitespace-insensitive for Keyword/Niche; uppercase for Geo codes
    out["k_norm"] = out["Keyword"].str.strip().str.casefold()
    out["n_norm"] = out["Niche"].str.strip().str.casefold()
    out["g_norm"] = out["Geo"].str.strip().str.upper()
    return out


def _merge_revctr_into(df: pd.DataFrame, today_roll: pd.DataFrame) -> pd.DataFrame:
    """Left-merge RevCTR from today's rollup by (Keyword, Niche, Geo)."""
    if df is None or df.empty or today_roll is None or today_roll.empty:
        return df
    keys = ["Keyword", "Niche", "Geo"]
    if (
        not all(k in today_roll.columns for k in keys)
        or "RevCTR" not in today_roll.columns
    ):
        return df
    return df.merge(today_roll[keys + ["RevCTR"]], on=keys, how="left")


# =====================
# Scoring & segments
# =====================
def cpc_cap_calc(rpc_today, rpc_prior, clicks_today, K, factor, volume_clicks):
    rpc_stab = ((clicks_today * rpc_today) + (K * rpc_prior)) / max(
        (clicks_today + K), 1e-9
    )
    cap = rpc_stab * factor
    if volume_clicks < 50:
        cap -= 0.05
    elif volume_clicks > 500:
        cap += 0.05
    cap = max(0.05, min(0.90 * max(rpc_today, 0.0), cap))
    return round(cap, 2)


def winners_DoD(hist: pd.DataFrame, min_clicks: int, K: float, factor_DoD: float):
    if hist.empty:
        return pd.DataFrame()

    df = hist.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 2:
        return pd.DataFrame()

    today, yest = dates[-1], dates[-2]

    # normalized keys
    df = _add_norm_cols(df)

    # today vs yesterday
    d0 = df[df["date"] == today].copy()
    d1 = df[df["date"] == yest][["k_norm", "n_norm", "g_norm", "Clicks", "RPC"]].rename(
        columns={"Clicks": "Clicks_prior", "RPC": "RPC_prior"}
    )

    out = d0.merge(d1, on=["k_norm", "n_norm", "g_norm"], how="left")

    # 7-day baseline prior to today (as a smarter fallback)
    prev7 = df[(df["date"] < today) & (df["date"] >= today - pd.Timedelta(days=7))]
    base7 = prev7.groupby(["k_norm", "n_norm", "g_norm"], as_index=False).agg(
        RPC_baseline=("RPC", "mean")
    )
    out = out.merge(base7, on=["k_norm", "n_norm", "g_norm"], how="left")

    # fill priors
    out["Clicks_prior"] = out["Clicks_prior"].fillna(0)
    if len(out):
        out["RPC_prior"] = out["RPC_prior"].fillna(out["RPC_baseline"])
        out["RPC_prior"] = out["RPC_prior"].fillna(out["RPC"].median())
    else:
        out["RPC_prior"] = 0.0

    # metrics
    out["rpc_lift_pct"] = (out["RPC"] - out["RPC_prior"]) / out["RPC_prior"].replace(
        0, 1
    )
    out["cpc_cap"] = out.apply(
        lambda r: cpc_cap_calc(
            r["RPC"], r["RPC_prior"], r["Clicks"], K, factor_DoD, r["Clicks"]
        ),
        axis=1,
    )

    # filter & sort
    out = out[out["Clicks"] >= min_clicks].copy()
    out.sort_values(
        ["rpc_lift_pct", "RPC", "Clicks"], ascending=[False, False, False], inplace=True
    )

    # clean up helper cols
    out.drop(
        columns=[
            c
            for c in ("k_norm", "n_norm", "g_norm", "RPC_baseline")
            if c in out.columns
        ],
        inplace=True,
    )

    return out


def winners_WoW(hist: pd.DataFrame, min_clicks: int, K: float, factor_WoW: float):
    if hist.empty:
        return pd.DataFrame()

    df = hist.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    last_day = df["date"].max()
    if pd.isna(last_day):
        return pd.DataFrame()

    this_week_start = last_day - timedelta(days=6)
    prev_week_end = this_week_start - timedelta(days=1)
    prev_week_start = prev_week_end - timedelta(days=6)

    # normalized keys
    df = _add_norm_cols(df)

    wk = df[(df["date"] >= this_week_start) & (df["date"] <= last_day)].copy()
    pw = df[(df["date"] >= prev_week_start) & (df["date"] <= prev_week_end)].copy()

    if wk.empty:
        return pd.DataFrame()

    # aggregate current week, keep a representative display value for text cols
    w0 = wk.groupby(["k_norm", "n_norm", "g_norm"], as_index=False).agg(
        Clicks=("Clicks", "sum"),
        RPC=("RPC", "mean"),
        Keyword=("Keyword", "first"),
        Niche=("Niche", "first"),
        Geo=("Geo", "first"),
    )

    # aggregate previous week
    w1 = pw.groupby(["k_norm", "n_norm", "g_norm"], as_index=False).agg(
        Clicks_prior=("Clicks", "sum"), RPC_prior=("RPC", "mean")
    )

    out = w0.merge(w1, on=["k_norm", "n_norm", "g_norm"], how="left")

    # 28-day baseline before this week as smarter fallback
    prev28 = df[
        (df["date"] < this_week_start)
        & (df["date"] >= this_week_start - pd.Timedelta(days=28))
    ]
    base28 = prev28.groupby(["k_norm", "n_norm", "g_norm"], as_index=False).agg(
        RPC_baseline=("RPC", "mean")
    )
    out = out.merge(base28, on=["k_norm", "n_norm", "g_norm"], how="left")

    # fill priors
    out["Clicks_prior"] = out["Clicks_prior"].fillna(0)
    if len(out):
        out["RPC_prior"] = out["RPC_prior"].fillna(out["RPC_baseline"])
        out["RPC_prior"] = out["RPC_prior"].fillna(out["RPC"].median())
    else:
        out["RPC_prior"] = 0.0

    # metrics
    out["rpc_lift_pct"] = (out["RPC"] - out["RPC_prior"]) / out["RPC_prior"].replace(
        0, 1
    )
    out["cpc_cap"] = out.apply(
        lambda r: cpc_cap_calc(
            r["RPC"], r["RPC_prior"], r["Clicks"], K, factor_WoW, r["Clicks"]
        ),
        axis=1,
    )

    # filter & sort
    out = out[out["Clicks"] >= min_clicks].copy()
    out.sort_values(
        ["rpc_lift_pct", "RPC", "Clicks"], ascending=[False, False, False], inplace=True
    )

    # clean up helper cols
    out.drop(
        columns=[
            c
            for c in ("k_norm", "n_norm", "g_norm", "RPC_baseline")
            if c in out.columns
        ],
        inplace=True,
    )

    return out

def surges_low_to_high(
    hist: pd.DataFrame,
    min_clicks: int,
    baseline_weeks: int,
    surge_threshold: float,
    K: float,
    factor_surge: float,
):
    if hist.empty:
        return pd.DataFrame()
    df = hist.copy()
    df["date"] = pd.to_datetime(df["date"])
    last_day = df["date"].max()
    baseline_start = last_day - timedelta(days=7 * baseline_weeks)
    base = df[(df["date"] >= baseline_start) & (df["date"] < last_day)]
    today = df[df["date"] == last_day]
    if base.empty or today.empty:
        return pd.DataFrame()
    b = base.groupby(["Keyword", "Niche", "Geo"], as_index=False).agg(
        base_clicks=("Clicks", "sum"), base_rpc=("RPC", "mean")
    )
    t = today[["Keyword", "Niche", "Geo", "Clicks", "RPC"]]
    out = t.merge(b, on=["Keyword", "Niche", "Geo"], how="left").fillna(
        {"base_clicks": 0, "base_rpc": t["RPC"].median() if len(t) else 0}
    )
    if len(out) >= 4:
        q1 = out["base_rpc"].quantile(0.25)
        out = out[out["base_rpc"] <= q1]
    out["lift_vs_base_pct"] = (out["RPC"] - out["base_rpc"]) / out["base_rpc"].replace(
        0, 1
    )
    out = out[
        (out["lift_vs_base_pct"] >= surge_threshold) & (out["Clicks"] >= min_clicks)
    ]
    out["cpc_cap"] = out.apply(
        lambda r: cpc_cap_calc(
            r["RPC"], r["base_rpc"], r["Clicks"], K, factor_surge, r["Clicks"]
        ),
        axis=1,
    )
    out.sort_values(
        ["lift_vs_base_pct", "RPC", "Clicks"],
        ascending=[False, False, False],
        inplace=True,
    )
    return out


def write_plan_payload(dod: pd.DataFrame, wow: pd.DataFrame, sur: pd.DataFrame):
    top_dod = (
        dod.head(10)[
            ["Keyword", "Niche", "Geo", "RPC", "Clicks", "rpc_lift_pct", "cpc_cap"]
        ]
        if not dod.empty
        else pd.DataFrame()
    )
    top_wow = (
        wow.head(10)[
            ["Keyword", "Niche", "Geo", "RPC", "Clicks", "rpc_lift_pct", "cpc_cap"]
        ]
        if not wow.empty
        else pd.DataFrame()
    )
    top_sur = (
        sur.head(10)[
            ["Keyword", "Niche", "Geo", "RPC", "Clicks", "lift_vs_base_pct", "cpc_cap"]
        ]
        if not sur.empty
        else pd.DataFrame()
    )
    lines = [
        "Arb LaunchOps — Daily Launch Summary",
        "\nDoD Winners (Top 10):\n"
        + (top_dod.to_csv(index=False) if not top_dod.empty else "(none)"),
        "\nWoW Winners (Top 10):\n"
        + (top_wow.to_csv(index=False) if not top_wow.empty else "(none)"),
        "\nSurges (Top 10):\n"
        + (top_sur.to_csv(index=False) if not top_sur.empty else "(none)"),
    ]
    txt = "\n".join(lines)
    with open(os.path.join(OUTPUTS, "chat_payload.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    plan = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dod": top_dod.to_dict(orient="records"),
        "wow": top_wow.to_dict(orient="records"),
        "surges": top_sur.to_dict(orient="records"),
    }
    with open(os.path.join(OUTPUTS, "plan_input.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

# =====================
# UI — Daily Scanner
# =====================
st.title("Arb LaunchOps — Daily Trend Scanner & Planner")

with st.expander("Daily Scanner", expanded=True):
    file = st.file_uploader(
        "Upload Daily CSV/XLSX", type=["csv", "xlsx"], accept_multiple_files=False
    )

    # Safe default for raw before any run
    raw = st.session_state.get("raw_df", pd.DataFrame())

    c1, c2, c3 = st.columns(3)
    with c1:
        min_clicks = st.number_input("Min Clicks", 0, 1_000_000, 100, 1)
        baseline_weeks = st.number_input("Surge Baseline (weeks)", 1, 52, 8, 1)
        default_geo = st.text_input("Default GEO", "US")
        device_filter = st.text_input("Filter by Device equals (optional)", "")
        name_filter = st.text_input("Filter by Name equals (optional)", "")
        build_plan = st.checkbox("Build plan files after scan", True)
        prevent_relaunch = st.checkbox(
            "Prevent relaunch if seen in last N days (applies to PLAN only)", True
        )
        prevent_days = st.number_input("N days window (dedupe)", 1, 90, 30, 1)
    with c2:
        rpc_lift_gate = st.number_input(
            "Lift Gate (rpc_lift_pct)", 0.0, 5.0, 0.25, 0.01, format="%.2f"
        )
        surge_gate = st.number_input(
            "Surge Gate (lift_vs_base)", 0.0, 5.0, 0.50, 0.01, format="%.2f"
        )
        K = st.number_input("CPC K (shrinkage)", 0.0, 10.0, 3.0, 0.1)
    with c3:
        factor_DoD = st.number_input("Factor DoD", 0.0, 2.0, 0.70, 0.01)
        factor_WoW = st.number_input("Factor WoW", 0.0, 2.0, 0.75, 0.01)
        factor_Surge = st.number_input("Factor Surge", 0.0, 2.0, 0.60, 0.01)

    # --- Custom Blocklist manager (persisted) ---
    with st.expander("Custom Blocklist (persisted)", expanded=False):
        st.caption(
            "Add keywords/phrases to permanently exclude from the **Daily Plan**. Stored in outputs/custom_blocklist.txt"
        )
        cur = load_custom_blocklist()
        st.write({"count": len(cur)})
        if cur:
            st.dataframe(pd.DataFrame({"blocked": cur}), use_container_width=True, height=200)

        add_text = st.text_area("Add terms (comma or newline separated)", "", height=80)
        cba, cbb, cbc = st.columns([1, 1, 1])
        with cba:
            if st.button("Add terms", key="bl_add"):
                new_list = add_block_terms(add_text)
                st.success(f"Added. Total blocked: {len(new_list)}")
        rem_sel = st.multiselect("Remove selected blocked terms", cur, key="bl_remove_sel")
        with cbb:
            if st.button("Remove selected", key="bl_remove_btn"):
                new_list = remove_block_terms(rem_sel)
                st.success(f"Removed. Total blocked: {len(new_list)}")
        with cbc:
            if st.button("Clear all", key="bl_clear_all"):
                save_custom_blocklist([])
                st.success("Custom blocklist cleared.")

    # --- Run button (keep this AFTER the blocklist panel) ---
    run = st.button("Run Scan", key="run_scan_btn")

    if run and file is not None:
        # Ingest
        raw = load_csv(file.read())
        raw = _coerce_numeric_cols(raw)

        # Normalize once (optional device filter applied here)
        raw, cols = ensure_core_columns(
            raw,
            default_geo=default_geo,
            device_filter=(device_filter.strip() or None),
        )

        # Optional Name filter (only if a 'Name' column actually exists)
        if name_filter.strip() and "Name" in raw.columns:
            raw = raw[raw["Name"].astype(str).str.lower() == name_filter.strip().lower()]

        # Re-coerce numerics after any filtering
        raw = _coerce_numeric_cols(raw)
        st.session_state["raw_df"] = raw

        today = date.today()

        # History update
        hist = load_history()
        today_roll = rollup_today(raw, cols, today)
        if today_roll is None or today_roll.empty or "date" not in today_roll.columns:
            st.error("No rows in today's rollup. Check your upload/column mapping or filters.")
            st.stop()

        if not hist.empty:
            day_key = today.strftime("%Y-%m-%d")
            if "date" in hist.columns:
                hist = hist[hist["date"].astype(str) != day_key]
            hist = pd.concat([hist, today_roll], ignore_index=True)
        else:
            hist = today_roll.copy()

        save_history(hist)
        save_csv(today_roll, "rollup_daily.csv")

        # Winners
        dod = winners_DoD(hist, min_clicks=min_clicks, K=K, factor_DoD=factor_DoD)
        wow = winners_WoW(hist, min_clicks=min_clicks, K=K, factor_WoW=factor_WoW)
        sur = surges_low_to_high(
            hist,
            min_clicks=min_clicks,
            baseline_weeks=int(baseline_weeks),
            surge_threshold=surge_gate,
            K=K,
            factor_surge=factor_Surge,
        )

        # 1) Apply lift gate on the raw winners (no dedupe yet)
        cols_dod = ["Keyword", "Niche", "Geo", "RPC", "Clicks", "rpc_lift_pct", "cpc_cap"]
        cols_wow = ["Keyword", "Niche", "Geo", "RPC", "Clicks", "rpc_lift_pct", "cpc_cap"]
        cols_sur = ["Keyword", "Niche", "Geo", "RPC", "Clicks", "lift_vs_base_pct", "cpc_cap"]

        if not dod.empty:
            dod = dod[dod["rpc_lift_pct"] >= float(rpc_lift_gate)]
        if not wow.empty:
            wow = wow[wow["rpc_lift_pct"] >= float(rpc_lift_gate)]

        # 2) Always write CSVs from the raw winners (pre-dedupe)
        save_csv(dod if not dod.empty else pd.DataFrame(columns=cols_dod), "winners_today.csv")
        save_csv(wow if not wow.empty else pd.DataFrame(columns=cols_wow), "winners_week.csv")
        save_csv(sur if not sur.empty else pd.DataFrame(columns=cols_sur), "surges_low_to_high.csv")

        # 3) Make raw winners available to other panels
        st.session_state["dod"] = dod.copy() if isinstance(dod, pd.DataFrame) else pd.DataFrame(columns=cols_dod)
        st.session_state["wow"] = wow.copy() if isinstance(wow, pd.DataFrame) else pd.DataFrame(columns=cols_wow)
        st.session_state["sur"] = sur.copy() if isinstance(sur, pd.DataFrame) else pd.DataFrame(columns=cols_sur)
        st.session_state["today_roll"] = today_roll.copy() if isinstance(today_roll, pd.DataFrame) else pd.DataFrame()

        # 4) Build PLAN copies and apply Prevent Relaunch ONLY to the plan
        dod_plan = dod.copy()
        wow_plan = wow.copy()
        sur_plan = sur.copy()

        # Daily plan (top N, blacklist, re-dedupe) with refilling
        TOP_N_PLAN = st.number_input("Daily plan size", 5, 200, 25, 5, key="plan_top_n")
        plan_pool = rank_and_pick_top(dod, wow, sur, top_n=TOP_N_PLAN * 4)  # grab extra slack
        patt = compile_blacklist()
        plan_pool = apply_blacklist(plan_pool, patt, text_cols=("Keyword", "Niche"))
        if not plan_pool.empty:
            plan_pool.sort_values(["lift_metric", "RPC", "Clicks"], ascending=[False, False, False], inplace=True)
            plan_top = pick_topN_avoiding_recent(
                plan_pool, TOP_N_PLAN, prevent_relaunch, prevent_days
            )
        else:
            plan_top = pd.DataFrame()

        if not plan_top.empty and "segment" in plan_top.columns:
            dod_plan = plan_top[plan_top["segment"] == "DoD"].copy()
            wow_plan = plan_top[plan_top["segment"] == "WoW"].copy()
            sur_plan = plan_top[plan_top["segment"] == "Surge"].copy()
        else:
            dod_plan = pd.DataFrame()
            wow_plan = pd.DataFrame()
            sur_plan = pd.DataFrame()

        # Add RevCTR from today's rollup
        tr = st.session_state.get("today_roll", pd.DataFrame())
        dod_plan = _merge_revctr_into(dod_plan, tr)
        wow_plan = _merge_revctr_into(wow_plan, tr)
        sur_plan = _merge_revctr_into(sur_plan, tr)

        # Keep plan copies for sidebar
        st.session_state["dod_plan"] = dod_plan.copy()
        st.session_state["wow_plan"] = wow_plan.copy()
        st.session_state["sur_plan"] = sur_plan.copy()

        # Write plan payloads + single daily CSV (deduped)
        if build_plan:
            write_plan_payload(dod_plan, wow_plan, sur_plan)
            plan_cols = [
                "segment",
                "Keyword",
                "Niche",
                "Geo",
                "RPC",
                "Clicks",
                "cpc_cap",
                "lift_metric",
                "RevCTR",
            ]

            def _pick_cols(df):
                if df is None or df.empty:
                    return pd.DataFrame(columns=plan_cols)
                for col in plan_cols:
                    if col not in df.columns:
                        df[col] = None
                return df[plan_cols]

            daily_plan = pd.concat(
                [_pick_cols(dod_plan), _pick_cols(wow_plan), _pick_cols(sur_plan)],
                ignore_index=True,
            )
            daily_plan_path = os.path.join(OUTPUTS, "daily_plan.csv")
            daily_plan.to_csv(daily_plan_path, index=False)

            # --- Sanity: how many of these appeared in last 30 days (Keyword+Geo only)
            try:
                _llp = os.path.join(OUTPUTS, "launch_log.parquet")
                if os.path.exists(_llp) and not daily_plan.empty:
                    _log = pd.read_parquet(_llp)
                    _log["launched_at"] = pd.to_datetime(_log["launched_at"], errors="coerce")
                    _cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)
                    _recent = _log[_log["launched_at"] >= _cutoff]
                    _keys_recent = set(zip(_recent["Keyword"].astype(str), _recent["Geo"].astype(str)))
                    _dupes_in_plan = sum(
                        (str(r["Keyword"]), str(r.get("Geo", ""))) in _keys_recent
                        for _, r in daily_plan.iterrows()
                    )
                    st.caption(f"Plan sanity: {int(_dupes_in_plan)} of {len(daily_plan)} appear in launch_log within 30 days.")
            except Exception:
                pass

            # --- Append today's plan to launch_log.parquet ---
            try:
                ll_path = os.path.join(OUTPUTS, "launch_log.parquet")
                log_now = pd.Timestamp.now()
                to_log = daily_plan.copy()
                for col in ["Keyword", "Niche", "Geo", "Device"]:
                    if col not in to_log.columns:
                        to_log[col] = ""
                to_log = to_log[["Keyword", "Niche", "Geo", "Device"]].copy()
                to_log["launched_at"] = log_now
                to_log["source"] = "daily_plan"

                if os.path.exists(ll_path):
                    _prior = pd.read_parquet(ll_path)
                    to_log = pd.concat([_prior, to_log], ignore_index=True)

                to_log.to_parquet(ll_path, index=False)
            except Exception as e:
                st.warning(f"Could not append to launch log: {e}")

            st.success(f"Saved deduped plan CSV (top {TOP_N_PLAN}) → {daily_plan_path}")

        st.success("Scan complete.")
        st.caption(f"Outputs folder: {OUTPUTS}")

# --- Show tables in UI (persist across reruns) ---
cA, cB, cC = st.columns(3)
with cA:
    st.caption("DoD Winners")
    st.dataframe(st.session_state.get("dod"), use_container_width=True)
with cB:
    st.caption("WoW Winners")
    st.dataframe(st.session_state.get("wow"), use_container_width=True)
with cC:
    st.caption("Surges (Low → High)")
    st.dataframe(st.session_state.get("sur"), use_container_width=True)

st.divider()

# =====================
# Public Signals (Trends + Suggest + RSS + Modifiers) — robust seed sourcing
# =====================

# --- Public Signals — Settings (edit feeds.yaml / modifiers.yaml) ---
with st.expander(
    "Public Signals — Settings (feeds.yaml / modifiers.yaml)", expanded=False
):
    FEEDS_PATH = os.path.join(BASE_DIR, "feeds.yaml")
    MODS_PATH = os.path.join(BASE_DIR, "modifiers.yaml")

    def _safe_load_yaml(path, default):
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            else:
                data = {}
        except Exception as e:
            st.warning(f"Could not read {os.path.basename(path)}: {e}")
            data = {}
        out = default.copy()
        out.update({k: v for k, v in (data or {}).items() if v is not None})
        return out

    def _safe_dump_yaml(path, data):
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
            return True
        except Exception as e:
            st.error(f"Failed to save {os.path.basename(path)}: {e}")
            return False

    # Defaults shown if files missing/corrupt
    default_feeds = {
        "feeds": [
            "https://www.bankrate.com/rss/",
            "https://www.theverge.com/rss/index.xml",
            "https://www.cnet.com/rss/news/",
        ]
    }
    default_mods = {
        "modifiers": [
            {"term": "prices", "weight": 1.0, "cpc_adj_pct": "+5"},
            {"term": "deals", "weight": 0.9, "cpc_adj_pct": "+5"},
            {"term": "near me", "weight": 0.6, "cpc_adj_pct": "-10"},
            {"term": "2025", "weight": 0.7, "cpc_adj_pct": "0"},
            {"term": "guide", "weight": 0.5, "cpc_adj_pct": "0"},
            {"term": "no contract", "weight": 0.6, "cpc_adj_pct": "0"},
            {"term": "for seniors", "weight": 0.8, "cpc_adj_pct": "0"},
        ]
    }

    feeds_obj = _safe_load_yaml(FEEDS_PATH, default_feeds)
    mods_obj = _safe_load_yaml(MODS_PATH, default_mods)

    c1, c2 = st.columns(2)
    with c1:
        st.caption(f"feeds.yaml • {len(feeds_obj.get('feeds', []))} feeds")
        feeds_text = st.text_area(
            "Edit feeds.yaml",
            value=yaml.safe_dump(feeds_obj, sort_keys=False, allow_unicode=True),
            height=220,
            key="feeds_yaml_editor",
        )
        if st.button("Save feeds.yaml", key="save_feeds_yaml"):
            try:
                parsed = yaml.safe_load(feeds_text) or {}
                if not isinstance(parsed.get("feeds", []), list):
                    raise ValueError("Expected top-level key 'feeds' with a list.")
                if _safe_dump_yaml(FEEDS_PATH, parsed):
                    st.success(f"Saved {FEEDS_PATH}")
            except Exception as e:
                st.error(f"Invalid YAML: {e}")

    with c2:
        st.caption(f"modifiers.yaml • {len(mods_obj.get('modifiers', []))} modifiers")
        mods_text = st.text_area(
            "Edit modifiers.yaml",
            value=yaml.safe_dump(mods_obj, sort_keys=False, allow_unicode=True),
            height=220,
            key="mods_yaml_editor",
        )
        if st.button("Save modifiers.yaml", key="save_mods_yaml"):
            try:
                parsed = yaml.safe_load(mods_text) or {}
                if not isinstance(parsed.get("modifiers", []), list):
                    raise ValueError("Expected top-level key 'modifiers' with a list.")
                # Normalize cpc_adj_pct to strings with sign (e.g., "+5" / "-10" / "0")
                for m in parsed["modifiers"]:
                    if "cpc_adj_pct" in m:
                        m["cpc_adj_pct"] = str(m["cpc_adj_pct"]).strip()
                        if m["cpc_adj_pct"] not in ("0", "+0", "-0"):
                            if not m["cpc_adj_pct"].startswith(("+", "-")):
                                m["cpc_adj_pct"] = "+" + m["cpc_adj_pct"]
                if _safe_dump_yaml(MODS_PATH, parsed):
                    st.success(f"Saved {MODS_PATH}")
            except Exception as e:
                st.error(f"Invalid YAML: {e}")

    st.caption("Quick add")
    f_add = st.text_input("Add a feed URL", key="feed_add")
    m_term = st.text_input("Add modifier term", key="mod_add_term")
    m_weight = st.number_input("Weight", 0.0, 5.0, 0.5, 0.1, key="mod_add_weight")
    m_pct = st.text_input("CPC % adjust (e.g., +5, -10, 0)", "+0", key="mod_add_pct")

    c3, c4, c5 = st.columns([1, 1, 2])
    with c3:
        if st.button("➕ Add feed"):
            feeds_obj["feeds"] = feeds_obj.get("feeds", [])
            if f_add:
                feeds_obj["feeds"].append(f_add.strip())
                _safe_dump_yaml(FEEDS_PATH, feeds_obj)
                st.success("Feed added.")
    with c4:
        if st.button("➕ Add modifier"):
            mods_obj["modifiers"] = mods_obj.get("modifiers", [])
            if m_term:
                mods_obj["modifiers"].append(
                    {
                        "term": m_term.strip(),
                        "weight": float(m_weight),
                        "cpc_adj_pct": m_pct.strip(),
                    }
                )
                _safe_dump_yaml(MODS_PATH, mods_obj)
                st.success("Modifier added.")
    with c5:
        if st.button("Restore defaults"):
            _safe_dump_yaml(FEEDS_PATH, default_feeds)
            _safe_dump_yaml(MODS_PATH, default_mods)
            st.success("Defaults restored.")

with st.expander("Public Signals (Trends + Suggest + RSS + Modifiers)", expanded=False):
    import os
    import re
    import yaml
    import socket
    import time
    import json
    import requests
    import pandas as pd

    # Controls
    aug_geo = st.text_input("AUG GEO", "US", key="aug_geo")
    aug_timeframe = st.selectbox(
        "AUG Timeframe", ["7d", "14d", "30d", "60d"], index=0, key="aug_tf"
    )
    max_seeds = st.number_input(
        "Max seeds", 1, 1000, 10, 1, key="aug_max_seeds"
    )  # default 10 for speed
    top_per_seed = st.number_input(
        "Top per seed (per source)", 1, 50, 5, 1, key="aug_top_per_seed"
    )

    # Filters
    min_score = st.slider("Min score", 0.0, 1.0, 0.35, 0.01, key="aug_min_score")
    multi_only = st.checkbox(
        "Only multi-signal hits", value=False, key="aug_multi_only"
    )

    manual_seeds_raw = st.text_area(
    "Manual seed keywords (comma/newline separated)",
    "",
    height=100,
    key="manual_seeds",  # <-- critical: populate session_state["manual_seeds"]
)
    manual_seeds = [
        s.strip() for s in re.split(r"[\n,]", manual_seeds_raw) if s.strip()
    ]

    # apply blacklist to manual seeds too
    patt = compile_blacklist()
    if patt:
        manual_seeds = [s for s in manual_seeds if not patt.search(s)]

    run_aug = st.button("Run Public Signals", key="public_signals_btn")

    # ---------- helpers: config & seed sourcing ----------
    def _read_yaml(path, key):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get(key, [])
        except Exception:
            return []

    def _detect_keyword_col(df):
        cand = {c.lower().strip(): c for c in df.columns}
        for k in ("keyword", "kw"):
            if k in cand:
                return cand[k]
        return "Keyword" if "Keyword" in df.columns else None

    def _collect_seeds(limit: int):
        """Collect seeds from winners/rollups/history, de-dupe, then filter with blacklist."""

        def _detect_keyword_col(df):
            cand = {c.lower().strip(): c for c in df.columns}
            for k in ("keyword", "kw"):
                if k in cand:
                    return cand[k]
            return "Keyword" if "Keyword" in df.columns else None

        seeds, source = [], ""

        # 1) winners in session_state (supports both full + RPC-only keys)
        for key in ("dod", "wow", "sur", "dod_rpc", "wow_rpc", "sur_rpc"):
            df = st.session_state.get(key)
            if isinstance(df, pd.DataFrame) and not df.empty:
                kw_col = _detect_keyword_col(df)
                if kw_col:
                    seeds.extend(df[kw_col].astype(str).tolist())
                    source = source or f"session:{key}"

        # 2) today's rollup in memory (supports both)
        if not seeds:
            for key in ("today_roll", "today_roll_rpc"):
                tr = st.session_state.get(key)
                if isinstance(tr, pd.DataFrame) and not tr.empty:
                    kw_col = _detect_keyword_col(tr)
                    if kw_col:
                        if "Clicks" in tr.columns:
                            seeds.extend(
                                tr.sort_values("Clicks", ascending=False)[kw_col]
                                .astype(str)
                                .tolist()
                            )
                        else:
                            seeds.extend(tr[kw_col].astype(str).tolist())
                        source = f"session:{key}"
                        break

        # 3) on-disk rollups (supports both)
        if not seeds:
            for fname in ("rollup_daily.csv", "rollup_daily_rpc.csv"):
                path = os.path.join(OUTPUTS, fname)
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    kw_col = _detect_keyword_col(df)
                    if kw_col:
                        if "Clicks" in df.columns:
                            seeds.extend(
                                df.sort_values("Clicks", ascending=False)[kw_col]
                                .astype(str)
                                .tolist()
                            )
                        else:
                            seeds.extend(df[kw_col].astype(str).tolist())
                        source = f"file:{fname}"
                        break

        # 4) history latest day
        if not seeds:
            hist = load_history()
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                hist["date"] = pd.to_datetime(hist["date"], errors="coerce")
                latest = hist["date"].max()
                if pd.notna(latest):
                    day = hist[hist["date"] == latest]
                    kw_col = _detect_keyword_col(day)
                    if kw_col:
                        if "Clicks" in day.columns:
                            seeds.extend(
                                day.sort_values("Clicks", ascending=False)[kw_col]
                                .astype(str)
                                .tolist()
                            )
                        else:
                            seeds.extend(day[kw_col].astype(str).tolist())
                        source = "file:history(latest)"

        # De-dupe (case-insensitive), drop empties
        seen, out = set(), []
        for s in seeds:
            s = str(s).strip()
            key = s.casefold()
            if s and key not in seen:
                seen.add(key)
                out.append(s)

        # APPLY BLACKLIST to seeds here (immediate effect)
        patt = compile_blacklist()
        if patt:
            out = [s for s in out if not patt.search(s)]

        return out[: int(limit)], source

    # ---------- helpers: scoring ----------

    # --- Similarity helpers (bias toward seed-like candidates) ---
    STOPWORDS = set(
        "a an the for of to with and or & near in on at by from 2025 2024".split()
    )

    def _tokens(s: str):
        return [
            t
            for t in re.findall(r"[a-z0-9]+", str(s).lower())
            if t not in STOPWORDS and len(t) > 2
        ]

    def _text_overlap(a: str, b: str) -> float:
        A, B = set(_tokens(a)), set(_tokens(b))
        return 0.0 if not A or not B else len(A & B) / max(1, len(A | B))

    def _nearest_cpc_cap(pool_df, target_rpc):
        if pool_df is None or pool_df.empty or "cpc_cap" not in pool_df.columns:
            return 0.25
        tmp = pool_df.copy()
        tmp["diff"] = (tmp.get("RPC", 0.0) - float(target_rpc)) ** 2
        tmp.sort_values("diff", inplace=True)
        return float(tmp.iloc[0].get("cpc_cap", 0.25))

    def _trend_norm(scores):
        if not scores:
            return []
        vals = [v for _, v in scores]
        mx = max(vals) or 1.0
        return [(q, (v / mx)) for q, v in scores]

    def _modifier_weight(cand, mod_list):
        cand_l = cand.lower()
        best = ("", 0.0, 0)
        for m in mod_list or []:
            term = str(m.get("term", "")).lower().strip()
            if term and term in cand_l:
                w = float(m.get("weight", 0.0))
                adj = int(str(m.get("cpc_adj_pct", 0)).replace("+", ""))
                if w > best[1]:
                    best = (term, w, adj)
        return best

    def _merge_with_manual(auto_list, manual_list, cap):
        """Manual first, then autos; preserve order; de-dupe; cap to 'cap'."""
        seen = set()
        out = []
        for s in (manual_list or []) + [x for x in (auto_list or []) if x not in (manual_list or [])]:
            if s not in seen:
                seen.add(s)
                out.append(s)
            if cap is not None and len(out) >= int(cap):
                break
        return out

    # ---------- preview seeds (ALWAYS visible, includes manual) ----------
    _preview, _src = _collect_seeds(max_seeds)

    # Parse manual seeds from the text area and merge for preview
    _manual_text = st.session_state.get("manual_seeds", manual_seeds_raw)
    _manual_list = [s.strip() for s in re.split(r"[\n,]", _manual_text) if s.strip()]
    _preview = _merge_with_manual(_preview, _manual_list, max_seeds)

    st.caption(
        f"Seed source: {_src or 'none'}; manual added: {len(_manual_list)}; total preview: {len(_preview)}"
    )
    st.write({"seed_preview": _preview[:10], "seed_count": len(_preview)})

    if run_aug:
        # timeouts / limits
        try:
            import feedparser
            from pytrends.request import TrendReq
        except Exception:
            st.error(
                "Missing deps. In Terminal run:  python3 -m pip install pytrends feedparser pyyaml requests"
            )
            st.stop()

        socket.setdefaulttimeout(6)  # RSS fetch timeout
        TRENDS_TIMEOUT = (4, 20)  # (connect, read)
        SUGGEST_TIMEOUT = 8
        SUGGEST_PER_SEED = int(top_per_seed)

        feeds = _read_yaml(os.path.join(BASE_DIR, "feeds.yaml"), "feeds")
        modifiers = _read_yaml(os.path.join(BASE_DIR, "modifiers.yaml"), "modifiers")

        seeds, _source = _collect_seeds(max_seeds)

        # ---- (#4) Merge manual seeds from the text area (dedup, preserve order) ----
        manual_text = st.session_state.get("manual_seeds", manual_seeds_raw)
        manual_list = [s.strip() for s in re.split(r"[\n,]", manual_text) if s.strip()]
        if manual_list:
            seen = set()
            merged = []
            for s in list(seeds) + manual_list:
                if s not in seen:
                    seen.add(s)
                    merged.append(s)
            seeds = merged

        if not seeds:
            st.warning(
                "No seeds found from winners, rollup, or history, and none entered manually."
            )
            st.stop()

        # winners pool → CPC mapping base
        pool_parts = []
        for key in ("dod", "wow", "sur"):
            df_ss = st.session_state.get(key)
            if isinstance(df_ss, pd.DataFrame) and not df_ss.empty:
                pool_parts.append(df_ss)
        winners_pool = (
            pd.concat(pool_parts, ignore_index=True) if pool_parts else pd.DataFrame()
        )

        # For seed-RPC weighting
        if "RPC" in winners_pool.columns and not winners_pool.empty:
            rpc_min = float(winners_pool["RPC"].min())
            rpc_max = float(winners_pool["RPC"].max())
        else:
            rpc_min, rpc_max = 0.0, 1.0

        # --- adapters ---
        def _trends(seed: str):
            try:
                pytrends = TrendReq(hl="en-US", tz=360, timeout=TRENDS_TIMEOUT)
                timeframe_map = {
                    "7d": "now 7-d",
                    "14d": "now 7-d",
                    "30d": "today 1-m",
                    "60d": "today 3-m",
                }
                pytrends.build_payload(
                    [seed],
                    timeframe=timeframe_map.get(aug_timeframe, "now 7-d"),
                    geo=aug_geo,
                )
                rq = pytrends.related_queries()
                rising = rq.get(seed, {}).get("rising")
                if rising is None or rising.empty:
                    return []
                res = []
                for _, r in rising.head(int(top_per_seed)).iterrows():
                    q = str(r.get("query", "")).strip()
                    v = float(r.get("value", 0.0))
                    if q:
                        res.append((q, v))
                # normalize 0..1
                if not res:
                    return []
                vals = [v for _, v in res]
                mx = max(vals) or 1.0
                return [(q, v / mx) for q, v in res]
            except Exception:
                return []

        def _suggest(seed: str):
            def g(q):
                try:
                    r = requests.get(
                        "https://suggestqueries.google.com/complete/search",
                        params={"client": "firefox", "q": q},
                        timeout=SUGGEST_TIMEOUT,
                    )
                    j = r.json()
                    return [s for s in j[1][:SUGGEST_PER_SEED] if isinstance(s, str)]
                except Exception:
                    return []

            def b(q):
                try:
                    r = requests.get(
                        "https://api.bing.com/osjson.aspx",
                        params={"query": q},
                        timeout=SUGGEST_TIMEOUT,
                    )
                    j = r.json()
                    return [s for s in j[1][:SUGGEST_PER_SEED] if isinstance(s, str)]
                except Exception:
                    return []

            def yt(q):
                try:
                    r = requests.get(
                        "https://suggestqueries.google.com/complete/search",
                        params={"client": "firefox", "ds": "yt", "q": q},
                        timeout=SUGGEST_TIMEOUT,
                    )
                    j = r.json()
                    return [s for s in j[1][:SUGGEST_PER_SEED] if isinstance(s, str)]
                except Exception:
                    return []

            out = []
            intents = ["", " prices", " deals", " best", " cheap", " near me", " 2025", " guide"]
            for mod in intents:
                q = (seed + mod).strip()
                out.extend(g(q))
                out.extend(b(q))
                out.extend(yt(q))
            seen_s = set()
            uniq = []
            for s in out:
                if s not in seen_s:
                    seen_s.add(s)
                    uniq.append(s)
            return uniq[: SUGGEST_PER_SEED * 3]

        def _rss_candidates(cands):
            if not feeds:
                return {c: 0.0 for c in cands}
            out_map = {}
            for cand in cands:
                hits = 0
                total = 0
                patt = re.compile(r"\b" + re.escape(cand.lower()) + r"\b")
                for url in feeds:
                    try:
                        total += 1
                        d = feedparser.parse(url)
                        titles = [e.get("title", "") for e in (d.entries or [])][:50]
                        joined = " || ".join(titles).lower()
                        if patt.search(joined):
                            hits += 1
                    except Exception:
                        pass
                out_map[cand] = (hits / total) if total else 0.0
            return out_map

        # ---- run with progress + partial writes ----
        rows = []
        total = len(seeds)
        prog = st.progress(0, text="Starting Public Signals…")
        status = st.status(
            "Collecting signals from Trends + Suggest + RSS + Modifiers…",
            state="running",
        )
        start_t = time.time()
        write_every = max(1, min(total, int(max_seeds)) // 4)
        tmp_path = os.path.join(EXPANSIONS, "expansion_candidates.tmp.csv")

        for i, seed in enumerate(seeds[: int(max_seeds)], start=1):
            # seed RPC baseline for CPC cap mapping
            if not winners_pool.empty:
                seed_rows = winners_pool[
                    winners_pool.get("Keyword", "").astype(str).str.lower()
                    == str(seed).lower()
                ]
                seed_rpc = (
                    float(seed_rows["RPC"].mean()) if not seed_rows.empty else 0.0
                )
            else:
                seed_rpc = 0.0
            base_cap = _nearest_cpc_cap(winners_pool, seed_rpc)

            # T1: Trends
            t1 = _trends(seed)  # [(q, norm_score)]
            # T2: Suggest
            t2 = _suggest(seed)  # [q]
            # Union for RSS + modifiers
            union = set([q for q, _ in t1]) | set(t2)

            # T3: RSS overlap
            rss_map = _rss_candidates(list(union))

            # T4: Modifier injection
            injected = []
            for m in modifiers or []:
                term = str(m.get("term", "")).strip()
                if term:
                    injected.append(f"{seed} {term}".strip())
            union |= set(injected)

            trend_lookup = {q: s for q, s in t1}
            suggest_set = set(t2)

            for q in list(union):
                tn = float(trend_lookup.get(q, 0.0))  # Trends [0..1]
                ov = float(rss_map.get(q, 0.0))  # RSS overlap [0..1]
                came_from_suggest = q in suggest_set
                base = 0.18 if came_from_suggest else 0.0

                # Similarity to seed + normalized seed RPC
                sim = _text_overlap(q, seed)  # [0..1]
                if rpc_max <= rpc_min:
                    rpc_norm = 0.0
                else:
                    rpc_norm = max(
                        0.0, min(1.0, (seed_rpc - rpc_min) / (rpc_max - rpc_min))
                    )

                # Modifier impact
                mod_term, mod_w, mod_adj = _modifier_weight(q, modifiers)
                rec_cap = round(base_cap * (1.0 + (mod_adj / 100.0)), 2)

                # Final score (keeps Trend/RSS on top, then similarity & seed RPC, then modifiers)
                raw = (
                    (0.45 * tn)
                    + (0.25 * ov)
                    + (0.12 * sim)
                    + (0.10 * rpc_norm)
                    + (0.08 * mod_w)
                    + base
                )
                score = round(min(1.0, raw), 3)

                rows.append(
                    {
                        "seed": seed,
                        "candidate": q,
                        "source_flags": json.dumps(
                            {
                                "trends": bool(tn),
                                "suggest": came_from_suggest,
                                "rss": ov > 0.0,
                                "injected": q in set(injected),
                            }
                        ),
                        "geo": aug_geo,
                        "window": aug_timeframe,
                        "trend_norm": round(tn, 3),
                        "overlap": round(ov, 3),
                        "similarity": round(sim, 3),
                        "seed_rpc_norm": round(rpc_norm, 3),
                        "modifier": mod_term,
                        "modifier_weight": round(mod_w, 3),
                        "score": score,
                        "rec_cpc_cap": rec_cap,
                    }
                )
            pct = int((i / min(total, int(max_seeds))) * 100)
            elapsed = int(time.time() - start_t)
            prog.progress(
                pct, text=f"{i}/{min(total, int(max_seeds))} seeds · {elapsed}s elapsed"
            )
            if i % write_every == 0 and rows:
                tmp = pd.DataFrame(rows).sort_values(
                    ["score", "trend_norm"], ascending=[False, False]
                )
                tmp.to_csv(tmp_path, index=False)

        # ---------- finalize (write ALL + filtered, show table) ----------
        if rows:
            df_all = pd.DataFrame(rows).sort_values(
                ["score", "trend_norm"], ascending=[False, False]
            )

            # derive signal count from source_flags JSON
            def _count_signals(flags_json):
                try:
                    d = json.loads(flags_json)
                    return sum(1 for v in d.values() if v)
                except Exception:
                    return 0

            df_all["signals"] = df_all["source_flags"].apply(_count_signals)

            # 1) ALWAYS save the COMPLETE, UNFILTERED set
            all_out = os.path.join(EXPANSIONS, "expansion_candidates.csv")
            df_all.to_csv(all_out, index=False)

            # 2) Apply UI filters ONLY for what's displayed, and also save a filtered copy
            df_filtered = df_all
            if multi_only:
                df_filtered = df_filtered[df_filtered["signals"] >= 2]
            df_filtered = df_filtered[df_filtered["score"] >= float(min_score)]

            # soft fallback if nothing passes: relax threshold a bit
            if df_filtered.empty and not df_all.empty:
                relaxed = max(0.10, float(min_score) * 0.5)
                df_filtered = df_all[df_all["score"] >= relaxed]

            filt_out = os.path.join(EXPANSIONS, "expansion_candidates_filtered.csv")
            df_filtered.to_csv(filt_out, index=False)

            if df_filtered.empty:
                status.update(
                    label="No candidates met your filters (min score / multi-signal). Saved all rows to expansion_candidates.csv",
                    state="complete",
                )
                st.info(
                    "Try lowering Min score, adding more seeds/feeds, or widening timeframe."
                )
            else:
                # save final (non-RPC) expansion candidates
                outp = os.path.join(EXPANSIONS, "expansion_candidates.csv")
                df_filtered.to_csv(outp, index=False)

                # clean up temp file if present
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

                status.update(
                    label=f"Done. Saved {len(df_filtered)} candidates to {outp} (min_score={min_score}, multi_only={multi_only})",
                    state="complete",
                )
                # quick source breakdown
                st.write(
                    {
                        "rows_saved": int(len(df_filtered)),
                        "from_trends": int(
                            (
                                df_filtered["source_flags"].str.contains(
                                    '"trends": true'
                                )
                            ).sum()
                        ),
                        "from_suggest": int(
                            (
                                df_filtered["source_flags"].str.contains(
                                    '"suggest": true'
                                )
                            ).sum()
                        ),
                        "from_rss": int(
                            (
                                df_filtered["source_flags"].str.contains('"rss": true')
                            ).sum()
                        ),
                    }
                )
                st.dataframe(df_filtered, use_container_width=True)

            # clean up tmp if present
            try:
                os.remove(os.path.join(EXPANSIONS, "expansion_candidates.tmp.csv"))
            except Exception:
                pass
        else:
            status.update(
                label="No candidates produced by any source.", state="complete"
            )

st.caption(f"Outputs folder: {OUTPUTS}")
st.caption(f"Expansions folder: {EXPANSIONS}")

# ---- Sidebar snapshot (rendered AFTER all logic so counts are fresh) ----
with st.sidebar:

    def _cnt(df):
        return 0 if not isinstance(df, pd.DataFrame) else int(len(df))

    raw_dod = _cnt(st.session_state.get("dod"))
    raw_wow = _cnt(st.session_state.get("wow"))
    raw_sur = _cnt(st.session_state.get("sur"))
    plan_dod = _cnt(st.session_state.get("dod_plan"))
    plan_wow = _cnt(st.session_state.get("wow_plan"))
    plan_sur = _cnt(st.session_state.get("sur_plan"))
    rollup_rows = _cnt(st.session_state.get("today_roll"))

    st.caption("Scan snapshot")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("RAW")
        st.write({"DoD": raw_dod, "WoW": raw_wow, "Surges": raw_sur})
    with c2:
        st.subheader("PLAN")
        st.write({"DoD": plan_dod, "WoW": plan_wow, "Surges": plan_sur})

    st.caption(f"Rollup rows today: {rollup_rows}")

    # --- Daily Plan preview (top 10) ---
    dp_path = os.path.join(OUTPUTS, "daily_plan.csv")
    try:
        if os.path.exists(dp_path):
            dp = pd.read_csv(dp_path)
            st.divider()
            st.caption(f"Daily Plan preview ({len(dp)} rows)")
            if not dp.empty:
                preview_cols = [
                    c
                    for c in [
                        "segment",
                        "Keyword",
                        "Niche",
                        "Geo",
                        "cpc_cap",
                        "lift_metric",
                    ]
                    if c in dp.columns
                ]
                st.dataframe(
                    dp[preview_cols].head(10), use_container_width=True, height=240
                )
                st.download_button(
                    "Download daily_plan.csv",
                    data=dp.to_csv(index=False),
                    file_name="daily_plan.csv",
                    mime="text/csv",
                    key="dl_daily_plan",
                )
            else:
                st.write("(daily_plan.csv is empty)")
        else:
            st.caption("Daily Plan: (not created yet)")
    except Exception as e:
        st.warning(f"Daily Plan preview error: {e}")

# Keep your existing captions:
st.caption(
    "Outputs: rollup_daily.csv, winners_today.csv, winners_week.csv, surges_low_to_high.csv, history.parquet, chat_payload.txt, plan_input.json"
)
st.caption(f"Outputs folder: {OUTPUTS}")
