import re
import sqlite3
from datetime import datetime
from io import StringIO
import json
import pandas as pd
import streamlit as st

DB_DEFAULT = "keyword_library.db"

# last_keywords will now store tuples: (keyword, geo)
if "last_keywords" not in st.session_state:
    st.session_state["last_keywords"] = []

def norm_kw(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def ensure_schema(conn: sqlite3.Connection) -> None:
    # Create or migrate schema so uniqueness is (keyword, geo) for tested table
    tbl = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='keyword_metrics';"
    ).fetchone()

    if not tbl:
        conn.execute(
            """
            CREATE TABLE keyword_metrics (
              keyword TEXT NOT NULL,
              geo TEXT NOT NULL,
              clicks INTEGER NOT NULL DEFAULT 0,
              cpc REAL,
              sell_rpc REAL,
              ctr REAL,
              last_updated TEXT,
              PRIMARY KEY (keyword, geo)
            );
            """
        )
    else:
        cols = [r[1].lower() for r in conn.execute("PRAGMA table_info(keyword_metrics);").fetchall()]
        if "geo" not in cols:
            conn.execute("BEGIN;")
            conn.execute("ALTER TABLE keyword_metrics RENAME TO keyword_metrics_old;")
            conn.execute(
                """
                CREATE TABLE keyword_metrics (
                  keyword TEXT NOT NULL,
                  geo TEXT NOT NULL,
                  clicks INTEGER NOT NULL DEFAULT 0,
                  cpc REAL,
                  sell_rpc REAL,
                  ctr REAL,
                  last_updated TEXT,
                  PRIMARY KEY (keyword, geo)
                );
                """
            )
            conn.execute(
                """
                INSERT INTO keyword_metrics (keyword, geo, clicks, cpc, sell_rpc, ctr, last_updated)
                SELECT keyword, 'XX', clicks, cpc, sell_rpc, ctr, last_updated
                FROM keyword_metrics_old;
                """
            )
            conn.execute("DROP TABLE keyword_metrics_old;")
            conn.execute("COMMIT;")

    # Research table for SEMrush and Keywords Everywhere
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS keyword_research (
          keyword TEXT NOT NULL,
          geo TEXT NOT NULL,
          source TEXT NOT NULL,
          search_volume INTEGER,
          cpc_est REAL,
          competition REAL,
          raw_json TEXT,
          imported_at TEXT NOT NULL,
          PRIMARY KEY (keyword, geo, source)
        );
        """
    )

def upsert_keywords_only(conn: sqlite3.Connection, keywords: list[str], geo: str) -> tuple[int, int]:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    geo = (geo or "XX").strip().upper()

    kws = [norm_kw(k) for k in keywords]
    kws = [k for k in kws if k]
    if not kws:
        return 0, 0

    df = pd.DataFrame({"keyword": kws}).drop_duplicates()
    batch = df["keyword"].tolist()
    rows = [(k, geo, now) for k in batch]

    # Count which already exist for this geo
    existing = set()
    cur = conn.cursor()
    chunk = 400
    for i in range(0, len(batch), chunk):
        part = batch[i:i + chunk]
        q = ",".join(["?"] * len(part))
        cur.execute(
            f"SELECT keyword FROM keyword_metrics WHERE geo = ? AND keyword IN ({q})",
            [geo] + part,
        )
        existing.update(r[0] for r in cur.fetchall())

    with conn:
        cur = conn.cursor()
        cur.execute("BEGIN;")
        cur.executemany(
            """
            INSERT INTO keyword_metrics (keyword, geo, last_updated)
            VALUES (?, ?, ?)
            ON CONFLICT(keyword, geo) DO UPDATE SET
              last_updated = excluded.last_updated
            """,
            rows,
        )
        cur.execute("COMMIT;")

    inserted = 0
    updated = 0
    for k in batch:
        if k in existing:
            updated += 1
        else:
            inserted += 1
    return inserted, updated

def parse_kw_optional_rpc(text: str) -> list[tuple[str, float | None]]:
    lines = re.split(r"[\r\n]+", (text or "").strip())
    out: list[tuple[str, float | None]] = []

    for line in lines:
        s = (line or "").strip()
        if not s:
            continue

        kw_raw = s
        rpc_val: float | None = None

        if "\t" in s:
            parts = [p.strip() for p in s.split("\t", 1)]
            if len(parts) == 2 and parts[0]:
                kw_raw = parts[0]
                rpc_raw = parts[1]
                rpc_raw = rpc_raw.replace("$", "").replace(",", "").strip()
                try:
                    rpc_val = float(rpc_raw)
                except Exception:
                    rpc_val = None

        else:
            if "," in s:
                left, right = s.rsplit(",", 1)
                left = left.strip()
                right = right.strip()
                if left and right:
                    rpc_try = right.replace("$", "").replace(",", "").strip()
                    try:
                        rpc_val = float(rpc_try)
                        kw_raw = left
                    except Exception:
                        kw_raw = s
                        rpc_val = None

        kw = norm_kw(kw_raw)
        if not kw:
            continue

        out.append((kw, rpc_val))

    return out

def upsert_research_manual_optional_cpc(
    conn: sqlite3.Connection,
    items: list[tuple[str, float | None]],
    geo: str
) -> tuple[int, int]:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    geo = (geo or "XX").strip().upper()
    source = "manual"

    if not items:
        return 0, 0

    df = pd.DataFrame(items, columns=["keyword", "cpc_est"]).drop_duplicates(subset=["keyword"])
    df["geo"] = geo
    df["source"] = source
    df["imported_at"] = now
    df["search_volume"] = pd.NA
    df["competition"] = pd.NA
    df["raw_json"] = None

    batch = list(df[["keyword", "geo", "source"]].itertuples(index=False, name=None))

    existing = set()
    cur = conn.cursor()
    chunk = 250
    for i in range(0, len(batch), chunk):
        part = batch[i:i + chunk]
        placeholders = ",".join(["(?, ?, ?)"] * len(part))
        flat = []
        for k, g, s in part:
            flat.extend([k, g, s])
        cur.execute(
            f"SELECT keyword, geo, source FROM keyword_research WHERE (keyword, geo, source) IN ({placeholders})",
            flat,
        )
        existing.update((r[0], r[1], r[2]) for r in cur.fetchall())

    rows = []
    for r in df.itertuples(index=False):
        rows.append(
            (
                r.keyword,
                r.geo,
                r.source,
                None,
                None if pd.isna(r.cpc_est) else float(r.cpc_est),
                None,
                r.raw_json,
                r.imported_at,
            )
        )

    with conn:
        cur = conn.cursor()
        cur.execute("BEGIN;")
        cur.executemany(
            """
            INSERT INTO keyword_research
            (keyword, geo, source, search_volume, cpc_est, competition, raw_json, imported_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(keyword, geo, source) DO UPDATE SET
              cpc_est = COALESCE(excluded.cpc_est, keyword_research.cpc_est),
              imported_at = excluded.imported_at
            """,
            rows,
        )
        cur.execute("COMMIT;")

    inserted = 0
    updated = 0
    for key in batch:
        if key in existing:
            updated += 1
        else:
            inserted += 1
    return inserted, updated

def upsert_keywords_optional_sell_rpc(
    conn: sqlite3.Connection,
    items: list[tuple[str, float | None]],
    geo: str
) -> tuple[int, int]:
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    geo = (geo or "XX").strip().upper()

    if not items:
        return 0, 0

    df = pd.DataFrame(items, columns=["keyword", "sell_rpc"]).drop_duplicates(subset=["keyword"])
    batch = df["keyword"].tolist()

    existing = set()
    cur = conn.cursor()
    chunk = 400
    for i in range(0, len(batch), chunk):
        part = batch[i:i + chunk]
        q = ",".join(["?"] * len(part))
        cur.execute(
            f"SELECT keyword FROM keyword_metrics WHERE geo = ? AND keyword IN ({q})",
            [geo] + part,
        )
        existing.update(r[0] for r in cur.fetchall())

    rows = []
    for r in df.itertuples(index=False):
        srpc = None if pd.isna(r.sell_rpc) else float(r.sell_rpc)
        rows.append((r.keyword, geo, srpc, now))

    with conn:
        cur = conn.cursor()
        cur.execute("BEGIN;")
        cur.executemany(
            """
            INSERT INTO keyword_metrics (keyword, geo, sell_rpc, last_updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(keyword, geo) DO UPDATE SET
              sell_rpc = COALESCE(excluded.sell_rpc, keyword_metrics.sell_rpc),
              last_updated = excluded.last_updated
            """,
            rows,
        )
        cur.execute("COMMIT;")

    inserted = 0
    updated = 0
    for k in batch:
        if k in existing:
            updated += 1
        else:
            inserted += 1
    return inserted, updated

def search(conn: sqlite3.Connection, q: str, limit: int, geo: str | None = None) -> pd.DataFrame:
    qn = norm_kw(q)
    if not qn:
        return pd.DataFrame(columns=["keyword", "geo", "clicks", "cpc", "sell_rpc", "ctr", "last_updated"])

    like = f"%{qn}%"
    if geo and geo.strip().upper() != "ALL":
        g = geo.strip().upper()
        df = pd.read_sql_query(
            """
            SELECT keyword, geo, clicks, cpc, sell_rpc, ctr, last_updated
            FROM keyword_metrics
            WHERE geo = ? AND keyword LIKE ?
            ORDER BY
              CASE WHEN keyword = ? THEN 0
                   WHEN keyword LIKE ? THEN 1
                   ELSE 2 END,
              clicks DESC,
              sell_rpc DESC
            LIMIT ?;
            """,
            conn,
            params=(g, like, qn, f"{qn}%", limit),
        )
        return df

    df = pd.read_sql_query(
        """
        SELECT keyword, geo, clicks, cpc, sell_rpc, ctr, last_updated
        FROM keyword_metrics
        WHERE keyword LIKE ?
        ORDER BY
          CASE WHEN keyword = ? THEN 0
               WHEN keyword LIKE ? THEN 1
               ELSE 2 END,
          clicks DESC,
          sell_rpc DESC
        LIMIT ?;
        """,
        conn,
        params=(like, qn, f"{qn}%", limit),
    )
    return df

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = StringIO()
    df.to_csv(out, index=False)
    return out.getvalue().encode("utf-8")

st.set_page_config(page_title="Keyword Library", layout="wide")
st.title("Keyword Library")

with st.sidebar:
    st.subheader("Database")
    db_path = st.text_input("SQLite DB Path", value=DB_DEFAULT)
    limit = st.slider("Search Limit", min_value=25, max_value=2000, value=250, step=25)

conn = connect(db_path)
ensure_schema(conn)

left, right = st.columns([3, 7])

with left:
    st.subheader("Add Keywords")
    st.caption("Default saves to Research. Optional: keyword,rpc or keyword<TAB>rpc")

    paste_geo = st.selectbox("Geo For Paste Add", ["US", "GB", "DE", "FR", "XX"], index=0)
    paste_bucket = st.selectbox("Save Pasted Keywords To", ["research", "tested"], index=0)

    def _clear_paste():
        st.session_state["paste_box"] = ""

    def _add_paste_keywords():
        text = st.session_state.get("paste_box", "")
        items = parse_kw_optional_rpc(text)

        if paste_bucket == "research":
            inserted, updated = upsert_research_manual_optional_cpc(conn, items, paste_geo)
        else:
            inserted, updated = upsert_keywords_optional_sell_rpc(conn, items, paste_geo)

        st.session_state["last_keywords"] = [(k, paste_geo) for (k, _) in items]
        st.session_state["toast_msg"] = f"Inserted {inserted} New Keywords Updated {updated}"
        st.session_state["paste_box"] = ""

    st.text_area(
        "Paste Keywords",
        height=220,
        placeholder="keyword one,RPC\nkeyword two\nkeyword three<tab>RPC",
        key="paste_box",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.button("Add Or Update", type="primary", on_click=_add_paste_keywords)
    with col_b:
        st.button("Clear Box", on_click=_clear_paste)

    if st.session_state.get("toast_msg"):
        st.success(st.session_state["toast_msg"])
        st.session_state["toast_msg"] = ""

    st.divider()
    st.subheader("Import Performance CSV")
    st.caption("Uses: Campaign Concept, Primus Country Code, Clicks, CPC, Sell RPC, CTR")
    up = st.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)

    def read_perf_csv(file) -> pd.DataFrame:
        df = pd.read_csv(file)

        def canon(c: str) -> str:
            c = str(c).strip().lower()
            c = re.sub(r"\s+", " ", c)
            return c

        cols = {canon(c): c for c in df.columns}

        required = {
            "keyword": "campaign concept",
            "geo": "primus country code",
            "clicks": "clicks",
        }

        optional = {
            "cpc": "cpc",
            "sell_rpc": "sell rpc",
            "ctr": "ctr",
        }

        missing = [k for k, src in required.items() if src not in cols]
        if missing:
            raise ValueError(
                "Missing required fields: "
                + ", ".join(missing)
                + " | Detected: "
                + ", ".join(list(df.columns))
            )

        data = {
            "keyword": df[cols[required["keyword"]]],
            "geo": df[cols[required["geo"]]],
            "clicks": df[cols[required["clicks"]]],
        }

        for field, src in optional.items():
            if src in cols:
                data[field] = df[cols[src]]
            else:
                data[field] = None

        df = pd.DataFrame(data)

        df.columns = ["keyword", "geo", "clicks", "cpc", "sell_rpc", "ctr"]

        df["keyword"] = df["keyword"].map(norm_kw)
        df["geo"] = df["geo"].astype(str).str.strip().str.upper()

        df = df[df["keyword"] != ""]
        df = df[df["geo"] != ""]

        def clean_num(s: pd.Series) -> pd.Series:
            s = s.astype(str).str.strip()
            s = s.str.replace(",", "", regex=False)
            s = s.str.replace("$", "", regex=False)
            s = s.str.replace("%", "", regex=False)
            s = s.replace({"": None, "none": None, "nan": None, "n/a": None, "na": None, "-": None, "—": None})
            return pd.to_numeric(s, errors="coerce")

        df["cpc"] = clean_num(df["cpc"])
        df["sell_rpc"] = clean_num(df["sell_rpc"])
        df["ctr"] = clean_num(df["ctr"])
        df["clicks"] = pd.to_numeric(df["clicks"], errors="coerce").fillna(0).astype(int)

        # Merge duplicates inside the file by (keyword, geo)
        df = df.groupby(["keyword", "geo"], as_index=False).agg(
            clicks=("clicks", "sum"),
            cpc=("cpc", "mean"),
            sell_rpc=("sell_rpc", "mean"),
            ctr=("ctr", "mean"),
        )

        return df

    def upsert_metrics(conn: sqlite3.Connection, df: pd.DataFrame) -> tuple[int, int]:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        df = df.drop_duplicates(subset=["keyword", "geo"]).copy()

        batch = list(df[["keyword", "geo"]].itertuples(index=False, name=None))

        existing = set()
        cur = conn.cursor()
        chunk = 400
        for i in range(0, len(batch), chunk):
            part = batch[i:i + chunk]
            placeholders = ",".join(["(?, ?)"] * len(part))
            flat = []
            for k, g in part:
                flat.extend([k, g])

            cur.execute(
                f"SELECT keyword, geo FROM keyword_metrics WHERE (keyword, geo) IN ({placeholders})",
                flat,
            )
            existing.update((r[0], r[1]) for r in cur.fetchall())

        rows = []
        for r in df.itertuples(index=False):
            rows.append(
                (
                    r.keyword,
                    r.geo,
                    int(r.clicks),
                    None if pd.isna(r.cpc) else float(r.cpc),
                    None if pd.isna(r.sell_rpc) else float(r.sell_rpc),
                    None if pd.isna(r.ctr) else float(r.ctr),
                    now,
                )
            )

        with conn:
            cur = conn.cursor()
            cur.execute("BEGIN;")
            cur.executemany(
                """
                INSERT INTO keyword_metrics (keyword, geo, clicks, cpc, sell_rpc, ctr, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(keyword, geo) DO UPDATE SET
                  clicks = excluded.clicks,
                  cpc = excluded.cpc,
                  sell_rpc = excluded.sell_rpc,
                  ctr = excluded.ctr,
                  last_updated = excluded.last_updated
                """,
                rows,
            )
            cur.execute("COMMIT;")

        inserted = 0
        updated = 0
        for k, g in batch:
            if (k, g) in existing:
                updated += 1
            else:
                inserted += 1
        return inserted, updated

    if up is not None:
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            run_import = st.button("Import CSV", type="primary")
        with col_u2:
            st.caption("Imports add new keywords and update existing ones (by keyword+geo)")

        if run_import:
            try:
                perf = read_perf_csv(up)
                ins, upd = upsert_metrics(conn, perf)
                st.success(f"CSV Imported Rows {len(perf)} Inserted {ins} Updated {upd}")
                st.session_state["last_keywords"] = list(perf[["keyword", "geo"]].itertuples(index=False, name=None))
            except Exception as e:
                st.error(str(e))

    st.divider()
    st.subheader("Import Research CSV")
    source = "auto"
    research_geo = st.selectbox("Geo For Research Import", ["ALL", "US", "GB", "DE", "FR"], index=0)
    up_r = st.file_uploader("Upload Research CSV", type=["csv"], accept_multiple_files=False, key="research_uploader")

    def _canon(c: str) -> str:
        c = str(c).strip().lower()
        c = re.sub(r"[\.\,\(\)\[\]\{\}\/\\]+", " ", c)
        c = re.sub(r"[^a-z0-9\s]+", " ", c)
        c = re.sub(r"\s+", " ", c).strip()
        return c    

    def read_research_csv(file, source: str, geo_choice: str) -> tuple[pd.DataFrame, str]:
        df0 = pd.read_csv(file)
        cols = {_canon(c): c for c in df0.columns}

        def get_col(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            return None

        kw_col = get_col("keyword")
        if not kw_col:
            raise ValueError("Research CSV missing Keyword column")

        # --- auto-detect source from headers (SEMrush / Keywords Everywhere / SpyFu) ---
        has_ke = ("vol us" in cols) or ("vol uk" in cols)
        has_spyfu = ("broad cost per click" in cols) or ("total monthly clicks" in cols)
        has_semrush = ("search volume" in cols) or ("volume" in cols) or ("cpc usd" in cols) or ("cpc gbp" in cols) or ("cpc eur" in cols)

        detected = None
        if has_ke:
            detected = "keywordseverywhere"
        elif has_spyfu:
            detected = "spyfu"
        elif has_semrush:
            detected = "semrush"

        if detected is None:
            raise ValueError(
                "Could not detect research source from headers. Detected columns: "
                + ", ".join(list(df0.columns))
            )

        # --- parse per source ---
        if detected == "semrush":
            vol_col = get_col("search volume", "volume")
            cpc_usd = get_col("cpc usd")
            cpc_gbp = get_col("cpc gbp")
            cpc_eur = get_col("cpc eur")
            cpc_any = get_col("cpc")

            if not vol_col:
                raise ValueError("SEMrush CSV missing Volume or Search Volume")

            def cpc_col_for_geo(g: str):
                g = g.upper()
                if g == "US":
                    return cpc_usd or cpc_any
                if g == "GB":
                    return cpc_gbp or cpc_any
                if g in ("DE", "FR"):
                    return cpc_eur or cpc_any
                return cpc_any or cpc_usd or cpc_gbp or cpc_eur

            geos = ["US", "GB", "DE", "FR"] if geo_choice == "ALL" else [geo_choice]
            frames = []
            for g in geos:
                cpc_col = cpc_col_for_geo(g)
                out = pd.DataFrame()
                out["keyword"] = df0[kw_col].map(norm_kw)
                out["geo"] = g
                out["search_volume"] = pd.to_numeric(df0[vol_col], errors="coerce")
                out["cpc_est"] = pd.to_numeric(df0[cpc_col], errors="coerce") if cpc_col else pd.NA
                out["competition"] = pd.NA
                out["raw_json"] = df0.apply(lambda r: json.dumps(r.to_dict(), ensure_ascii=False), axis=1)
                frames.append(out)

            df = pd.concat(frames, ignore_index=True)

        elif detected == "keywordseverywhere":
            # canonicalized headers to support Keyword, Vol (US), Vol (UK), CPC ($), Cmp.
            vol_us = get_col("vol us")
            vol_uk = get_col("vol uk")
            cpc_col = get_col("cpc $", "cpc")
            cmp_col = get_col("cmp")

            if not (vol_us or vol_uk):
                raise ValueError("Keywords Everywhere CSV missing Vol (US) or Vol (UK)")
            if not cpc_col:
                raise ValueError("Keywords Everywhere CSV missing CPC ($) or CPC")
            if not cmp_col:
                raise ValueError("Keywords Everywhere CSV missing Cmp or Cmp.")

            def add_geo(g: str, vol_col_name: str):
                out = pd.DataFrame()
                out["keyword"] = df0[kw_col].map(norm_kw)
                out["geo"] = g
                out["search_volume"] = pd.to_numeric(df0[vol_col_name], errors="coerce") if vol_col_name else pd.NA
                out["cpc_est"] = pd.to_numeric(df0[cpc_col], errors="coerce")
                out["competition"] = pd.to_numeric(df0[cmp_col], errors="coerce")
                out["raw_json"] = df0.apply(lambda r: json.dumps(r.to_dict(), ensure_ascii=False), axis=1)
                return out

            if geo_choice == "ALL":
                frames = []
                if vol_us is not None:
                    frames.append(add_geo("US", vol_us))
                if vol_uk is not None:
                    frames.append(add_geo("GB", vol_uk))
                df = pd.concat(frames, ignore_index=True)
            else:
                if geo_choice == "US":
                    if vol_us is None:
                        raise ValueError("This file has no Vol (US) column")
                    df = add_geo("US", vol_us)
                elif geo_choice == "GB":
                    if vol_uk is None:
                        raise ValueError("This file has no Vol (UK) column")
                    df = add_geo("GB", vol_uk)
                else:
                    raise ValueError("Keywords Everywhere supports US or GB or ALL")

        else:
            # SpyFu: Keyword, Search Volume, Total Monthly Clicks, Broad Cost Per Click
            sv_col = get_col("search volume")
            cpc_col = get_col("broad cost per click")

            if not sv_col:
                raise ValueError("SpyFu CSV missing Search Volume")
            if not cpc_col:
                raise ValueError("SpyFu CSV missing Broad Cost Per Click")

            g = geo_choice if geo_choice != "ALL" else "US"
            df = pd.DataFrame()
            df["keyword"] = df0[kw_col].map(norm_kw)
            df["geo"] = g
            df["search_volume"] = pd.to_numeric(df0[sv_col], errors="coerce")
            df["cpc_est"] = pd.to_numeric(df0[cpc_col], errors="coerce")
            df["competition"] = pd.NA
            df["raw_json"] = df0.apply(lambda r: json.dumps(r.to_dict(), ensure_ascii=False), axis=1)

        df = df[df["keyword"] != ""]
        df = df[df["geo"] != ""]
        df["geo"] = df["geo"].astype(str).str.strip().str.upper()

        # merge duplicates inside the file by (keyword, geo)
        df = df.groupby(["keyword", "geo"], as_index=False).agg(
            search_volume=("search_volume", "max"),
            cpc_est=("cpc_est", "mean"),
            competition=("competition", "mean"),
            raw_json=("raw_json", "last"),
        )

        return df, detected

    def upsert_research(conn: sqlite3.Connection, df: pd.DataFrame, source: str) -> tuple[int, int]:
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        source = (source or "other").strip().lower()

        df = df.copy()
        df["source"] = source
        df = df.drop_duplicates(subset=["keyword", "geo", "source"])

        batch = list(df[["keyword", "geo", "source"]].itertuples(index=False, name=None))

        existing = set()
        cur = conn.cursor()
        chunk = 250
        for i in range(0, len(batch), chunk):
            part = batch[i:i + chunk]
            placeholders = ",".join(["(?, ?, ?)"] * len(part))
            flat = []
            for k, g, s in part:
                flat.extend([k, g, s])
            cur.execute(
                f"SELECT keyword, geo, source FROM keyword_research WHERE (keyword, geo, source) IN ({placeholders})",
                flat,
            )
            existing.update((r[0], r[1], r[2]) for r in cur.fetchall())

        rows = []
        for r in df.itertuples(index=False):
            rows.append(
                (
                    r.keyword,
                    r.geo,
                    r.source,
                    None if pd.isna(r.search_volume) else int(r.search_volume),
                    None if pd.isna(r.cpc_est) else float(r.cpc_est),
                    None if pd.isna(r.competition) else float(r.competition),
                    r.raw_json,
                    now,
                )
            )

        with conn:
            cur = conn.cursor()
            cur.execute("BEGIN;")
            cur.executemany(
                """
                INSERT INTO keyword_research
                (keyword, geo, source, search_volume, cpc_est, competition, raw_json, imported_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(keyword, geo, source) DO UPDATE SET
                  search_volume = excluded.search_volume,
                  cpc_est = excluded.cpc_est,
                  competition = excluded.competition,
                  raw_json = excluded.raw_json,
                  imported_at = excluded.imported_at
                """,
                rows,
            )
            cur.execute("COMMIT;")

        inserted = 0
        updated = 0
        for key in batch:
            if key in existing:
                updated += 1
            else:
                inserted += 1
        return inserted, updated

    if up_r is not None:
        if st.button("Import Research CSV", type="primary"):
            try:
                rdf, detected_source = read_research_csv(up_r, source, research_geo)
                ins, upd = upsert_research(conn, rdf, detected_source)
                st.success(f"Research Imported Rows {len(rdf)} Inserted {ins} Updated {upd} Source {detected_source}")
                st.session_state["last_keywords"] = list(rdf[["keyword", "geo"]].itertuples(index=False, name=None))
            except Exception as e:
                st.error(str(e))

with right:
    st.subheader("Search")

    mode = st.selectbox("View", ["tested", "research"], index=0)
    geo_filter = st.selectbox("Geo Filter", ["ALL", "US", "GB", "DE", "FR", "XX"], index=0)
    min_sell_rpc = st.number_input("Min Sell RPC", min_value=0.0, value=0.0, step=0.1)

    q = st.text_input(
        "Search Term",
        placeholder="type part of a keyword (blank shows last added)",
    )

    if mode == "tested":
        if q.strip():
            df = search(conn, q, limit, geo=geo_filter)
        elif st.session_state["last_keywords"]:
            pairs = st.session_state["last_keywords"]
            placeholders = ",".join(["(?, ?)"] * len(pairs))
            flat = []
            for k, g in pairs:
                flat.extend([k, g])

            df = pd.read_sql_query(
                f"""
                SELECT keyword, geo, clicks, cpc, sell_rpc, ctr, last_updated
                FROM keyword_metrics
                WHERE (keyword, geo) IN ({placeholders})
                ORDER BY last_updated DESC
                """,
                conn,
                params=flat,
            )
        else:
            df = pd.DataFrame(
                columns=["keyword", "geo", "clicks", "cpc", "sell_rpc", "ctr", "last_updated"]
            )
        if min_sell_rpc > 0:
            df = df[df["sell_rpc"].fillna(0) >= float(min_sell_rpc)]

    else:
        if q.strip():
            like = f"%{norm_kw(q)}%"
            if geo_filter != "ALL":
                df = pd.read_sql_query(
                    """
                    SELECT keyword, geo, source, search_volume, cpc_est, competition, imported_at
                    FROM keyword_research
                    WHERE geo = ? AND keyword LIKE ?
                    ORDER BY imported_at DESC
                    LIMIT ?;
                    """,
                    conn,
                    params=(geo_filter, like, limit),
                )
            else:
                df = pd.read_sql_query(
                    """
                    SELECT keyword, geo, source, search_volume, cpc_est, competition, imported_at
                    FROM keyword_research
                    WHERE keyword LIKE ?
                    ORDER BY imported_at DESC
                    LIMIT ?;
                    """,
                    conn,
                    params=(like, limit),
                )
        elif st.session_state["last_keywords"]:
            pairs = st.session_state["last_keywords"]
            placeholders = ",".join(["(?, ?)"] * len(pairs))
            flat = []
            for k, g in pairs:
                flat.extend([k, g])

            df = pd.read_sql_query(
                f"""
                SELECT keyword, geo, source, search_volume, cpc_est, competition, imported_at
                FROM keyword_research
                WHERE (keyword, geo) IN ({placeholders})
                ORDER BY imported_at DESC
                """,
                conn,
                params=flat,
            )
        else:
            df = pd.DataFrame(
                columns=["keyword", "geo", "source", "search_volume", "cpc_est", "competition", "imported_at"]
            )

    st.dataframe(
        df,
        use_container_width=True,
        height=520,
        column_config={
            "keyword": st.column_config.TextColumn("keyword", width="large"),
            "geo": st.column_config.TextColumn("geo", width="small"),
            "clicks": st.column_config.NumberColumn("clicks", width="small"),
            "cpc": st.column_config.NumberColumn("cpc", format="%.2f", width="small"),
            "sell_rpc": st.column_config.NumberColumn("sell_rpc", format="%.2f", width="small"),
            "ctr": st.column_config.NumberColumn("ctr", format="%.2f", width="small"),
            "last_updated": st.column_config.TextColumn("last_updated", width="medium"),
        },
    )

    if not df.empty:
        # Copy block stays "easy to copy": keyword lines only
        kw_block = "\n".join(df["keyword"].tolist())
        st.caption("Copy Block")
        st.text_area("Copy These Keywords", value=kw_block, height=160)

        st.download_button(
            "Download Search Results CSV",
            data=df_to_csv_bytes(df),
            file_name="keyword_search_results.csv",
            mime="text/csv",
        )

st.divider()
st.subheader("Export Full Library")
export_all = st.button("Generate Full CSV Export")

if export_all:
    full_df = pd.read_sql_query(
        """
        SELECT keyword, geo, clicks, cpc, sell_rpc, ctr, last_updated
        FROM keyword_metrics
        ORDER BY clicks DESC, sell_rpc DESC;
        """,
        conn,
    )
    st.success(f"Rows {len(full_df)}")
    st.download_button(
        "Download Full Library CSV",
        data=df_to_csv_bytes(full_df),
        file_name="keyword_library_export.csv",
        mime="text/csv",
    )