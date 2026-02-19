import os
import re
import io
import sys
import json
import csv
import time
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse

import pandas as pd
from google.cloud import storage

import google.auth
from google.auth.transport.requests import AuthorizedSession

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("native-ad-enricher")


# -----------------
# Time helpers
# -----------------
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_day() -> str:
    return utc_now().date().isoformat()


def utc_ts_compact() -> str:
    return utc_now().strftime("%Y%m%d-%H%M%S")


# -----------------
# GCS helpers
# -----------------
def _gcs_client() -> storage.Client:
    return storage.Client()


def _write_gcs_bytes(bucket: str, blob_name: str, data: bytes, content_type: str):
    b = _gcs_client().bucket(bucket).blob(blob_name)
    b.upload_from_string(data, content_type=content_type)


def _list_blobs(bucket: str, prefix: str):
    return list(_gcs_client().bucket(bucket).list_blobs(prefix=prefix))


# -----------------
# URL helpers
# -----------------
def _safe_domain(u: str) -> str:
    try:
        h = urlparse(u).netloc.lower()
        return h[4:] if h.startswith("www.") else h
    except Exception:
        return ""


def _safe_path(u: str) -> str:
    try:
        return (urlparse(u).path or "").lower()
    except Exception:
        return ""


# -----------------
# Editorial detection (tightened, but does NOT “ban” news sites outright)
# -----------------
NEWS_DOMAINS = {
    "nbcnews.com",
    "cnn.com",
    "foxnews.com",
    "nytimes.com",
    "washingtonpost.com",
    "reuters.com",
    "apnews.com",
    "theguardian.com",
    "bbc.co.uk",
    "bbc.com",
    "usatoday.com",
    "msnbc.com",
    "news.yahoo.com",
    "politico.com",
    "axios.com",
    "wsj.com",
    # commonly-seen “news wrapper / short” domains
    "ms.now",
}

EDITORIAL_MARKERS = [
    "breaking",
    "live updates",
    "exclusive",
    "interview",
    "debate",
    "opinion",
    "analysis",
    "watch:",
    "video:",
    "podcast",
    "what we know",
    "explainer",
    "transcript",
    "latest updates",
    "fact check",
    "fact-check",
]

BAD_HEADLINE_PATTERNS = [
    r"click to share",
    r"opens in new window",
    r"\blive updates?\b",
    r"\bwhat we know\b",
    r"\btranscript\b",
    r"\bopinion\b",
    r"\banalysis\b",
    r"\binterview\b",
    r"\bwatch:\b",
    r"\bvideo:\b",
    r"\bpodcast\b",
    r"\bnewsletter\b",
]

def _is_bad_headline(headline: str) -> bool:
    if not headline:
        return False
    hl = headline.lower().strip()
    return any(re.search(p, hl) for p in BAD_HEADLINE_PATTERNS)


def _domain_is_news(dest_domain: str) -> bool:
    """
    True if dest_domain is exactly a known news domain OR a subdomain of one.
    e.g. "www.cnn.com" normalized -> "cnn.com" earlier, but also handles "foo.nbcnews.com".
    """
    if not dest_domain:
        return False
    d = dest_domain.lower().strip()
    for nd in NEWS_DOMAINS:
        if d == nd or d.endswith("." + nd):
            return True
    return False

def _looks_editorial(headline: str, dest_domain: str) -> bool:
    if not headline:
        return False

    hl = headline.lower().strip()

    # Strong editorial markers in the text
    if any(m in hl for m in EDITORIAL_MARKERS):
        return True

    # Strong politics/news tokens regardless of domain (this is what you’re seeing leak in)
    if re.search(r"\b(trump|biden|democrat|republican|maga|doj|fbi|fed|powell|white house|administration|senate|congress|election|supreme court|ukraine|israel|gaza)\b", hl):
        return True

    # Classic news framing verbs regardless of domain
    if re.search(r"\b(says|report|reports|reported|amid|after|before|during|vows|slam|accuses|probe|criminal)\b", hl):
        return True

    # News-domain becomes editorial even faster (kept, but no longer required)
    if _domain_is_news(dest_domain):
        return True

    return False



def _extract_cta_word(text: str) -> str:
    if not text:
        return ""
    ctas = [
        "search",
        "see",
        "discover",
        "learn",
        "read",
        "compare",
        "check",
        "get",
        "find",
        "take a look",
        "take a peek",
        "view",
        "browse",
        "shop",
    ]
    t = text.lower()
    for c in ctas:
        if c in t:
            return c.replace(" ", "_")
    return ""


# -----------------
# Scoring (tightened to push down news/editorial)
# -----------------
def _score_row(row: dict) -> int:
    headline = (row.get("headline") or "").strip()
    network = (row.get("network") or "").strip()
    dest = (row.get("dest_domain") or "").strip()

    score = 0

    # Network signal
    if network in ("Taboola", "Outbrain"):
        score += 35
    elif network:
        score += 15
    else:
        score += 0

    # Headline length (ads tend to be mid-length)
    n = len(headline)
    if 35 <= n <= 120:
        score += 25
    elif 25 <= n < 35 or 120 < n <= 170:
        score += 12
    else:
        score += 0

    # CTA presence
    if row.get("cta_word"):
        score += 12

    # Commercial intent tokens
    hl = headline.lower()
    if re.search(r"\b(cost|price|rates|eligible|paying|save|savings|quote|quotes|insurance|bank|banks|interest)\b", hl):
        score += 10

    # Destination domain present
    if dest:
        score += 8

    # Heavy penalty for editorial patterns (still keep the row)
    if _looks_editorial(headline, dest):
        score -= 60

    # Additional penalty for pure news framing verbs (even off news domains)
    if re.search(r"\b(says|report|reports|reported|amid|after|before|during|vows|slam|accuses)\b", hl):
        score -= 10

    # Clamp
    score = max(0, min(100, score))
    return int(score)


# -----------------
# Raw loader
# -----------------
def _parse_day_from_blobname(blob_name: str, raw_root: str) -> str:
    """
    Expects: <raw_root>/<YYYY-MM-DD>/native_<ts>.csv
    """
    s = blob_name
    if not s.startswith(raw_root.rstrip("/") + "/"):
        return ""
    parts = s.split("/")
    raw_parts = raw_root.strip("/").split("/")
    if len(parts) < len(raw_parts) + 2:
        return ""
    day = parts[len(raw_parts)]
    return day if re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) else ""


def load_raw_for_day(bucket: str, raw_root: str, day: str) -> pd.DataFrame:
    day_prefix = f"{raw_root.rstrip('/')}/{day}/"
    blobs = [b for b in _list_blobs(bucket, day_prefix) if b.name.endswith(".csv") and "/native_" in b.name]
    if not blobs:
        return pd.DataFrame()

    dfs = []
    for b in sorted(blobs, key=lambda x: x.name):
        try:
            txt = b.download_as_text()
            df = pd.read_csv(io.StringIO(txt))
            df["raw_blob"] = b.name
            dfs.append(df)
        except Exception as e:
            log.info(f"skip blob read failed blob={b.name} err={e}")

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


# -----------------
# Enrich
# -----------------
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    # Ensure expected cols exist
    for col in [
        "source",
        "page_url",
        "network",
        "headline",
        "ad_description",
        "image_url",
        "original_url",
        "hash_id",
        "date_seen",
        "ts_seen",
        "hash_recurrence",
        "raw_blob",
    ]:
        if col not in df.columns:
            df[col] = ""

    df["dest_domain"] = df["original_url"].astype(str).apply(_safe_domain)
    df["dest_path"] = df["original_url"].astype(str).apply(_safe_path)
    df["source_domain"] = df["page_url"].astype(str).apply(_safe_domain)

    # Normalize search-engine set (dest_domain already stripped of www.)
    search_domains = {"search.yahoo.com", "bing.com", "google.com", "duckduckgo.com"}
    df["is_search_engine"] = df["dest_domain"].isin(search_domains).astype("int64")

    def _is_arb(u: str) -> int:
        s = (u or "").lower()
        return 1 if any(x in s for x in ("yhs/r", "startsearch", "searchfeed", "trk.", "click.php", "token=")) else 0

    df["is_arb_redirect"] = df["original_url"].astype(str).apply(_is_arb).astype("int64")

    df["cta_word"] = df["headline"].astype(str).apply(_extract_cta_word)
    df["looks_editorial"] = df.apply(
        lambda r: 1 if _looks_editorial((r.get("headline") or ""), (r.get("dest_domain") or "")) else 0,
        axis=1,
    ).astype("int64")

    df["is_bad_headline"] = df["headline"].astype(str).apply(lambda s: 1 if _is_bad_headline(s) else 0).astype("int64")

    # Score
    df["quality_score"] = df.apply(lambda r: _score_row(r.to_dict()), axis=1).astype("int64")

    # Niche hint (keep for now)
    def _niche(h: str) -> str:
        t = (h or "").lower()
        if re.search(r"\b(window|windows|roof|gutter|hvac|garage)\b", t):
            return "home_services"
        if re.search(r"\b(cd rates|savings|bank|interest|ira|social security|ss)\b", t):
            return "finance"
        if re.search(r"\b(car|vehicle|dashcam|warranty|auto insurance)\b", t):
            return "auto"
        if re.search(r"\b(weight loss|dementia|edema|serum)\b", t):
            return "health"
        if re.search(r"\b(dating|singles)\b", t):
            return "dating"
        return "other"

    df["niche_hint"] = df["headline"].astype(str).apply(_niche)

    # Type coercions for BigQuery
    df["date_seen"] = pd.to_datetime(df["date_seen"], errors="coerce").dt.date
    df["ts_seen"] = pd.to_datetime(df["ts_seen"], errors="coerce", utc=True)
    df["hash_recurrence"] = pd.to_numeric(df["hash_recurrence"], errors="coerce").fillna(0).astype("int64")

    return df


# -----------------
# BigQuery REST helpers (no extra deps)
# -----------------
def _bq_session():
    creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return AuthorizedSession(creds)


def _bq_insert_load_job(project: str, location: str, job_body: dict) -> dict:
    sess = _bq_session()
    url = f"https://bigquery.googleapis.com/bigquery/v2/projects/{project}/jobs"
    r = sess.post(url, json=job_body)
    r.raise_for_status()
    return r.json()


def _bq_get_job(project: str, location: str, job_id: str) -> dict:
    sess = _bq_session()
    url = f"https://bigquery.googleapis.com/bigquery/v2/projects/{project}/jobs/{job_id}"
    r = sess.get(url, params={"location": location})
    r.raise_for_status()
    return r.json()


def _bq_wait_job(project: str, location: str, job_id: str, poll_s: int = 2, timeout_s: int = 900) -> dict:
    t0 = time.time()
    while True:
        j = _bq_get_job(project, location, job_id)
        state = (((j.get("status") or {}).get("state")) or "").upper()
        if state == "DONE":
            err = (j.get("status") or {}).get("errorResult")
            if err:
                raise RuntimeError(f"BigQuery job failed: {err}")
            return j
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"BigQuery job timed out: {job_id}")
        time.sleep(poll_s)


def _bq_query(project: str, location: str, sql: str) -> dict:
    sess = _bq_session()
    url = f"https://bigquery.googleapis.com/bigquery/v2/projects/{project}/queries"
    payload = {"query": sql, "useLegacySql": False, "location": location}
    r = sess.post(url, json=payload)
    r.raise_for_status()
    out = r.json()

    # If not complete, poll using jobReference, then fetch results
    if not out.get("jobComplete", True):
        job_id = out["jobReference"]["jobId"]
        _bq_wait_job(project, location, job_id)
        r2 = sess.get(url, params={"jobId": job_id, "location": location})
        r2.raise_for_status()
        return r2.json()

    return out


def _query_to_csv_bytes(qr: dict) -> bytes:
    fields = [f["name"] for f in (qr.get("schema", {}).get("fields") or [])]
    rows = qr.get("rows") or []
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(fields)
    for r in rows:
        vals = [c.get("v") for c in (r.get("f") or [])]
        w.writerow(vals)
    return buf.getvalue().encode("utf-8")


# -----------------
# BigQuery rollup SQL (filters out low-quality/news by default)
# -----------------
def rollup_sql(project: str, dataset: str, table: str, days_back: int) -> str:
    return f"""
    WITH maxd AS (
      SELECT MAX(date_seen) AS max_day
      FROM `{project}.{dataset}.{table}`
    ),
    base AS (
      SELECT
        * EXCEPT(rn)
      FROM (
        SELECT
          t.*,
          ROW_NUMBER() OVER (
            PARTITION BY t.hash_id
            ORDER BY t.ts_seen DESC
          ) AS rn
        FROM `{project}.{dataset}.{table}` t, maxd
        WHERE t.date_seen >= DATE_SUB(maxd.max_day, INTERVAL {days_back} DAY)
          AND t.headline IS NOT NULL
          AND t.headline != ""
      )
      WHERE rn = 1
    )
    SELECT
      headline,
      dest_domain,
      network,
      COUNT(1) AS impressions,
      ROUND(AVG(CAST(quality_score AS FLOAT64)), 1) AS avg_score,
      MAX(CAST(is_search_engine AS INT64)) AS any_search,
      MAX(CAST(is_arb_redirect AS INT64)) AS any_arb,
      ANY_VALUE(niche_hint) AS niche
    FROM base
    WHERE quality_score >= 70
      AND COALESCE(CAST(is_bad_headline AS INT64), 0) = 0
      AND COALESCE(CAST(looks_editorial AS INT64), 0) = 0
    GROUP BY headline, dest_domain, network
    ORDER BY impressions DESC, avg_score DESC
    """


def latest_ads_sql(project: str, dataset: str, table: str) -> str:
    return f"""
    WITH base AS (
      SELECT
        *,
        COALESCE(
          NULLIF(CAST(hash_id AS STRING), ''),
          TO_HEX(SHA256(CONCAT(
            IFNULL(CAST(headline AS STRING), ''), '|',
            IFNULL(CAST(dest_domain AS STRING), ''), '|',
            IFNULL(CAST(network AS STRING), '')
          )))
        ) AS dedupe_key
      FROM `{project}.{dataset}.{table}`
      WHERE date_seen = (SELECT MAX(date_seen) FROM `{project}.{dataset}.{table}`)
    )
    SELECT * EXCEPT(dedupe_key)
    FROM base
    QUALIFY ROW_NUMBER() OVER (PARTITION BY dedupe_key ORDER BY ts_seen DESC) = 1
    ORDER BY ts_seen DESC
    LIMIT 5000
    """


# -----------------
# Main
# -----------------
def main():
    bucket = os.environ.get("BUCKET_NAME", "").strip()
    if not bucket:
        print("ERROR: BUCKET_NAME env var required", file=sys.stderr)
        sys.exit(2)

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    if not project:
        print("ERROR: GOOGLE_CLOUD_PROJECT env var required", file=sys.stderr)
        sys.exit(2)

    # BigQuery destination
    bq_dataset = os.environ.get("BQ_DATASET", "arb_native_intel").strip()
    bq_table = os.environ.get("BQ_TABLE", "ads_enriched_history").strip()
    bq_location = os.environ.get("BQ_LOCATION", "US").strip()

    # GCS roots
    raw_root = os.environ.get("RAW_ROOT", "native/daily").strip()
    summaries_root = os.environ.get("SUMMARIES_ROOT", "summaries").strip()

    target_day = os.environ.get("TARGET_DAY", "").strip() or utc_day()
    run_ts = os.environ.get("RUN_TS", "").strip() or utc_ts_compact()

    log.info(f"enrich start bucket={bucket} target_day={target_day} raw_root={raw_root}")

    raw_df = load_raw_for_day(bucket=bucket, raw_root=raw_root, day=target_day)
    if raw_df.empty:
        log.info("enrich: no raw rows found; still producing empty rollups")
        enriched_df = pd.DataFrame()
    else:
        enriched_df = enrich(raw_df)

    # If empty, still write empty rollups so UI doesn't break
    if enriched_df.empty:
        empty_roll = "headline,dest_domain,network,impressions,avg_score,any_search,any_arb,niche\n"
        _write_gcs_bytes(bucket, f"{summaries_root.rstrip('/')}/last_7_days.csv", empty_roll.encode("utf-8"), "text/csv")
        _write_gcs_bytes(bucket, f"{summaries_root.rstrip('/')}/last_30_days.csv", empty_roll.encode("utf-8"), "text/csv")
        sys.exit(0)

    # Stage enriched csv to GCS (temporary artifact to load into BigQuery)
    staging_blob = f"native_enriched/staging/{target_day}/enriched_{run_ts}.csv"
    # Keep staged CSV aligned to the existing BigQuery table schema
    BQ_COLS = [
        "source",
        "page_url",
        "network",
        "headline",
        "ad_description",
        "image_url",
        "original_url",
        "hash_id",
        "date_seen",
        "ts_seen",
        "hash_recurrence",
        "raw_blob",
        "dest_domain",
        "dest_path",
        "source_domain",
        "is_search_engine",
        "is_arb_redirect",
        "cta_word",
        "looks_editorial",
        "quality_score",
        "niche_hint",
        "is_bad_headline",
    ]

    staged_df = enriched_df.copy()
    for c in BQ_COLS:
        if c not in staged_df.columns:
            staged_df[c] = ""

    staged_df = staged_df[BQ_COLS]
    csv_bytes = staged_df.to_csv(index=False).encode("utf-8")

    _write_gcs_bytes(bucket, staging_blob, csv_bytes, "text/csv")
    staging_uri = f"gs://{bucket}/{staging_blob}"
    log.info(f"staged enriched csv: {staging_uri}")

    # BigQuery LOAD job: append into destination table
    job_body = {
        "configuration": {
            "load": {
                "sourceUris": [staging_uri],
                "destinationTable": {
                    "projectId": project,
                    "datasetId": bq_dataset,
                    "tableId": bq_table,
                },
                "writeDisposition": "WRITE_APPEND",
                "sourceFormat": "CSV",
                "skipLeadingRows": 1,
                "allowQuotedNewlines": True,
                "allowJaggedRows": True,
            }
        }
    }

    job = _bq_insert_load_job(project, bq_location, job_body)
    job_id = job["jobReference"]["jobId"]
    log.info(f"bq load job submitted: {job_id}")
    _bq_wait_job(project, bq_location, job_id, poll_s=2, timeout_s=900)
    log.info("bq load job DONE")

    # Build 7/30 rollups from BigQuery and write to GCS summaries for the UI
    q7 = _bq_query(project, bq_location, rollup_sql(project, bq_dataset, bq_table, 7))
    q30 = _bq_query(project, bq_location, rollup_sql(project, bq_dataset, bq_table, 30))

    _write_gcs_bytes(bucket, f"{summaries_root.rstrip('/')}/last_7_days.csv", _query_to_csv_bytes(q7), "text/csv")
    _write_gcs_bytes(bucket, f"{summaries_root.rstrip('/')}/last_30_days.csv", _query_to_csv_bytes(q30), "text/csv")

    # Latest snapshots for UI (optional but keeps existing UI paths stable)
    latest_ads = _bq_query(project, bq_location, latest_ads_sql(project, bq_dataset, bq_table))
    _write_gcs_bytes(bucket, "native_enriched/latest/ads_latest.csv", _query_to_csv_bytes(latest_ads), "text/csv")

    # latest rollup: use 7d (fast + “what’s hot lately”)
    latest_roll = _bq_query(project, bq_location, rollup_sql(project, bq_dataset, bq_table, 7))
    _write_gcs_bytes(bucket, "native_enriched/latest/rollup_latest.csv", _query_to_csv_bytes(latest_roll), "text/csv")

    # State marker
    state = {
        "status": "done",
        "target_day": target_day,
        "run_ts": run_ts,
        "raw_rows": int(len(raw_df)),
        "enriched_rows": int(len(enriched_df)),
        "staging_uri": staging_uri,
        "bq_table": f"{project}.{bq_dataset}.{bq_table}",
        "gcs_sum7": f"gs://{bucket}/{summaries_root.rstrip('/')}/last_7_days.csv",
        "gcs_sum30": f"gs://{bucket}/{summaries_root.rstrip('/')}/last_30_days.csv",
    }
    _write_gcs_bytes(
        bucket,
        f"{summaries_root.rstrip('/')}/enricher_last_run.json",
        json.dumps(state, indent=2).encode("utf-8"),
        "application/json",
    )

    log.info("enrich done")
    sys.exit(0)


if __name__ == "__main__":
    main()