import os
import re
import json
import hashlib
import asyncio
import threading
import logging
import sys
import ssl
from datetime import datetime, timezone
from urllib.parse import urlparse, urljoin

import yaml
import pandas as pd
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from google.cloud import storage
from playwright.async_api import async_playwright

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("native-ad-scan")

app = FastAPI()

@app.get("/_health", include_in_schema=False)
def _health():
    return {"ok": True}

@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"ok": True}

BUCKET = os.environ.get("BUCKET_NAME", "")
PREFIX = os.environ.get("GCS_PREFIX", "native")

MAX_PAGES_PER_SITE = int(os.environ.get("MAX_PAGES_PER_SITE", "2"))
SITE_TIMEOUT_MS = int(os.environ.get("SITE_TIMEOUT_MS", "45000"))
SCROLL_STEPS = int(os.environ.get("SCROLL_STEPS", "14"))
SCROLL_SLEEP = float(os.environ.get("SCROLL_SLEEP", "0.9"))
POST_WAIT = float(os.environ.get("POST_WAIT", "3.0"))

TABOOLA_SELECTORS = [
    '[id*="taboola"]', '[class*="taboola"]', 'div[id^="taboola-"]', '[data-taboola]',
    'iframe[src*="taboola"]', 'iframe[name*="taboola"]',
]
OUTBRAIN_SELECTORS = [
    '[id*="outbrain"]', '[class*="outbrain"]', 'div[class^="OUTBRAIN"]',
    '[data-ob-widget]', '[data-ob-template]',
    'iframe[src*="outbrain"]', 'iframe[name*="outbrain"]',
]
GENERIC_SELECTORS = [
    'div.trc_rbox_container', 'div.trc-content-sponsored',
    'div:has(a[href*="taboola.com"])', 'div:has(a[href*="outbrain.com"])',
]
CONTAINER_CSS = ", ".join(TABOOLA_SELECTORS + OUTBRAIN_SELECTORS + GENERIC_SELECTORS)

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

def utc_ts_compact() -> str:
    return utc_now().strftime("%Y%m%d-%H%M%S")

def utc_iso_seconds() -> str:
    return utc_now().isoformat(timespec="seconds").replace("+00:00", "Z")

def utc_day() -> str:
    return utc_now().date().isoformat()

def _host(u: str) -> str:
    try:
        return urlparse(u).netloc.lower().lstrip("www.")
    except Exception:
        return ""

def _same_host(a: str, b: str) -> bool:
    ha, hb = _host(a), _host(b)
    return bool(ha) and bool(hb) and (ha == hb or ha.endswith("." + hb) or hb.endswith("." + ha))

def _clean(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    s = s.replace("|", " ")
    s = re.sub(r"(?i)\b(search ads?)\b", "", s)
    s = re.sub(r"(?i)\b(sponsored|ad choices|adchoice|advertisement|promoted)\b", "", s)
    s = s.replace("Learn More", "").replace("LEARN MORE", "")
    return re.sub(r"\s+", " ", s).strip(" -|:/")

def _hash_id(headline: str, img: str, href: str) -> str:
    return hashlib.md5((headline + "|" + (img or "") + "|" + (href or "")).encode("utf-8")).hexdigest()

def _detect_network_from_container(html_low: str) -> str:
    if not html_low:
        return "Unknown"
    if "taboola" in html_low or "data-taboola" in html_low or "trc.taboola" in html_low:
        return "Taboola"
    if "outbrain" in html_low or "data-ob" in html_low or "ob-widget" in html_low:
        return "Outbrain"
    return "Unknown"

def _is_bad_href(href: str) -> bool:
    h = (href or "").lower()
    if not h.startswith(("http://", "https://")):
        return True

    if any(x in h for x in (
        "facebook.com/dialog/share", "twitter.com/intent", "linkedin.com/share",
        "pinterest.com/pin/create", "reddit.com/submit", "mailto:", "wa.me", "api.whatsapp.com",
    )):
        return True

    p = urlparse(h)
    host = (p.netloc or "").lower()
    query = (p.query or "").lower()
    path = (p.path or "").lower()

    if host == "popup.taboola.com":
        return True
    if "template=colorbox" in query:
        return True

    if host in ("www.outbrain.com", "outbrain.com") and ("what-is" in path or "what-is" in query):
        return True
    if host in ("www.taboola.com", "taboola.com") and ("what-is" in path or "what-is" in query):
        return True

    return False

def _is_bad_headline(headline: str) -> bool:
    hl = (headline or "").lower()
    if not hl:
        return True
    if "learn about this recommendation" in hl:
        return True
    if "opens dialog" in hl:
        return True
    if hl.startswith(("share to ", "share on ")):
        return True
    return False

# --- GCS helpers with retries/timeouts ---
from google.api_core.retry import Retry, if_exception_type
from google.api_core import exceptions as gexc

try:
    from requests.exceptions import SSLError as RequestsSSLError
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import ReadTimeout as RequestsReadTimeout
    from requests.exceptions import Timeout as RequestsTimeout
except Exception:
    RequestsSSLError = None
    RequestsConnectionError = None
    RequestsReadTimeout = None
    RequestsTimeout = None

_GCS_CLIENT = None
_GCS_LOCK = threading.Lock()

def gcs_client() -> storage.Client:
    global _GCS_CLIENT
    if _GCS_CLIENT is None:
        with _GCS_LOCK:
            if _GCS_CLIENT is None:
                _GCS_CLIENT = storage.Client()
    return _GCS_CLIENT

def _retryable_exc_types():
    types = [
        gexc.TooManyRequests,
        gexc.ServiceUnavailable,
        gexc.InternalServerError,
        gexc.BadGateway,
        gexc.GatewayTimeout,
        gexc.GoogleAPICallError,
        gexc.RetryError,
        ssl.SSLError,
        ConnectionError,
        TimeoutError,
        OSError,
    ]
    if RequestsSSLError is not None:
        types.append(RequestsSSLError)
    if RequestsConnectionError is not None:
        types.append(RequestsConnectionError)
    if RequestsReadTimeout is not None:
        types.append(RequestsReadTimeout)
    if RequestsTimeout is not None:
        types.append(RequestsTimeout)
    return tuple({t for t in types if isinstance(t, type)})

def _gcs_retry() -> Retry:
    return Retry(
        initial=1.0,
        maximum=30.0,
        multiplier=2.0,
        deadline=300.0,
        predicate=if_exception_type(*_retryable_exc_types()),
    )

def gcs_read_json(bucket: str, blob_name: str):
    try:
        b = gcs_client().bucket(bucket).blob(blob_name)
        if not b.exists(retry=_gcs_retry(), timeout=30):
            return None
        txt = b.download_as_text(retry=_gcs_retry(), timeout=60)
        return json.loads(txt)
    except Exception:
        return None

def gcs_write_text(bucket: str, blob_name: str, text: str, content_type="text/plain"):
    b = gcs_client().bucket(bucket).blob(blob_name)
    b.upload_from_string(text, content_type=content_type, retry=_gcs_retry(), timeout=180)

def gcs_write_bytes(bucket: str, blob_name: str, data: bytes, content_type="application/octet-stream"):
    b = gcs_client().bucket(bucket).blob(blob_name)
    b.upload_from_string(data, content_type=content_type, retry=_gcs_retry(), timeout=240)
# --- end GCS helpers ---

async def dismiss_common_banners(page):
    for sel in (
        "#onetrust-accept-btn-handler",
        "button#onetrust-accept-btn-handler",
        "button:has-text('Accept All')",
        "button:has-text('Accept')",
        "button:has-text('I Agree')",
        "button:has-text('Agree')",
        "button:has-text('Continue')",
        "button:has-text('OK')",
    ):
        try:
            loc = page.locator(sel)
            if await loc.count():
                await loc.first.click(timeout=2000)
                break
        except Exception:
            pass

async def scroll_for_widgets(page):
    for _ in range(SCROLL_STEPS):
        await page.mouse.wheel(0, 1400)
        await asyncio.sleep(SCROLL_SLEEP)
    await asyncio.sleep(POST_WAIT)

async def extract_from_page(page, src_name: str, src_url: str):
    rows = []
    containers = await page.locator(CONTAINER_CSS).all()

    for fr in page.frames:
        if fr is page.main_frame:
            continue
        tag = ((fr.url or "") + "|" + (fr.name or "")).lower()
        if "taboola" in tag or "outbrain" in tag or "trc." in tag:
            try:
                containers += await fr.locator(CONTAINER_CSS).all()
            except Exception:
                pass

    containers = containers[:20]
    seen = set()

    for el in containers:
        try:
            html_chunk = await el.evaluate("(n)=>n.outerHTML")
        except Exception:
            html_chunk = ""
        html_low = (html_chunk or "").lower()
        network_guess = _detect_network_from_container(html_low)

        links = el.locator("a[href]")
        link_count = min(await links.count(), 90)

        for i in range(link_count):
            a = links.nth(i)

            href = (await a.get_attribute("href")) or ""
            if href.startswith("/"):
                href = urljoin(page.url, href)
            if href.startswith("//"):
                href = "https:" + href

            if _is_bad_href(href):
                continue
            if _same_host(href, src_url):
                continue

            txt = ""
            for attr in ("data-item-title", "aria-label", "title"):
                v = await a.get_attribute(attr)
                if v and v.strip():
                    txt = v.strip()
                    break
            if not txt:
                try:
                    txt = await a.inner_text()
                except Exception:
                    txt = await a.evaluate("(n)=>n.textContent || ''")

            headline = _clean(txt)

            if _is_bad_headline(headline):
                continue

            # Desktop behavior: reject too-short or weirdly long strings
            if len(headline) < 25 or len(headline) > 200:
                continue

            img_url = ""
            img = a.locator("img").first
            try:
                if await img.count():
                    img_url = (await img.get_attribute("src")) or (await img.get_attribute("data-src")) or ""
            except Exception:
                pass

            key = (headline, href)
            if key in seen:
                continue
            seen.add(key)

            # Try to refine network if the href itself contains hints
            hlow = href.lower()
            net = network_guess
            if net == "Unknown":
                if "taboola" in hlow:
                    net = "Taboola"
                elif "outbrain" in hlow:
                    net = "Outbrain"

            rows.append({
                "source": src_name,
                "page_url": page.url,
                "network": net,
                "headline": headline,
                "ad_description": "",
                "image_url": img_url,
                "original_url": href,
                "hash_id": _hash_id(headline, img_url, href),
                "date_seen": utc_day(),
                "ts_seen": utc_iso_seconds(),
            })

    return rows

async def scan_sources():
    with open("sources.yaml", "r", encoding="utf-8") as f:
        sources = yaml.safe_load(f).get("sources", [])

    all_rows = []
    diag = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="en-US",
            viewport={"width": 1366, "height": 900},
            timezone_id="America/Los_Angeles",
        )

        try:
            for src in sources:
                src_name = src.get("name")
                src_url = src.get("url")
                if not src_name or not src_url:
                    diag.append({"source": src_name or "UNKNOWN", "kept": 0, "tried_pages": [], "error": "Invalid source entry"})
                    continue

                page = await context.new_page()
                kept = 0
                tried_pages = []
                err = None

                log.info(f"scan source start source={src_name} url={src_url}")

                try:
                    await page.goto(src_url, wait_until="domcontentloaded", timeout=SITE_TIMEOUT_MS)
                    tried_pages.append(page.url)
                    await dismiss_common_banners(page)
                    await scroll_for_widgets(page)

                    rows = await extract_from_page(page, src_name, src_url)
                    all_rows.extend(rows)
                    kept += len(rows)

                    if kept == 0 and MAX_PAGES_PER_SITE > 1:
                        try:
                            html = await page.content()
                            soup = BeautifulSoup(html, "html.parser")
                            cand = None
                            for a in soup.select("a[href]"):
                                href = a.get("href") or ""
                                if href.startswith("/"):
                                    href = urljoin(src_url, href)
                                if href.startswith("http") and _same_host(href, src_url) and any(
                                    x in href for x in ("/news", "/politics", "/us", "/world", "/business", "/health", "/weather")
                                ):
                                    cand = href
                                    break
                            if cand:
                                await page.goto(cand, wait_until="domcontentloaded", timeout=SITE_TIMEOUT_MS)
                                tried_pages.append(page.url)
                                await dismiss_common_banners(page)
                                await scroll_for_widgets(page)
                                rows2 = await extract_from_page(page, src_name, src_url)
                                all_rows.extend(rows2)
                                kept += len(rows2)
                        except Exception as e:
                            log.info(f"scan source second-page failed source={src_name} err={e}")

                except Exception as e:
                    err = str(e)

                finally:
                    entry = {"source": src_name, "kept": kept, "tried_pages": tried_pages}
                    if err:
                        entry["error"] = err
                        log.info(f"scan source error source={src_name} err={err}")
                    else:
                        log.info(f"scan source done source={src_name} kept={kept}")
                    diag.append(entry)

                    try:
                        await page.close()
                    except Exception:
                        pass

        finally:
            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                pass

    return all_rows, diag

async def run_once_to_gcs(run_id: str, day: str, run_ts: str):
    if not BUCKET:
        raise RuntimeError("Missing BUCKET_NAME env var")

    ping_blob = f"{PREFIX}/_ping/{day}/ping_{run_ts}.txt"
    state_blob = f"{PREFIX}/state/last_run.json"
    latest_hash_blob = f"{PREFIX}/state/latest_hashes.json"

    gcs_write_text(BUCKET, ping_blob, f"run_id={run_id}\n", content_type="text/plain")
    gcs_write_text(
        BUCKET,
        state_blob,
        json.dumps({"run_id": run_id, "status": "running", "ts": run_ts}, indent=2),
        content_type="application/json",
    )

    rows, diag = await scan_sources()
    df = pd.DataFrame(rows)

    prev = gcs_read_json(BUCKET, latest_hash_blob) or {}
    prev_hashes = set(prev.get("hashes", []))

    if not df.empty and "hash_id" in df.columns:
        df["hash_recurrence"] = df["hash_id"].apply(lambda h: 1 if h in prev_hashes else 0)
    else:
        df = pd.DataFrame(columns=[
            "source", "page_url", "network", "headline", "ad_description", "image_url",
            "original_url", "hash_id", "date_seen", "ts_seen", "hash_recurrence"
        ])

    out_blob = f"{PREFIX}/daily/{day}/native_{run_ts}.csv"
    diag_blob = f"{PREFIX}/daily/{day}/diag_{run_ts}.json"

    gcs_write_bytes(BUCKET, out_blob, df.to_csv(index=False).encode("utf-8"), content_type="text/csv")
    gcs_write_text(BUCKET, diag_blob, json.dumps(diag, indent=2), content_type="application/json")

    hashes_now = df["hash_id"].dropna().unique().tolist() if "hash_id" in df.columns else []
    gcs_write_text(
        BUCKET,
        latest_hash_blob,
        json.dumps({"updated": run_ts, "hashes": hashes_now}),
        content_type="application/json",
    )

    result_state = {
        "run_id": run_id,
        "status": "done",
        "ts": run_ts,
        "rows": int(len(df)),
        "recurring": int(df["hash_recurrence"].sum()) if "hash_recurrence" in df.columns else 0,
        "gcs_csv": f"gs://{BUCKET}/{out_blob}",
        "gcs_diag": f"gs://{BUCKET}/{diag_blob}",
        "gcs_ping": f"gs://{BUCKET}/{ping_blob}",
    }
    gcs_write_text(BUCKET, state_blob, json.dumps(result_state, indent=2), content_type="application/json")
    return result_state

@app.get("/run", include_in_schema=False)
async def run_http():
    if not BUCKET:
        raise HTTPException(status_code=500, detail="Missing BUCKET_NAME env var")

    day = utc_day()
    run_ts = utc_ts_compact()
    run_id = f"{day}_{run_ts}"

    log.info(f"run start run_id={run_id} day={day} bucket={BUCKET} prefix={PREFIX}")

    def _thread_runner():
        try:
            asyncio.run(run_once_to_gcs(run_id=run_id, day=day, run_ts=run_ts))
            log.info(f"run done run_id={run_id}")
        except Exception as e:
            err_blob = f"{PREFIX}/daily/{day}/error_{run_ts}.json"
            state_blob = f"{PREFIX}/state/last_run.json"
            gcs_write_text(
                BUCKET,
                err_blob,
                json.dumps({"run_id": run_id, "error": str(e), "ts": run_ts}, indent=2),
                content_type="application/json",
            )
            gcs_write_text(
                BUCKET,
                state_blob,
                json.dumps({"run_id": run_id, "status": "error", "error": str(e), "ts": run_ts}, indent=2),
                content_type="application/json",
            )
            log.exception(f"run failed run_id={run_id}: {e}")

    threading.Thread(target=_thread_runner, daemon=True).start()
    return {"ok": True, "run_id": run_id}

def main():
    if "--job" not in sys.argv:
        return

    if not BUCKET:
        print("ERROR: Missing BUCKET_NAME env var", file=sys.stderr)
        sys.exit(2)

    day = utc_day()
    run_ts = utc_ts_compact()
    run_id = f"{day}_{run_ts}"

    log.info(f"job start run_id={run_id} day={day} bucket={BUCKET} prefix={PREFIX}")
    try:
        result = asyncio.run(run_once_to_gcs(run_id=run_id, day=day, run_ts=run_ts))
        log.info(f"job done run_id={run_id} rows={result.get('rows')} gcs_csv={result.get('gcs_csv')}")
        sys.exit(0)
    except Exception as e:
        log.exception(f"job failed run_id={run_id}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()