# ### ✅ FINAL — scraper_app.py (cleaned & fixed)

import streamlit as st
import os
import hashlib
import pandas as pd
import re
import yaml
from bs4 import BeautifulSoup
from datetime import date, datetime
import asyncio
from playwright.async_api import async_playwright
import requests
import altair as alt
from urllib.parse import urlparse, urljoin, quote_plus

# --------------------------------------------------
# Host helpers (single, de-duped versions)
# --------------------------------------------------
def _host(u: str) -> str:
    try:
        return urlparse(u).netloc.lower().lstrip("www.")
    except Exception:
        return ""

def same_host(u1: str, u2: str) -> bool:
    h1, h2 = _host(u1), _host(u2)
    return bool(h1) and bool(h2) and (h1.endswith(h2) or h2.endswith(h1))

# tracker/portal domains we don't want as FINAL destinations
_TRACKER_HOSTS = {
    "adclick.g.doubleclick.net",
    "googleadservices.com",
    "googlesyndication.com",
}

def is_external_ad(url: str, src_url: str) -> bool:
    """True if url is absolute http(s), not same-site as src_url."""
    if not url or not url.startswith(("http://", "https://")):
        return False
    if same_host(url, src_url):
        return False
    return True

def should_resolve(url: str) -> bool:
    """Decide if we should follow redirects for this url to improve landing clarity."""
    h = _host(url)
    if not h:
        return False
    if h in _TRACKER_HOSTS or "yhs/r" in url:
        return True
    return False

# --------------------------------------------------
# Global Constants
# --------------------------------------------------
EXPORTS_DIR = os.path.join(os.getcwd(), "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

# Tuning knobs for reliability/speed
RESOLVE_DEST_URL = True           # follow redirects selectively (see should_resolve)
GOTO_TIMEOUT_MS = 35000           # page.goto timeout per site
SCROLL_STEPS = 8                  # how many scrolls to trigger lazy-load
SCROLL_SLEEP_S = 1.2              # delay between scrolls
POST_INJECT_WAIT_S = 2.5          # wait after scrolling for Taboola/Outbrain to inject
SITE_DEADLINE_S = 60              # hard per-site watchdog (skip site if it exceeds this)
MAX_CONTAINERS = 16               # cap how many native containers per site
MAX_LINKS_PER_CONTAINER = 40      # cap links per container

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")

def make_output_path(prefix):
    filename = f"{prefix}_creatives_{timestamp()}.csv"
    return os.path.join(EXPORTS_DIR, filename)

def hash_creative(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def resolve_final_url(url, use_playwright=False):
    try:
        # first a quick HEAD chain
        response = requests.head(url, allow_redirects=True, timeout=6)
        return response.url
    except Exception:
        use_playwright = True

    if use_playwright:
        try:
            return asyncio.run(fetch_with_playwright(url))
        except Exception:
            return url
    return url

async def fetch_with_playwright(url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto(url, timeout=15000)
        final_url = page.url
        await browser.close()
        return final_url

async def async_resolve_final_url(url):
    try:
        return await fetch_with_playwright(url)
    except:
        return url
    
def unwrap_yahoo_redirect(url: str) -> str:
    """Best-effort to extract final destination from Yahoo redirect URLs."""
    try:
        from urllib.parse import urlparse, parse_qs, unquote

        pu = urlparse(url)
        host = pu.netloc.lower()

        # r.search.yahoo.com pattern often encodes final URL in the path as /RU=<enc>/...
        if "r.search.yahoo.com" in host:
            # Try path-based RU=
            m = re.search(r"/RU=([^/]+)", pu.path)
            if m:
                return unquote(m.group(1))
            # Fallback: sometimes query has u= or url=
            q = parse_qs(pu.query)
            for key in ("u", "url", "RU"):
                if key in q and q[key]:
                    return unquote(q[key][0])

        # search.yahoo.com/yhs/r?url=<enc> or …&u=<enc>
        if "search.yahoo.com" in host and "/yhs/r" in pu.path:
            q = parse_qs(pu.query)
            for key in ("url", "u", "target", "dest"):
                if key in q and q[key]:
                    return unquote(q[key][0])

        return url
    except Exception:
        return url    

def classify_cluster(text):
    text = text.lower()
    if any(x in text for x in ["cd rate", "savings", "bank", "certificate", "finance", "mortgage", "ira"]):
        return "Finance"
    if any(x in text for x in ["vacation", "trip", "resort", "flights", "cruise"]):
        return "Travel"
    if any(x in text for x in ["arthritis", "eczema", "medication", "symptom", "joint pain", "health", "dental", "implants"]):
        return "Health"
    if any(x in text for x in ["lawsuit", "accident", "lawyer", "mesothelioma", "legal"]):
        return "Legal"
    if any(x in text for x in ["tech", "ai", "cloud", "software", "iphone", "nvidia", "saas"]):
        return "Tech"
    if any(x in text for x in ["car", "truck", "suv", "vehicle", "ev", "toyota", "gmc"]):
        return "Automotive"
    if any(x in text for x in ["shower", "bath", "roof", "contractor", "kitchen"]):
        return "Home"
    return "Other"

def assign_cluster(row):
    return classify_cluster(f"{row['headline']} {row['ad_description']} {row.get('dest_url', '')}")

def load_prior_hashes(scan_type, current_file):
    prior_csvs = [
        f for f in os.listdir(EXPORTS_DIR)
        if f.endswith(".csv") and scan_type in f and "_creatives_" in f
        and os.path.join(EXPORTS_DIR, f) != current_file
    ]
    hashes = set()
    for f in prior_csvs:
        try:
            df = pd.read_csv(os.path.join(EXPORTS_DIR, f), usecols=["hash_id"])
            hashes.update(df["hash_id"].dropna().unique())
        except:
            continue
    return hashes

def summarize_column(files, column_name, keyword):
    dfs = []
    for f in files:
        if keyword in f:
            try:
                df = pd.read_csv(os.path.join(EXPORTS_DIR, f), usecols=[column_name])
                dfs.append(df)
            except:
                continue
    if not dfs:
        return pd.DataFrame(columns=[column_name.capitalize(), "Count"])
    df_all = pd.concat(dfs)
    summary = df_all[column_name].value_counts().reset_index()
    summary.columns = [column_name.capitalize(), "Count"]
    return summary

def recurrence_table(files, column_name, keyword):
    dfs = []
    for f in files:
        if keyword in f:
            try:
                df = pd.read_csv(os.path.join(EXPORTS_DIR, f), usecols=[column_name, "date_seen"])
                dfs.append(df)
            except:
                continue
    if not dfs:
        return pd.DataFrame()
    df_all = pd.concat(dfs)
    df_all["date_seen"] = pd.to_datetime(df_all["date_seen"], errors="coerce").dt.date
    grouped = df_all.groupby([column_name, "date_seen"]).size().reset_index(name="count")
    pivot = grouped.pivot_table(index=column_name, columns="date_seen", values="count", fill_value=0)
    pivot["Total"] = pivot.sum(axis=1)
    return pivot.sort_values("Total", ascending=False)

# --------------------------------------------------
# Native Ad Scan Logic
# --------------------------------------------------
def detect_ad_network_from_html(html_low: str) -> str:
    if "taboola" in html_low or "data-taboola" in html_low:
        return "Taboola"
    if "outbrain" in html_low or "ob-widget" in html_low or "data-ob" in html_low:
        return "Outbrain"
    return "Unknown"

def _clean_text(raw: str) -> str:
    # normalize all whitespace to single spaces; strip ad boilerplate
    txt = re.sub(r"\s+", " ", (raw or "")).strip()
    txt = re.sub(r"(?i)\b(SPONSORED|Sponsored|Ad|Ads|Search ads ?/?|Learn More)\b", "", txt)
    txt = txt.replace("| |", "|").strip(" -:|").strip()
    return txt

def _split_brand_headline_desc(text_cleaned: str):
    """Legacy-style, robust splitter with guardrails so a long headline doesn't become 'brand'."""
    brand, headline, desc = "", "", ""

    # prefer separators if they look brandlike on the left
    if " | " in text_cleaned or " / " in text_cleaned:
        sep = " | " if " | " in text_cleaned else " / "
        parts = [p.strip() for p in text_cleaned.split(sep) if p.strip()]
        if len(parts) >= 2:
            left, right = parts[0], parts[1]
            # left is brand only if reasonably short (≤ 28 chars, ≤ 4 words)
            if len(left) <= 28 and len(left.split()) <= 4:
                brand, headline = left, right
                if len(parts) > 2:
                    desc = " ".join(parts[2:])
            else:
                # left is actually headline; keep rest as desc
                headline = left
                if len(parts) > 1:
                    desc = " ".join(parts[1:])
    elif " - " in text_cleaned:
        p = text_cleaned.split(" - ", 1)
        if len(p) == 2:
            headline, desc = p[0].strip(), p[1].strip()
        else:
            headline = text_cleaned
    else:
        headline = text_cleaned

    # punctuation-based fallback to get desc if missing
    if not desc:
        m = re.match(r"^(.*?[.!?])\s+(.*)$", headline)
        if m:
            headline, desc = m.group(1).strip(), m.group(2).strip()

    return brand.strip(), headline.strip(), desc.strip()

async def async_run_native_scan(output_file):
    DATE_SEEN = date.today().isoformat()

    with open("sources.yaml", "r") as f:
        sources = yaml.safe_load(f).get("sources", [])

    all_rows = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        # Block heavy assets but allow scripts so Taboola/Outbrain can inject
        async def _block(route):
            r = route.request
            if r.resource_type in {"image", "stylesheet", "font"}:
                await route.abort()
            else:
                await route.continue_()

        await context.route("**/*", _block)

        async def _scan_one_site(src):
            url = src["url"]
            name = src.get("name", _host(url)) or _host(url)
            st.write(f"🌐 Scanning: {url}")

            page = await context.new_page()
            try:
                await page.goto(url, timeout=GOTO_TIMEOUT_MS)

                # trigger lazy-load
                for _ in range(SCROLL_STEPS):
                    await page.mouse.wheel(0, 1000)
                    await asyncio.sleep(SCROLL_SLEEP_S)
                await asyncio.sleep(POST_INJECT_WAIT_S)

                selectors = [
                    '[id*="taboola"]', '[class*="taboola"]', '[id*="outbrain"]', '[class*="outbrain"]',
                    '[data-ob-widget]', '[data-type*="sponsored"]', 'div[data-ads*="true"]',
                    'div:has(a[href*="taboola.com"])', 'div:has(a[href*="outbrain.com"])',
                    '[data-taboola]', '[data-ob-template]', 'div[id^="taboola-"]', 'div[class^="OUTBRAIN"]',
                ]
                container_css = ", ".join(selectors)

                # find containers in main doc and iframes
                containers = await page.locator(container_css).all()
                for fr in page.frames:
                    if fr != page.main_frame:
                        containers += await fr.locator(container_css).all()

                st.write(f"🔎 {name}: found {len(containers)} potential native ad containers")
                containers = containers[:MAX_CONTAINERS]

                seen = set()
                kept = 0

                for el in containers:
                    # identify probable network
                    try:
                        html_chunk = await el.evaluate("(n) => n.outerHTML")
                    except Exception:
                        html_chunk = ""
                    network = detect_ad_network_from_html((html_chunk or "").lower())

                    # links inside this container
                    link_loc = el.locator("a[href]")
                    link_count = min(await link_loc.count(), MAX_LINKS_PER_CONTAINER)

                    for idx in range(link_count):
                        link = link_loc.nth(idx)
                        href = (await link.get_attribute("href")) or ""
                        if not is_external_ad(href, url):
                            continue

                        # text content — normalize whitespace early
                        try:
                            raw_text = await link.inner_text()
                        except Exception:
                            raw_text = await link.evaluate("(n) => (n.textContent || '')")
                        text = _clean_text(raw_text)

                        # guard rails on length
                        if not text or len(text) < 25 or len(text) > 220:
                            continue

                        # de-dupe
                        key = f"{text}||{href}"
                        if key in seen:
                            continue
                        seen.add(key)

                        # image (best-effort; we don't block functionality if missing)
                        image_url = ""
                        img = link.locator("img").first
                        try:
                            if await img.count():
                                image_url = (await img.get_attribute("src")) or ""
                        except Exception:
                            image_url = ""

                        # split into brand/headline/desc (legacy-like)
                        brand, headline, desc = _split_brand_headline_desc(text)

                        # CTA detection
                        cta_text = ""
                        for w in ("See", "Discover", "Search", "Take", "Learn", "Explore", "Read"):
                            if re.search(rf"\b{w}\b", text, re.IGNORECASE):
                                cta_text = w
                                break

                        # editorial filters (similar to legacy)
                        headline_low = headline.lower()
                        if any(bad in headline_low for bad in [
                            "video file", "breaking", "exclusive", "update", "interview", "report"
                        ]):
                            continue
                        # if same-site AND no CTA AND no desc, likely editorial promo
                        if same_host(href, url) and not cta_text and not desc:
                            continue

                        # final sanity
                        headline = re.sub(r"\s+", " ", headline).strip()
                        desc = re.sub(r"\s+", " ", (desc or "")).strip()
                        if not headline or len(headline.split()) < 3:
                            continue
                        if headline.lower() in {"ad", "sponsored", "ad ad"}:
                            continue

                        # follow redirects only if it looks like a tracker
                        dest = href
                        if RESOLVE_DEST_URL and should_resolve(href):
                            dest = await async_resolve_final_url(href)
                            if _host(dest) in _TRACKER_HOSTS:
                                # still a tracker; keep original
                                dest = href

                        row = {
                            "source": name,
                            "type": "ad",
                            "position": f"native_ad_{len(all_rows)+1}",
                            "brand": brand or name,
                            "headline": headline,
                            "ad_description": desc,
                            "image_url": image_url,
                            "original_url": href,
                            "dest_url": dest,
                            "hash_id": hash_creative(headline + image_url),
                            "date_seen": DATE_SEEN,
                            "network": network,
                        }
                        all_rows.append(row)
                        kept += 1

                st.write(f"✅ {name}: kept {kept} creatives after filtering")
            finally:
                await page.close()

        # run each site with a per-site watchdog
        for src in sources:
            url = src["url"]
            try:
                await asyncio.wait_for(_scan_one_site(src), timeout=SITE_DEADLINE_S)
            except asyncio.TimeoutError:
                st.warning(f"[⏱️ Native] Timed out scanning {url} after {SITE_DEADLINE_S}s — skipping.")
            except Exception as e:
                st.warning(f"[❌ Native] Error scanning {url}: {e}")

        await context.close()
        await browser.close()

    # write results (or an empty CSV with schema) after Playwright is closed
    if not all_rows:
        st.warning("⚠️ No native ads detected in this scan.")
        cols = [
            "source","type","position","brand","headline","ad_description",
            "image_url","original_url","dest_url","hash_id","date_seen","network",
            "cluster","hash_recurrence"
        ]
        pd.DataFrame(columns=cols).to_csv(output_file, index=False)
        return

    df = pd.DataFrame(all_rows)
    df["cluster"] = df.apply(assign_cluster, axis=1)
    df["hash_recurrence"] = df["hash_id"].apply(
        lambda h: 1 if h in load_prior_hashes("native", output_file) else 0
    )
    df.to_csv(output_file, index=False)
    st.success(f"✅ Saved native results: {output_file}")

def run_yahoo_scan(output_file, seeds=None):
    """
    Yahoo scan (search.yahoo.com only).
    - Optional `seeds` list of queries.
    - If no seeds provided or blank, uses a robust default set.
    - Extracts only Yahoo Search redirect links (r.search.yahoo.com / yhs/r).
    - Always writes a CSV (even if empty).
    """
    from urllib.parse import urlparse, urljoin, urlencode

    DATE_SEEN = date.today().isoformat()
    seeds = [s.strip() for s in (seeds or []) if s and s.strip()]
    if not seeds:
        # Strong defaults that consistently return results/ads
        seeds = [
            "cd rates", "mortgage rates", "best credit cards", "car insurance quotes",
            "injury lawyer", "mesothelioma lawyer", "dental implants cost",
            "home solar quotes", "replacement windows", "cheap flights",
            "personal loans", "online mba", "hearing aids", "cloud software"
        ]

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    def fetch(url, params=None):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15, allow_redirects=True)
            if r.status_code == 200 and r.text.strip():
                return r.text, r.url
            else:
                q = f"?{urlencode(params)}" if params else ""
                st.warning(f"[Yahoo] Non-200 or empty response for {url}{q} (status={r.status_code})")
                return None, url
        except Exception as e:
            q = f"?{urlencode(params)}" if params else ""
            st.warning(f"[Yahoo] Request failed for {url}{q} — {e}")
            return None, url

    def _resolve_yahoo_redirect(href: str) -> str:
        """Follow Yahoo Search redirectors to the true destination."""
        try:
            r = requests.get(href, headers=HEADERS, allow_redirects=True, timeout=8)
            return r.url or href
        except Exception:
            return href

    def _brand_from(url: str) -> str:
        try:
            host = urlparse(url).netloc.lower()
            return host.lstrip("www.")
        except Exception:
            return "unknown"

    # Only search.yahoo.com
    start_pages = []
    for q in seeds:
        start_pages.append(("https://search.yahoo.com/search", {"p": q}))
        start_pages.append(("https://search.yahoo.com/search", {"p": q, "fr2": "news"}))

    data = []
    seen = set()

    for base, params in start_pages:
        html, base_url = fetch(base, params=params)
        if not html:
            continue

        soup = BeautifulSoup(html, "html.parser")

        # Yahoo redirectors only
        anchors = []
        anchors.extend(soup.select('a[href^="https://r.search.yahoo.com/"]'))
        anchors.extend(soup.select('a[href^="https://search.yahoo.com/yhs/r"]'))

        for a in anchors:
            href = (a.get("href") or "").strip()
            if not href:
                continue
            href = urljoin(base_url, href)

            text = a.get_text(" ", strip=True)
            if not text or len(text) < 18 or len(text) > 220:
                continue

            key = (text, href)
            if key in seen:
                continue
            seen.add(key)

            final_url = _resolve_yahoo_redirect(href)
            brand = _brand_from(final_url)

            row = {
                "source": "yahoo",
                "type": "ad",
                "position": f"yahoo_ad_{len(data)+1}",
                "brand": brand,
                "headline": text,
                "ad_description": "",
                "cta_text": "Read",
                "image_url": "",
                "dest_url": final_url,
                "original_url": href,
                "hash_id": hash_creative(text + href),
                "date_seen": DATE_SEEN,
                "network": "Yahoo Search",
            }
            data.append(row)

    # Always write something
    if not data:
        st.warning("⚠️ No Yahoo ads found on this scan.")
        cols = [
            "source","type","position","brand","headline","ad_description","cta_text",
            "image_url","dest_url","original_url","hash_id","date_seen","network",
            "cluster","hash_recurrence"
        ]
        pd.DataFrame(columns=cols).to_csv(output_file, index=False)
        return

    df = pd.DataFrame(data)
    df["cluster"] = df.apply(assign_cluster, axis=1)
    prior_hashes = load_prior_hashes("yahoo", output_file)
    df["hash_recurrence"] = df["hash_id"].apply(lambda h: 1 if h in prior_hashes else 0)
    df.to_csv(output_file, index=False)
    
# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="Scraper App", layout="centered")
st.title("📡 Ad Creative Scraper App")
st.write("Choose which scan to run:")
yahoo_seeds_input = st.text_input("Yahoo seed keywords (comma-separated)", value="")

run_native = st.button("Run Native Scan")
run_yahoo = st.button("Run Yahoo Scan")
run_both = st.button("Run Both")

seeds = [s.strip() for s in yahoo_seeds_input.split(",") if s.strip()]

if run_native:
    with st.spinner("Running Native Scan..."):
        out_path = make_output_path("native")
        asyncio.run(async_run_native_scan(out_path))
        if os.path.exists(out_path):
            df = pd.read_csv(out_path)
            st.success("Native scan complete.")
            st.write("✅ Native Ad Results")
            st.dataframe(df[["headline", "ad_description", "original_url", "dest_url", "network"]].head(25))
            st.download_button("Download Native CSV", open(out_path, "rb"), file_name=os.path.basename(out_path))
        else:
            st.warning("⚠️ No results were saved. Check scan output or errors.")

if run_yahoo:
    with st.spinner("Running Yahoo Scan..."):
        out_path = make_output_path("yahoo")
        run_yahoo_scan(out_path, seeds=seeds)
        if os.path.exists(out_path):
            df = pd.read_csv(out_path)
            st.success("Yahoo scan complete.")
            st.write("✅ Yahoo Ad Results")
            st.dataframe(df[["headline", "ad_description", "original_url", "dest_url"]].head(25))
            st.download_button("Download Yahoo CSV", open(out_path, "rb"), file_name=os.path.basename(out_path))
        else:
            st.warning("⚠️ No Yahoo results were saved.")

if run_both:
    with st.spinner("Running Both Scans..."):
        n_path = make_output_path("native")
        y_path = make_output_path("yahoo")

        asyncio.run(async_run_native_scan(n_path))
        run_yahoo_scan(y_path, seeds=seeds)

        st.success("Both scans complete.")

        # Preview Native
        df_n = pd.read_csv(n_path)
        st.write("✅ Native Ad Results")
        st.dataframe(df_n[["headline", "ad_description", "original_url", "dest_url"]].head(25))
        st.download_button("Download Native CSV", open(n_path, "rb"), file_name=os.path.basename(n_path))

        # Preview Yahoo
        df_y = pd.read_csv(y_path)
        st.write("✅ Yahoo Ad Results")
        st.dataframe(df_y[["headline", "ad_description", "original_url", "dest_url"]].head(25))
        st.download_button("Download Yahoo CSV", open(y_path, "rb"), file_name=os.path.basename(y_path))

# --------------------------------------------------
# Browse Past Yahoo Scan Files
# --------------------------------------------------
csv_files = sorted(
    [f for f in os.listdir(EXPORTS_DIR) if f.endswith(".csv") and "_creatives_" in f],
    reverse=True
)

st.markdown("### 🗂 Browse Past Yahoo Scans")

yahoo_files = [f for f in csv_files if "yahoo" in f]

if yahoo_files:
    selected_yahoo_file = st.selectbox("Select a Yahoo scan to view:", sorted(yahoo_files, reverse=True))
    df_yahoo_selected = pd.read_csv(os.path.join(EXPORTS_DIR, selected_yahoo_file))
    st.write(f"✅ Showing: `{selected_yahoo_file}` ({len(df_yahoo_selected)} rows)")
    st.dataframe(df_yahoo_selected.head(25))
else:
    st.info("No Yahoo scan CSVs found in exports/")

# --------------------------------------------------
# Trend Display Helpers
# --------------------------------------------------
def display_trends(source_label, csv_files):
    headlines = summarize_column(csv_files, "headline", source_label)
    descs = summarize_column(csv_files, "ad_description", source_label)
    emoji = "📡" if source_label == "native" else "📰"
    with st.expander(f"{emoji} {source_label.capitalize()} Trends"):
        st.subheader("Top Headlines")
        st.dataframe(headlines)
        st.subheader("Top Ad Descriptions")
        st.dataframe(descs)

# --------------------------------------------------
# Headline & Description Trend Summary Dashboard
# --------------------------------------------------
st.markdown("### 📈 Headline & Description Trends")
display_trends("native", csv_files)
display_trends("yahoo", csv_files)

# --- Yahoo clusters (safe) ---
cluster_summary = pd.DataFrame()
with st.expander("🧠 Yahoo Clusters"):
    try:
        cluster_summary = summarize_column(csv_files, "cluster", "yahoo")
        st.subheader("Top Yahoo Clusters")
        st.dataframe(cluster_summary)
    except Exception:
        st.info("No cluster data available yet.")

if not cluster_summary.empty:
    bar_chart = alt.Chart(cluster_summary).mark_bar().encode(
        x=alt.X("Cluster:N", sort="-y"),
        y="Count:Q",
        tooltip=["Cluster:N", "Count:Q"]
    ).properties(width="container", height=300)
    st.altair_chart(bar_chart, use_container_width=True)

# --------------------------------------------------
# Recurrence Display Helper
# --------------------------------------------------
def display_recurrence(source_label, csv_files):
    emoji = "📡" if source_label == "native" else "📰"
    with st.expander(f"{emoji} {source_label.capitalize()} Headline Recurrence Over Time"):
        st.dataframe(recurrence_table(csv_files, "headline", source_label))
    with st.expander(f"{emoji} {source_label.capitalize()} Description Recurrence Over Time"):
        st.dataframe(recurrence_table(csv_files, "ad_description", source_label))

# --------------------------------------------------
# Recurrence Over Time Dashboard
# --------------------------------------------------
st.markdown("### 📆 Recurrence Over Time")