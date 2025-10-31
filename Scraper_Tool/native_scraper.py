# native_scraper.py

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import hashlib
import pandas as pd
from datetime import date
import time
import yaml

def load_sources(config_path="sources.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config["sources"]

def hash_creative(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def is_external_ad(url, source):
    if not url.startswith("http"):
        return False
    if source in url:
        return False
    if "nbcnews.com" in url or "msn.com" in url:
        return False
    return True

import re
from urllib.parse import urlparse

def clean_text(text):
    text = text.replace("Learn More", "")
    text = text.replace("SPONSORED", "")
    text = re.sub(r"(?i)search ads ?", "", text)
    text = text.replace("|", "")
    return text.strip()

def detect_ad_network(block):
    """
    Detect whether the ad block belongs to Taboola, Outbrain, or unknown.
    Checks id, class, data attributes, and raw HTML.
    """
    # Pull ID and class attributes
    id_attr = block.get("id", "")
    class_attr = block.get("class", [])
    if isinstance(class_attr, list):
        class_attr = " ".join(class_attr)

    # Combine key attrs + HTML for string search
    combined_attrs = f"{id_attr} {class_attr}".lower()
    block_html = str(block).lower()

    # Search for known Taboola indicators
    if (
        "taboola" in combined_attrs
        or "taboola.com" in block_html
        or "data-taboola" in block_html
    ):
        return "Taboola"

    # Search for known Outbrain indicators
    if (
        "outbrain" in combined_attrs
        or "ob-widget" in combined_attrs
        or "outbrain.com" in block_html
        or "data-ob" in block_html
    ):
        return "Outbrain"

    return "Unknown"

def extract_native_ads(soup, source):
    rows = []
    today = date.today().isoformat()

    # Narrow selectors to real ad blocks (avoid editorial content)
    ad_blocks = soup.select(
        ', '.join([
            '[id*="taboola"]',
            '[class*="taboola"]',
            '[id*="outbrain"]',
            '[class*="outbrain"]',
            '[data-ob-widget]',
            '[data-type*="sponsored"]',
            'div[data-ads*="true"]',
            'div:has(a[href*="yahoo.com/yhs/r"])',
            'div:has(a[href*="taboola.com"])',
            'div:has(a[href*="outbrain.com"])',
            'div:has(a[href*="startsearch"]):not([role="navigation"])',
            'div:has(a[href*="trendinganswers"])',
            'div:has(a[href*="frequentsearches"])',
            'div:has(a[href*="exploreanswers"])',
            'div:has(a[href*="advisorbooth"])',
            'div:has(a[href*="wellnessgaze"])',
            'div:has(a[href*="searchlynk"])',
            'div:has(a[href*="popularsearches"])',
            'div:has(a[href*="top5-search.com"])',
        ])
    )

    print(f"[ℹ] Found {len(ad_blocks)} potential ad blocks on {source}")
    seen = set()

    for i, block in enumerate(ad_blocks):
        links = block.select("a[href]")
        for link in links:
            href = link.get("href", "").strip()
            if not href.startswith("http") or not is_external_ad(href, source):
                continue

            text = link.get_text(separator=" ", strip=True)
            if not text or len(text) < 25 or len(text) > 200:
                continue

            # Avoid duplicates
            key = text + href
            if key in seen:
                continue
            seen.add(key)

            brand = ""
            headline = ""
            desc = ""

            # --- CLEANUP ---
            text_cleaned = (
                text.replace("SPONSORED", "")
                    .replace("Sponsored", "")
                    .replace("Ad", "")
                    .replace("ads", "")
                    .replace("Search s /", "")
                    .replace("Search ads /", "")
                    .replace("Search ads", "")
                    .strip(" -:|")
                    .strip()
            )

            # --- SPLITTING LOGIC ---
            if " / " in text_cleaned or " | " in text_cleaned:
                sep = " / " if " / " in text_cleaned else " | "
                parts = [p.strip() for p in text_cleaned.split(sep)]
                if len(parts) == 2:
                    brand, headline = parts
                elif len(parts) > 2:
                    brand = parts[0]
                    headline = parts[1]
                    desc = " ".join(parts[2:])
            elif " - " in text_cleaned:
                split_parts = text_cleaned.split(" - ", 1)
                headline = split_parts[0].strip()
                desc = split_parts[1].strip()
            else:
                headline = text_cleaned

            # 🧠 Punctuation-based fallback
            if not desc:
                punct_split = re.match(r"^(.*?[.!?])\s+(.*)", headline)
                if punct_split:
                    headline = punct_split.group(1).strip()
                    desc = punct_split.group(2).strip()

            network = detect_ad_network(block)

            # --- CTA detection ---
            cta_text = ""
            for word in ["See", "Discover", "Search", "Take", "Learn", "Explore", "Read"]:
                if re.search(rf"\b{word}\b", text_cleaned, re.IGNORECASE):
                    cta_text = word
                    break

            # --- Image ---
            image_url = ""
            img = link.find("img")
            if img and img.has_attr("src"):
                image_url = img["src"]

            # --- Hash ID ---
            hash_id = hash_creative(headline + image_url)

            # --- Editorial content filter ---
            if any(bad in headline.lower() for bad in [
                "nfl", "nba", "nbc news", "trump", "peace plan", "video file", "cnn",
                "breaking", "interview", "debate", "update", "exclusive", "report"
            ]):
                continue

            # --- STRONGER FILTER FOR EDITORIAL OUTBRAIN LINKS ---
            is_mostly_editorial = (
                not cta_text  # No CTA word found
                and source in href  # Destination is the same domain (e.g. foxnews.com)
                and len(desc) == 0  # No ad-style description
            )
            if is_mostly_editorial:
                continue

            rows.append({
                "keyword": "",
                "source": source,
                "type": "ad",
                "position": f"native_ad_{i+1}",
                "brand": brand,
                "headline": headline.strip(),
                "ad_description": desc.strip(),
                "cta_text": cta_text,
                "image_url": image_url,
                "dest_url": href,
                "hash_id": hash_id,
                "cluster": "",
                "date_seen": today,
                "network": network
            })

    return rows

def scan_native_source(url, source_name="msn.com"):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        print(f"[🔍] Scanning: {url} ({source_name})")
        page.goto(url, timeout=50000)

        # Scroll slowly to trigger lazy-load
        for _ in range(10):
            page.mouse.wheel(0, 1000)
            time.sleep(1.5)

        # Extra wait for Taboola JS to inject content
        time.sleep(4)

        html = page.content()

        soup = BeautifulSoup(html, "html.parser")
        
        # Save snapshot for debug
     #  with open(f"page_{source_name}.html", "w", encoding="utf-8") as f:
     #      f.write(html)
        
        results = extract_native_ads(soup, source_name)
        browser.close()
        return results

def write_csv(data, filename="native_creatives.csv"):
    df = pd.DataFrame(data, columns=[
    "source", "network", "type", "position",
    "brand", "headline", "ad_description", "cta_text",
    "image_url", "dest_url", "hash_id",
    "cluster", "date_seen"
    ])

    df.to_csv(filename, index=False)
    print(f"[✔] Wrote {len(data)} native creatives to {filename}")