# yahoo_scraper.py

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import hashlib
import pandas as pd
from datetime import date
import time
import re

def hash_creative(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def extract_creatives(soup, keyword):
    rows = []
    today = date.today().isoformat()

    # === Extract Organic Results ===
    organic_results = soup.select("ol.searchCenterMiddle li div.dd.algo")
    for i, result in enumerate(organic_results[:6]):
        headline = result.select_one("h3").text.strip() if result.select_one("h3") else ""
        snippet = result.select_one("p").text.strip() if result.select_one("p") else ""
        link = result.select_one("a")["href"] if result.select_one("a") else ""
        hash_id = hash_creative(headline + snippet)

        rows.append({
            "keyword": keyword,
            "source": "yahoo.com",
            "type": "organic",
            "position": f"organic_{i+1}",
            "headline": headline,
            "body": snippet,
            "cta_text": "",
            "image_url": "",
            "dest_url": link,
            "hash_id": hash_id,
            "cluster": "",
            "date_seen": today
        })

    # === Extract Top Ad Blocks ===
    ad_blocks = soup.select("div.compText.ad")
    for i, ad in enumerate(ad_blocks):
        headline = ad.select_one("h3").text.strip() if ad.select_one("h3") else ""
        snippet = ad.select_one("p").text.strip() if ad.select_one("p") else ""
        link = ad.select_one("a")["href"] if ad.select_one("a") else ""

        # === Broad CTA Phrase Detection ===
        cta_phrases = {
            "search": "Search Now",
            "learn more": "Learn More",
            "see": "See Options",
            "take a look": "Take A Look",
            "check": "Check It Out",
            "compare": "Compare Deals",
            "get a quote": "Get A Quote",
            "find": "Find Offers",
            "view": "View Rates",
            "book": "Book Now",
            "apply": "Apply Now",
            "save": "Save Today",
            "grab": "Grab This Deal"
        }

        text_block = f"{headline} {snippet}".lower()
        for phrase, label in cta_phrases.items():
            if phrase in snippet_lower:
                cta_text = label
                break

        hash_id = hash_creative(headline + snippet)

        rows.append({
            "keyword": keyword,
            "source": "yahoo.com",
            "type": "ad",
            "position": f"top_ads_{i+1}",
            "headline": headline,
            "body": snippet,
            "cta_text": cta_text,
            "image_url": "",
            "dest_url": link,
            "hash_id": hash_id,
            "cluster": "",
            "date_seen": today
        })

    return rows

def scan_yahoo_keywords(keywords):
    all_rows = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        for keyword in keywords:
            url = f"https://search.yahoo.com/search?p={keyword.replace(' ', '+')}"
            print(f"Scanning: {keyword}")
            page.goto(url, timeout=10000)
            time.sleep(2)
            soup = BeautifulSoup(page.content(), "html.parser")
            rows = extract_creatives(soup, keyword)
            all_rows.extend(rows)

        browser.close()
    return all_rows

def write_csv(data, path="yahoo_creatives.csv"):
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    print(f"[✔] Wrote {len(data)} creatives to {path}")