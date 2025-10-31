import os
import io
import gzip
import json
from pathlib import Path
from datetime import date, datetime
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv
import re

BASE_DIR = Path(__file__).resolve().parent

# load .env from MSADS_TEST only
load_dotenv(dotenv_path=BASE_DIR / ".env", override=False)

SCOPE = "openid offline_access https://ads.microsoft.com/msads.manage"
if "https://ads.microsoft.com/msads.manage" not in SCOPE:
    raise RuntimeError(f"Invalid SCOPE: {SCOPE}")

NETWORK = "OwnedAndOperatedAndSyndicatedSearch"

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def make_session():
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "POST"),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "KeywordMiner/1.0"})
    return s


SESSION = make_session()

TOK_FILE = BASE_DIR / "msads_tokens.json"

# note the comma here
AUTH_TENANT = os.getenv("MSADS_AUTH_TENANT", "common").strip() or "common"
AUTH_URL = f"https://login.microsoftonline.com/{AUTH_TENANT}/oauth2/v2.0/authorize"
TOKEN_URL = f"https://login.microsoftonline.com/{AUTH_TENANT}/oauth2/v2.0/token"

# we want this exact redirect
REDIRECT_URI = os.getenv(
    "MSADS_REDIRECT_URI",
    "https://localhost:8000/auth/callback",
)


def _normalize_secret(val: str | None) -> str | None:
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    if s.lower() in {"none", "null", "nil", "replace_me"}:
        return None
    return s


# ids and secret loaded from env
CLIENT_ID = os.getenv("MSADS_CLIENT_ID") or os.getenv("CLIENT_ID") or ""
CLIENT_SECRET = os.getenv("MSADS_CLIENT_SECRET") or os.getenv("CLIENT_SECRET") or ""
CLIENT_SECRET = _normalize_secret(CLIENT_SECRET)

# Markets and locales used by the UI
MARKETS = [
    ("United States", "English", "EN"),
    ("United Kingdom", "English", "EN"),
    ("Germany", "German", "DE"),
    ("France", "French", "FR"),
    ("Italy", "Italian", "IT"),
    ("Spain", "Spanish", "ES"),
    ("Netherlands", "Dutch", "NL"),
    ("Austria", "German", "DE"),
    ("Denmark", "Danish", "DA"),
    ("Brazil", "Portuguese", "PT"),
    ("Australia", "English", "EN"),
]

DEFAULT_URLS_BY_MARKET = {
    "United States": "https://www.amazon.com/Best-Sellers/zgbs",
    "United Kingdom": "https://www.amazon.co.uk/gp/bestsellers/",
    "Germany": "https://www.amazon.de/gp/bestsellers/",
    "France": "https://www.amazon.fr/gp/bestsellers/",
    "Italy": "https://www.amazon.it/gp/bestsellers/",
    "Spain": "https://www.amazon.es/gp/bestsellers/",
    "Netherlands": "https://www.amazon.nl/gp/bestsellers/",
    "Austria": "https://www.amazon.de/gp/bestsellers/",
    "Denmark": "https://www.amazon.de/gp/bestsellers/",
    "Brazil": "https://www.amazon.com.br/gp/bestsellers/",
    "Australia": "https://www.amazon.com.au/gp/bestsellers/",
}

PUBLISHER_COUNTRIES = {
    "United States": ["US"],
    "United Kingdom": ["GB"],
    "Germany": ["DE"],
    "France": ["FR"],
    "Italy": ["IT"],
    "Spain": ["ES"],
    "Netherlands": ["NL"],
    "Austria": ["AT"],
    "Denmark": ["DK"],
    "Brazil": ["BR"],
    "Australia": ["AU"],
}

MARKET_LANGUAGE = {
    "United States": "English",
    "United Kingdom": "English",
    "Germany": "German",
    "France": "French",
    "Italy": "Italian",
    "Spain": "Spanish",
    "Netherlands": "Dutch",
    "Austria": "German",
    "Denmark": "Danish",
    "Brazil": "Portuguese",
    "Australia": "English",
}

# if token file exists we can pull client id and secret from it
if TOK_FILE.exists():
    try:
        tok_data = json.loads(TOK_FILE.read_text())
        if not CLIENT_ID:
            CLIENT_ID = tok_data.get("client_id", "")
        if not CLIENT_SECRET:
            CLIENT_SECRET = _normalize_secret(tok_data.get("client_secret"))
    except Exception:
        pass


def _st_info(msg: str) -> None:
    try:
        st.info(msg)
    except Exception:
        print(msg)


def refresh_tokens(client_id: str, client_secret: str, refresh_token: str) -> dict:
    if not client_id:
        raise RuntimeError("Missing CLIENT_ID for refresh")
    if not client_secret:
        raise RuntimeError("Missing CLIENT_SECRET for refresh")
    if not refresh_token:
        raise RuntimeError("Missing refresh_token for refresh")

    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
    }
    r = SESSION.post(TOKEN_URL, data=data, timeout=60)
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        _st_info(f"Token refresh failed {err}")
        r.raise_for_status()
    return r.json()


def ensure_access_token(client_id: str, client_secret: str):
    if not TOK_FILE.exists():
        st.error("msads_tokens.json not found. Run auth_once.py first.")
        st.stop()
    tokens = json.loads(TOK_FILE.read_text())
    rt = tokens.get("refresh_token", "")
    if not rt:
        st.error("msads_tokens.json is missing refresh_token. Run auth_once.py again.")
        st.stop()
    new_tokens = refresh_tokens(client_id, client_secret, rt)
    # persist client id and secret so next run can find them
    new_tokens["client_id"] = client_id
    new_tokens["client_secret"] = client_secret
    TOK_FILE.write_text(json.dumps(new_tokens, indent=2))
    return new_tokens["access_token"], new_tokens.get("refresh_token", "")


def get_ids_with_sdk(client_id: str, client_secret: str, developer_token: str, refresh_token: str):
    from bingads.authorization import OAuthWebAuthCodeGrant, AuthorizationData
    from bingads.service_client import ServiceClient

    oauth = OAuthWebAuthCodeGrant(
        client_id=client_id,
        client_secret=client_secret,
        redirection_uri=REDIRECT_URI,
        env="production",
    )
    oauth.request_oauth_tokens_by_refresh_token(refresh_token)

    auth = AuthorizationData(
        developer_token=developer_token,
        authentication=oauth,
    )

    cm = ServiceClient(
        service="CustomerManagementService",
        version=13,
        authorization_data=auth,
    )

    infos = cm.GetAccountsInfo(CustomerId=None).AccountInfo
    if not infos:
        st.error("No accessible accounts for this login.")
        st.stop()

    account_id = infos[0].Id
    acct = cm.GetAccount(AccountId=account_id)
    customer_id = acct.ParentCustomerId
    return account_id, customer_id

# ---------- Device KPIs ----------
def get_device_kpis_rest(access_token, developer_token, account_id, customer_id,
                         keywords, devices, language, publisher_countries,
                         time_interval="Last30Days", match_types=("Exact",)):
    """
    Returns {(keyword, device): {...}} via AdInsight v13 HistoricalKeywordPerformance.
    """
    if not keywords:
        return {}

    url = "https://adinsight.api.bingads.microsoft.com/AdInsight/v13/HistoricalKeywordPerformance/Query"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "DeveloperToken": developer_token,
        "CustomerAccountId": str(account_id),
        "CustomerId": str(customer_id),
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
    }

    api_interval = _map_kpi_interval(time_interval)

    out = {}
    for i in range(0, len(keywords), 200):
        chunk = [k for k in keywords[i:i+200] if k]
        if not chunk:
            continue

        payload = {
            "Keywords": chunk,
            "TimeInterval": api_interval,
            "TargetAdPosition": "All",
            "MatchTypes": list(match_types),
            "Language": language,
            "PublisherCountries": publisher_countries,
        }
        if devices:
            payload["Devices"] = list(devices)

        r = SESSION.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            try:
                st.warning(f"Device KPIs fetch failed: {r.status_code} {r.text[:400]}")
            except Exception:
                pass
            continue

        data = r.json() or {}
        for item in (data.get("KeywordHistoricalPerformances") or []):
            kw = item.get("Keyword")
            for kpi in (item.get("KeywordKPIs") or []):
                dev = kpi.get("Device")
                if kw and dev:
                    out[(kw, dev)] = {
                        "Clicks":      kpi.get("Clicks"),
                        "Impressions": kpi.get("Impressions"),
                        "CTR":         kpi.get("CTR"),
                        "AverageCPC":  kpi.get("AverageCPC"),
                        "AverageBid":  kpi.get("AverageBid"),
                        "TotalCost":   kpi.get("TotalCost"),
                    }
    return out

def _map_kpi_interval(val: str) -> str:
v = (val or “”).strip()
if v in (“LastWeek”,“LastMonth”,“LastDay”):
return v
if v in (“Last7Days”,):
return “LastWeek”
if v in (“Last30Days”,“Last3Months”):
return “LastMonth”
return “LastMonth”

# ---------- Geo file (v3.0) ----------
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_geos_csv_rest(access_token, developer_token, account_id, customer_id, language_locale="en"):
    """
    Calls CampaignManagement/v13/GeoLocationsFileUrl/Query (REST) to get a CSV URL.
    The file may be GZip or plain CSV; we try gzip and fall back to plain.
    """
    url = "https://campaign.api.bingads.microsoft.com/CampaignManagement/v13/GeoLocationsFileUrl/Query"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "DeveloperToken": developer_token,
        "CustomerAccountId": str(account_id),
        "CustomerId": str(customer_id),
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
    }
    payload = {
        "Version": "2.0",
        "LanguageLocale": language_locale,
        "CompressionType": "GZip"
    }

    r = SESSION.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"GeoFileUrl {r.status_code}: {r.text[:400]}")
    file_url = (r.json() or {}).get("FileUrl")
    if not file_url:
        raise RuntimeError("GeoFileUrl: response missing FileUrl")

    r2 = SESSION.get(file_url, timeout=120)
    r2.raise_for_status()
    content = r2.content

    is_gzip = content[:2] == b"\x1f\x8b"
    if is_gzip:
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
                csv_bytes = gz.read()
        except Exception:
            csv_bytes = content
    else:
        csv_bytes = content

    df = pd.read_csv(io.BytesIO(csv_bytes), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    return df

def resolve_country_id(df_geos, country_name):
    # Expect: Location Id, Bing Display Name, Location Type, Status
    need = {"Location Id","Bing Display Name","Location Type","Status"}
    if not need.issubset(set(df_geos.columns)):
        return None
    d = df_geos[(df_geos["Location Type"].str.lower()=="country") & (df_geos["Status"].str.lower()=="active")].copy()
    exact = d[d["Bing Display Name"].str.fullmatch(country_name, case=False, na=False)]
    if len(exact):
        return int(exact.iloc[0]["Location Id"])
    suffix = d[d["Bing Display Name"].str.endswith(country_name, na=False)]
    if len(suffix):
        return int(suffix.iloc[0]["Location Id"])
    return None

# ---------- Ideas & Bids ----------
def get_keyword_ideas(
    access_token,
    developer_token,
    account_id,
    customer_id,
    seeds,
    language_name,
    location_ids,
    devices=None,
    network="OwnedAndOperatedAndSyndicatedSearch",
):
    """Call AdInsight v13 KeywordIdeas/Query and return the raw 'KeywordIdeas' list."""
    url = "https://adinsight.api.bingads.microsoft.com/AdInsight/v13/KeywordIdeas/Query"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "DeveloperToken": developer_token,
        "CustomerAccountId": str(account_id),
        "CustomerId": str(customer_id),
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
    }

    payload = {
        "ExpandIdeas": True,
        "IdeaAttributes": ["Keyword", "Competition", "MonthlySearchCounts", "SuggestedBid"],
        "SearchParameters": [
            {"Type": "QuerySearchParameter", "Queries": seeds},
            {"Type": "LanguageSearchParameter", "Languages": [{"Language": language_name}]},
            {"Type": "LocationSearchParameter", "Locations": [{"LocationId": int(l)} for l in location_ids]},
            {"Type": "NetworkSearchParameter", "Network": {"Network": network}},
        ],
    }

    if devices and set(devices) != {"Computers", "Smartphones", "Tablets"}:
        payload["SearchParameters"].append({
            "Type": "DeviceSearchParameter",
            "Devices": list(devices),
        })

    r = SESSION.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"KeywordIdeas {r.status_code}: {r.text[:400]}")
    data = r.json() or {}
    return data.get("KeywordIdeas") or []

def get_estimated_bids(access_token, developer_token, account_id, customer_id, keywords, lang_code, location_id):
    url = "https://adinsight.api.bingads.microsoft.com/AdInsight/v13/EstimatedBid/QueryByKeywords"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "DeveloperToken": developer_token,
        "CustomerAccountId": str(account_id),
        "CustomerId": str(customer_id),
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8",
    }
    out = {}
    for i in range(0, len(keywords), 200):
        chunk = keywords[i:i+200]
        payload = {
            "Keywords": [{"KeywordText": k, "MatchTypes": ["Exact"]} for k in chunk],
            "TargetPositionForAds": "MainLine1",
            "Language": lang_code,
            "LocationIds": [int(location_id)],
            "EntityLevelBid": "Keyword",
        }
        r = SESSION.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code != 200:
            continue
        data = r.json() or {}
        for item in (data.get("KeywordEstimatedBids") or []):
            kw = item.get("Keyword")
            ests = item.get("EstimatedBids") or []
            pick = next((e for e in ests if (e.get("MatchType") or "").lower() == "exact"), ests[0] if ests else None)
            if pick:
                out[kw] = {
                    "first_page_min_bid": float(pick.get("EstimatedMinBid") or 0),
                    "avg_cpc_est": float(pick.get("AverageCPC") or 0),
                }
    return out

# ---------- Transform helpers ----------
def comp_to_float(c):
    if c is None:
        return 0.0
    if isinstance(c,(int,float)):
        return float(c)
    s = str(c).strip().lower()
    if s in ("low","verylow","lowest"):
        return 0.2
    if s in ("medium","moderate","mid"):
        return 0.5
    if s in ("high","veryhigh","highest"):
        return 0.8
    try:
        return float(s)
    except:
        return 0.0

def bid_to_float(b):
    if b is None:
        return 0.0
    if isinstance(b,(int,float,str)):
        try:
            return float(b)
        except:
            return 0.0
    if isinstance(b,dict):
        val = b.get("Amount") or b.get("amount") or 0
        try:
            return float(val)
        except:
            return 0.0
    return 0.0

def month_total(msc):
    if not msc:
        return 0
    if isinstance(msc,list) and all(not isinstance(x,dict) for x in msc):
        try:
            return int(sum(float(x) for x in msc))
        except:
            return 0
    try:
        return sum(int(x.get("Count",0)) for x in (msc or []) if isinstance(x,dict))
    except:
        return 0

def recent_total(msc, months=3):
    """Sum of the most recent `months` (0..months-1), based on API order (latest first)."""
    if not msc:
        return 0
    arr = list(msc)[:months]
    try:
        return int(sum(float(x) for x in arr))
    except:
        return 0

def months_indexed(msc):
    """
    Map each item in MonthlySearchCounts to a real month end.
    We assume the list ends at the most recent complete month.
    """
    if not msc:
        return []
    n = len(msc)
    today = pd.Timestamp.today().normalize()
    last_complete = (today.to_period("M") - 1).to_timestamp()
    months = pd.period_range(end=last_complete.to_period("M"), periods=n, freq="M").to_timestamp()
    return list(zip(months, msc))

def window_total(msc, start_month: pd.Timestamp, end_month: pd.Timestamp):
    """Sum counts where month within [start_month, end_month] inclusive."""
    if not msc:
        return 0
    pairs = months_indexed(msc)
    tot = 0
    for m, v in pairs:
        if start_month <= m <= end_month:
            try:
                tot += float(v)
            except:
                pass
    return int(tot)

def score_item(item, use_window=False, window_val=0):
    comp = comp_to_float(item.get("Competition"))
    bid  = bid_to_float(item.get("SuggestedBid"))
    vol  = window_val if use_window else month_total(item.get("MonthlySearchCounts"))
    return (bid + 0.01) * (0.5 + comp) * (1 + vol/1000.0)

def _safe_unique(seq):
    """Order-preserving unique filter."""
    return list(dict.fromkeys([s for s in seq if s]))

def _normalize_lines_to_list(text):
    """Turn textarea/upload text to a clean list of strings."""
    return [s.strip() for s in text.replace(",", "\n").splitlines() if s.strip()]

# --- FREE providers ---

def ddg_autosuggest(query: str, limit: int = 8):
    """DuckDuckGo public suggestions (no key). Returns a list of strings."""
    if not query:
        return []
    try:
        url = "https://duckduckgo.com/ac/"
        r = SESSION.get(url, params={"q": query, "type": "list"}, timeout=10)
        if r.status_code != 200:
            return []
        js = r.json()
        terms = []
        if isinstance(js, list):
            # Some responses are a list of dicts with "phrase"; some a list of strings
            for it in js:
                if isinstance(it, str):
                    terms.append(it.strip())
                elif isinstance(it, dict):
                    t = (it.get("phrase") or it.get("Phrase") or "").strip()
                    if t:
                        terms.append(t)
        return _safe_unique([t for t in terms if t])[:limit]
    except Exception:
        return []

def wiki_trending_seeds(lang_code: str = "en", day: str | None = None, limit: int = 50):
    """
    Wikimedia Pageviews Top Articles (free). lang_code like 'en','de','fr'...
    day format 'YYYY/MM/DD'. If None, use (today - 2 days) for availability.
    """
    try:
        if day is None:
            day = (pd.Timestamp.today() - pd.Timedelta(days=2)).strftime("%Y/%m/%d")
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/{lang_code}.wikipedia/all-access/{day}"
        r = SESSION.get(url, timeout=12)
        if r.status_code != 200:
            return []
        items = (r.json() or {}).get("items") or []
        if not items:
            return []
        articles = items[0].get("articles") or []
        titles = []
        for a in articles:
            t = (a.get("article") or "").strip()
            if not t:
                continue
            # Filter out namespaces (contain ':') and main-page noise
            if ":" in t:
                continue
            t = t.replace("_", " ")
            if t.lower() in ("main page", "special:search"):
                continue
            titles.append(t)
        return titles[:limit]
    except Exception:
        return []

# Map our markets to Wikipedia language codes (rough)
WIKI_LANG_BY_MARKET = {
    "United States": "en", "United Kingdom": "en",
    "Germany": "de", "France": "fr", "Italy": "it", "Spain": "es",
    "Netherlands": "nl", "Austria": "de", "Denmark": "da",
    "Brazil": "pt", "Australia": "en",
}

def expand_seeds_from_urls(access_token, developer_token, account_id, customer_id,
                           urls, language_name, location_id, devices=None, per_url_limit=50):
    """
    Use AdInsight KeywordIdeas/Query with UrlSearchParameter to expand from competitor/product URLs.
    Returns a list of keywords (strings).
    """
    out = []
    for u in urls:
        if not u.strip():
            continue
        payload = {
            "ExpandIdeas": True,
            "IdeaAttributes": ["Keyword"],
            "SearchParameters": [
                {"Type": "UrlSearchParameter", "Url": u.strip()},
                {"Type": "LanguageSearchParameter", "Languages": [{"Language": language_name}]},
                {"Type": "LocationSearchParameter", "Locations": [{"LocationId": int(location_id)}]},
                {"Type": "NetworkSearchParameter", "Network": {"Network": NETWORK}},
            ],
        }
        if devices and set(devices) != {"Computers", "Smartphones", "Tablets"}:
            payload["SearchParameters"].append({"Type": "DeviceSearchParameter", "Devices": list(devices)})

        try:
            url = "https://adinsight.api.bingads.microsoft.com/AdInsight/v13/KeywordIdeas/Query"
            headers = {
                "Authorization": f"Bearer {access_token}",
                "DeveloperToken": developer_token,
                "CustomerAccountId": str(account_id),
                "CustomerId": str(customer_id),
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
            }
            r = SESSION.post(url, headers=headers, json=payload, timeout=30)
            if r.status_code != 200:
                continue
            ideas = (r.json() or {}).get("KeywordIdeas") or []
            kws = [i.get("Keyword") for i in ideas if (i.get("Keyword") or "").strip()]
            out.extend(kws[:per_url_limit])
        except Exception:
            continue
    return _safe_unique(out)

# Helper to prefill default URL safely (no post-instantiation writes)
def _fill_default_urls_if_empty():
    if st.session_state.get("seed_expand_urls") and not st.session_state.get("urls_text", "").strip():
        first_market = (st.session_state.get("markets_pick") or ["United States"])[0]
        auto_url = DEFAULT_URLS_BY_MARKET.get(first_market, DEFAULT_URLS_BY_MARKET["United States"])
        st.session_state["urls_text"] = auto_url

# ---------- UI ----------
st.set_page_config(page_title="Microsoft Ads Keyword Miner — Multi-Market", layout="wide")
st.title("Microsoft Ads Keyword Miner — Multi-Market")

with st.sidebar:
    st.subheader("Credentials (editable; pre-filled from env/.env)")
    CLIENT_ID = st.text_input("Client ID", os.getenv("MSADS_CLIENT_ID",""))
    CLIENT_SECRET = st.text_input("Client Secret", os.getenv("MSADS_CLIENT_SECRET",""), type="password")
    DEVELOPER_TOKEN = st.text_input("Developer Token", os.getenv("MSADS_DEV_TOKEN",""), type="password")
    save_env = st.checkbox("Save these to .env for next time", value=False, key="save_env")

    auto_ids = st.checkbox("Auto-detect Account/Customer IDs", value=True, key="auto_ids")
    ACCOUNT_ID = st.text_input("Account ID (if not auto)", "")
    CUSTOMER_ID = st.text_input("Customer ID (if not auto)", "")

# Normalize so the rest of the app sees None, not "None"/"null"
CLIENT_SECRET = _normalize_secret(CLIENT_SECRET)
st.caption(f"Auth mode: {'Public (no secret)' if CLIENT_SECRET is None else 'Confidential (with secret)'} — Tenant: {AUTH_TENANT}")

# Save creds to .env if requested
if save_env:
    env_path = Path(".env")
    def _v(x):  # avoid writing the string "None"
        return "" if x is None else str(x)
    lines = [
        f"MSADS_CLIENT_ID={_v(CLIENT_ID)}\n",
        f"MSADS_CLIENT_SECRET={_v(CLIENT_SECRET)}\n",
        f"MSADS_DEV_TOKEN={_v(DEVELOPER_TOKEN)}\n",
    ]
    try:
        env_path.write_text("".join(lines), encoding="utf-8")
        st.success("Saved credentials to .env")
    except Exception as ex:
        st.warning(f"Could not write .env: {ex}")

# --- Devices ---
device_labels = ["Desktop", "Mobile", "Tablet"]
device_map = {"Desktop": "Computers", "Mobile": "Smartphones", "Tablet": "Tablets"}
pick_devices = st.sidebar.multiselect("Devices", device_labels, default=device_labels)
api_devices = [device_map[d] for d in pick_devices]  # e.g. ["Computers","Smartphones","Tablets"]

st.sidebar.markdown("---")
add_device_kpis = st.sidebar.checkbox("Add device-level KPIs (historical)", value=False, key="add_device_kpis")

time_window = st.sidebar.selectbox(
    "Historical window (for device KPIs)",
    ["Last7Days", "Last30Days", "LastMonth", "Last3Months"],
    index=1
)

match_types = st.sidebar.multiselect("Match types for KPIs", ["Exact", "Phrase", "Broad"], default=["Exact"])

st.markdown("#### Seeds")
seeds_str = st.text_area(
    "One keyword per line",
    value="",  # start empty
    placeholder="e.g.\nroof repair near me\nbilling software\nfleet tracking",
    height=120,
)
st.caption("You can leave this empty if you use Seed Discovery below.")

# --- Seed Discovery (single, keyed) ---
with st.expander("Seed Discovery (auto-expansion)", expanded=False):
    use_url_expand = st.checkbox(
        "Expand from competitor/product URLs (Microsoft Ads)",
        value=False,
        key="seed_expand_urls",
        on_change=_fill_default_urls_if_empty
    )

    # If checkbox is already True (e.g., from prior run), prefill before creating the textarea
    if st.session_state.get("seed_expand_urls"):
        _fill_default_urls_if_empty()

    urls_text = st.text_area(
        "URLs (one per line)",
        value=st.session_state.get("urls_text", ""),
        height=80,
        help="We’ll ask Ad Insight to generate ideas from these URLs. Try Amazon Best Sellers for your market.",
        placeholder="https://www.amazon.com/Best-Sellers/zgbs",
        key="urls_text"
    )

    per_url_limit = st.slider("Max ideas per URL", 10, 200, 50, 10, key="per_url_limit")

    use_ddg = st.checkbox(
        "Use FREE Autosuggest to expand typed seeds (DuckDuckGo)",
        value=False,
        key="seed_expand_ddg"
    )
    ddg_limit = st.slider("Max autosuggest per typed seed", 3, 20, 8, 1, key="seed_expand_ddg_limit")

    use_wiki = st.checkbox(
        "Add FREE trending topics (Wikipedia Pageviews) as extra seeds",
        value=False,
        key="seed_expand_wiki"
    )
    wiki_limit = st.slider("Max trending topics per market", 10, 100, 30, 10, key="seed_expand_wiki_limit")

st.caption("Note: by default, only the seeds you type above are used.")

# Optional: load from seeds.txt or an uploaded file
use_file = st.checkbox("Load seeds from a file (optional)", value=False, key="use_file")
combine_mode = "Append"
file_seeds = []

if use_file:
    # Prefer an uploaded file; fall back to seeds.txt if present
    up = st.file_uploader("Upload seeds (.txt or .csv, one keyword per line)", type=["txt","csv"])
    if up is not None:
        try:
            text = up.read().decode("utf-8", errors="ignore")
            file_seeds = [s.strip() for s in text.replace(",", "\n").splitlines() if s.strip()]
        except Exception as ex:
            st.warning(f"Could not read uploaded file: {ex}")
    elif Path("seeds.txt").exists():
        try:
            text = Path("seeds.txt").read_text(encoding="utf-8", errors="ignore")
            file_seeds = [s.strip() for s in text.splitlines() if s.strip()]
            st.info("Loaded seeds from local seeds.txt")
        except Exception as ex:
            st.warning(f"Could not read seeds.txt: {ex}")

    combine_mode = st.radio(
        "How to combine file seeds with typed seeds",
        ["Append", "Replace"],
        index=0,
        horizontal=True,
        key="combine_mode"
    )

# Build the final seeds list (typed first, then optional file)
typed_seeds = [s.strip() for s in seeds_str.splitlines() if s.strip()]
if use_file and file_seeds:
    if combine_mode == "Replace":
        seeds = list(dict.fromkeys(file_seeds))  # de-dup, keep order
    else:
        seeds = list(dict.fromkeys(typed_seeds + file_seeds))
else:
    seeds = typed_seeds

st.caption(f"Using {len(seeds)} seed(s).")

st.markdown("#### Markets")
options = [m[0] for m in MARKETS]
if "markets_pick" not in st.session_state:
    st.session_state["markets_pick"] = ["United States"]
markets_pick = st.multiselect(
    "Select one or more",
    options=options,
    default=st.session_state["markets_pick"],
)
st.session_state["markets_pick"] = markets_pick

st.markdown("#### Date range (used for volume and score)")
mode = st.radio(
    "How do you want to aggregate volume?",
    ["All available months", "Last N months", "Absolute range (month to month)"],
    index=0,
)
months_n = None
start_month = end_month = None
if mode == "Last N months":
    months_n = st.slider("N (months)", 1, 12, 3, 1)
elif mode == "Absolute range (month to month)":
    col1, col2 = st.columns(2)
    with col1:
        d1 = st.date_input("Start month", value=date.today().replace(day=1))
    with col2:
        d2 = st.date_input("End month", value=date.today().replace(day=1))
    # normalize to first-of-month
    start_month = pd.Timestamp(d1.year, d1.month, 1)
    end_month   = pd.Timestamp(d2.year, d2.month, 1)
    if end_month < start_month:
        st.error("End month must be >= Start month.")
        st.stop()

# Advanced scoring option: default = use selected window; override to lifetime volume if needed
with st.expander("Advanced scoring (optional)", expanded=False):
    use_lifetime_for_score = st.checkbox(
        "Override and use lifetime (all months) volume for score",
        value=False,
        help="When ON, the score ignores the selected date range and uses total historical volume.",
        key="use_lifetime_for_score"
    )

estimate_bids = st.checkbox("Also fetch first-page bid estimates (MainLine1)", key="estimate_bids")
topN_bids = st.number_input("How many top ideas per market to price?", min_value=50, max_value=1000, value=200, step=50)

colA, _ = st.columns([1,2])
with colA:
    lang_locale = st.selectbox("Geo CSV language locale (for country names)", options=["en","de","fr","it","es","pt-BR"], index=0)

with st.expander("Filters (novelty & quality)", expanded=False):
    existing_up = st.file_uploader("Existing keywords to exclude (txt/csv; one per line or comma-separated)", type=["txt","csv"])
    include_default_blocklist = st.checkbox("Include default brand/sensitive blocklist", value=True, key="include_default_blocklist")
    extra_blocklist = st.text_area("Extra blocklist patterns (regex, one per line)", value="", height=80)
    min_window_vol = st.number_input("Minimum volume in selected window (0 = no minimum)", min_value=0, max_value=1_000_000, value=0, step=10)
    max_kw_len = st.slider("Maximum keyword length", 10, 80, 49, 1)

# ---- Run (form submit is more reliable) ----
with st.form("run_form", clear_on_submit=False):
    submitted = st.form_submit_button("Run")

if submitted:
    try:
        st.info("Run started…")

        # --- Validate creds ---
        problems = []
        if not CLIENT_ID:
            problems.append("Client ID")
        if not DEVELOPER_TOKEN:
            problems.append("Developer Token")
        if problems:
            raise ValueError(f"Missing: {', '.join(problems)}. Fill credentials in the sidebar.")

        # --- Tokens ---
        with st.spinner("Refreshing access token…"):
            access_token, refresh_token = ensure_access_token(CLIENT_ID, CLIENT_SECRET)

        # --- IDs ---
        if auto_ids:
            try:
                with st.spinner("Detecting Account/Customer IDs…"):
                    account_id, customer_id = get_ids_with_sdk(
                        CLIENT_ID, CLIENT_SECRET, DEVELOPER_TOKEN, refresh_token
                    )
            except Exception as e:
                st.error(
                    "Couldn’t auto-detect Account/Customer IDs.\n\n"
                    "Most common causes:\n"
                    "• Developer Token is invalid, pending, or Sandbox-only\n"
                    "• The signed-in Microsoft Ads user has no access to any account\n"
                    "• Tokens were issued for a different app (CLIENT_ID/SECRET mismatch)\n\n"
                    "Turn OFF auto-detect and enter Account ID + Customer ID manually in the sidebar."
                )
                if ACCOUNT_ID and CUSTOMER_ID:
                    account_id, customer_id = int(ACCOUNT_ID), int(CUSTOMER_ID)
                    st.info(f"Falling back to manual IDs: AccountId={account_id}  CustomerId={customer_id}")
                else:
                    st.stop()
        else:
            if not (ACCOUNT_ID and CUSTOMER_ID):
                raise ValueError("Provide Account ID and Customer ID or enable auto-detect.")
            account_id, customer_id = int(ACCOUNT_ID), int(CUSTOMER_ID)

        # --- Geo CSV ---
        with st.spinner("Downloading Geo locations CSV…"):
            df_geos = fetch_geos_csv_rest(
                access_token, DEVELOPER_TOKEN, account_id, customer_id, language_locale=lang_locale
            )

        # --- Market plan (needed for expansions) ---
        market_plan = []
        for name, lang_full, lang_code in MARKETS:
            if name not in st.session_state["markets_pick"]:
                continue
            lid = resolve_country_id(df_geos, name)
            if not lid:
                st.warning(f"Could not resolve LocationId for {name}. Skipping.")
                continue
            market_plan.append((name, lang_full, lang_code, lid))
        if not market_plan:
            raise ValueError("No valid markets selected.")

        # --- Seed Discovery expansions (optional) ---
        seeds = list(typed_seeds)  # start with typed
        urls_list = _normalize_lines_to_list(st.session_state.get("urls_text", ""))
        expanded_from_urls = []
        expanded_from_ddg = []
        expanded_from_wiki = []

        if st.session_state.get("seed_expand_urls") and urls_list:
            st.info("Expanding seeds from URLs via Ad Insight…")
            for (mname, lang_full, lang_code, lid) in market_plan:
                kws = expand_seeds_from_urls(
                    access_token, DEVELOPER_TOKEN, account_id, customer_id,
                    urls=urls_list, language_name=lang_full, location_id=lid,
                    devices=api_devices, per_url_limit=int(st.session_state.get("per_url_limit", 50))
                )
                expanded_from_urls.extend(kws)
            expanded_from_urls = _safe_unique(expanded_from_urls)

        if st.session_state.get("seed_expand_ddg") and typed_seeds:
            st.info("Expanding seeds via DuckDuckGo (free)…")
            lim = int(st.session_state.get("seed_expand_ddg_limit", 8))
            for q in typed_seeds:
                expanded_from_ddg.extend(ddg_autosuggest(q, limit=lim))
            expanded_from_ddg = _safe_unique(expanded_from_ddg)

        if st.session_state.get("seed_expand_wiki"):
            st.info("Adding trending topics from Wikipedia Pageviews…")
            lim = int(st.session_state.get("seed_expand_wiki_limit", 30))
            for (mname, _, _, _) in market_plan:
                lang = WIKI_LANG_BY_MARKET.get(mname, "en")
                expanded_from_wiki.extend(wiki_trending_seeds(lang_code=lang, limit=lim))
            expanded_from_wiki = _safe_unique(expanded_from_wiki)

        seeds = _safe_unique(seeds + expanded_from_urls + expanded_from_ddg + expanded_from_wiki)
        if not seeds:
            raise ValueError("No seeds available after expansion. Enter seeds or enable Seed Discovery.")

        with st.expander("Seeds used (preview)", expanded=False):
            st.write({
                "typed_seeds": len(typed_seeds),
                "expanded_from_urls": len(expanded_from_urls),
                "expanded_from_ddg": len(expanded_from_ddg),
                "expanded_from_wiki": len(expanded_from_wiki),
                "total_seeds_used": len(seeds),
            })
            st.dataframe(pd.DataFrame({"seed": seeds}).head(200), use_container_width=True, hide_index=True)

        # --- Build novelty/quality filters ---
        existing_set = set()
        if 'existing_up' in locals() and existing_up is not None:
            try:
                text = existing_up.read().decode("utf-8", errors="ignore")
                existing_set = set(s.lower() for s in _normalize_lines_to_list(text))
            except Exception as ex:
                st.warning(f"Could not read uploaded existing keywords: {ex}")

        block_patterns = []
        if st.session_state.get("include_default_blocklist", True):
            block_patterns += [
                r"\bamazon\b", r"\bgoogle\b", r"\bfacebook\b", r"\btesla\b",
                r"\biphone\b", r"\bgalaxy\b", r"\boxempic\b", r"\bviagra\b",
                r"\binsulin\b", r"\bpfizer\b", r"\bmoderna\b",
            ]
        if 'extra_blocklist' in locals() and extra_blocklist.strip():
            block_patterns += [ln.strip() for ln in extra_blocklist.splitlines() if ln.strip()]

        block_regs = [re.compile(p, re.I) for p in block_patterns]

        # --- Run summary / fingerprint ---
        ts_str = pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')
        run_params = {
            "timestamp": ts_str,
            "account_id": int(account_id),
            "customer_id": int(customer_id),
            "network": NETWORK,
            "markets": [m[0] for m in market_plan],
            "geo_csv_locale": lang_locale,
            "devices": api_devices,
            "estimate_bids": bool(st.session_state.get("estimate_bids")),
            "add_device_kpis": bool(st.session_state.get("add_device_kpis")),
            "time_window_kpi": time_window,
            "volume_mode": mode,
            "months_n": int(months_n or 0),
            "advanced_use_lifetime_for_score": bool(st.session_state.get("use_lifetime_for_score")),
            "topN_bids": int(topN_bids),
            "seeds": seeds,
        }

        import hashlib
        run_id = hashlib.sha1(json.dumps(run_params, sort_keys=True).encode("utf-8")).hexdigest()[:8]

        with st.expander("Run summary", expanded=False):
            st.write({"run_id": run_id, **{k: v for k, v in run_params.items() if k != "seeds"}})
            st.write({"seeds_sample": seeds[:10], "seeds_count": len(seeds)})

        # --- Ideas per market ---
        all_rows = []
        progress = st.progress(0.0, text="Fetching keyword ideas…")
        for i, (mname, lang_full, lang_code, lid) in enumerate(market_plan, 1):
            ideas = get_keyword_ideas(
                access_token, DEVELOPER_TOKEN, account_id, customer_id,
                seeds, lang_full, [lid],
                devices=api_devices,
                network=NETWORK,
            )
            for ki in ideas:
                msc = ki.get("MonthlySearchCounts")
                total_all = month_total(msc)

                # windowed totals
                if mode == "All available months":
                    win_val = total_all
                elif mode == "Last N months":
                    win_val = recent_total(msc, months=months_n or 3)
                else:
                    win_val = window_total(msc, start_month, end_month)

                comp_num = comp_to_float(ki.get("Competition"))
                bid_amt  = bid_to_float(ki.get("SuggestedBid"))

                # Default: score uses selected window; advanced toggle can force lifetime volume
                base_vol = total_all if st.session_state.get("use_lifetime_for_score") else win_val
                score = (bid_amt + 0.01) * (0.5 + comp_num) * (1 + base_vol / 1000.0)

                kw_text = (ki.get("Keyword") or "").strip()
                if not kw_text:
                    continue
                if len(kw_text) > int(max_kw_len):
                    continue
                if min_window_vol and (win_val < int(min_window_vol)):
                    continue
                if kw_text.lower() in existing_set:
                    continue
                if any(r.search(kw_text) for r in block_regs):
                    continue

                all_rows.append({
                    "market": mname,
                    "language": lang_full,
                    "location_id": lid,
                    "keyword": kw_text,
                    "competition_raw": ki.get("Competition"),
                    "competition_num": round(comp_num, 3),
                    "suggested_bid": round(bid_amt, 2),
                    "monthly_total_all": total_all,
                    "window_total": win_val,
                    "_score": round(score, 4),
                })
            progress.progress(i/len(market_plan), text=f"Fetched {mname}")

        if not all_rows:
            st.warning("No ideas returned.")
            st.stop()

        # --- Table before optional bids/KPIs ---
        df = pd.DataFrame(all_rows).sort_values(
            ["market","_score"], ascending=[True,False]
        ).reset_index(drop=True)

        # --- Optional: first-page bids (MainLine1) ---
        if st.session_state.get("estimate_bids"):
            st.info("Estimating first-page bids (MainLine1)…")
            if "first_page_min_bid" not in df.columns:
                df["first_page_min_bid"] = pd.NA
            if "avg_cpc_est" not in df.columns:
                df["avg_cpc_est"] = pd.NA

            for (mname, lang_full, lang_code, lid) in market_plan:
                subset = df[df["market"]==mname].nlargest(int(topN_bids), "_score")
                kw_list = subset["keyword"].dropna().astype(str).str.strip().tolist()
                if not kw_list:
                    st.write(f"{mname}: no keywords to price.")
                    continue

                bmap = get_estimated_bids(
                    access_token, DEVELOPER_TOKEN, account_id, customer_id,
                    kw_list, lang_code, lid
                )
                norm_map = { (k or "").strip().lower(): v for k, v in bmap.items() }
                mask = (df["market"] == mname)
                lowered = df.loc[mask, "keyword"].astype(str).str.strip().str.lower()
                df.loc[mask, "first_page_min_bid"] = lowered.map(lambda x: (norm_map.get(x) or {}).get("first_page_min_bid"))
                df.loc[mask, "avg_cpc_est"]       = lowered.map(lambda x: (norm_map.get(x) or {}).get("avg_cpc_est"))
                st.write(f"{mname}: priced {len(bmap)} of {len(kw_list)} requested keywords")

        # --- Optional: device-level KPIs (HistoricalKeywordPerformance) ---
        if st.session_state.get("add_device_kpis") and not df.empty and api_devices:
            st.info("Fetching device-level KPIs…")
            # Ensure device columns exist
            for dev_api, label in [("Computers","Desktop"), ("Smartphones","Mobile"), ("Tablets","Tablet")]:
                if dev_api in api_devices:
                    for suf in ("impr","ctr","cpc"):
                        col = f"{label}_{suf}"
                        if col not in df.columns:
                            df[col] = pd.NA

            for (mname, lang_full, lang_code, lid) in market_plan:
                subset = df[df["market"]==mname].nlargest(200, "_score")
                kw_list = subset["keyword"].dropna().astype(str).str.strip().tolist()
                if not kw_list:
                    st.write(f"{mname}: no keywords for device KPIs.")
                    continue

                lang = MARKET_LANGUAGE.get(mname, lang_full)
                pubs = PUBLISHER_COUNTRIES.get(mname, [])
                kmap = get_device_kpis_rest(
                    access_token, DEVELOPER_TOKEN, account_id, customer_id,
                    kw_list, api_devices, lang, pubs,
                    time_interval=time_window,  # pass raw; mapper handles it
                    match_types=tuple(match_types)
                )

                # assign back into df
                for kw in kw_list:
                    for dev_api, label in [("Computers","Desktop"), ("Smartphones","Mobile"), ("Tablets","Tablet")]:
                        if dev_api not in api_devices:
                            continue
                        vals = kmap.get((kw, dev_api))
                        if not vals:
                            continue
                        mask = (df["market"]==mname) & (df["keyword"]==kw)
                        if "Impressions" in vals:
                            df.loc[mask, f"{label}_impr"] = vals["Impressions"]
                        if "CTR" in vals:
                            df.loc[mask, f"{label}_ctr"] = vals["CTR"]
                        if "AverageCPC" in vals:
                            df.loc[mask, f"{label}_cpc"] = vals["AverageCPC"]

                st.write(f"{mname}: KPIs fetched for ~{len(kmap)} device-keyword pairs.")

        # --- Output ---
        # Ensure optional columns exist even if estimate_bids/device KPIs were skipped
        for c in ("first_page_min_bid", "avg_cpc_est"):
            if c not in df.columns:
                df[c] = pd.NA

        base_cols = [
            "market","keyword","_score",
            "competition_raw","competition_num",
            "suggested_bid","first_page_min_bid","avg_cpc_est",
            "window_total","monthly_total_all",
            "language","location_id"
        ]
        dev_cols = [c for c in (
            "Desktop_impr","Desktop_ctr","Desktop_cpc",
            "Mobile_impr","Mobile_ctr","Mobile_cpc",
            "Tablet_impr","Tablet_ctr","Tablet_cpc"
        ) if c in df.columns]

        show_cols = [c for c in base_cols if c in df.columns] + dev_cols

        # Export controls so CSV matches the view
        with st.expander("Export options", expanded=False):
            default_sort_idx = show_cols.index("_score") if "_score" in show_cols else 0
            sort_col = st.selectbox("Sort by", show_cols, index=default_sort_idx)
            sort_asc = st.checkbox("Ascending", value=False, key="sort_asc")
            top_per_market = st.number_input("Top N per market (0 = all rows)", min_value=0, max_value=10000, value=0, step=50)

        # Build the export dataframe to mirror the chosen sort/limit
        df_export = df[show_cols].copy()
        if top_per_market and "market" in df_export.columns:
            df_export = (
                df_export.groupby("market", as_index=False, group_keys=False)
                .apply(lambda g: g.sort_values(sort_col, ascending=sort_asc).head(int(top_per_market)))
            )
        df_export = df_export.sort_values(sort_col, ascending=sort_asc).reset_index(drop=True)

        st.success(f"Done. {len(df_export)} rows.")
        st.dataframe(df_export, use_container_width=True, hide_index=True)

        # Bytes for both download and disk write (so they always match)
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv_bytes,
            file_name="msads_keyword_ideas_multi_market.csv",
            mime="text/csv",
            key=f"dl_csv_{run_id}"  # ← unique per run
        )

        # --- Auto-save (single file) ---
        # Save exactly one CSV next to the app, so it's easy to find and overwrite each run.
        root_csv = Path("msads_keyword_ideas.csv")
        try:
            root_csv.write_bytes(csv_bytes)
            mtime_str = datetime.fromtimestamp(root_csv.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            st.info(f"Auto-saved file:\n- {root_csv.resolve().as_posix()} (mtime {mtime_str})")
        except Exception as ex:
            st.warning(f"Auto-save failed: {ex}")

    except Exception as e:
        st.exception(e)