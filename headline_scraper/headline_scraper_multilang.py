import sys
import csv
import time
import re
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from urllib import robotparser

USER_AGENT = "HeadlineScraperBot/0.2 (+https://example.com/bot-info)"
REQUEST_TIMEOUT = 10
MAX_PAGES_PER_TARGET = 600       # safety cap per domain/scope
SLEEP_BETWEEN_REQUESTS = 0.2     # throttle a bit


NICHE_RULES = {
    "automotive": [
        " car ", " cars ", " suv ", " suvs ", " crossover ", " truck ", " trucks ",
        " vehicle ", " vehicles ", " tyre ", " tire ", " tires ", " cuv ",
        " pickup ", " motorhome ", " rv ", " camper ", " motorhome ",
        " financing ", " motorcycle financing ",
        # brand/model signals from your list (classification only)
        " toyota ", " honda ", " ford ", " chevy ", " chevrolet ", " nissan ",
        " jeep ", " kia ", " subaru ", " lexus ", " buick ", " ram ", " dodge ",
        " tacoma ", " wrangler ", " rav4 ", " grand cherokee ", " silverado ",
        " tahoe ", " explorer ", " accord ", " altima ", " frontier ",
        " telluride ", " envoy ", " montana ", " nautlius ", " nautilus ",
        " terra ", " crosstrek ", " prado ", " highlander ", " kluger ", " picanto ",
        " f-150 ", " f 150 ", " f-450 ", " f 450 ", " ranger ",
        " tacozilla ", " colorado ", " armada ", " terra ",
        " police impound cars ", " police auction cars ", " seized cars ",
        " police impound boats "
    ],

    "home_services_garden": [
        " garden ", " landscaping ", " lawn care ", " lawn service ",
        " tree trimming ", " tree removal ", " stump removal ",
        " patio ", " paving ", " concrete crack ", " concrete cracks ",
        " fence ", " fencing ", " awning ", " awnings ",
        " home cleaning ", " maid ", " cleaning services ", " house cleaning ",
        " bathroom remodeling ", " bathroom update ", " bath remodel ",
        " kitchen remodel ", " kitchen renovation ", " kitchen cabinets ",
        " window replacement ", " replacement windows ", " window grants ",
        " attic insulation ", " spray foam insulation ", " insulation ",
        " mold remediation ", " water damage ", " leak repair ",
        " air conditioning ", " ac repair ", " hvac ", " ductless ac ",
        " pool installation ", " swimming pool ", " plunge pool ",
        " outdoor lighting ", " backyard ", " yard ", " lawn weed ",
        " pest control ", " cockroaches ", " moles ",
        " power washing ", " pressure washing ",
        " scaffolding ", " flooring installation ", " epoxy floor ",
        " rubber floor ", " vent cleaning ", " drain cleaning ",
        " home renovation ", " home repair ", " home remodel ",
        " quonset hut ", " prefab cabin ", " pre-fab cabin ",
        " water storage tank ", " storage tanks "
    ],

    "housing_property": [
        " abandoned houses ", " foreclosed homes ", " foreclosures ",
        " bank-owned ", " police impound boats ",
        " rent-to-own ", " rent to own ", " rent-to-buy ", " rent to buy ",
        " social housing ", " section 8 ", " section 202 ",
        " apartments for seniors ", " retirement bungalows ",
        " mobile homes ", " modular homes ", " container homes ",
        " prefabricated homes ", " pre-fab homes ",
        " granny annexes ", " accessory dwelling ", " backyard apartment ",
        " home value ", " property value ", " property values ",
        " house value ", " homeownership ", " down payment ",
        " cabins ", " mountain cabins ", " blue ridge cabins "
    ],

    "health_medical": [
        " health ", " healthcare ", " medical ", " medicine ",
        " hospital ", " clinic ", " doctor ", " physician ", " nurse ",
        " dementia ", " autism ", " asthma ", " glaucoma ",
        " lung cancer ", " heart failure ", " nafld ", " nash ",
        " pulmonary hypertension ", " plaque psoriasis ", " actinic keratosis ",
        " obesity ", " osteoarthritis ", " osteoporosis ",
        " angioedema ", " choledocholithiasis ", " cholangitis ",
        " amyloidosis ", " sma ", " multiple sclerosis ",
        " cancer ", " tumor ", " tumours ",
        " dentures ", " invisible dentures ", " tooth replacement ",
        " dental implants ", " screwless implants ",
        " dental clips ", " cosmetic dentistry ",
        " blepharitis ", " demodex ", " eyelash mites ",
        " incontinence ", " neuropathy ", " sciatica ",
        " arthritis clinic ", " arthritis clinics ",
        " antidepressants ", " depression ", " brain power test ",
        " glaucoma specialists ", " ophthalmology ",
        " supplements for men 50+ ", " multivitamin ", " healthy ageing ",
        " laser fat removal ", " non-surgical fat removal ",
        " tummy tuck ", " abdominoplasty ",
        " ed treatment ", " erectile dysfunction ",
        " probiotics ", " adaptogens ",
        " tinnitus ", " sleep apnea ", " cpap ",
        " clinical trial ", " clinical trials ",
        " tetanus toxoid ", " urinary tract infections ", " uti ",
        " bile duct stone ", " bile duct stones ",
        " weight loss options ", " weight loss tips "
    ],

    "supplements_wellness": [
        " supplements ", " multivitamin ", " vitamins ",
        " wellness ", " fitness ", " gym shoes ",
        " tai chi ", " yoga ", " meditation ", " breathwork ",
        " sleep masks ", " sleep chairs ",
        " wellness gadgets ", " spa ", " body spa ",
        " sound therapy ", " hot girl walk ",
        " mindfulness ", " mental wellness ", " resilience ",
        " bedtime drinks ", " hydration tracking "
    ],

    "sexual_health": [
        " erectile dysfunction ", " ed treatment ", " ed solutions ",
        " prostate health "
    ],

    "telecom_broadband_tv": [
        " broadband ", " fixed wireless ", " ipv6 ", " telecom ",
        " sky tv ", " internet providers ", " 5g internet ",
        " wi-fi ", " wifi ", " wi fi ", " portable wifi ",
        " portable wi-fi ", " satellite internet ",
        " mobile plans ", " family cell phone plans ",
        " data caps ", " cable internet ", " dsl internet ",
        " tv packages ", " streaming ", " smart speaker "
    ],

    "dating_relationships": [
        " dating ", " finding love ", " relationships ",
        " love after 50 ", " love after 60 ",
        " speed dating ", " citas rápidas ", " online dating ",
        " zodiac sign loves ", " how your sign loves "
    ],

    "senior_living_mobility": [
        " seniors ", " over 50 ", " over 60 ", " 55+ ",
        " retirement bungalows ", " retirement destinations ",
        " assisted living ", " dementia care facilities ",
        " portable stairlifts ", " mobile stairlifts ",
        " stairlift ", " stair lift ",
        " mobility scooters ", " scooters ",
        " medicare ", " medicaid ", " centrelink ",
        " utilities-included apartments for seniors ",
        " superannuation "
    ],

    "finance_investing": [
        " cd rates ", " cd rate ", " 6-month cd ",
        " high-interest savings ", " high yield savings ",
        " savings accounts ", " checking account bonuses ",
        " bank bonuses ", " bank account bonus ",
        " tax advantaged ", " tax efficient ",
        " etf ", " etfs ", " mutual funds ",
        " compound interest ", " dollar-cost averaging ",
        " gold ira ", " gold iras ", " ira kits ",
        " retirement planning ", " pension ", " state pension ",
        " reverse mortgage ", " home equity ",
        " credit cards ", " balance transfer ",
        " instant approval credit cards ",
        " bad credit ", " no credit ",
        " prepaid debit cards ", " debit cards ",
        " personal loans ", " emergency cash ",
        " grants ", " scholarships ", " small farms grants ",
        " disability grants ", " financial aid ",
        " utilities assistance ", " energy assistance ",
        " pet insurance ", " dog insurance ",
        " auto insurance ", " car insurance ",
        " renters insurance ", " vision insurance ",
        " life insurance ", " mortgage refinancing ",
        " debt relief ", " irs debt ", " debt forgiveness ",
        " budgeting ", " passive income ", " financial independence ",
        " checking account ", " online bank ",
        " cash rewards ", " cash back card ", " cashback credit card "
    ],

    "b2b_industrial": [
        " industrial machines ", " industrial ", " logistics ",
        " warehouse ", " warehouse picking ", " warehouse sale ",
        " equipment operator ", " forklift operator ",
        " welding programs ", " laser welding ", " portable laser welding ",
        " food packing ", " packaging & logistics ",
        " waste management ", " manufacturing efficiency ",
        " smart factories ", " industry 4.0 ",
        " maintenance strategies ", " supply chain ",
        " ecommerce website development ", " online product advertising ",
        " content marketing ", " inventory management ",
        " network security ", " cybersecurity ",
        " managed it support ", " ac repair business ",
        " hvac maintenance contracts ",
        " attendance systems ", " workforce efficiency ",
        " dashboard creation ", " data analytics ",
        " advertising tools ", " ai tools for visibility "
    ],

    "law_legal": [
        " lawyer ", " lawyers ",
        " attorney ", " attorneys ",
        " law firm ", " law firms ",
        " law office ", " law offices ",
        " legal help ", " legal advice ",
        " legal services ", " legal representation ",
        " lawsuit ", " lawsuits ",
        " settlement ", " settlements ",
        " class action ",
        " compensation claim ", " compensation claims ",
        " civil forfeiture ", " eminent domain ",
        " dui charges ", " divorce online ",
        " personal injury ", " maximize personal injury compensation "
    ],

    "b2b_software": [
        " software ", " saas ", " platform ", " cloud service ",
        " business software ", " enterprise software ", " b2b software ",
        " crm ", " crm software ", " erp ", " erp software ",
        " hris ", " hr software ", " hrm software ",
        " accounting software ", " bookkeeping software ",
        " payroll software ", " billing software ", " invoicing software ",
        " marketing software ", " marketing automation ",
        " email marketing software ", " lead generation software ",
        " sales software ", " sales engagement ", " sales enablement ",
        " call center software ", " contact center software ",
        " voip software ",
        " webinar software ", " webinar platform ",
        " observability platform ", " log management ",
        " backup software ", " disaster recovery software ",
        " gdpr compliance software ", " compliance software ",
        " grc software ",
        " document management software ",
        " esignature software ", " e-signature software ",
        " online fax service ", " fax service online ",
        " help desk software ", " ticketing software ",
        " customer support software ", " service desk software ",
        " ad blocker ", " online payment systems "
    ],

    # new helpful buckets based on your list:
    "travel_tourism": [
        " vacation ", " vacations ", " resort ", " resorts ",
        " safari ", " golf vacation ", " honeymoon ",
        " cruise ", " cruises ", " yacht rental ", " private jet ",
        " villas ", " jacuzzis ", " spa retreat ",
        " tour ", " tours ", " train trips ", " scenic train ",
        " cabins ", " lodges ", " wellness retreat ",
        " destinations ", " getaway ", " travel insurance ",
        " flights ", " booking flights ", " bucket list travel ",
        " northern lights ", " glamping ",
        " themed hotels ", " castle stay ",
        " national parks ", " wildlife tours ",
        " digital nomads ", " slow travel "
    ],

    "education_career": [
        " mba ", " master's ", " masters ", " doctorate ", " doctoral ",
        " phd ", " online degree ", " online degrees ",
        " high school diploma ", " early childhood education ",
        " electrical engineering degree ", " civil engineering degree ",
        " computer science courses ",
        " scholarships ", " mba scholarships ",
        " tuition assistance ", " military tuition assistance ",
        " anatomy knowledge ", " stem courses ",
        " vocational training ", " electrician courses ",
        " game development degrees ", " yoga teaching certification ",
        " graphic design courses ", " veterinary assistant ",
        " nursing programs ", " accelerated nursing ",
        " college ", " colleges ", " universities ", " campus ",
        " online learning ", " hybrid learning ",
        " cdl training ", " trucking career ",
        " aviation training ", " aviation mechanics ",
        " security field careers ", " caregiver careers ",
        " warehouse careers ", " moving industry careers ",
        " restaurant industry careers ",
        " laundry industry careers "
    ],

    "fashion_beauty": [
        " dresses ", " maxi dresses ", " a-line dresses ", " sheath dresses ",
        " bohemian dresses ", " boho dresses ",
        " micro bikini ", " bikini styles ", " daring bikini ",
        " bras ", " full-coverage bras ",
        " hair transplant ", " rhinoplasty ",
        " anti-aging cream ", " anti aging cream ",
        " skincare ", " skin care ", " eye creams ", " niacinamide ",
        " hair volume ", " thinning hair ", " hairstyles ",
        " mother of the bride ", " plus size boho ",
        " shapewear ", " slides ",
        " comfortable pants ", " pull-on pants ",
        " outfit ", " wardrobe ",
        " nail art ", " makeup ",
        " beauty standards ", " beauty ",
        " wigs ", " hairpieces ", " wiglets ",
        " lingerie ", " underwear ", " nighties ",
        " fashion ", " mature women "
    ],

    "pets_animals": [
        " french bulldogs ", " chihuahua puppies ", " small dogs ",
        " dog insurance ", " pet insurance ",
        " kennel ", " outdoor kennels ",
        " leafcutter ants ", " australian quokka ",
        " wildlife destinations "
    ],

    "zodiac_astrology": [
        " zodiac sign ", " star sign ", " horoscope ",
        " astrology ", " brunch horoscope "
    ],

    "entertainment_popculture": [
        " memes ", " netflix ", " horror movies ",
        " trivia games ", " metaverse concerts ",
        " hollywood ", " celebrity ", " red carpet ",
        " broadway ", " movie trailers ", " blockbusters ",
        " esports ", " gaming titles ", " video game soundtracks ",
        " interactive theater ", " live performances "
    ],

    "general_lifestyle": [
        " decluttering ", " hygge ", " cozy home ",
        " diy projects ", " mason jars ",
        " home office space ", " wardrobe ",
        " minimalism ", " digital decluttering ",
        " mindfulness in everyday life ",
        " balanced lifestyle ", " mental well-being "
    ]
}

def get_session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def get_robots_parser(session, netloc, scheme="https"):
    robots_url = f"{scheme}://{netloc}/robots.txt"
    rp = robotparser.RobotFileParser()
    try:
        resp = session.get(robots_url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            rp.parse("")
    except Exception:
        rp.parse("")
    return rp


def normalize_target_line(line):
    line = line.strip()
    if not line:
        return None

    parsed = urlparse(line)
    if not parsed.scheme:
        domain = line.strip("/ ")
        base_url = f"https://{domain}"
        scope_prefix = ""
        netloc = domain
    else:
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        scope_prefix = parsed.path
        if scope_prefix and not scope_prefix.endswith("/"):
            scope_prefix += "/"
        netloc = parsed.netloc

    return {
        "base_url": base_url,
        "scope_prefix": scope_prefix,
        "netloc": netloc,
    }


def is_html_response(resp):
    ctype = resp.headers.get("Content-Type", "")
    return "text/html" in ctype or "application/xhtml" in ctype


def is_asset_url(path):
    asset_exts = (
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg",
        ".css", ".js", ".ico", ".woff", ".woff2", ".ttf",
        ".eot", ".otf", ".pdf", ".zip", ".rar", ".mp4",
        ".mp3", ".avi", ".mov", ".m4v", ".webm"
    )
    return any(path.lower().endswith(ext) for ext in asset_exts)


def looks_like_article(path):
    if is_asset_url(path):
        return False

    if any(seg in path for seg in ("/tag/", "/tags/", "/category/", "/categories/", "/author/", "/page/")):
        return False

    segments = [s for s in path.split("/") if s]
    if not segments:
        return False

    last = segments[-1]

    if re.search(r"[A-Za-z0-9]-[A-Za-z0-9]", last):
        return True

    if re.search(r"/\d{4}/\d{2}/\d{2}/", path):
        return True

    if len(segments) >= 2 and len(last) > 6:
        return True

    return False


def extract_headline_from_soup(soup):
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        return h1.get_text(strip=True)

    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()

    title = soup.find("title")
    if title and title.get_text(strip=True):
        return title.get_text(strip=True)

    h2 = soup.find("h2")
    if h2 and h2.get_text(strip=True):
        return h2.get_text(strip=True)

    return None


def extract_categories_and_tags_from_soup(soup):
    bits = []

    for sel in ["nav.breadcrumb", ".breadcrumbs", "ol.breadcrumb"]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True)
            if txt:
                bits.append(txt)

    for sel in ["a[rel=category]", ".post-categories a", ".cat-links a", ".tag-cloud a", "a.tag"]:
        for a in soup.select(sel):
            txt = a.get_text(" ", strip=True)
            if txt:
                bits.append(txt)

    if not bits:
        return ""
    return " | ".join(sorted(set(bits)))


def classify_niche(context_text):
    if not context_text:
        return "unknown"

    # pad with spaces so patterns like " car " match at edges too
    text = " " + context_text.lower() + " "

    best_niche = "unknown"
    best_score = 0

    for niche, patterns in NICHE_RULES.items():
        score = 0
        for pat in patterns:
            if pat in text:
                score += 1
        if score > best_score:
            best_score = score
            best_niche = niche

    return best_niche


def guess_language_from_path(path):
    segments = [s for s in path.split("/") if s]
    if not segments:
        return ""
    first = segments[0]
    if len(first) in (2, 5) and first.replace("-", "").isalpha():
        return first.lower()
    return ""


def crawl_and_extract(session, target, writer):
    base_url = target["base_url"]
    scope_prefix = target["scope_prefix"]
    netloc = target["netloc"]

    rp = get_robots_parser(session, netloc)

    start_url = base_url + scope_prefix
    if not start_url.endswith("/"):
        start_url += "/"

    to_visit = [start_url]
    visited = set()

    pages_seen = 0
    headlines_written = 0

    print(f"Processing target: {netloc} (scope: {scope_prefix or 'whole domain'})")

    while to_visit and pages_seen < MAX_PAGES_PER_TARGET:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        parsed = urlparse(url)
        if not parsed.netloc.endswith(netloc):
            continue
        if scope_prefix and not parsed.path.startswith(scope_prefix):
            continue
        if is_asset_url(parsed.path):
            continue
        if not rp.can_fetch(USER_AGENT, url):
            continue

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
        except Exception:
            continue

        if resp.status_code != 200 or not is_html_response(resp):
            continue

        pages_seen += 1

        soup = BeautifulSoup(resp.text, "lxml")

        if looks_like_article(parsed.path):
            headline = extract_headline_from_soup(soup)
            if headline:
                cats_tags = extract_categories_and_tags_from_soup(soup)
                context = " | ".join([headline, parsed.path, cats_tags])
                niche = classify_niche(context)
                lang_code = guess_language_from_path(parsed.path)

                writer.writerow({
                    "headline": headline,
                    "language": lang_code,
                    "niche": niche,
                })
                headlines_written += 1

        for a in soup.find_all("a", href=True):
            href = a["href"]
            full = urljoin(url, href)
            p = urlparse(full)
            if not p.netloc.endswith(netloc):
                continue
            if scope_prefix and not p.path.startswith(scope_prefix):
                continue
            if full not in visited and not is_asset_url(p.path):
                to_visit.append(full)

        if pages_seen % 20 == 0:
            print(f"  visited {pages_seen} pages, captured {headlines_written} headlines so far")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    print(f"Finished {netloc} [{scope_prefix or 'root'}]: visited {pages_seen} pages, wrote {headlines_written} headlines")


def main(input_file, output_csv):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    targets = []
    for line in lines:
        t = normalize_target_line(line)
        if t:
            targets.append(t)

    if not targets:
        print("No valid targets found in input file.")
        return

    session = get_session()

    with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
        fieldnames = ["headline", "language", "niche"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for target in targets:
            crawl_and_extract(session, target, writer)

    print(f"Done. Results written to {output_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python headline_scraper_multilang.py targets.txt output.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])