from yahoo_scraper import scan_yahoo_keywords, write_csv

keywords = [
    "2026 hybrid suvs",
    "solar panel contractor",
    "crossover suv deals",
    "cd rates"
]

results = scan_yahoo_keywords(keywords)
write_csv(results)