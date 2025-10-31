from native_scraper import scan_native_source, write_csv, load_sources

all_results = []

for entry in load_sources():
    url = entry["url"]
    name = entry["name"]
    try:
        results = scan_native_source(url, name)
        all_results.extend(results)
    except Exception as e:
        print(f"[!] Error scanning {name}: {e}")

write_csv(all_results)