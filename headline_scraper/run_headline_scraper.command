#!/bin/zsh
cd /Users/sampierce/arb_tools/headline_scraper
source ../.venv/bin/activate
python headline_scraper_multilang.py targets.txt headlines_multilang.csv
echo ""
echo "Done. Press any key to close this window."
read -n 1
