#!/bin/zsh

CMD_ARGS="${@:2}"

case "$1" in
  trend)
    cd ~/arb_tools/trend_scanner || exit 1
    source .venv/bin/activate
    streamlit run app.py $CMD_ARGS
    ;;
  trend_rpc)
    cd ~/arb_tools/trend_scanner || exit 1
    source .venv/bin/activate
    streamlit run app_rpc_only.py $CMD_ARGS
    ;;
  msads)
    cd ~/arb_tools/MSADS_TEST || exit 1
    source .venv/bin/activate
    streamlit run ui_keyword_ideas.py $CMD_ARGS
    ;;
  scrape)
    cd ~/arb_tools/Scraper_Tool || exit 1
    source .venv/bin/activate
    streamlit run scraper_app.py $CMD_ARGS
    ;;
  *)
    echo "Usage: ./run.sh trend|trend_rpc|msads|scrape [streamlit args]"
    ;;
esac