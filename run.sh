#!/usr/bin/env bash
set -euo pipefail

APP="${1:-}"
PORT_ARG="${2:-}"   # optional custom port like 8616
shift || true
shift || true
EXTRA_ARGS=("$@")   # any extra Streamlit flags get passed through

ROOT="$HOME/arb_tools"

default_port() {
  case "$1" in
    trend)      echo 8616 ;;
    trend_rpc)  echo 8617 ;;
    trend_intl) echo 8618 ;;
    msads)      echo 8620 ;;
    scraper|scrape) echo 8621 ;;
    *) echo 8619 ;;
  esac
}

ensure_free_port() {
  local p="$1"
  local tries=30
  while lsof -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1; do
    p=$((p+1))
    tries=$((tries-1))
    [[ $tries -le 0 ]] && { echo "No free port found" >&2; exit 1; }
  done
  echo "$p"
}

ensure_venv() {
  local dir="$1"
  cd "$dir"
  if [[ ! -d .venv ]]; then
    python3 -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install --upgrade pip wheel >/dev/null
}

install_requirements_if_any() {
  if [[ -f requirements.txt ]]; then
    python -m pip install -r requirements.txt
  fi
}

ensure_min_deps_trend() {
  python - <<'PY' || python -m pip install streamlit pandas requests pyyaml pytrends feedparser
import importlib
for m in ("streamlit","pandas","requests","yaml","pytrends","feedparser"):
    importlib.import_module(m)
PY
}

ensure_min_deps_msads() {
  python - <<'PY' || python -m pip install streamlit pandas requests python-dotenv bingads
import importlib
for m in ("streamlit","pandas","requests","dotenv","bingads"):
    importlib.import_module(m)
PY
}

ensure_min_deps_scraper() {
  python - <<'PY' || python -m pip install streamlit pandas requests beautifulsoup4 pyyaml playwright
import importlib
for m in ("streamlit","pandas","requests","bs4","yaml","playwright"):
    importlib.import_module(m)
PY
  # Install Playwright browsers if not present
  python - <<'PY' || python -m playwright install
try:
    from playwright.sync_api import sync_playwright  # noqa
    ok=True
except Exception:
    ok=False
if not ok:
    raise SystemExit(1)
PY
}

launch_streamlit() {
  local dir="$1"
  local entry="$2"
  local port="$3"

  # Forward a proper port arg even if user passed a raw value
  exec streamlit run "$entry" --server.port "$port" "${EXTRA_ARGS[@]}"
}

usage() {
  echo "Usage:"
  echo "  ./run.sh trend [port] [extra streamlit args]"
  echo "  ./run.sh trend_rpc [port] [args]"
  echo "  ./run.sh trend_intl [port] [args]"
  echo "  ./run.sh msads [port] [args]"
  echo "  ./run.sh scraper [port] [args]"
  echo "Aliases:"
  echo "  scrape == scraper"
}

[[ -z "$APP" ]] && { usage; exit 1; }

PORT="${PORT_ARG:-$(default_port "$APP")}"
PORT="$(ensure_free_port "$PORT")"

case "$APP" in
  trend)
    ensure_venv "$ROOT/trend_scanner"
    install_requirements_if_any || true
    ensure_min_deps_trend
    launch_streamlit "$ROOT/trend_scanner" "app.py" "$PORT"
    ;;

  trend_rpc)
    ensure_venv "$ROOT/trend_scanner"
    install_requirements_if_any || true
    ensure_min_deps_trend
    launch_streamlit "$ROOT/trend_scanner" "app_rpc_only.py" "$PORT"
    ;;

  trend_intl)
    ensure_venv "$ROOT/trend_scanner"
    install_requirements_if_any || true
    ensure_min_deps_trend
    launch_streamlit "$ROOT/trend_scanner" "app_international.py" "$PORT"
    ;;

  msads)
    ensure_venv "$ROOT/MSADS_TEST"
    install_requirements_if_any || true
    ensure_min_deps_msads
    launch_streamlit "$ROOT/MSADS_TEST" "ui_keyword_ideas.py" "$PORT"
    ;;

  scraper|scrape)
    ensure_venv "$ROOT/Scraper_Tool"
    install_requirements_if_any || true
    ensure_min_deps_scraper
    launch_streamlit "$ROOT/Scraper_Tool" "scraper_app.py" "$PORT"
    ;;

  *)
    usage
    exit 1
    ;;
esac