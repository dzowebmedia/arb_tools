#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python export_history_agg.py --in app.py/history.parquet app_rpc_only.py/history.parquet --outdir exports --prefix history_full
