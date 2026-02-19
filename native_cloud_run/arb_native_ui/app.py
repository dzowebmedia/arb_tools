import os
import pandas as pd
import streamlit as st
from google.cloud import storage

st.set_page_config(page_title="Arb Native UI", layout="wide")

BUCKET = os.environ.get("ARB_BUCKET", "arb-native-ad-scan-data")
LATEST_ADS = os.environ.get("ARB_LATEST_ADS", "native_enriched/latest/ads_latest.csv")
LATEST_ROLLUP = os.environ.get("ARB_LATEST_ROLLUP", "native_enriched/latest/rollup_latest.csv")
SUM_7 = os.environ.get("ARB_SUM_7", "summaries/last_7_days.csv")
SUM_30 = os.environ.get("ARB_SUM_30", "summaries/last_30_days.csv")

@st.cache_data(ttl=120)
def load_csv_from_gcs(bucket: str, blob: str) -> pd.DataFrame:
    try:
        client = storage.Client()
        obj = client.bucket(bucket).blob(blob)

        # One call instead of exists()+download(): just try download and handle NotFound.
        data = obj.download_as_bytes()
        return pd.read_csv(pd.io.common.BytesIO(data))

    except Exception as e:
        msg = str(e)

        # Friendly auth-expired handling (ADC reauth)
        if "Reauthentication is needed" in msg or "RefreshError" in msg:
            st.error(
                "Google auth expired for this UI.\n\n"
                "In Terminal - cd ~/native_cloud_run/arb_native_ui - "
                " - source .venv/bin/activate - " 
                "gcloud auth application-default login\n\n"
                " - Then click 'Refresh Now' in the sidebar."
            )
            return pd.DataFrame()

        # Missing file handling
        if "No such object" in msg or "404" in msg or "NotFound" in msg:
            return pd.DataFrame()

        # Anything else: show a short error but keep the app alive
        st.error(f"GCS read error for gs://{bucket}/{blob}\n\n{e}")
        return pd.DataFrame()

st.title("Native + Arbitrage Scanner UI")

with st.sidebar:
    st.header("Data Source")
    st.write(f"Bucket: **{BUCKET}**")
    if st.button("Refresh Now"):
        st.cache_data.clear()

tabs = st.tabs(["Latest Snapshot", "7 Days Rollup", "30 Days Rollup"])

with tabs[0]:
    st.subheader("Latest Rollup (Headlines x Domain x Network)")
    roll = load_csv_from_gcs(BUCKET, LATEST_ROLLUP)
    if roll.empty:
        st.warning("Latest rollup not found.")
    else:
        st.dataframe(roll, use_container_width=True, height=500)

    st.subheader("Latest Ads (Raw Enriched Rows)")
    ads = load_csv_from_gcs(BUCKET, LATEST_ADS)
    if ads.empty:
        st.warning("Latest ads file not found.")
    else:
        st.dataframe(ads, use_container_width=True, height=500)

with tabs[1]:
    st.subheader("Last 7 Days Rollup")
    df7 = load_csv_from_gcs(BUCKET, SUM_7)
    if df7.empty:
        st.warning("7-day summary not found.")
    else:
        st.dataframe(df7, use_container_width=True, height=650)

with tabs[2]:
    st.subheader("Last 30 Days Rollup")
    df30 = load_csv_from_gcs(BUCKET, SUM_30)
    if df30.empty:
        st.warning("30-day summary not found.")
    else:
        st.dataframe(df30, use_container_width=True, height=650)