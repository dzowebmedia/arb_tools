# streamlit_app.py

import streamlit as st
import sqlite3
import pandas as pd
from collections import Counter

DB_PATH = "creatives.db"

st.set_page_config(page_title="Creative Monitor", layout="wide")

@st.cache_data
def load_creatives():
    with sqlite3.connect(DB_PATH) as conn:
        creatives = pd.read_sql_query("SELECT * FROM creatives", conn)
        appearances = pd.read_sql_query("SELECT * FROM appearances", conn)
    return creatives, appearances

def show_top_ctas(creatives):
    cta_series = creatives["cta_text"].dropna().str.strip()
    cta_series = cta_series[cta_series != ""]
    ctas = Counter(cta_series)
    top_ctas = ctas.most_common(15)

    if top_ctas:
        st.subheader("📈 Top CTA Phrases")
        df = pd.DataFrame(top_ctas, columns=["CTA Phrase", "Count"])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No CTA phrases detected in the current dataset.")

def show_top_creatives(creatives):
    st.subheader("🔥 Most Recurring Creatives")
    recurring = creatives[creatives["seen_count"] > 1].sort_values("seen_count", ascending=False)
    if not recurring.empty:
        st.dataframe(recurring[[
            "headline", "body", "cta_text", "seen_count", "first_seen", "last_seen"
        ]], use_container_width=True)
    else:
        st.info("No recurring creatives yet — keep scanning.")

def show_novel_creatives(creatives):
    st.subheader("🧪 Novel / One-Time Creatives")
    novel = creatives[creatives["seen_count"] == 1]
    if not novel.empty:
        st.dataframe(novel[[
            "headline", "body", "cta_text", "first_seen"
        ]], use_container_width=True)
    else:
        st.info("No novel creatives found.")

def filter_by_keyword(appearances, creatives):
    keywords = appearances["keyword"].unique().tolist()
    keyword = st.sidebar.selectbox("Filter by keyword", ["(All)"] + sorted(keywords))

    if keyword != "(All)":
        filtered_hashes = appearances[appearances["keyword"] == keyword]["hash_id"].unique()
        creatives = creatives[creatives["hash_id"].isin(filtered_hashes)]

    return creatives

# === Streamlit Layout ===

st.title("🔍 Ad Creative Monitoring Dashboard")

creatives, appearances = load_creatives()
filtered = filter_by_keyword(appearances, creatives)

col1, col2 = st.columns(2)
with col1:
    show_top_ctas(filtered)
with col2:
    show_top_creatives(filtered)

st.divider()
show_novel_creatives(filtered)