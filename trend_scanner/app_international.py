import os
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(__file__)
st.set_page_config(page_title="Arb LaunchOps — International",layout="wide")

st.title("Arb LaunchOps — International")
st.caption("Restored placeholder. Add real localization rules later.")

lang = st.selectbox("Language",["US English","German","Spanish","French","Portuguese"])
uploaded = st.file_uploader("Upload base keywords",type=["csv","xlsx"])

if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    st.write(df.head())
else:
    st.info("Upload a keyword sheet to start")
