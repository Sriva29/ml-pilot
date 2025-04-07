import streamlit as st
import pandas as pd
from urllib.parse import urlparse, parse_qs

st.set_page_config(page_title="ML Assistant", page_icon="🤖", layout="centered")

learner_id = st.query_params.get("learner_id", "anonymous")

# Title and intro
st.title(f"🧠 Welcome to Your ML Learning Assistant, {learner_id}!")

st.markdown("""
Welcome, future ML wizard!  
Upload your dataset below, and I’ll guide you step by step through the entire machine learning workflow — no stress, just clarity.
""")

# Upload CSV
uploaded_file = st.file_uploader("📁 Upload a CSV file to get started", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.subheader("🔍 Here's a quick preview:")
        st.dataframe(df.head())

        st.info("Next step: You’ll be asked questions about your dataset and what you want to achieve. Stay tuned!")
    except Exception as e:
        st.error(f"❌ Something went wrong: {e}")
