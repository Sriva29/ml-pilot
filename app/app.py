import streamlit as st
import pandas as pd
import os

# Set page config
st.set_page_config(page_title="ML Coach", page_icon="🧠", layout="centered")

# Session state setup
if "project_name" not in st.session_state:
    st.session_state["project_name"] = None

if "dataset" not in st.session_state:
    st.session_state["dataset"] = None

# --- UI Starts ---
st.title("🧠 ML Coach - Let’s Prep That Data!")

# Step 1: Ask about the project
if not st.session_state["project_name"]:
    st.chat_message("assistant").write("Hi there! 👋 What ML project are you tackling today?")
    project_name = st.text_input("Your ML project:", placeholder="e.g., Predicting customer churn")

    if project_name:
        st.session_state["project_name"] = project_name
        st.success(f"Great! You’re working on: **{project_name}**")
        st.balloons()

# Step 2: Ask for CSV upload
if st.session_state["project_name"]:
    st.markdown("### 📂 Now upload your dataset (.csv):")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["dataset"] = df
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(df.head())

        # Save file to /data/uploaded if needed later
        os.makedirs("data/uploaded", exist_ok=True)
        with open(f"data/uploaded/{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
