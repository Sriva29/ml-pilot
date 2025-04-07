# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="ML Assistant", page_icon="🤖", layout="centered")


# Handle authentication from LTI
token = st.experimental_get_query_params().get("token", [None])[0]
if token:
    st.session_state["token"] = token

# Main app interface
st.title("ML Mentor")

# Module navigation with steps
module = st.sidebar.selectbox(
    "Learning Module", 
    ["Data Understanding", "Feature Engineering", "Model Selection", "Evaluation"]
)

# Module-specific UI components
if module == "Data Understanding":
    st.header("Upload and Explore Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        # Call backend API
        response = requests.post(
            "http://localhost:8000/ml/analyze_data",
            files={"file": uploaded_file},
            headers={"Authorization": f"Bearer {st.session_state.get('token')}"}
        )
        
        # Display analysis results
        analysis = response.json()
        st.write(analysis["message"])
        
        # Show data preview
        df = pd.read_csv(uploaded_file)
        st.subheader("Data Preview")
        st.dataframe(df.head())