import streamlit as st

st.set_page_config(page_title="DPWH Dashboard", layout="wide")

st.sidebar.write("Developed by Elly Olegario")


st.title("DPWH Project Risk and Anomaly Dashboard")
st.markdown("""
Welcome to the **DPWH Project Risk and Anomaly Dashboard**.

This application allows users to:
- check whether a project is anomalous
- review project-specific details
- analyze regional project trends
- assess contractor performance

Use the sidebar to navigate through the available sections.
""")
st.info("Select a page from the sidebar to continue.")
