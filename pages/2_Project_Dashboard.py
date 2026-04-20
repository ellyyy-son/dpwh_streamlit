from datasets import load_dataset
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Project Specific Dashboard", layout="wide")

st.title("Project Specific Dashboard")
st.sidebar.write("Developed by Elly Olegario")

# ----------------------------
# CSS for colored text metrics
# ----------------------------
st.markdown("""
<style>
.metric-green {
    color: #16a34a !important;
    font-weight: 700;
    font-size: 28px;
}
.metric-orange {
    color: #f59e0b !important;
    font-weight: 700;
    font-size: 28px;
}
.metric-red {
    color: #dc2626 !important;
    font-weight: 700;
    font-size: 28px;
}
.metric-label {
    font-size: 14px;
    color: #6b7280;
    margin-bottom: 2px;
}
.metric-card {
    padding: 8px 0 16px 0;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load your data
# ----------------------------
@st.cache_data
def load_data():
    data_files = {
        "main": "hf://datasets/ell-ws/dpwh-tracking/final_results.csv",
    }

    dataset = load_dataset(
        "csv",
        data_files=data_files,
        token=st.secrets["HF_TOKEN"]
    )

    return dataset["main"].to_pandas()

df = load_data()

# ----------------------------
# Expected columns
# ----------------------------
expected_columns = [
    "contractId", "description", "category", "componentCategories", "status",
    "budget", "progress", "location", "contractor", "startDate",
    "completionDate", "infraYear", "programName", "sourceOfFunds", "isLive",
    "latitude", "longitude", "reportCount", "hasSatelliteImage",
    "effective_end_date", "duration_days", "is_completed", "region",
    "province", "category_grouped", "anomaly_label", "anomaly_score",
    "anomaly_score_scaled", "is_anomaly", "prob_completed",
    "prob_not_completed", "risk_score", "risk_level"
]

missing_cols = [col for col in expected_columns if col not in df.columns]
if missing_cols:
    st.warning(f"Missing columns in dataset: {missing_cols}")

available_columns = [col for col in expected_columns if col in df.columns]
df = df[available_columns].copy()

# ----------------------------
# Data preparation
# ----------------------------
date_cols = ["startDate", "completionDate", "effective_end_date"]
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")

numeric_cols = [
    "budget", "progress", "latitude", "longitude", "reportCount",
    "duration_days", "anomaly_score", "anomaly_score_scaled",
    "prob_completed", "prob_not_completed", "risk_score"
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# ----------------------------
# Helper functions
# ----------------------------
def get_risk_class(score):
    if pd.isna(score):
        return "metric-label"
    if score < 30:
        return "metric-green"
    elif score < 60:
        return "metric-orange"
    return "metric-red"

def get_risk_level(score):
    if pd.isna(score):
        return "N/A"
    if score < 30:
        return "Low"
    elif score < 60:
        return "Medium"
    return "High"

def get_anomaly_class(is_anomaly):
    if pd.isna(is_anomaly):
        return "metric-label"
    return "metric-red" if int(is_anomaly) == 1 else "metric-green"

def format_metric(label, value, css_class):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="{css_class}">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# Sidebar filters
# ----------------------------
st.sidebar.header("Filters")

filtered_df = df.copy()

if "region" in df.columns:
    region_options = ["All"] + sorted(df["region"].dropna().astype(str).unique().tolist())
    selected_region = st.sidebar.selectbox("Region", region_options)
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["region"].astype(str) == selected_region]

if "province" in df.columns:
    province_options = ["All"] + sorted(filtered_df["province"].dropna().astype(str).unique().tolist())
    selected_province = st.sidebar.selectbox("Province", province_options)
    if selected_province != "All":
        filtered_df = filtered_df[filtered_df["province"].astype(str) == selected_province]

if "status" in df.columns:
    status_options = ["All"] + sorted(filtered_df["status"].dropna().astype(str).unique().tolist())
    selected_status = st.sidebar.selectbox("Status", status_options)
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["status"].astype(str) == selected_status]

if "contractId" not in filtered_df.columns or filtered_df.empty:
    st.error("No data available for selection.")
    st.stop()

project_ids = filtered_df["contractId"].dropna().astype(str).unique().tolist()
selected_contract = st.sidebar.selectbox("Select Contract ID", sorted(project_ids))

project_df = filtered_df[filtered_df["contractId"].astype(str) == selected_contract]

if project_df.empty:
    st.warning("No project found.")
    st.stop()

project = project_df.iloc[0]

# ----------------------------
# Header
# ----------------------------
st.subheader(f"Contract ID: {project.get('contractId', 'N/A')}")
st.caption(project.get("description", "No description available"))

progress_value = project.get("progress", np.nan)
budget_value = project.get("budget", np.nan)
risk_score_header = project.get("risk_score", np.nan)
anomaly_value = project.get("anomaly_score", np.nan)
status_value = project.get("status", "N/A")

risk_level_header = get_risk_level(risk_score_header)
risk_class_header = get_risk_class(risk_score_header)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    format_metric("Status", status_value, "metric-green")

with col2:
    format_metric(
        "Progress",
        f"{progress_value:.1f}%" if pd.notna(progress_value) else "N/A",
        "metric-green"
    )

with col3:
    format_metric(
        "Budget",
        f"₱ {budget_value:,.2f}" if pd.notna(budget_value) else "N/A",
        "metric-green"
    )

with col4:
    format_metric(
        "Risk Level",
        risk_level_header,
        risk_class_header
    )

with col5:
    anomaly_header_class = (
        "metric-red" if pd.notna(anomaly_value) and anomaly_value > 0.7
        else "metric-orange" if pd.notna(anomaly_value)
        else "metric-label"
    )
    format_metric(
        "Anomaly Score",
        f"{anomaly_value:.3f}" if pd.notna(anomaly_value) else "N/A",
        anomaly_header_class
    )

if "progress" in project.index and pd.notna(project["progress"]):
    st.progress(min(max(float(project["progress"]) / 100, 0), 1))

# ----------------------------
# Project details
# ----------------------------
st.markdown("### Project Details")
details_data = {
    "Description": project.get("description"),
    "Category": project.get("category"),
    "Component Categories": project.get("componentCategories"),
    "Category Grouped": project.get("category_grouped"),
    "Location": project.get("location"),
    "Region": project.get("region"),
    "Province": project.get("province"),
    "Contractor": project.get("contractor"),
    "Program Name": project.get("programName"),
    "Source of Funds": project.get("sourceOfFunds"),
    "Infrastructure Year": project.get("infraYear"),
    "Start Date": project.get("startDate"),
    "Completion Date": project.get("completionDate"),
    "Effective End Date": project.get("effective_end_date"),
    "Duration (Days)": project.get("duration_days"),
    "Is Live": project.get("isLive"),
    "Is Completed": project.get("is_completed"),
    "Has Satellite Image": project.get("hasSatelliteImage"),
    "Report Count": project.get("reportCount"),
}

details_df = pd.DataFrame(
    {"Field": list(details_data.keys()), "Value": list(details_data.values())}
)
st.dataframe(details_df, use_container_width=True, hide_index=True)

# ----------------------------
# Risk & Anomaly
# ----------------------------
st.markdown("### Risk & Anomaly Overview")

risk_score = project.get("risk_score", np.nan)
prob_completed = project.get("prob_completed", np.nan)
prob_not_completed = project.get("prob_not_completed", np.nan)
anomaly_label = project.get("anomaly_label", np.nan)
is_anomaly = project.get("is_anomaly", np.nan)
anomaly_score = project.get("anomaly_score", np.nan)
anomaly_score_scaled = project.get("anomaly_score_scaled", np.nan)

risk_class = get_risk_class(risk_score)
anomaly_class = get_anomaly_class(is_anomaly)

col1, col2 = st.columns(2)

with col1:
    format_metric(
        "Risk Score",
        f"{risk_score:.2f}%" if pd.notna(risk_score) else "N/A",
        risk_class
    )

with col2:
    format_metric(
        "Risk Level",
        get_risk_level(risk_score),
        risk_class
    )

col4, col5, col6 = st.columns(3)

with col4:
    format_metric(
        "Anomaly Label",
        f"{int(anomaly_label)}" if pd.notna(anomaly_label) else "N/A",
        anomaly_class
    )

with col5:
    format_metric(
        "Is Anomaly",
        "Yes" if pd.notna(is_anomaly) and int(is_anomaly) == 1 else "No" if pd.notna(is_anomaly) else "N/A",
        anomaly_class
    )

with col6:
    format_metric(
        "Anomaly Score",
        f"{anomaly_score:.3f}" if pd.notna(anomaly_score) else "N/A",
        anomaly_class
    )

# ----------------------------
# Project Location
# ----------------------------
st.markdown("### Project Location")

lat = project.get("latitude", np.nan)
lon = project.get("longitude", np.nan)

if pd.notna(lat) and pd.notna(lon):
    map_df = pd.DataFrame({"latitude": [lat], "longitude": [lon]})
    st.map(map_df, latitude="latitude", longitude="longitude")
else:
    st.info("No latitude/longitude available for this project.")

# ----------------------------
# Full record
# ----------------------------
st.markdown("### Full Project Record")
st.dataframe(project_df, use_container_width=True, hide_index=True)