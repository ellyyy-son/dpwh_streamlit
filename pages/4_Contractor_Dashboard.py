import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Contractor Specific Dashboard", layout="wide")

st.title("Contractor Specific Dashboard")
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
# Load data
# ----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("dataset/final_results_exploded.csv")

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
    "prob_completed", "prob_not_completed", "risk_score", "is_anomaly",
    "is_completed"
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

if "contractor" not in df.columns:
    st.error("The dataset does not contain a 'contractor' column.")
    st.stop()

contractor_options = sorted(df["contractor"].dropna().astype(str).unique().tolist())
selected_contractor = st.sidebar.selectbox("Select Contractor", contractor_options)

filtered_df = filtered_df[filtered_df["contractor"].astype(str) == selected_contractor]

if "region" in filtered_df.columns:
    region_options = ["All"] + sorted(filtered_df["region"].dropna().astype(str).unique().tolist())
    selected_region = st.sidebar.selectbox("Region", region_options)
    if selected_region != "All":
        filtered_df = filtered_df[filtered_df["region"].astype(str) == selected_region]

if "province" in filtered_df.columns:
    province_options = ["All"] + sorted(filtered_df["province"].dropna().astype(str).unique().tolist())
    selected_province = st.sidebar.selectbox("Province", province_options)
    if selected_province != "All":
        filtered_df = filtered_df[filtered_df["province"].astype(str) == selected_province]

if "status" in filtered_df.columns:
    status_options = ["All"] + sorted(filtered_df["status"].dropna().astype(str).unique().tolist())
    selected_status = st.sidebar.selectbox("Status", status_options)
    if selected_status != "All":
        filtered_df = filtered_df[filtered_df["status"].astype(str) == selected_status]

if filtered_df.empty:
    st.warning("No projects found for the selected filters.")
    st.stop()

# ----------------------------
# Contractor summary values
# ----------------------------
total_projects = len(filtered_df)
total_budget = filtered_df["budget"].sum() if "budget" in filtered_df.columns else np.nan
avg_progress = filtered_df["progress"].mean() if "progress" in filtered_df.columns else np.nan
avg_risk = filtered_df["risk_score"].mean() if "risk_score" in filtered_df.columns else np.nan
completion_rate = filtered_df["is_completed"].mean() if "is_completed" in filtered_df.columns else np.nan
anomaly_count = filtered_df["is_anomaly"].sum() if "is_anomaly" in filtered_df.columns else np.nan
avg_prob_completed = filtered_df["prob_completed"].mean() if "prob_completed" in filtered_df.columns else np.nan
avg_prob_not_completed = filtered_df["prob_not_completed"].mean() if "prob_not_completed" in filtered_df.columns else np.nan

risk_class = get_risk_class(avg_risk)

# ----------------------------
# Header
# ----------------------------
st.subheader(f"Contractor: {selected_contractor}")
st.caption(f"{total_projects:,} project(s) for the selected contractor")

# ----------------------------
# Top metrics
# ----------------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    format_metric("Total Projects", f"{total_projects:,}", "metric-green")

with col2:
    format_metric(
        "Total Budget",
        f"₱ {total_budget:,.2f}" if pd.notna(total_budget) else "N/A",
        "metric-green"
    )

with col3:
    format_metric(
        "Average Progress",
        f"{avg_progress:.2f}%" if pd.notna(avg_progress) else "N/A",
        "metric-green"
    )

with col4:
    format_metric(
        "Completion Rate",
        f"{completion_rate:.2%}" if pd.notna(completion_rate) else "N/A",
        "metric-green" if pd.notna(completion_rate) and completion_rate >= 0.7 else "metric-orange"
    )

st.markdown("### Risk & Anomaly Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    format_metric(
        "Average Risk Score",
        f"{avg_risk:.2f}%" if pd.notna(avg_risk) else "N/A",
        risk_class
    )

with col2:
    format_metric(
        "Risk Level",
        get_risk_level(avg_risk),
        risk_class
    )

with col3:
    format_metric(
        "Anomaly Count",
        f"{int(anomaly_count):,}" if pd.notna(anomaly_count) else "N/A",
        "metric-red" if pd.notna(anomaly_count) and anomaly_count > 0 else "metric-green"
    )

with col4:
    format_metric(
        "Avg Prob Not Completed",
        f"{avg_prob_not_completed:.2%}" if pd.notna(avg_prob_not_completed) else "N/A",
        risk_class
    )

# ----------------------------
# Region breakdown
# ----------------------------
st.markdown("### Region Breakdown")
if "region" in filtered_df.columns:
    region_summary = (
        filtered_df.groupby("region", dropna=False)
        .agg(
            project_count=("contractId", "count"),
            total_budget=("budget", "sum"),
            avg_risk_score=("risk_score", "mean"),
            anomaly_count=("is_anomaly", "sum"),
            avg_progress=("progress", "mean"),
        )
        .reset_index()
        .sort_values(["project_count", "avg_risk_score"], ascending=[False, False])
    )
    st.dataframe(region_summary, use_container_width=True, hide_index=True)

# ----------------------------
# Category breakdown
# ----------------------------
st.markdown("### Category Breakdown")
if "category_grouped" in filtered_df.columns:
    category_summary = (
        filtered_df.groupby("category_grouped", dropna=False)
        .agg(
            project_count=("contractId", "count"),
            total_budget=("budget", "sum"),
            avg_risk_score=("risk_score", "mean"),
            anomaly_count=("is_anomaly", "sum"),
            avg_progress=("progress", "mean"),
        )
        .reset_index()
        .sort_values(["project_count", "avg_risk_score"], ascending=[False, False])
    )
    st.dataframe(category_summary, use_container_width=True, hide_index=True)

# ----------------------------
# Project list
# ----------------------------
st.markdown("### Contractor Projects")
project_cols = [
    col for col in [
        "contractId", "description", "region", "province", "status",
        "budget", "progress", "risk_score", "risk_level",
        "prob_completed", "prob_not_completed",
        "is_anomaly", "anomaly_score_scaled"
    ] if col in filtered_df.columns
]

project_list = (
    filtered_df[project_cols]
    .sort_values(["risk_score", "anomaly_score_scaled"], ascending=[False, False])
)

st.dataframe(project_list, use_container_width=True, hide_index=True)

# ----------------------------
# High-risk projects
# ----------------------------
st.markdown("### Top High-Risk Projects")
high_risk_cols = [
    col for col in [
        "contractId", "description", "region", "province", "status",
        "budget", "progress", "risk_score", "risk_level",
        "prob_not_completed", "is_anomaly", "anomaly_score_scaled",
        "latitude", "longitude"
    ] if col in filtered_df.columns
]

if "risk_score" in filtered_df.columns:
    high_risk_projects = (
        filtered_df[filtered_df["risk_score"] >= 60][high_risk_cols]
        .sort_values(["risk_score", "anomaly_score_scaled"], ascending=[False, False])
        .head(20)
    )
else:
    high_risk_projects = pd.DataFrame(columns=high_risk_cols)

st.dataframe(high_risk_projects, use_container_width=True, hide_index=True)

st.markdown("#### High-Risk Projects Map")
if {"latitude", "longitude"}.issubset(high_risk_projects.columns):
    high_risk_map = high_risk_projects.copy()
    high_risk_map["latitude"] = pd.to_numeric(high_risk_map["latitude"], errors="coerce")
    high_risk_map["longitude"] = pd.to_numeric(high_risk_map["longitude"], errors="coerce")
    high_risk_map = high_risk_map.dropna(subset=["latitude", "longitude"])

    if not high_risk_map.empty:
        st.map(
            high_risk_map[["latitude", "longitude"]],
            latitude="latitude",
            longitude="longitude"
        )
    else:
        st.info("No valid coordinates available for high-risk projects.")
else:
    st.info("Latitude/longitude columns are not available for high-risk projects.")

# ----------------------------
# Anomalous projects
# ----------------------------
if "is_anomaly" in filtered_df.columns:
    st.markdown("### Anomalous Projects")
    anomaly_projects = (
        filtered_df[filtered_df["is_anomaly"] == 1][high_risk_cols]
        .sort_values("anomaly_score_scaled", ascending=False)
        .head(20)
    )

    st.dataframe(anomaly_projects, use_container_width=True, hide_index=True)

    st.markdown("#### Anomalous Projects Map")
    if {"latitude", "longitude"}.issubset(anomaly_projects.columns):
        anomaly_map = anomaly_projects.copy()
        anomaly_map["latitude"] = pd.to_numeric(anomaly_map["latitude"], errors="coerce")
        anomaly_map["longitude"] = pd.to_numeric(anomaly_map["longitude"], errors="coerce")
        anomaly_map = anomaly_map.dropna(subset=["latitude", "longitude"])

        if not anomaly_map.empty:
            st.map(
                anomaly_map[["latitude", "longitude"]],
                latitude="latitude",
                longitude="longitude"
            )
        else:
            st.info("No valid coordinates available for anomalous projects.")
    else:
        st.info("Latitude/longitude columns are not available for anomalous projects.")

# ----------------------------
# Full contractor dataset
# ----------------------------
st.markdown("### Full Contractor Dataset")
st.dataframe(filtered_df, use_container_width=True, hide_index=True)