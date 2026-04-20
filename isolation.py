import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Project Anomaly Checker", page_icon="🚩", layout="centered")

st.title("🚩 Project Anomaly Checker")
st.write(
    "Enter a project's details below to check whether it looks anomalous "
    "based on your trained Isolation Forest model."
)

@st.cache_resource
def load_bundle(path: str = "isolation_forest_bundle.pkl"):
    bundle = joblib.load(path)
    return bundle["model"], bundle["preprocessor"]

try:
    model, preprocessor = load_bundle()
except Exception as e:
    st.error(
        "Could not load `isolation_forest_bundle.pkl`. Make sure the file is in the same folder as this app."
    )
    st.exception(e)
    st.stop()

st.subheader("Project Inputs")

with st.form("project_form"):
    budget = st.number_input("Budget", min_value=0.0, value=1000000.0, step=1000.0)
    progress = st.number_input("Progress", min_value=0.0, max_value=100.0, value=50.0, step=1.0)
    latitude = st.number_input("Latitude", value=14.5995, format="%.6f")
    longitude = st.number_input("Longitude", value=120.9842, format="%.6f")
    duration_days = st.number_input("Duration (days)", min_value=0, value=180, step=1)

    region = st.selectbox(
        "Region",
        [
            "Central Office",
            "Cordillera Administrative Region",
            "National Capital Region",
            "Negros Island Region",
            "Region I",
            "Region II",
            "Region III",
            "Region IV-A",
            "Region IV-B",
            "Region V",
            "Region VI",
            "Region VII",
            "Region VIII",
            "Region IX",
            "Region X",
            "Region XI",
            "Region XII",
            "Region XIII",
        ],
    )
    category_grouped = st.selectbox(
        "Category Grouped",
        [
            "Buildings and Facilities",
            "Roads",
            "Flood Control and Drainage",
            "Bridges",
            "OTHER",
            "Water Provision and Storage",
            "GAA 2025 SSP",
            "GAA 2025 OO-1",
            "GAA 2025 OO-2",
            "Bridges, Roads",
        ],
    )
    program_name = st.selectbox(
        "Program Name",
        [
            "Regular Infra",
            "Outside Infra",
        ],
    )

    is_live = st.selectbox("Is Live?", options=[True, False], index=0)
    has_satellite_image = st.selectbox("Has Satellite Image?", options=[True, False], index=1)
    is_completed = st.selectbox("Is Completed?", options=[True, False], index=1)

    submitted = st.form_submit_button("Check Anomaly")

if submitted:
    input_df = pd.DataFrame([
        {
            "budget": budget,
            "progress": progress,
            "latitude": latitude,
            "longitude": longitude,
            "duration_days": duration_days,
            "region": region,
            "category_grouped": category_grouped,
            "programName": program_name,
            "isLive": float(is_live),
            "hasSatelliteImage": float(has_satellite_image),
            "is_completed": float(is_completed),
        }
    ])

    try:
        X_processed = preprocessor.transform(input_df)
        anomaly_label = model.predict(X_processed)[0]
        anomaly_score = model.decision_function(X_processed)[0]
        anomaly_score_scaled = -1 * anomaly_score

        st.subheader("Result")

        if anomaly_label == -1:
            st.error("This project was flagged as anomalous.")
        else:
            st.success("This project was not flagged as anomalous.")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Anomaly Label", "Anomalous" if anomaly_label == -1 else "Normal")
        with col2:
            st.metric("Scaled Anomaly Score", f"{anomaly_score_scaled:.4f}")

        with st.expander("See raw model outputs"):
            st.write(
                {
                    "anomaly_label": int(anomaly_label),
                    "anomaly_score": float(anomaly_score),
                    "anomaly_score_scaled": float(anomaly_score_scaled),
                }
            )

        st.subheader("Submitted Project Data")
        st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed. Check that your saved preprocessor expects the same input columns used in this form.")
        st.exception(e)

st.markdown("---")
st.caption(
    "Expected file: `isolation_forest_bundle.pkl` containing a dictionary with keys `model` and `preprocessor`."
)

st.code(
    """# Example: create the bundle in Colab\n"
    "joblib.dump({\"model\": iso_model, \"preprocessor\": iso_preprocessor}, \"isolation_forest_bundle.pkl\")\n"
    "# Then put that file beside this Streamlit app""",
    language="python",
)
