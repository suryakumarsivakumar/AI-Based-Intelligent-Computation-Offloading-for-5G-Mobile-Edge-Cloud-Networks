import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI 5G Offloading",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "offloading_model_v1.pkl"
FEATURES_PATH = BASE_DIR / "offloading_features_v1.pkl"


# ============================================================
# TITLE
# ============================================================

st.title("📡 AI-Based Intelligent Computation Offloading")

st.caption(
    "AI-powered Mobile–Edge–Cloud computation offloading using "
    "5G network, device, task and infrastructure parameters."
)

st.divider()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, features


# ============================================================
# LOAD SHAP EXPLAINER
# ============================================================

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)


model, features = load_model()
explainer = load_explainer(model)


# ============================================================
# CLASS MAPPING
# ============================================================

destination_map = {
    0: "Mobile",
    1: "Edge",
    2: "Cloud"
}


# ============================================================
# INPUT PARAMETERS
# ============================================================

st.header("⚙️ Input Parameters")

col1, col2, col3 = st.columns(3)


# ============================================================
# 5G NETWORK
# ============================================================

with col1:

    st.subheader("📶 5G Network")

    cqi = st.slider(
        "CQI",
        min_value=0,
        max_value=15,
        value=10
    )

    dl_mcs = st.slider(
        "DL MCS",
        min_value=0,
        max_value=28,
        value=17
    )

    ul_mcs = st.slider(
        "UL MCS",
        min_value=0,
        max_value=28,
        value=20
    )

    dl_brate = st.number_input(
        "DL Bitrate",
        min_value=0.0,
        value=58.2
    )

    ul_brate = st.number_input(
        "UL Bitrate",
        min_value=0.0,
        value=98.0
    )

    dl_error = st.number_input(
        "DL Error",
        min_value=0.0,
        value=0.0
    )

    ul_error = st.number_input(
        "UL Error",
        min_value=0.0,
        value=0.0
    )

    snr = st.number_input(
        "SNR",
        value=15.0
    )

    rsrp = st.number_input(
        "RSRP",
        value=-95.0
    )

    crc_delay = st.number_input(
        "CRC Delay",
        min_value=0.0,
        value=2.0
    )

    harq_delay = st.number_input(
        "HARQ Delay",
        min_value=0.0,
        value=2.0
    )

    network_quality = st.selectbox(
        "Network Quality",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Poor",
            1: "Medium",
            2: "Good"
        }[x],
        index=1
    )


# ============================================================
# DEVICE
# ============================================================

with col2:

    st.subheader("📱 Device")

    battery_level = st.slider(
        "Battery Level (%)",
        min_value=0,
        max_value=100,
        value=45
    )

    cpu_load = st.slider(
        "CPU Load (%)",
        min_value=0,
        max_value=100,
        value=70
    )

    ram_usage = st.slider(
        "RAM Usage (%)",
        min_value=0,
        max_value=100,
        value=65
    )

    device_temperature = st.number_input(
        "Device Temperature (°C)",
        value=40.0
    )


# ============================================================
# TASK
# ============================================================

with col2:

    st.subheader("🧠 Task")

    task_size_mb = st.number_input(
        "Task Size (MB)",
        min_value=1.0,
        value=500.0
    )

    task_complexity = st.slider(
        "Task Complexity",
        min_value=1,
        max_value=10,
        value=6
    )

    latency_requirement_ms = st.number_input(
        "Latency Requirement (ms)",
        min_value=1.0,
        value=50.0
    )

    bandwidth_requirement = st.number_input(
        "Bandwidth Requirement",
        min_value=0.0,
        value=40.0
    )


# ============================================================
# INFRASTRUCTURE
# ============================================================

with col3:

    st.subheader("☁️ Infrastructure")

    mobility_speed = st.number_input(
        "Mobility Speed",
        min_value=0.0,
        value=20.0
    )

    edge_distance_km = st.number_input(
        "Edge Distance (km)",
        min_value=0.0,
        value=2.0
    )

    cloud_distance_km = st.number_input(
        "Cloud Distance (km)",
        min_value=0.0,
        value=300.0
    )


# ============================================================
# CREATE INPUT DATA
# ============================================================

input_data = {
    "cqi": cqi,
    "dl_mcs": dl_mcs,
    "ul_mcs": ul_mcs,
    "dl_brate": dl_brate,
    "ul_brate": ul_brate,
    "dl_error": dl_error,
    "ul_error": ul_error,
    "snr": snr,
    "rsrp": rsrp,
    "crc_delay": crc_delay,
    "harq_delay": harq_delay,
    "battery_level": battery_level,
    "cpu_load": cpu_load,
    "ram_usage": ram_usage,
    "device_temperature": device_temperature,
    "task_size_mb": task_size_mb,
    "task_complexity": task_complexity,
    "latency_requirement_ms": latency_requirement_ms,
    "bandwidth_requirement": bandwidth_requirement,
    "mobility_speed": mobility_speed,
    "edge_distance_km": edge_distance_km,
    "cloud_distance_km": cloud_distance_km,
    "network_quality": network_quality
}


# ============================================================
# CREATE DATAFRAME
# ============================================================

input_df = pd.DataFrame([input_data])

# Force exact training feature order
input_df = input_df[features]


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

predict_button = st.button(
    "🚀 Predict Offloading Decision",
    type="primary",
    use_container_width=True
)


# ============================================================
# RUN PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # MODEL PREDICTION
    # --------------------------------------------------------

    prediction = int(
        model.predict(input_df)[0]
    )

    probabilities = model.predict_proba(
        input_df
    )[0]

    destination = destination_map[prediction]

    confidence = probabilities[prediction] * 100


    # ========================================================
    # AI OFFLOADING DECISION
    # ========================================================

    st.divider()

    st.header("🤖 AI Offloading Decision")

    result_col1, result_col2 = st.columns(2)


    # ========================================================
    # RECOMMENDED LOCATION
    # ========================================================

    with result_col1:

        st.subheader(
            "Recommended Execution Location"
        )

        if destination == "Mobile":

            st.success(
                "📱 MOBILE"
            )

        elif destination == "Edge":

            st.info(
                "🖥️ EDGE"
            )

        else:

            st.warning(
                "☁️ CLOUD"
            )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )


    # ========================================================
    # PREDICTION PROBABILITY
    # ========================================================

    with result_col2:

        st.subheader(
            "Prediction Probability"
        )

        probability_df = pd.DataFrame(
            {
                "Destination": [
                    "Mobile",
                    "Edge",
                    "Cloud"
                ],
                "Probability (%)": [
                    probabilities[0] * 100,
                    probabilities[1] * 100,
                    probabilities[2] * 100
                ]
            }
        )

        st.bar_chart(
            probability_df.set_index(
                "Destination"
            )
        )


    # ========================================================
    # DESTINATION PROBABILITIES
    # ========================================================

    st.subheader(
        "📊 Destination Probabilities"
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.metric(
            "📱 Mobile",
            f"{probabilities[0] * 100:.2f}%"
        )

    with p2:

        st.metric(
            "🖥️ Edge",
            f"{probabilities[1] * 100:.2f}%"
        )

    with p3:

        st.metric(
            "☁️ Cloud",
            f"{probabilities[2] * 100:.2f}%"
        )


    # ========================================================
    # SHAP EXPLANATION
    # ========================================================

    st.divider()

    st.header(
        "🔍 Explainable AI — SHAP"
    )

    st.write(
        f"The AI selected **{destination}**. "
        "SHAP explains which input features influenced "
        "this prediction."
    )


    # ========================================================
    # CALCULATE SHAP
    # ========================================================

    try:

        shap_values = explainer.shap_values(
            input_df,
            check_additivity=False
        )

    except Exception as e:

        st.error(
            f"SHAP calculation failed: {e}"
        )

        st.stop()


    # ========================================================
    # CONVERT SHAP OUTPUT
    # ========================================================

    shap_array = np.asarray(
        shap_values
    )


    # ========================================================
    # HANDLE YOUR SHAP FORMAT
    #
    # Your test_shap.py produced:
    #
    # (1, 23, 3)
    #
    # 1  = sample
    # 23 = features
    # 3  = classes
    #
    # ========================================================

    if shap_array.ndim == 3:

        selected_shap = shap_array[
            0,
            :,
            prediction
        ]

    elif shap_array.ndim == 2:

        selected_shap = shap_array[
            0
        ]

    elif isinstance(
        shap_values,
        list
    ):

        selected_shap = np.asarray(
            shap_values[prediction]
        )[0]

    else:

        st.error(
            "Unexpected SHAP output shape: "
            f"{shap_array.shape}"
        )

        st.stop()


    # ========================================================
    # VERIFY SHAP FEATURE COUNT
    # ========================================================

    if len(selected_shap) != len(features):

        st.error(
            "SHAP feature count does not match "
            "the model feature count."
        )

        st.write(
            "Model features:",
            len(features)
        )

        st.write(
            "SHAP values:",
            len(selected_shap)
        )

        st.stop()


    # ========================================================
    # CREATE SHAP DATAFRAME
    # ========================================================

    shap_df = pd.DataFrame(
        {
            "Feature": features,
            "Value": input_df.iloc[0].values,
            "SHAP Value": selected_shap
        }
    )


    # ========================================================
    # ABSOLUTE SHAP
    # ========================================================

    shap_df["Absolute SHAP"] = (
        shap_df["SHAP Value"].abs()
    )


    # Sort by importance
    shap_df = shap_df.sort_values(
        "Absolute SHAP",
        ascending=False
    )


    # ========================================================
    # TOP FEATURES
    # ========================================================

    st.subheader(
        f"Top Features Influencing "
        f"{destination} Decision"
    )

    display_df = shap_df[
        [
            "Feature",
            "Value",
            "SHAP Value"
        ]
    ].head(10)


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # SHAP BAR CHART
    # ========================================================

    st.subheader(
        "SHAP Feature Contribution"
    )

    chart_df = (
        shap_df
        .head(10)
        .sort_values(
            "SHAP Value"
        )
        .set_index(
            "Feature"
        )
    )


    st.bar_chart(
        chart_df["SHAP Value"]
    )


    # ========================================================
    # SHAP INTERPRETATION
    # ========================================================

    st.subheader(
        "🧠 Interpretation"
    )

    st.caption(
        "Positive SHAP values push the prediction "
        "toward the selected destination. Negative "
        "values push the prediction away from it."
    )


    top_features = shap_df.head(5)


    for _, row in top_features.iterrows():

        feature_name = row[
            "Feature"
        ]

        shap_value = row[
            "SHAP Value"
        ]


        if shap_value > 0:

            st.write(
                f"🔼 **{feature_name}** supports "
                f"the **{destination}** decision "
                f"(SHAP = {shap_value:.4f})."
            )

        else:

            st.write(
                f"🔽 **{feature_name}** pushes "
                f"against the **{destination}** decision "
                f"(SHAP = {shap_value:.4f})."
            )


    # ========================================================
    # VIEW INPUT PARAMETERS
    # ========================================================

    with st.expander(
        "View Input Parameters"
    ):

        st.dataframe(
            input_df.T.rename(
                columns={
                    0: "Value"
                }
            ),
            use_container_width=True
        )