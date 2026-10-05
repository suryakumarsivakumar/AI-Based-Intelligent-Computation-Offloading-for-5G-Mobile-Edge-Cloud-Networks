import pandas as pd
import numpy as np
import joblib
import shap

print("=" * 60)
print("SHAP TEST")
print("=" * 60)

# Load model
model = joblib.load("offloading_model_v1.pkl")

# Load features
features = joblib.load("offloading_features_v1.pkl")

print("Model loaded")
print("Features:", len(features))

# Create test input
input_data = {
    "cqi": 10,
    "dl_mcs": 17,
    "ul_mcs": 20,
    "dl_brate": 58.2,
    "ul_brate": 98.0,
    "dl_error": 0,
    "ul_error": 0,
    "snr": 15,
    "rsrp": -95,
    "crc_delay": 2,
    "harq_delay": 2,

    "battery_level": 45,
    "cpu_load": 70,
    "ram_usage": 65,
    "device_temperature": 40,

    "task_size_mb": 500,
    "task_complexity": 6,
    "latency_requirement_ms": 50,
    "bandwidth_requirement": 40,

    "mobility_speed": 20,
    "edge_distance_km": 2,
    "cloud_distance_km": 300,

    "network_quality": 1
}

X = pd.DataFrame(
    [input_data],
    columns=features
)

print("Input created")

# Prediction
prediction = model.predict(X)[0]

print("Prediction:", prediction)

print(
    "Prediction probabilities:",
    model.predict_proba(X)
)

# --------------------------------------------------
# SHAP
# --------------------------------------------------

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(model)

print("Explainer created")

print("\nCalculating SHAP values...")

shap_values = explainer.shap_values(
    X,
    check_additivity=False
)

print("SHAP calculation successful!")

print(
    "SHAP type:",
    type(shap_values)
)

print(
    "SHAP shape:",
    np.asarray(shap_values).shape
)

print("\nSHAP TEST PASSED")