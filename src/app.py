import os
import numpy as np
import streamlit as st
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Load model artifacts ───────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


@st.cache_resource
def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    threshold_path = os.path.join(MODEL_DIR, "threshold.txt")

    # Try repo-root level scaler (Docker copies it there)
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(BASE_DIR, "scaler.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    threshold = 0.45
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            threshold = float(f.read().strip())

    return model, scaler, threshold


try:
    model, scaler, threshold = load_artifacts()
    model_loaded = True
    load_note = "Running with locally saved model (Streamlit Cloud mode)"
except Exception as e:
    model_loaded = False
    load_note = f"⚠️ Model not loaded: {e}"

# ── UI ─────────────────────────────────────────────────────────────────────────
st.title("Fraud Detector 🔍")
st.caption("Check credit card transactions for potential fraud.")
st.info(load_note)

with st.sidebar:
    st.header("Transaction Features")
    amount = st.number_input("Transaction Amount", min_value=0.0, value=500.0, step=0.01)
    hour = st.slider("Hour of day", 0, 23, 14)

    st.subheader("V-Features")
    v_features = []
    for i in range(1, 29):
        val = st.number_input(f"V{i}", value=0.0, step=0.01, key=f"v{i}", format="%.2f")
        v_features.append(val)

# ── Prediction ─────────────────────────────────────────────────────────────────
if st.button("Check Transaction", type="primary", use_container_width=False):
    if not model_loaded:
        st.error("Model is not loaded. Please train the model first.")
    else:
        # Feature order must match training: Amount, V1…V28
        raw = np.array([[amount] + v_features])
        scaled = scaler.transform(raw)
        prob = model.predict_proba(scaled)[0][1]
        pred = int(prob >= threshold)

        col1, col2, col3 = st.columns(3)
        col1.metric("Fraud Probability", f"{prob:.4f}")
        col2.metric("Decision Threshold", f"{threshold:.4f}")

        with col3:
            st.write("**Verdict**")
            if pred == 1:
                st.error("FRAUD DETECTED — Block this transaction!")
            else:
                st.success("✅ Transaction Approved — Looks Legitimate")
