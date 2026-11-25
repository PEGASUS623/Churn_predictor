# frontend.py - Streamlit UI that calls the churn FastAPI service
import os

import requests
import streamlit as st

DEFAULT_API = "https://churn-predictor-3qqk.onrender.com"

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")


def get_api_url() -> str:
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"].rstrip("/")
    except Exception:
        pass
    return os.environ.get("API_URL", DEFAULT_API).rstrip("/")


API_URL = get_api_url()

st.title("📉 Customer Churn Predictor")
st.write("Enter the customer's details to estimate how likely they are to leave.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer profile")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_label = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0,
                              value=70.0, step=0.5, format="%.2f")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])

with col2:
    st.subheader("Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes"], disabled=(phone == "No"))
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    no_net = internet == "No"
    security = st.selectbox("Online Security", ["No", "Yes"], disabled=no_net)
    backup = st.selectbox("Online Backup", ["No", "Yes"], disabled=no_net)
    device = st.selectbox("Device Protection", ["No", "Yes"], disabled=no_net)
    tech = st.selectbox("Tech Support", ["No", "Yes"], disabled=no_net)
    tv = st.selectbox("Streaming TV", ["No", "Yes"], disabled=no_net)
    movies = st.selectbox("Streaming Movies", ["No", "Yes"], disabled=no_net)

if st.button("Predict", type="primary", use_container_width=True):
    payload = {
        "gender": gender,
        "SeniorCitizen": 1 if senior_label == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
    }

    try:
        with st.spinner("Contacting the prediction API (first call can take up to a minute while it wakes up)..."):
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=90)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not get a prediction from the API at {API_URL}.")
        st.caption(str(e))
    else:
        prob = result["churn_probability"]
        thr = result["threshold"]
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Churn probability", f"{prob:.1%}")
        m2.metric("Decision threshold", f"{thr:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))

        if result["will_churn"]:
            st.error("⚠️ High risk: this customer is likely to churn.")
        else:
            st.success("✅ Low risk: this customer is unlikely to churn.")
        st.caption("Customers at or above the threshold are flagged. The threshold was "
                   "chosen on training data to balance precision and recall (F1).")
