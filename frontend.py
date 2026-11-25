# frontend.py  -  Streamlit UI that calls the FastAPI service
import os

import requests
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")


def get_api_url() -> str:
    # On Streamlit Cloud: set API_URL in the app's Secrets.
    # Locally: defaults to the uvicorn server on your machine.
    try:
        if "API_URL" in st.secrets:
            return st.secrets["API_URL"].rstrip("/")
    except Exception:
        pass
    return os.environ.get("API_URL", "http://127.0.0.1:8000").rstrip("/")


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
    monthly = st.number_input(
        "Monthly Charges", min_value=18.0, max_value=120.0,
        value=70.0, step=0.5, format="%.2f")
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"])

with col2:
    st.subheader("Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.selectbox("Multiple Lines", ["No", "Yes"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["No", "Yes"])
    backup = st.selectbox("Online Backup", ["No", "Yes"])
    device = st.selectbox("Device Protection", ["No", "Yes"])
    tech = st.selectbox("Tech Support", ["No", "Yes"])
    tv = st.selectbox("Streaming TV", ["No", "Yes"])
    movies = st.selectbox("Streaming Movies", ["No", "Yes"])

if st.button("Predict", type="primary"):
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
        # Long timeout: free hosts "sleep" and take ~1 minute to wake up
        with st.spinner("Contacting the prediction API..."):
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=90)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Could not get a prediction from the API at {API_URL}.")
        st.caption(str(e))
    else:
        prob = result["churn_probability"]
        st.metric("Churn probability", f"{prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))

        if result["will_churn"]:
            st.error("⚠️ This customer is likely to churn.")
        else:
            st.success("✅ This customer is unlikely to churn.")

        st.caption(f"Decision threshold: {result['threshold']} (customers at or above it are flagged).")