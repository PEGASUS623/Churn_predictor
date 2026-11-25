# main.py  -  FastAPI service that serves the churn model
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Load once at startup (not on every request)
model = joblib.load("churn_model.joblib")
threshold = float(joblib.load("threshold.joblib"))

app = FastAPI(title="Churn Prediction API", version="1.0")

# Lets a browser-based frontend on another domain call the API.
# For production, replace "*" with your frontend's URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

YesNo = Literal["Yes", "No"]
AddOn = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=72)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: AddOn
    OnlineBackup: AddOn
    DeviceProtection: AddOn
    TechSupport: AddOn
    StreamingTV: AddOn
    StreamingMovies: AddOn
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"]
    MonthlyCharges: float = Field(ge=18, le=120)


class Prediction(BaseModel):
    churn_probability: float
    will_churn: bool
    threshold: float


SERVICE_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]


@app.get("/")
def health():
    return {"status": "ok", "message": "Churn API is running. See /docs."}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])

    # Keep categories consistent with how the training data is coded
    if row.loc[0, "PhoneService"] == "No":
        row.loc[0, "MultipleLines"] = "No phone service"
    if row.loc[0, "InternetService"] == "No":
        for col in SERVICE_COLS:
            row.loc[0, col] = "No internet service"

    # Same engineered features as train.py
    row["NumServices"] = (row[SERVICE_COLS] == "Yes").sum(axis=1)
    row["AutoPay"] = row["PaymentMethod"].str.contains("automatic").astype(int)

    prob = float(model.predict_proba(row)[0, 1])
    return Prediction(
        churn_probability=round(prob, 4),
        will_churn=prob >= threshold,
        threshold=round(threshold, 3),
    )