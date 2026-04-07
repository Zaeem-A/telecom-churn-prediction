from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import uuid
from datetime import datetime

from src.api.schemas import CustomerFeatures, ChurnPrediction, ChurnReason
from src.models.explainer import ChurnExplainer
from src.monitoring.logger import log_prediction

app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predicts customer churn probability with explainability via SHAP values.",
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model at startup
pipeline = joblib.load("models/churn_model.joblib")
explainer = ChurnExplainer(pipeline)

FEATURE_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "charges_per_tenure",
                "num_services", "Contract", "InternetService", "PaymentMethod",
                "TechSupport", "OnlineSecurity"]

def compute_derived(data: CustomerFeatures) -> pd.DataFrame:
    d = data.model_dump()
    d["charges_per_tenure"] = d["MonthlyCharges"] / (d["tenure"] + 1)
    # Simplified num_services for API (not all service fields exposed — extend as needed)
    d["num_services"] = 0
    return pd.DataFrame([d])[FEATURE_COLS]

def risk_level(prob: float) -> str:
    if prob < 0.3: return "LOW"
    if prob < 0.6: return "MEDIUM"
    return "HIGH"

@app.post("/predict", response_model=ChurnPrediction, tags=["Prediction"])
async def predict_churn(customer: CustomerFeatures):
    try:
        X = compute_derived(customer)
        prob = float(pipeline.predict_proba(X)[0, 1])
        reasons_raw = explainer.get_top_reasons(X)[0]
        reasons = [ChurnReason(**r) for r in reasons_raw]

        prediction = ChurnPrediction(
            customer_id=str(uuid.uuid4()),
            churn_probability=round(prob, 4),
            churn_predicted=prob >= 0.5,
            risk_level=risk_level(prob),
            top_reasons=reasons,
        )

        # Log to DB async
        await log_prediction(customer.model_dump(), prediction.model_dump())
        return prediction

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "model_version": "lightgbm-v1"}
