from pydantic import BaseModel, Field
from typing import Literal

class CustomerFeatures(BaseModel):
    tenure: int = Field(..., ge=0, le=72, example=12)
    MonthlyCharges: float = Field(..., example=65.5)
    TotalCharges: float = Field(..., example=786.0)
    Contract: Literal["Month-to-month", "One year", "Two year"] = "Month-to-month"
    InternetService: Literal["DSL", "Fiber optic", "No"] = "Fiber optic"
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ] = "Electronic check"
    TechSupport: Literal["Yes", "No", "No internet service"] = "No"
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = "No"

class ChurnReason(BaseModel):
    feature: str
    impact: float
    direction: str

class ChurnPrediction(BaseModel):
    customer_id: str | None = None
    churn_probability: float
    churn_predicted: bool
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    top_reasons: list[ChurnReason]
