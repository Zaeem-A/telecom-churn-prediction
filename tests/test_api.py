from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_returns_valid_response():
    payload = {
        "tenure": 3,
        "MonthlyCharges": 85.0,
        "TotalCharges": 255.0,
        "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod": "Electronic check",
        "TechSupport": "No",
        "OnlineSecurity": "No",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert len(body["top_reasons"]) == 3
