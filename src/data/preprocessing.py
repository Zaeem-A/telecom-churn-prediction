import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges",
                    "charges_per_tenure", "num_services"]
CATEGORICAL_FEATURES = ["Contract", "InternetService", "PaymentMethod",
                         "TechSupport", "OnlineSecurity"]

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ratio: how much per month relative to tenure
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    # Count of value-added services
    service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                    "TechSupport", "StreamingTV", "StreamingMovies"]
    df["num_services"] = df[service_cols].apply(
        lambda row: (row == "Yes").sum(), axis=1
    )
    return df

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ])
