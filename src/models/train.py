import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from src.data.preprocessing import load_and_clean, engineer_features, build_preprocessor

def train(data_path: str, model_output_path: str):
    df = load_and_clean(data_path)
    df = engineer_features(df)

    X = df[["tenure", "MonthlyCharges", "TotalCharges", "charges_per_tenure",
             "num_services", "Contract", "InternetService", "PaymentMethod",
             "TechSupport", "OnlineSecurity"]]
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()

    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "xgboost": XGBClassifier(scale_pos_weight=3, use_label_encoder=False,
                                  eval_metric="logloss", random_state=42),
        "lightgbm": LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "auc": roc_auc_score(y_test, proba),
            "f1": f1_score(y_test, preds),
        }
        print(f"{name}: AUC={results[name]['auc']:.4f} F1={results[name]['f1']:.4f}")

    # Save best model (LightGBM is your production choice)
    best_pipe = Pipeline([("preprocessor", preprocessor), ("model", models["lightgbm"])])
    best_pipe.fit(X_train, y_train)
    joblib.dump(best_pipe, model_output_path)
    print(f"Model saved to {model_output_path}")
    return results

if __name__ == "__main__":
    train("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv", "models/churn_model.joblib")
