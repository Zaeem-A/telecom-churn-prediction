import shap
import numpy as np
import pandas as pd

class ChurnExplainer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.model = pipeline.named_steps["model"]
        self.preprocessor = pipeline.named_steps["preprocessor"]
        # Build explainer on the raw model, not the pipeline
        self.explainer = shap.TreeExplainer(self.model)

    def get_top_reasons(self, X_raw: pd.DataFrame, top_n: int = 3) -> list[dict]:
        X_transformed = self.preprocessor.transform(X_raw)
        shap_values = self.explainer.shap_values(X_transformed)
        
        # Get feature names after one-hot encoding
        feature_names = (
            self.preprocessor.transformers_[0][2] +  # numeric
            list(self.preprocessor.named_transformers_["cat"]
                 .get_feature_names_out(self.preprocessor.transformers_[1][2]))
        )

        # For binary classification, shap_values may be a list
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        
        reasons = []
        for i in range(len(X_raw)):
            row_shap = sv[i]
            top_indices = np.argsort(np.abs(row_shap))[::-1][:top_n]
            reasons.append([
                {
                    "feature": feature_names[j],
                    "impact": float(row_shap[j]),
                    "direction": "increases" if row_shap[j] > 0 else "decreases",
                }
                for j in top_indices
            ])
        return reasons
