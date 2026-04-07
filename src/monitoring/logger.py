import asyncpg
import os
import json
from datetime import datetime

DB_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/churndb")

async def log_prediction(features: dict, prediction: dict):
    try:
        conn = await asyncpg.connect(DB_URL)
        await conn.execute("""
            INSERT INTO prediction_logs 
                (timestamp, input_features, churn_probability, churn_predicted, risk_level)
            VALUES ($1, $2, $3, $4, $5)
        """,
            datetime.utcnow(),
            json.dumps(features),
            prediction["churn_probability"],
            prediction["churn_predicted"],
            prediction["risk_level"],
        )
        await conn.close()
    except Exception as e:
        # Don't let logging failures break predictions
        print(f"[WARNING] Failed to log prediction: {e}")
