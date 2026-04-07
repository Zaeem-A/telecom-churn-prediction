CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    input_features JSONB,
    churn_probability FLOAT,
    churn_predicted BOOLEAN,
    risk_level VARCHAR(10)
);

-- Index for time-series monitoring queries
CREATE INDEX idx_timestamp ON prediction_logs(timestamp);
CREATE INDEX idx_risk ON prediction_logs(risk_level);
