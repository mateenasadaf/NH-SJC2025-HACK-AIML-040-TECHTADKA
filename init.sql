create table fraud_alerts (
  id SERIAL primary key,
  created_at TIMESTAMPTZ default now(),
  user_id TEXT,
  amount FLOAT,
  fraud_score FLOAT,
  details JSONB -- This will store the full SHAP explanation
);