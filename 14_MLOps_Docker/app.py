"""FastAPI tabanlı churn tahmin servisi."""
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Churn Prediction API", version="1.0.0")

class Customer(BaseModel):
    tenure: int = Field(ge=0, le=100)
    monthly_charges: float = Field(gt=0)
    month_to_month: bool

@app.get("/health")
def health():
    return {"status": "ok", "model_version": "rule-baseline-1"}

@app.post("/predict")
def predict(customer: Customer):
    score = (
        .25
        + .35 * customer.month_to_month
        + .002 * customer.monthly_charges
        - .006 * customer.tenure
    )
    score = min(max(score, 0), 1)
    return {
        "churn_probability": round(score, 4),
        "prediction": int(score >= .5),
    }
