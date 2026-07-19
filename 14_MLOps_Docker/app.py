"""Model yaşam döngüsü özellikleri içeren FastAPI churn tahmin servisi."""
from __future__ import annotations
import logging
import math
import os
import time
from collections import deque
from datetime import datetime, timezone
from typing import Literal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("churn-api")
MODEL_VERSION = os.getenv("MODEL_VERSION", "rule-baseline-2.0.0")
START_TIME = time.time()
RECENT_PREDICTIONS = deque(maxlen=1000)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Müşteri özelliklerinden churn olasılığı üreten örnek MLOps servisi.",
    version=MODEL_VERSION,
)

class CustomerFeatures(BaseModel):
    customer_id: str = Field(min_length=1, max_length=50)
    tenure: int = Field(ge=0, le=100)
    monthly_charges: float = Field(gt=0, le=1000)
    total_charges: float = Field(ge=0)
    contract: Literal["Month-to-month", "One year", "Two year"]
    internet_service: Literal["Fiber optic", "DSL", "No"]
    payment_method: Literal[
        "Electronic check", "Credit card", "Bank transfer", "Mailed check"
    ]
    senior_citizen: bool = False
    partner: bool = False

    @field_validator("customer_id")
    @classmethod
    def normalize_id(cls, value: str) -> str:
        return value.strip().upper()

class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=500)

class Prediction(BaseModel):
    customer_id: str
    churn_probability: float
    prediction: int
    risk_level: Literal["low", "medium", "high"]
    model_version: str
    timestamp: str

def sigmoid(value: float) -> float:
    value = min(max(value, -30), 30)
    return 1 / (1 + math.exp(-value))

def calculate_probability(customer: CustomerFeatures) -> float:
    score = -1.5
    score += 1.20 if customer.contract == "Month-to-month" else -.30 if customer.contract == "One year" else -.75
    score += .60 if customer.internet_service == "Fiber optic" else 0
    score += .45 if customer.payment_method == "Electronic check" else 0
    score += .32 if customer.senior_citizen else 0
    score -= .18 if customer.partner else 0
    score += .007 * customer.monthly_charges
    score -= .035 * customer.tenure
    expected_total = customer.tenure * customer.monthly_charges
    if customer.tenure > 0 and customer.total_charges < expected_total * .5:
        score += .20
    return min(max(sigmoid(score), 0), 1)

def risk_level(probability: float) -> str:
    if probability >= .70:
        return "high"
    if probability >= .40:
        return "medium"
    return "low"

def make_prediction(customer: CustomerFeatures) -> Prediction:
    probability = calculate_probability(customer)
    result = Prediction(
        customer_id=customer.customer_id,
        churn_probability=round(probability, 5),
        prediction=int(probability >= .5),
        risk_level=risk_level(probability),
        model_version=MODEL_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    RECENT_PREDICTIONS.append(result)
    return result

@app.middleware("http")
async def request_logging(request: Request, call_next):
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("İstek işlenemedi")
        raise
    duration_ms = (time.perf_counter() - started) * 1000
    logger.info("%s %s status=%s duration_ms=%.2f", request.method, request.url.path, response.status_code, duration_ms)
    response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
    response.headers["X-Model-Version"] = MODEL_VERSION
    return response

@app.exception_handler(ValueError)
async def value_error_handler(_: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})

@app.get("/")
def root():
    return {
        "service": "Customer Churn Prediction API",
        "version": MODEL_VERSION,
        "documentation": "/docs",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "uptime_seconds": round(time.time() - START_TIME, 2),
    }

@app.get("/ready")
def readiness():
    return {"ready": True, "checks": {"model": "loaded", "memory": "available"}}

@app.post("/predict", response_model=Prediction)
def predict(customer: CustomerFeatures):
    return make_prediction(customer)

@app.post("/predict/batch", response_model=list[Prediction])
def batch_predict(request: BatchRequest):
    return [make_prediction(customer) for customer in request.customers]

@app.get("/metrics")
def metrics():
    if not RECENT_PREDICTIONS:
        return {"prediction_count": 0}
    probabilities = [item.churn_probability for item in RECENT_PREDICTIONS]
    return {
        "prediction_count": len(probabilities),
        "average_probability": round(sum(probabilities) / len(probabilities), 5),
        "high_risk_count": sum(item.risk_level == "high" for item in RECENT_PREDICTIONS),
        "positive_prediction_rate": round(sum(item.prediction for item in RECENT_PREDICTIONS) / len(probabilities), 5),
    }
