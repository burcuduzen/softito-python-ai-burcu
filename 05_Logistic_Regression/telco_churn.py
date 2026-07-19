"""Telco churn: kategorik dönüşüm ve sızıntısız Logistic Regression."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def create_demo_data(n: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    monthly = rng.normal(72, 24, n).clip(18, 130)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20])
    logit = 1.3 * (contract == "Month-to-month") + .012 * monthly - .035 * tenure - 1
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({
        "tenure": tenure, "MonthlyCharges": monthly, "Contract": contract, "Churn": churn
    })

if __name__ == "__main__":
    df = create_demo_data()
    X, y = df.drop(columns="Churn"), df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=42
    )
    prep = ColumnTransformer([
        ("numeric", StandardScaler(), ["tenure", "MonthlyCharges"]),
        ("category", OneHotEncoder(handle_unknown="ignore"), ["Contract"]),
    ])
    model = Pipeline([
        ("preprocess", prep),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    probability = model.predict_proba(X_test)[:, 1]
    print("ROC-AUC:", round(roc_auc_score(y_test, probability), 4))
    print(classification_report(y_test, probability >= .5, digits=4))
