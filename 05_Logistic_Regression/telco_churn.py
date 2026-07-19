"""Telco müşteri kaybı için uçtan uca Logistic Regression projesi."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "Churn"

def create_demo_data(n: int = 3000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 73, n)
    monthly = rng.normal(72, 24, n).clip(18, 130)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n, p=[.55, .25, .20]
    )
    internet = rng.choice(["Fiber optic", "DSL", "No"], n, p=[.48, .39, .13])
    payment = rng.choice(
        ["Electronic check", "Credit card", "Bank transfer", "Mailed check"], n
    )
    senior = rng.binomial(1, .17, n)
    partner = rng.choice(["Yes", "No"], n)
    total = (tenure * monthly + rng.normal(0, 80, n)).clip(0)
    logit = (
        -1.25 + 1.25 * (contract == "Month-to-month")
        + .55 * (internet == "Fiber optic")
        + .45 * (payment == "Electronic check")
        + .35 * senior - .034 * tenure + .006 * monthly
    )
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly.round(2),
        "TotalCharges": total.round(2),
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "SeniorCitizen": senior,
        "Partner": partner,
        TARGET: np.where(churn == 1, "Yes", "No"),
    })
    df.loc[rng.choice(n, 25, replace=False), "TotalCharges"] = np.nan
    return df

def load_data(path: Path | None) -> pd.DataFrame:
    return pd.read_csv(path) if path else create_demo_data()

def prepare_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    clean = df.drop_duplicates().copy()
    clean["TotalCharges"] = pd.to_numeric(clean["TotalCharges"], errors="coerce")
    if clean[TARGET].dtype == object:
        y = clean[TARGET].str.strip().map({"Yes": 1, "No": 0})
    else:
        y = clean[TARGET].astype(int)
    if y.isna().any():
        raise ValueError("Churn hedefinde tanınmayan değerler var.")
    return clean.drop(columns=TARGET), y

def build_pipeline(X: pd.DataFrame) -> Pipeline:
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])

def find_best_threshold(y_true, probability) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    f1_values = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-12
    )
    index = int(np.argmax(f1_values))
    return float(thresholds[index]), float(f1_values[index])

def save_figures(y_true, probability, threshold: float, output: Path) -> None:
    prediction = (probability >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, probability)
    fraction_positive, mean_prediction = calibration_curve(
        y_true, probability, n_bins=10
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    axes[0].plot(fpr, tpr, label=f"AUC={roc_auc_score(y_true, probability):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--")
    axes[0].legend(); axes[0].set_title("ROC Eğrisi")
    sns.heatmap(
        confusion_matrix(y_true, prediction), annot=True, fmt="d",
        cmap="Blues", cbar=False, ax=axes[1]
    )
    axes[1].set_title(f"Confusion Matrix (eşik={threshold:.2f})")
    axes[2].plot(mean_prediction, fraction_positive, marker="o")
    axes[2].plot([0, 1], [0, 1], "k--")
    axes[2].set_title("Kalibrasyon Eğrisi")
    fig.tight_layout()
    fig.savefig(output / "classification_diagnostics.png", dpi=170)
    plt.close(fig)

def run(input_path: Path | None, output: Path) -> None:
    X, y = prepare_target(load_data(input_path))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=RANDOM_STATE
    )
    pipeline = build_pipeline(X)
    search = GridSearchCV(
        pipeline,
        {"classifier__C": [.05, .1, .5, 1, 2, 10]},
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    probability = search.predict_proba(X_test)[:, 1]
    threshold, best_f1 = find_best_threshold(y_test, probability)
    prediction = (probability >= threshold).astype(int)
    output.mkdir(parents=True, exist_ok=True)
    metrics = {
        "best_C": search.best_params_["classifier__C"],
        "cv_roc_auc": search.best_score_,
        "test_roc_auc": roc_auc_score(y_test, probability),
        "best_threshold": threshold,
        "threshold_f1": best_f1,
        "classification_report": classification_report(
            y_test, prediction, output_dict=True
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "probability": probability,
        "prediction": prediction,
    }).to_csv(output / "test_predictions.csv", index=False)
    joblib.dump(search.best_estimator_, output / "churn_pipeline.joblib")
    save_figures(y_test, probability, threshold, output)
    print(json.dumps(metrics, indent=2))

def main() -> None:
    parser = argparse.ArgumentParser(description="Telco Churn Logistic Regression")
    parser.add_argument("--input", type=Path, help="Kaggle Telco CSV dosyası")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "outputs"
    )
    args = parser.parse_args()
    run(args.input, args.output)

if __name__ == "__main__":
    main()
