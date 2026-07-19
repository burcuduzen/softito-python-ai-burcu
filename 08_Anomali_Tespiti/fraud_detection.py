"""Finansal işlemlerde gözetimsiz ve yarı gözetimli anomali tespiti."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

RANDOM_STATE = 42
FEATURES = ["amount", "hour", "velocity_1h", "distance_from_home", "device_risk", "merchant_risk"]

def create_transactions(normal_count=6000, fraud_count=120, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    normal = pd.DataFrame({
        "amount": rng.lognormal(3.4, .7, normal_count),
        "hour": rng.integers(0, 24, normal_count),
        "velocity_1h": rng.poisson(1.2, normal_count),
        "distance_from_home": rng.exponential(8, normal_count),
        "device_risk": rng.beta(1.5, 8, normal_count),
        "merchant_risk": rng.beta(2, 9, normal_count),
        "Class": 0,
    })
    fraud = pd.DataFrame({
        "amount": rng.lognormal(5.0, 1.0, fraud_count),
        "hour": rng.choice([0, 1, 2, 3, 4, 22, 23], fraud_count),
        "velocity_1h": rng.poisson(6, fraud_count),
        "distance_from_home": rng.exponential(35, fraud_count),
        "device_risk": rng.beta(6, 2, fraud_count),
        "merchant_risk": rng.beta(5, 2, fraud_count),
        "Class": 1,
    })
    return pd.concat([normal, fraud], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)

def load_data(path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(path) if path else create_transactions()
    target = "Class"
    if target not in df:
        raise ValueError("Veri setinde Class hedef sütunu bulunmalıdır.")
    available = FEATURES if set(FEATURES).issubset(df.columns) else [
        column for column in df.select_dtypes(include="number").columns
        if column != target
    ]
    if len(available) < 3:
        raise ValueError("Yeterli sayısal özellik yok.")
    return df[available + [target]].dropna()

def choose_threshold(y_true, score) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    index = int(np.argmax(f1))
    return float(thresholds[index]), float(f1[index])

def evaluate_model(name, model, X_train_normal, X_test, y_test):
    model.fit(X_train_normal)
    if hasattr(model, "decision_function"):
        score = -model.decision_function(X_test)
    else:
        score = -model.score_samples(X_test)
    threshold, best_f1 = choose_threshold(y_test, score)
    prediction = (score >= threshold).astype(int)
    return {
        "name": name,
        "model": model,
        "score": score,
        "prediction": prediction,
        "threshold": threshold,
        "f1": best_f1,
        "roc_auc": roc_auc_score(y_test, score),
        "average_precision": average_precision_score(y_test, score),
    }

def save_diagnostics(y_test, result, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(result["score"][y_test == 0], bins=40, color="steelblue", label="Normal", ax=axes[0], stat="density")
    sns.histplot(result["score"][y_test == 1], bins=40, color="crimson", label="Fraud", ax=axes[0], stat="density")
    axes[0].axvline(result["threshold"], color="black", linestyle="--")
    axes[0].legend(); axes[0].set_title("Anomali Skoru Dağılımı")
    sns.heatmap(confusion_matrix(y_test, result["prediction"]), annot=True, fmt="d", cmap="Reds", cbar=False, ax=axes[1])
    axes[1].set_title(f"{result['name']} Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output / "fraud_diagnostics.png", dpi=170)
    plt.close(fig)

def run(input_path: Path | None, output: Path) -> None:
    df = load_data(input_path)
    feature_columns = [column for column in df.columns if column != "Class"]
    X, y = df[feature_columns], df["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.35, stratify=y, random_state=RANDOM_STATE
    )
    scaler = RobustScaler().fit(X_train[y_train == 0])
    train_normal = scaler.transform(X_train[y_train == 0])
    test_scaled = scaler.transform(X_test)
    contamination = max(float(y_train.mean()), .005)
    models = {
        "Isolation Forest": IsolationForest(n_estimators=350, contamination=contamination, random_state=RANDOM_STATE),
        "One-Class SVM": OneClassSVM(kernel="rbf", nu=contamination, gamma="scale"),
        "Local Outlier Factor": LocalOutlierFactor(n_neighbors=35, novelty=True, contamination=contamination),
    }
    results = [
        evaluate_model(name, model, train_normal, test_scaled, y_test.to_numpy())
        for name, model in models.items()
    ]
    best = max(results, key=lambda item: item["average_precision"])
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([{k: v for k, v in result.items() if k not in {"model", "score", "prediction"}} for result in results])
    table.to_csv(output / "model_comparison.csv", index=False)
    save_diagnostics(y_test.to_numpy(), best, output)
    joblib.dump({"scaler": scaler, "model": best["model"], "threshold": best["threshold"], "features": feature_columns}, output / "fraud_detector.joblib")
    report = classification_report(y_test, best["prediction"], output_dict=True)
    summary = {"best_model": best["name"], "metrics": table.to_dict(orient="records"), "classification_report": report}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.round(4).to_string(index=False))

def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud anomali tespiti")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    run(args.input, args.output)

if __name__ == "__main__":
    main()
