"""Klasik sınıflandırma algoritmalarının adil ve kapsamlı karşılaştırması."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    y = dataset.target.copy()
    X.columns = [column.lower().replace(" ", "_") for column in X.columns]
    return X, y

def build_models() -> dict[str, object]:
    scaled = lambda model: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])
    unscaled = lambda model: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])
    return {
        "KNN": scaled(KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "Gaussian NB": scaled(GaussianNB()),
        "RBF SVM": scaled(SVC(
            C=2.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=RANDOM_STATE
        )),
        "Decision Tree": unscaled(DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=6,
            class_weight="balanced", random_state=RANDOM_STATE
        )),
        "Random Forest": unscaled(RandomForestClassifier(
            n_estimators=350, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
        )),
        "AdaBoost": unscaled(AdaBoostClassifier(
            n_estimators=180, learning_rate=.05, random_state=RANDOM_STATE
        )),
        "Gradient Boosting": unscaled(GradientBoostingClassifier(
            n_estimators=180, learning_rate=.05, max_depth=2,
            random_state=RANDOM_STATE
        )),
    }

def probability_of_positive(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    decision = model.decision_function(X)
    return 1 / (1 + np.exp(-decision))

def evaluate_models(X_train, y_train, X_test, y_test):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy", "f1": "f1",
        "roc_auc": "roc_auc", "balanced_accuracy": "balanced_accuracy",
    }
    rows, fitted, predictions = [], {}, {}
    for name, model in build_models().items():
        start = time.perf_counter()
        cv_result = cross_validate(
            model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1
        )
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        pred = model.predict(X_test)
        probability = probability_of_positive(model, X_test)
        rows.append({
            "model": name,
            "test_accuracy": accuracy_score(y_test, pred),
            "test_balanced_accuracy": balanced_accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred),
            "test_recall": recall_score(y_test, pred),
            "test_f1": f1_score(y_test, pred),
            "test_roc_auc": roc_auc_score(y_test, probability),
            "cv_accuracy": cv_result["test_accuracy"].mean(),
            "cv_f1": cv_result["test_f1"].mean(),
            "cv_roc_auc": cv_result["test_roc_auc"].mean(),
            "fit_seconds": elapsed,
        })
        fitted[name] = model
        predictions[name] = (pred, probability)
    table = pd.DataFrame(rows).sort_values(
        ["test_roc_auc", "test_f1"], ascending=False
    ).reset_index(drop=True)
    return table, fitted, predictions

def save_roc_curves(y_test, predictions, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    for name, (_, probability) in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        auc = roc_auc_score(y_test, probability)
        ax.plot(fpr, tpr, label=f"{name} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Model ROC Eğrileri")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "roc_comparison.png", dpi=170)
    plt.close(fig)

def save_best_model_details(name, model, y_test, pred, output: Path) -> None:
    matrix = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set(title=f"En İyi Model: {name}", xlabel="Tahmin", ylabel="Gerçek")
    fig.tight_layout()
    fig.savefig(output / "best_model_confusion_matrix.png", dpi=170)
    plt.close(fig)
    report = classification_report(y_test, pred, output_dict=True)
    (output / "best_model_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    joblib.dump(model, output / "best_classifier.joblib")

def run(output: Path) -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, stratify=y, random_state=RANDOM_STATE
    )
    table, fitted, predictions = evaluate_models(X_train, y_train, X_test, y_test)
    output.mkdir(parents=True, exist_ok=True)
    table.round(5).to_csv(output / "model_comparison.csv", index=False)
    best_name = str(table.iloc[0]["model"])
    best_pred, _ = predictions[best_name]
    save_roc_curves(y_test, predictions, output)
    save_best_model_details(best_name, fitted[best_name], y_test, best_pred, output)
    summary = {
        "dataset_rows": len(X),
        "feature_count": X.shape[1],
        "positive_rate": float(y.mean()),
        "best_model": best_name,
        "best_test_roc_auc": float(table.iloc[0]["test_roc_auc"]),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(table.round(4).to_string(index=False))
    print("\nÖzet:", summary)

def main() -> None:
    parser = argparse.ArgumentParser(description="Klasik ML model karşılaştırması")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    run(args.output)

if __name__ == "__main__":
    main()
