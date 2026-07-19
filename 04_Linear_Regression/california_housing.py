"""California Housing için kapsamlı regresyon modelleme çalışması."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

def load_dataset(sample_size: int | None = None) -> tuple[pd.DataFrame, pd.Series]:
    dataset = fetch_california_housing(as_frame=True)
    X, y = dataset.data, dataset.target.rename("median_house_value")
    if sample_size and sample_size < len(X):
        indices = X.sample(sample_size, random_state=RANDOM_STATE).index
        X, y = X.loc[indices], y.loc[indices]
    return X, y

def build_models() -> dict[str, object]:
    scaled_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    return {
        "dummy_median": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="median")),
        ]),
        "linear_regression": Pipeline([
            *scaled_steps,
            ("model", LinearRegression()),
        ]),
        "ridge": Pipeline([
            *scaled_steps,
            ("model", Ridge(alpha=1.0)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=220, min_samples_leaf=2,
                n_jobs=-1, random_state=RANDOM_STATE
            )),
        ]),
    }

def regression_metrics(y_true, prediction) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, prediction)),
        "RMSE": float(mean_squared_error(y_true, prediction) ** .5),
        "R2": float(r2_score(y_true, prediction)),
    }

def compare_models(X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    rows = []
    fitted = {}
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    for name, model in build_models().items():
        cv = cross_validate(model, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
        model.fit(X_train, y_train)
        metrics = regression_metrics(y_test, model.predict(X_test))
        rows.append({
            "model": name,
            **metrics,
            "CV_MAE": float(-cv["test_mae"].mean()),
            "CV_RMSE": float(-cv["test_rmse"].mean()),
            "CV_R2": float(cv["test_r2"].mean()),
        })
        fitted[name] = model
    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return table, fitted

def tune_ridge(X_train, y_train) -> GridSearchCV:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", Ridge()),
    ])
    search = GridSearchCV(
        pipeline,
        {"model__alpha": np.logspace(-3, 3, 13)},
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return search.fit(X_train, y_train)

def save_diagnostics(model, X_test, y_test, output: Path) -> None:
    prediction = model.predict(X_test)
    residual = y_test - prediction
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(y_test, prediction, alpha=.35)
    limits = [min(y_test.min(), prediction.min()), max(y_test.max(), prediction.max())]
    axes[0].plot(limits, limits, "r--")
    axes[0].set(title="Gerçek ve Tahmin", xlabel="Gerçek", ylabel="Tahmin")
    sns.histplot(residual, bins=35, kde=True, ax=axes[1])
    axes[1].set_title("Artık Dağılımı")
    axes[2].scatter(prediction, residual, alpha=.35)
    axes[2].axhline(0, color="red", linestyle="--")
    axes[2].set(title="Tahmin-Artık", xlabel="Tahmin", ylabel="Artık")
    fig.tight_layout()
    fig.savefig(output / "regression_diagnostics.png", dpi=170)
    plt.close(fig)

def run(output: Path, sample_size: int | None) -> None:
    X, y = load_dataset(sample_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.20, random_state=RANDOM_STATE
    )
    comparison, fitted = compare_models(X_train, y_train, X_test, y_test)
    ridge_search = tune_ridge(X_train, y_train)
    tuned_metrics = regression_metrics(y_test, ridge_search.predict(X_test))
    output.mkdir(parents=True, exist_ok=True)
    comparison.round(5).to_csv(output / "model_comparison.csv", index=False)
    report = {
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "best_comparison_model": comparison.iloc[0]["model"],
        "tuned_ridge_alpha": ridge_search.best_params_["model__alpha"],
        "tuned_ridge_metrics": tuned_metrics,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    joblib.dump(ridge_search.best_estimator_, output / "ridge_model.joblib")
    best_name = comparison.iloc[0]["model"]
    save_diagnostics(fitted[best_name], X_test, y_test, output)
    print(comparison.round(4).to_string(index=False))
    print("\n", report)

def main() -> None:
    parser = argparse.ArgumentParser(description="California Housing regresyonu")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "outputs"
    )
    args = parser.parse_args()
    run(args.output, args.sample_size)

if __name__ == "__main__":
    main()
