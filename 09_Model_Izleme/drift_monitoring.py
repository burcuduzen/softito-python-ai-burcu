"""Üretim modeli için veri kalitesi, drift ve performans izleme sistemi."""
from __future__ import annotations
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

@dataclass
class DriftResult:
    feature: str
    feature_type: str
    score: float
    p_value: float | None
    status: str
    method: str

def population_stability_index(reference, current, bins=10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_ratio = np.histogram(reference, edges)[0] / len(reference)
    cur_ratio = np.histogram(current, edges)[0] / len(current)
    ref_ratio = np.clip(ref_ratio, 1e-6, None)
    cur_ratio = np.clip(cur_ratio, 1e-6, None)
    return float(np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio)))

def status_from_psi(score: float) -> str:
    if score >= .25:
        return "kritik"
    if score >= .10:
        return "uyarı"
    return "stabil"

def numeric_drift(feature: str, reference: pd.Series, current: pd.Series) -> DriftResult:
    ref = reference.dropna().astype(float)
    cur = current.dropna().astype(float)
    psi = population_stability_index(ref, cur)
    _, p_value = ks_2samp(ref, cur)
    status = status_from_psi(psi)
    if p_value < .01 and status == "stabil":
        status = "uyarı"
    return DriftResult(feature, "numeric", psi, float(p_value), status, "PSI + KS")

def categorical_drift(feature: str, reference: pd.Series, current: pd.Series) -> DriftResult:
    categories = sorted(set(reference.dropna().astype(str)) | set(current.dropna().astype(str)))
    ref_counts = reference.astype(str).value_counts().reindex(categories, fill_value=0)
    cur_counts = current.astype(str).value_counts().reindex(categories, fill_value=0)
    table = np.vstack([ref_counts, cur_counts])
    _, p_value, _, _ = chi2_contingency(table + 1e-6)
    ref_ratio = ref_counts / max(ref_counts.sum(), 1)
    cur_ratio = cur_counts / max(cur_counts.sum(), 1)
    tvd = float(.5 * np.abs(ref_ratio - cur_ratio).sum())
    status = "kritik" if tvd >= .20 else "uyarı" if tvd >= .10 or p_value < .01 else "stabil"
    return DriftResult(feature, "categorical", tvd, float(p_value), status, "TVD + Chi-square")

def data_quality_report(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "duplicate_rate": float(df.duplicated().mean()),
        "missing_rate": {column: float(value) for column, value in df.isna().mean().items()},
        "numeric_outlier_rate": {
            column: float(((df[column] < df[column].quantile(.01)) | (df[column] > df[column].quantile(.99))).mean())
            for column in df.select_dtypes(include="number").columns
        },
    }

def monitor_features(reference: pd.DataFrame, current: pd.DataFrame) -> list[DriftResult]:
    common = [column for column in reference.columns if column in current.columns]
    results = []
    for column in common:
        if pd.api.types.is_numeric_dtype(reference[column]):
            results.append(numeric_drift(column, reference[column], current[column]))
        else:
            results.append(categorical_drift(column, reference[column], current[column]))
    return results

def performance_report(y_true, probability, threshold=.5) -> dict:
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "f1": float(f1_score(y_true, prediction)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "positive_prediction_rate": float(prediction.mean()),
    }

def create_demo_batches(n=4000, seed=42):
    rng = np.random.default_rng(seed)
    reference = pd.DataFrame({
        "temperature": rng.normal(20, 5, n),
        "humidity": rng.normal(62, 12, n).clip(10, 100),
        "wind_speed": rng.gamma(2, 4, n),
        "region": rng.choice(["north", "south", "east", "west"], n),
    })
    current = pd.DataFrame({
        "temperature": rng.normal(23, 6, n // 2),
        "humidity": rng.normal(64, 14, n // 2).clip(10, 100),
        "wind_speed": rng.gamma(2.4, 4.5, n // 2),
        "region": rng.choice(["north", "south", "east", "west"], n // 2, p=[.15, .50, .20, .15]),
    })
    return reference, current

def save_chart(results: list[DriftResult], output: Path) -> None:
    frame = pd.DataFrame([asdict(result) for result in results])
    colors = frame["status"].map({"stabil": "seagreen", "uyarı": "orange", "kritik": "crimson"})
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(frame["feature"], frame["score"], color=colors)
    ax.set(title="Özellik Drift Skorları", ylabel="Drift skoru")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output / "drift_scores.png", dpi=170)
    plt.close(fig)

def run(reference_path: Path | None, current_path: Path | None, output: Path) -> None:
    if reference_path and current_path:
        reference, current = pd.read_csv(reference_path), pd.read_csv(current_path)
    else:
        reference, current = create_demo_batches()
    results = monitor_features(reference, current)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "reference_quality": data_quality_report(reference),
        "current_quality": data_quality_report(current),
        "drift": [asdict(result) for result in results],
        "overall_status": "kritik" if any(r.status == "kritik" for r in results) else "uyarı" if any(r.status == "uyarı" for r in results) else "stabil",
    }
    (output / "monitoring_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(report["drift"]).to_csv(output / "drift_table.csv", index=False)
    save_chart(results, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))

def main() -> None:
    parser = argparse.ArgumentParser(description="Model ve veri drift izleme")
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    run(args.reference, args.current, args.output)

if __name__ == "__main__":
    main()
