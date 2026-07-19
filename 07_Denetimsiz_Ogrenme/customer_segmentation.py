"""Müşteri davranış verisi üzerinde kapsamlı kümeleme ve segment profilleme."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

RANDOM_STATE = 42
FEATURES = ["age", "annual_income", "spending_score", "visits_per_month", "online_ratio"]

def create_demo_data(n: int = 1200, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    profiles = [
        {"age": 24, "income": 32, "spend": 78, "visits": 9, "online": .82},
        {"age": 45, "income": 82, "spend": 72, "visits": 6, "online": .55},
        {"age": 52, "income": 48, "spend": 28, "visits": 2, "online": .25},
        {"age": 31, "income": 105, "spend": 35, "visits": 4, "online": .70},
    ]
    sizes = rng.multinomial(n, [.27, .28, .25, .20])
    frames = []
    for segment, (profile, size) in enumerate(zip(profiles, sizes)):
        frames.append(pd.DataFrame({
            "customer_id": [f"C{segment}{i:05d}" for i in range(size)],
            "age": rng.normal(profile["age"], 5, size).clip(18, 75),
            "annual_income": rng.normal(profile["income"], 11, size).clip(12, 160),
            "spending_score": rng.normal(profile["spend"], 10, size).clip(1, 100),
            "visits_per_month": rng.poisson(profile["visits"], size).clip(0, 25),
            "online_ratio": rng.normal(profile["online"], .12, size).clip(0, 1),
        }))
    df = pd.concat(frames, ignore_index=True)
    outliers = pd.DataFrame({
        "customer_id": [f"OUT{i:03d}" for i in range(15)],
        "age": rng.integers(18, 75, 15),
        "annual_income": rng.choice([5, 190], 15),
        "spending_score": rng.choice([1, 100], 15),
        "visits_per_month": rng.choice([0, 30], 15),
        "online_ratio": rng.choice([0, 1], 15),
    })
    return pd.concat([df, outliers], ignore_index=True)

def load_data(path: Path | None) -> pd.DataFrame:
    df = pd.read_csv(path) if path else create_demo_data()
    missing = set(FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Eksik özellikler: {sorted(missing)}")
    return df.dropna(subset=FEATURES).drop_duplicates().copy()

def select_k(features: np.ndarray, k_min=2, k_max=9) -> pd.DataFrame:
    rows = []
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit_predict(features)
        rows.append({
            "k": k,
            "inertia": KMeans(n_clusters=k, n_init=30, random_state=RANDOM_STATE).fit(features).inertia_,
            "silhouette": silhouette_score(features, labels),
            "davies_bouldin": davies_bouldin_score(features, labels),
        })
    return pd.DataFrame(rows)

def compare_algorithms(features: np.ndarray, best_k: int):
    candidates = {
        "KMeans": KMeans(best_k, n_init=30, random_state=RANDOM_STATE),
        "Agglomerative": AgglomerativeClustering(best_k),
        "GaussianMixture": GaussianMixture(best_k, random_state=RANDOM_STATE),
        "DBSCAN": DBSCAN(eps=.7, min_samples=10),
    }
    rows, labels_by_model = [], {}
    for name, model in candidates.items():
        labels = model.fit_predict(features)
        labels_by_model[name] = labels
        valid = labels != -1
        cluster_count = len(set(labels[valid]))
        if cluster_count >= 2 and valid.sum() > cluster_count:
            sil = silhouette_score(features[valid], labels[valid])
            db = davies_bouldin_score(features[valid], labels[valid])
        else:
            sil, db = float("nan"), float("nan")
        rows.append({
            "model": name,
            "cluster_count": cluster_count,
            "noise_count": int((labels == -1).sum()),
            "silhouette": sil,
            "davies_bouldin": db,
        })
    return pd.DataFrame(rows).sort_values("silhouette", ascending=False), labels_by_model

def name_segments(profile: pd.DataFrame) -> dict[int, str]:
    names = {}
    for segment, row in profile.iterrows():
        if row["spending_score"] >= profile["spending_score"].median() and row["annual_income"] >= profile["annual_income"].median():
            name = "Değerli Müşteriler"
        elif row["spending_score"] >= profile["spending_score"].median():
            name = "Aktif Fırsat Avcıları"
        elif row["annual_income"] >= profile["annual_income"].median():
            name = "Potansiyel Premium"
        else:
            name = "Düşük Etkileşim"
        names[int(segment)] = name
    return names

def save_visuals(df: pd.DataFrame, labels_by_model: dict, output: Path) -> None:
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coordinates = pca.fit_transform(RobustScaler().fit_transform(df[FEATURES]))
    names = list(labels_by_model)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, name in zip(axes.flat, names):
        ax.scatter(coordinates[:, 0], coordinates[:, 1], c=labels_by_model[name], s=12, cmap="tab10", alpha=.65)
        ax.set_title(name)
    fig.suptitle(f"PCA Görünümü — Açıklanan varyans %{pca.explained_variance_ratio_.sum()*100:.1f}")
    fig.tight_layout()
    fig.savefig(output / "clustering_comparison.png", dpi=170)
    plt.close(fig)

def run(input_path: Path | None, output: Path) -> None:
    df = load_data(input_path)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df[FEATURES])
    k_scores = select_k(scaled)
    best_k = int(k_scores.sort_values(["silhouette", "davies_bouldin"], ascending=[False, True]).iloc[0]["k"])
    comparison, labels_by_model = compare_algorithms(scaled, best_k)
    best_model = str(comparison.iloc[0]["model"])
    df["segment"] = labels_by_model[best_model]
    profile = df[df["segment"] != -1].groupby("segment")[FEATURES].mean().round(2)
    segment_names = name_segments(profile)
    df["segment_name"] = df["segment"].map(segment_names).fillna("Aykırı Müşteri")
    output.mkdir(parents=True, exist_ok=True)
    df.to_csv(output / "segmented_customers.csv", index=False)
    profile.to_csv(output / "segment_profiles.csv")
    k_scores.to_csv(output / "k_selection.csv", index=False)
    comparison.to_csv(output / "algorithm_comparison.csv", index=False)
    save_visuals(df, labels_by_model, output)
    summary = {"best_k": best_k, "best_model": best_model, "segments": segment_names}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(comparison.round(4).to_string(index=False))
    print("\nSegment profilleri:\n", profile)

def main() -> None:
    parser = argparse.ArgumentParser(description="Müşteri segmentasyonu")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    run(args.input, args.output)

if __name__ == "__main__":
    main()
