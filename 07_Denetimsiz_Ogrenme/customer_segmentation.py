"""K-Means müşteri segmentasyonu ve silhouette ile k seçimi."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "income": rng.normal(55, 18, n).clip(12, 130),
        "spending_score": rng.beta(2, 2, n) * 100,
    })
    features = StandardScaler().fit_transform(df)
    scores = {
        k: silhouette_score(
            features, KMeans(k, n_init=20, random_state=42).fit_predict(features)
        )
        for k in range(2, 8)
    }
    best_k = max(scores, key=scores.get)
    df["segment"] = KMeans(best_k, n_init=20, random_state=42).fit_predict(features)
    output = Path(__file__).parent / "outputs"
    output.mkdir(exist_ok=True)
    df.to_csv(output / "segmented_customers.csv", index=False)
    print("En iyi k:", best_k, "silhouette:", round(scores[best_k], 4))
    print(df.groupby("segment").mean().round(2))
