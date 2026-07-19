"""Otel rezervasyonları için temizleme, EDA ve feature engineering."""
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_demo_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "hotel": rng.choice(["City Hotel", "Resort Hotel"], n, p=[0.65, 0.35]),
        "lead_time": rng.gamma(2, 35, n).astype(int),
        "adr": np.maximum(rng.normal(105, 38, n), 10),
        "adults": rng.integers(1, 4, n),
        "stays": rng.integers(1, 9, n),
        "is_canceled": rng.binomial(1, 0.31, n),
    })

def analyze(df: pd.DataFrame, output: Path) -> None:
    df = df.drop_duplicates().copy()
    df["revenue_potential"] = df["adr"] * df["stays"]
    output.mkdir(exist_ok=True)
    summary = df.groupby("hotel").agg(
        reservation_count=("hotel", "size"),
        cancel_rate=("is_canceled", "mean"),
        average_adr=("adr", "mean"),
    )
    summary.to_csv(output / "hotel_summary.csv")
    sns.set_theme(style="whitegrid")
    chart = sns.boxplot(data=df, x="hotel", y="adr", hue="is_canceled")
    chart.figure.tight_layout()
    chart.figure.savefig(output / "adr_by_hotel.png", dpi=160)
    plt.close()
    print(summary.round(3))

if __name__ == "__main__":
    analyze(create_demo_data(), Path(__file__).parent / "outputs")
