"""Otel rezervasyon verisi için uçtan uca keşifsel veri analizi."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REQUIRED_COLUMNS = {
    "hotel", "lead_time", "adr", "adults", "children",
    "stays_in_week_nights", "stays_in_weekend_nights", "is_canceled",
}

def create_demo_data(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hotel = rng.choice(["City Hotel", "Resort Hotel"], n, p=[.66, .34])
    lead_time = rng.gamma(shape=2.0, scale=42, size=n).astype(int)
    month = rng.choice(
        ["January", "March", "May", "July", "September", "December"], n
    )
    adults = rng.integers(1, 4, n)
    children = rng.choice([0, 1, 2, np.nan], n, p=[.73, .15, .10, .02])
    weekday = rng.integers(1, 8, n)
    weekend = rng.integers(0, 4, n)
    adr = np.maximum(
        25,
        rng.normal(95, 30, n)
        + (hotel == "City Hotel") * 18
        + np.isin(month, ["July", "September"]) * 25,
    )
    cancel_probability = 1 / (
        1 + np.exp(-(-1.8 + .009 * lead_time + .005 * adr))
    )
    is_canceled = rng.binomial(1, cancel_probability)
    market_segment = rng.choice(
        ["Online TA", "Offline TA", "Direct", "Corporate"], n, p=[.52, .18, .22, .08]
    )
    df = pd.DataFrame({
        "hotel": hotel,
        "arrival_date_month": month,
        "lead_time": lead_time,
        "adr": adr.round(2),
        "adults": adults,
        "children": children,
        "stays_in_week_nights": weekday,
        "stays_in_weekend_nights": weekend,
        "market_segment": market_segment,
        "is_canceled": is_canceled,
    })
    return pd.concat([df, df.iloc[:10]], ignore_index=True)

def load_data(path: Path | None) -> pd.DataFrame:
    if path is None:
        return create_demo_data()
    if not path.exists():
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {path}")
    return pd.read_csv(path)

def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Eksik zorunlu sütunlar: {sorted(missing)}")

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    validate_schema(df)
    before = {
        "rows": len(df),
        "duplicates": int(df.duplicated().sum()),
        "missing_values": int(df.isna().sum().sum()),
    }
    clean = df.drop_duplicates().copy()
    clean["children"] = clean["children"].fillna(0).astype(int)
    numeric = ["lead_time", "adr", "adults", "children"]
    for column in numeric:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
        clean[column] = clean[column].fillna(clean[column].median())
    clean = clean.query("adr > 0 and adults > 0")
    clean["total_nights"] = (
        clean["stays_in_week_nights"] + clean["stays_in_weekend_nights"]
    )
    clean["total_guests"] = clean["adults"] + clean["children"]
    clean["revenue_potential"] = clean["adr"] * clean["total_nights"]
    clean["lead_time_group"] = pd.cut(
        clean["lead_time"],
        bins=[-1, 7, 30, 90, float("inf")],
        labels=["Son dakika", "Kısa", "Orta", "Uzun"],
    )
    after = {
        "rows": len(clean),
        "duplicates": int(clean.duplicated().sum()),
        "missing_values": int(clean.isna().sum().sum()),
    }
    return clean, {"before": before, "after": after}

def create_summary(df: pd.DataFrame) -> dict:
    return {
        "row_count": len(df),
        "overall_cancel_rate": round(float(df["is_canceled"].mean()), 4),
        "average_daily_rate": round(float(df["adr"].mean()), 2),
        "average_lead_time": round(float(df["lead_time"].mean()), 2),
        "hotel_summary": (
            df.groupby("hotel")
            .agg(
                reservations=("hotel", "size"),
                cancel_rate=("is_canceled", "mean"),
                average_adr=("adr", "mean"),
                potential_revenue=("revenue_potential", "sum"),
            )
            .round(3)
            .reset_index()
            .to_dict(orient="records")
        ),
        "segment_cancel_rate": (
            df.groupby("market_segment")["is_canceled"]
            .mean()
            .sort_values(ascending=False)
            .round(4)
            .to_dict()
        ),
    }

def save_plots(df: pd.DataFrame, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="Set2")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(df, x="lead_time", hue="is_canceled", bins=35, ax=axes[0, 0])
    axes[0, 0].set_title("Rezervasyon Ön Süresi Dağılımı")
    sns.boxplot(df, x="hotel", y="adr", hue="is_canceled", ax=axes[0, 1])
    axes[0, 1].set_title("Otel Türüne Göre Günlük Fiyat")
    cancel_by_month = df.groupby("arrival_date_month")["is_canceled"].mean().reset_index()
    sns.barplot(cancel_by_month, x="arrival_date_month", y="is_canceled", ax=axes[1, 0])
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].set_title("Aylık İptal Oranı")
    numeric = df.select_dtypes(include="number").corr()
    sns.heatmap(numeric, cmap="coolwarm", center=0, ax=axes[1, 1])
    axes[1, 1].set_title("Korelasyon Matrisi")
    fig.tight_layout()
    fig.savefig(figure_dir / "hotel_eda_dashboard.png", dpi=170)
    plt.close(fig)

def run_analysis(input_path: Path | None, output_dir: Path) -> None:
    raw = load_data(input_path)
    clean, quality = clean_data(raw)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_dir / "clean_hotel_bookings.csv", index=False)
    summary = {"data_quality": quality, **create_summary(clean)}
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plots(clean, output_dir / "figures")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

def main() -> None:
    parser = argparse.ArgumentParser(description="Otel rezervasyon EDA")
    parser.add_argument("--input", type=Path, help="Kaggle hotel_bookings.csv")
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "outputs"
    )
    args = parser.parse_args()
    run_analysis(args.input, args.output)

if __name__ == "__main__":
    main()
