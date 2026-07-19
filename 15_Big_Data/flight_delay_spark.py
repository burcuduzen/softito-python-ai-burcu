"""US Flight Delays verisi için üretim tarzı PySpark analiz pipeline'ı."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

REQUIRED_COLUMNS = {
    "YEAR", "MONTH", "DAY", "AIRLINE", "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT", "DEPARTURE_DELAY", "ARRIVAL_DELAY",
    "CANCELLED", "DISTANCE",
}

def create_spark(app_name="FlightDelayAnalytics") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

def read_flights(spark: SparkSession, path: str) -> DataFrame:
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("mode", "PERMISSIVE")
        .csv(path)
    )
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Eksik uçuş sütunları: {sorted(missing)}")
    return df

def clean_flights(df: DataFrame) -> DataFrame:
    numeric_columns = [
        "YEAR", "MONTH", "DAY", "DEPARTURE_DELAY",
        "ARRIVAL_DELAY", "CANCELLED", "DISTANCE",
    ]
    clean = df
    for column in numeric_columns:
        clean = clean.withColumn(column, F.col(column).cast("double"))
    clean = (
        clean
        .filter(F.col("AIRLINE").isNotNull())
        .filter(F.col("ORIGIN_AIRPORT").isNotNull())
        .filter(F.col("DESTINATION_AIRPORT").isNotNull())
        .filter((F.col("MONTH") >= 1) & (F.col("MONTH") <= 12))
        .filter(F.col("DISTANCE") > 0)
        .withColumn("IS_DELAYED", (F.col("ARRIVAL_DELAY") >= 15).cast("int"))
        .withColumn("IS_SEVERELY_DELAYED", (F.col("ARRIVAL_DELAY") >= 60).cast("int"))
        .withColumn(
            "ROUTE",
            F.concat_ws("-", F.col("ORIGIN_AIRPORT"), F.col("DESTINATION_AIRPORT")),
        )
        .withColumn(
            "DISTANCE_GROUP",
            F.when(F.col("DISTANCE") < 500, "short")
            .when(F.col("DISTANCE") < 1500, "medium")
            .otherwise("long"),
        )
    )
    return clean

def data_quality(df: DataFrame) -> dict:
    row_count = df.count()
    null_expressions = [
        F.sum(F.col(column).isNull().cast("int")).alias(column)
        for column in df.columns
    ]
    null_counts = df.agg(*null_expressions).first().asDict()
    duplicate_count = row_count - df.dropDuplicates().count()
    return {
        "row_count": row_count,
        "column_count": len(df.columns),
        "duplicate_count": duplicate_count,
        "null_counts": null_counts,
    }

def airline_summary(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("AIRLINE")
        .agg(
            F.count("*").alias("flight_count"),
            F.round(F.avg("DEPARTURE_DELAY"), 2).alias("average_departure_delay"),
            F.round(F.avg("ARRIVAL_DELAY"), 2).alias("average_arrival_delay"),
            F.round(F.avg("IS_DELAYED"), 4).alias("delay_rate"),
            F.round(F.avg("IS_SEVERELY_DELAYED"), 4).alias("severe_delay_rate"),
            F.round(F.avg(F.col("CANCELLED").cast("double")), 4).alias("cancel_rate"),
            F.round(F.avg("DISTANCE"), 2).alias("average_distance"),
        )
        .orderBy(F.desc("delay_rate"))
    )

def route_summary(df: DataFrame, minimum_flights=100) -> DataFrame:
    return (
        df.groupBy("ROUTE")
        .agg(
            F.count("*").alias("flight_count"),
            F.round(F.avg("ARRIVAL_DELAY"), 2).alias("average_arrival_delay"),
            F.round(F.avg("IS_DELAYED"), 4).alias("delay_rate"),
        )
        .filter(F.col("flight_count") >= minimum_flights)
        .orderBy(F.desc("delay_rate"), F.desc("flight_count"))
    )

def monthly_summary(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("YEAR", "MONTH")
        .agg(
            F.count("*").alias("flight_count"),
            F.round(F.avg("ARRIVAL_DELAY"), 2).alias("average_arrival_delay"),
            F.round(F.avg("IS_DELAYED"), 4).alias("delay_rate"),
            F.round(F.avg(F.col("CANCELLED").cast("double")), 4).alias("cancel_rate"),
        )
        .orderBy("YEAR", "MONTH")
    )

def write_table(df: DataFrame, path: Path, partitions=1) -> None:
    (
        df.coalesce(partitions)
        .write.mode("overwrite")
        .option("header", True)
        .csv(str(path))
    )

def run(input_path: str, output: Path, minimum_route_flights=100) -> None:
    spark = create_spark()
    spark.sparkContext.setLogLevel("WARN")
    try:
        raw = read_flights(spark, input_path)
        quality_before = data_quality(raw)
        clean = clean_flights(raw).cache()
        quality_after = data_quality(clean)
        airlines = airline_summary(clean)
        routes = route_summary(clean, minimum_route_flights)
        months = monthly_summary(clean)
        output.mkdir(parents=True, exist_ok=True)
        write_table(airlines, output / "airline_summary")
        write_table(routes, output / "route_summary")
        write_table(months, output / "monthly_summary")
        airlines.show(30, truncate=False)
        routes.show(20, truncate=False)
        manifest = {
            "input": input_path,
            "quality_before": quality_before,
            "quality_after": quality_after,
            "output_tables": ["airline_summary", "route_summary", "monthly_summary"],
        }
        (output / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        clean.unpersist()
    finally:
        spark.stop()

def main():
    parser = argparse.ArgumentParser(description="Spark uçuş gecikme analizi")
    parser.add_argument("csv_path", help="flights.csv dosya yolu veya glob")
    parser.add_argument("--minimum-route-flights", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "outputs")
    args = parser.parse_args()
    run(args.csv_path, args.output, args.minimum_route_flights)

if __name__ == "__main__":
    main()
