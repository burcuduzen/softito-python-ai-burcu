"""Spark ile havayolu bazında gecikme ve iptal özeti."""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def analyze(path: str) -> None:
    spark = SparkSession.builder.appName("FlightDelayAnalysis").getOrCreate()
    flights = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .csv(path)
    )
    result = (
        flights.filter(F.col("AIRLINE").isNotNull())
        .groupBy("AIRLINE")
        .agg(
            F.count("*").alias("flight_count"),
            F.avg("ARRIVAL_DELAY").alias("average_arrival_delay"),
            F.avg(F.col("CANCELLED").cast("double")).alias("cancel_rate"),
        )
        .orderBy(F.desc("average_arrival_delay"))
    )
    result.show(30, truncate=False)
    (
        result.coalesce(1)
        .write.mode("overwrite")
        .option("header", True)
        .csv("outputs/airline_summary")
    )
    spark.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    analyze(parser.parse_args().csv_path)
