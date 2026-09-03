from pyspark import pipelines as dp
from pyspark.sql.functions import count, countDistinct, max

@dp.materialized_view
def flights_stats():
    df = spark.read.table("ingest_flights")

    return df.agg(
        count("*").alias("num_events"),
        countDistinct("icao24").alias("distinct_aircraft"),
        max("velocity").alias("max_velocity")
    )
