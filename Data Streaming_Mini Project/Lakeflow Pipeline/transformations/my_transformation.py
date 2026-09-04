from pyspark import pipelines as dp
from pyspark.sql import functions as F


# ============================================================
# BRONZE
# ============================================================

@dp.table(
    name="stream_bronze_trade_events"
)
def bronze_trade_events():

    return (
        spark.readStream
            .table("streaming_demo.trades.raw_trade_events")
            .withColumn(
                "_bronze_processed_at",
                F.current_timestamp()
            )
    )


# ============================================================
# SILVER
# ============================================================

@dp.table(
    name="stream_silver_trades"
)
def silver_trades():

    bronze = (
        spark.read
            .table("LIVE.stream_bronze_trade_events")
    )

    return (
        bronze

        # Clean string columns
        .withColumn(
            "side",
            F.upper(F.trim(F.col("side")))
        )

        # Convert event timestamp
        .withColumn(
            "event_time",
            F.to_timestamp(F.col("event_ts"))
        )

        # Calculate trade value
        .withColumn(
            "trade_value",
            F.col("quantity") * F.col("price")
        )

        # Data quality rules
        .filter(
            F.col("event_id").isNotNull()
        )
        .filter(
            F.col("event_time").isNotNull()
        )
        .filter(
            F.col("side").isin("BUY", "SELL")
        )
        .filter(
            F.col("quantity").isNotNull()
            & (F.col("quantity") > 0)
        )
        .filter(
            F.col("price").isNotNull()
            & (F.col("price") > 0)
        )
    )


# ============================================================
# GOLD
# ============================================================

@dp.table(
    name="stream_gold_symbol_metrics"
)
def gold_symbol_metrics():

    silver = (
        spark.read
            .table("LIVE.stream_silver_trades")
            .withWatermark(
                "event_time",
                "10 minutes"
            )
    )

    return (
        silver
        .groupBy(
            F.window(
                "event_time",
                "5 minutes"
            ),
            "symbol"
        )
        .agg(
            F.count("*").alias("trade_count"),

            F.sum(
                "trade_value"
            ).alias("gross_value"),

            F.sum(
                F.when(
                    F.col("side") == "BUY",
                    F.col("trade_value")
                ).otherwise(0)
            ).alias("buy_value"),

            F.sum(
                F.when(
                    F.col("side") == "SELL",
                    F.col("trade_value")
                ).otherwise(0)
            ).alias("sell_value")
        )
    )
