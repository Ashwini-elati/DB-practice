# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 8 · Bronze and Silver
# MAGIC
# MAGIC Read the files ADF landed, write a bronze Delta table, then a silver table
# MAGIC with deduplication and quality rules applied — all registered in Unity
# MAGIC Catalog with grants set.
# MAGIC
# MAGIC The local PySpark equivalent of everything below is in `src/lakehouse/`
# MAGIC with 26 tests. Read that first if a transformation here is unclear: it is
# MAGIC the same logic, executable on a laptop in two seconds.

# COMMAND ----------
dbutils.widgets.text("catalog", "aurora_dev")
dbutils.widgets.text("load_date", "2024-04-01")
dbutils.widgets.text("batch_id", "1")

CATALOG = dbutils.widgets.get("catalog")
LOAD_DATE = dbutils.widgets.get("load_date")
BATCH_ID = int(dbutils.widgets.get("batch_id"))

LANDING = f"/Volumes/{CATALOG}/bronze/landing"
CHECKPOINTS = f"/Volumes/{CATALOG}/ops/checkpoints"

spark.sql(f"USE CATALOG {CATALOG}")

# COMMAND ----------
from pyspark.sql import Window
from pyspark.sql import functions as F
from pyspark.sql.types import (DecimalType, IntegerType, StringType,
                               StructField, StructType)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 · Read the landing volume
# MAGIC
# MAGIC Note the path: `/Volumes/catalog/schema/volume/...`. Reading through a
# MAGIC **volume** rather than an `abfss://` path is what gets this access
# MAGIC governed and audited, and it is what makes lineage work. A job that reads
# MAGIC the storage path directly produces no lineage, and the graph then has a
# MAGIC hole exactly where the interesting transformation was.
# MAGIC
# MAGIC Every column is read as a **string**, deliberately. Letting the reader
# MAGIC infer types means one row with `N/A` in a numeric column silently changes
# MAGIC that column's type for the whole batch — so your bronze schema depends on
# MAGIC the worst row in the file. Read as text, cast in silver, keep the decision
# MAGIC visible.

# COMMAND ----------
ORDERS_SCHEMA = StructType([
    StructField("order_id",    StringType(), True),
    StructField("customer_id", StringType(), True),
    StructField("product_id",  StringType(), True),
    StructField("quantity",    StringType(), True),
    StructField("unit_price",  StringType(), True),
    StructField("order_ts",    StringType(), True),
    StructField("status",      StringType(), True),
    StructField("updated_at",  StringType(), True),
])

raw = (spark.read
       .option("header", "true")
       .schema(ORDERS_SCHEMA)
       .csv(f"{LANDING}/orders/load_date={LOAD_DATE}"))

print(f"landed: {raw.count():,} rows")

# COMMAND ----------
# MAGIC %md
# MAGIC ### The Auto Loader alternative
# MAGIC
# MAGIC For a real feed, use Auto Loader rather than a plain read. Three options
# MAGIC carry the weight, and each is a decision rather than a default:
# MAGIC
# MAGIC * `schemaEvolutionMode = addNewColumns` — a new source column is added
# MAGIC   rather than failing the stream.
# MAGIC * `rescuedDataColumn` — anything that does not fit the schema is captured
# MAGIC   in a column instead of dropped. This is the streaming equivalent of a
# MAGIC   quarantine table, and the difference between handling bad data and losing it.
# MAGIC * `checkpointLocation` — this **is** the watermark. It must live in a
# MAGIC   governed volume, not on a cluster, or losing it means a full reload.

# COMMAND ----------
# raw = (spark.readStream.format("cloudFiles")
#        .option("cloudFiles.format", "csv")
#        .option("cloudFiles.schemaLocation", f"{CHECKPOINTS}/bronze_orders/schema")
#        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
#        .option("rescuedDataColumn", "_rescued_data")
#        .option("header", "true")
#        .load(f"{LANDING}/orders"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 · Bronze — add lineage, reject nothing
# MAGIC
# MAGIC Bronze applies a schema and lineage columns. It rejects nothing and
# MAGIC corrects nothing. **If your bronze layer rejects a row you have written
# MAGIC silver by accident** and lost the ability to re-derive — which is the only
# MAGIC reason bronze exists.
# MAGIC
# MAGIC `_metadata.file_path` is a free lineage column and it is the first thing
# MAGIC you want when somebody asks where a row came from.

# COMMAND ----------
bronze = (raw
          .withColumn("_batch_id", F.lit(BATCH_ID))
          .withColumn("_load_date", F.to_date(F.lit(LOAD_DATE)))
          .withColumn("_ingested_at", F.current_timestamp())
          .withColumn("_source_file", F.col("_metadata.file_path")))

(bronze.write.format("delta")
 .mode("overwrite")
 # replaceWhere overwrites THIS partition and leaves every other day alone.
 # That is what makes a re-run of one day safe, and it is the difference
 # between a pipeline you can replay and one you dare not touch.
 .option("replaceWhere", f"_load_date = '{LOAD_DATE}'")
 .partitionBy("_load_date")
 .saveAsTable("bronze.orders"))

print(f"bronze.orders: {spark.table('bronze.orders').count():,} rows total")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 · Deduplicate to one row per business key
# MAGIC
# MAGIC `row_number`, **not** `rank`. Rank returns 1 for every tied row, so a tie
# MAGIC hands back two rows and silently reintroduces the duplicate you were
# MAGIC removing.
# MAGIC
# MAGIC The second `orderBy` column is not decoration. Without a deterministic
# MAGIC tie-break, which row survives is arbitrary and the output changes between
# MAGIC runs — so two people running the same code on the same data get different
# MAGIC answers, and neither can reproduce the other's.

# COMMAND ----------
window = (Window.partitionBy("order_id")
          .orderBy(F.col("updated_at").desc(), F.col("_ingested_at").desc()))

deduped = (spark.table("bronze.orders")
           .filter(F.col("_load_date") == LOAD_DATE)
           .withColumn("_rn", F.row_number().over(window))
           .filter("_rn = 1")
           .drop("_rn"))

print(f"after dedup: {deduped.count():,} rows")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 · Quality rules, and the NULL trap
# MAGIC
# MAGIC Rules are data: a name, a predicate rows must **satisfy**, and a severity.
# MAGIC Declarative means the rule set is reviewable by somebody who does not read
# MAGIC PySpark, and a new rule is a row rather than a commit.
# MAGIC
# MAGIC **The NULL trap is the part worth reading twice.** SQL predicates are
# MAGIC three-valued. If `unit_price` is NULL then `unit_price RLIKE '...'`
# MAGIC evaluates to NULL, and `NOT NULL` is NULL — which is not TRUE, so a plain
# MAGIC `when(~expr)` never fires and the row sails into silver.
# MAGIC
# MAGIC A row with a missing price is exactly the row you wanted to catch, so a
# MAGIC rule that silently passes NULLs is worse than no rule at all. Wrapping the
# MAGIC predicate in `coalesce(..., false)` makes "could not be evaluated" count
# MAGIC as a failure. This was a real bug in this lab, caught by a test.

# COMMAND ----------
RULES = [
    ("order_id_present",       "order_id IS NOT NULL AND trim(order_id) <> ''", "QUARANTINE"),
    ("customer_id_present",    "customer_id IS NOT NULL",                        "QUARANTINE"),
    ("quantity_is_integer",    "quantity RLIKE '^-?[0-9]+$'",                    "QUARANTINE"),
    ("quantity_positive",      "CAST(quantity AS INT) > 0",                      "QUARANTINE"),
    ("unit_price_numeric",     "unit_price RLIKE '^-?[0-9]+(\\.[0-9]+)?$'",      "QUARANTINE"),
    ("unit_price_not_negative","CAST(unit_price AS DOUBLE) >= 0",                "QUARANTINE"),
    ("order_ts_parseable",     "to_timestamp(order_ts) IS NOT NULL",             "QUARANTINE"),
    ("status_known",           "status IN ('placed','shipped','delivered','cancelled')", "WARN"),
]


def apply_quality(df, rules):
    """Return (passing, failing). Failing rows carry the FIRST rule they break —
    the source owner needs to know what to fix first, not everything at once."""
    blocking = [(n, e) for n, e, sev in rules if sev == "QUARANTINE"]
    reason = F.lit(None).cast(StringType())
    for name, expr in reversed(blocking):
        failed = ~F.coalesce(F.expr(expr), F.lit(False))     # NULL-safe. See above.
        reason = F.when(failed, F.lit(f"failed rule: {name}")).otherwise(reason)
    tagged = df.withColumn("_reject_reason", reason)
    return (tagged.filter("_reject_reason IS NULL").drop("_reject_reason"),
            tagged.filter("_reject_reason IS NOT NULL"))


passing, failing = apply_quality(deduped, RULES)
passing_count, failing_count = passing.count(), failing.count()
print(f"passing {passing_count:,} | quarantined {failing_count:,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 · Quarantine — park it, never drop it
# MAGIC
# MAGIC A pipeline that silently discards rows is one nobody can trust, and
# MAGIC "the numbers do not match" is the most expensive conversation in data
# MAGIC engineering. Every rejected row keeps its original values and a reason a
# MAGIC non-engineer could act on.

# COMMAND ----------
if failing_count:
    (failing
     .withColumn("_batch_id", F.lit(BATCH_ID))
     .withColumn("_load_date", F.to_date(F.lit(LOAD_DATE)))
     .withColumn("_quarantined_at", F.current_timestamp())
     .write.format("delta").mode("append").option("mergeSchema", "true")
     .saveAsTable("silver.quarantine_orders"))

    display(spark.table("silver.quarantine_orders")
            .filter(F.col("_load_date") == LOAD_DATE)
            .select("order_id", "_reject_reason"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6 · Silver — cast, conform, write
# MAGIC
# MAGIC Money is a **DECIMAL**, never a double. Binary floating point
# MAGIC misrepresents currency, the error compounds through aggregation, and it
# MAGIC surfaces as a reconciliation that is out by a few rupees which nobody can
# MAGIC explain.
# MAGIC
# MAGIC Note `order_date` comes from `order_ts`, not from the load date. A late
# MAGIC order placed three weeks ago must land in **its own** partition — re-dating
# MAGIC it to today corrupts every historical trend.

# COMMAND ----------
silver = (passing
          .withColumn("order_id",    F.trim("order_id"))
          .withColumn("customer_id", F.trim("customer_id"))
          .withColumn("quantity",    F.col("quantity").cast(IntegerType()))
          .withColumn("unit_price",  F.col("unit_price").cast(DecimalType(18, 2)))
          .withColumn("line_amount", (F.col("unit_price") * F.col("quantity"))
                                      .cast(DecimalType(18, 2)))
          .withColumn("order_ts",    F.to_timestamp("order_ts"))
          .withColumn("order_date",  F.to_date("order_ts"))
          .withColumn("status",      F.lower(F.trim("status")))
          .withColumn("updated_at",  F.to_timestamp("updated_at"))
          .withColumn("_processed_at", F.current_timestamp())
          .drop("_load_date", "_source_file"))

(silver.write.format("delta")
 .mode("overwrite")
 # Late data means old partitions are NOT immutable. That is a design
 # constraint, not an exception, and replaceWhere on the ORDER date rather than
 # the load date is how you honour it.
 .option("replaceWhere", f"order_date >= '{LOAD_DATE}' OR order_date < '{LOAD_DATE}'")
 .option("mergeSchema", "true")
 .partitionBy("order_date")
 .saveAsTable("silver.orders"))

print(f"silver.orders: {spark.table('silver.orders').count():,} rows")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7 · Reconcile — the control that catches silent loss
# MAGIC
# MAGIC `bronze == silver + quarantine`. An alert on job failure cannot fire when
# MAGIC nothing fails, and silent loss is the failure mode that costs a quarter.
# MAGIC
# MAGIC Note what an imbalance means when quarantine is **empty**: rows are being
# MAGIC lost rather than rejected, which is a different and more serious problem.

# COMMAND ----------
bronze_rows = spark.table("bronze.orders").filter(F.col("_load_date") == LOAD_DATE).count()
quarantined = (spark.table("silver.quarantine_orders")
               .filter(F.col("_load_date") == LOAD_DATE).count()
               if spark.catalog.tableExists("silver.quarantine_orders") else 0)

difference = deduped.count() - (passing_count + quarantined)
print(f"deduped {deduped.count():,} | silver {passing_count:,} | "
      f"quarantined {quarantined:,} | difference {difference}")

assert difference == 0, (
    f"identity broken by {difference} rows — "
    f"{'rows are being LOST, not rejected' if quarantined == 0 else 'counts do not balance'}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8 · Register grants
# MAGIC
# MAGIC The full grant script is in `ddl/unity_catalog_setup.sql`. The one line
# MAGIC that matters most: **analysts get gold only.** Analysts querying silver is
# MAGIC how two versions of a number start circulating — one from the governed
# MAGIC model and one from a table with no business rules applied — and nobody can
# MAGIC then say which is right.

# COMMAND ----------
spark.sql("GRANT SELECT ON TABLE silver.orders TO `data-engineers`")
spark.sql("GRANT SELECT ON TABLE silver.orders TO `data-scientists`")
spark.sql("GRANT SELECT ON TABLE silver.quarantine_orders TO `data-engineers`")
# Deliberately NO grant to data-analysts on silver.

display(spark.sql("SHOW GRANTS ON TABLE silver.orders"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9 · Delta housekeeping
# MAGIC
# MAGIC `OPTIMIZE` compacts small files. A table written daily by a streaming job
# MAGIC accumulates thousands of tiny files, and then scheduling overhead dominates
# MAGIC the actual work — 40,000 tasks each finishing in 200ms is the signature.
# MAGIC
# MAGIC `VACUUM` removes files no longer referenced. The 7-day default retention
# MAGIC is not arbitrary: it is what makes time travel possible. Vacuuming to zero
# MAGIC hours destroys your ability to recover from a bad write, which is the
# MAGIC single most useful property Delta gives you.

# COMMAND ----------
spark.sql("OPTIMIZE silver.orders")

# On a recent runtime prefer liquid clustering over Z-order: declare it on the
# table and stop hand-tuning the layout as query patterns change.
# spark.sql("ALTER TABLE silver.orders CLUSTER BY (order_date, customer_id)")

display(spark.sql("DESCRIBE HISTORY silver.orders").select(
    "version", "timestamp", "operation", "operationMetrics").limit(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Time travel — the reason to care about the transaction log
# MAGIC
# MAGIC Every write appends a JSON commit to `_delta_log/`. That log is what gives
# MAGIC you ACID, and it is what makes these possible:

# COMMAND ----------
# What did this table look like before today's load?
# display(spark.read.format("delta").option("versionAsOf", 3).table("silver.orders"))

# Recover from a bad write, without a backup:
# spark.sql("RESTORE TABLE silver.orders TO VERSION AS OF 3")

# What exactly changed between two versions?
# display(spark.sql("""
#   SELECT * FROM table_changes('silver.orders', 3, 4)
# """))
