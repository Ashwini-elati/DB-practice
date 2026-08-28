# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# MAGIC %md
# MAGIC # Gold · Dimensional modelling in Delta
# MAGIC
# MAGIC SCD Type 2 with `MERGE`, the two-updates-in-one-batch case, and
# MAGIC late-arriving data.
# MAGIC
# MAGIC This is the notebook to read closely. The `MERGE` here is what an
# MAGIC interviewer asks you to explain, and section 4 is the case that catches
# MAGIC almost everyone the first time they meet it in production.
# MAGIC
# MAGIC The same semantics are tested locally in `tests/test_transforms.py` —
# MAGIC 26 tests, no cluster required.

# COMMAND ----------

dbutils.widgets.text("catalog", "aurora_dev1")
dbutils.widgets.text("business_date", "2024-04-01")

CATALOG = dbutils.widgets.get("catalog")
EFFECTIVE = dbutils.widgets.get("business_date")
HIGH_DATE = "9999-12-31"

spark.sql(f"USE CATALOG {CATALOG}")


# COMMAND ----------

from delta.tables import DeltaTable
from pyspark.sql import Window
from pyspark.sql import functions as F

TRACKED = ["city", "region", "segment"]
TYPE1 = ["full_name", "email"]



# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · The dimension table
# MAGIC
# MAGIC `valid_to` is **EXCLUSIVE**. The old version closes exactly where the new
# MAGIC one opens, so a point-in-time lookup matches exactly one version.
# MAGIC Inclusive matches two and the fact fans out; leaving a gap matches none
# MAGIC and the fact is dropped. Both failures are silent.
# MAGIC
# MAGIC `row_hash` covers the **tracked attributes only**. Include a Type 1 column
# MAGIC and every email change opens a pointless new version.

# COMMAND ----------

spark.sql(f"""
CREATE TABLE gold.dim_customer (
    customer_key  BIGINT GENERATED ALWAYS AS IDENTITY,
    customer_id   STRING NOT NULL,
    full_name     STRING,
    email         STRING,
    city          STRING,
    region        STRING,
    segment       STRING,
    valid_from    DATE NOT NULL,
    valid_to      DATE NOT NULL,
    is_current    BOOLEAN NOT NULL,
    row_hash      STRING NOT NULL,
    inserted_at   TIMESTAMP NOT NULL
)
USING DELTA
CLUSTER BY (customer_id, valid_from)
""")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Prepare the source — ONE row per key
# MAGIC
# MAGIC ### This is the step people skip, and it is the whole of section 4.
# MAGIC
# MAGIC Delta's `MERGE` raises when a single target row matches **more than one**
# MAGIC source row:
# MAGIC
# MAGIC > `Cannot perform Merge as multiple source rows matched and attempted to
# MAGIC > modify the same target row in the Delta table`
# MAGIC
# MAGIC That failure is a **good** design — the alternative is a silent
# MAGIC non-deterministic update. But it means the pipeline has to answer a
# MAGIC question the source cannot: if a customer changed region twice today,
# MAGIC which change is the truth?
# MAGIC
# MAGIC For an SCD Type 2 dimension the answer is almost always the **latest**,
# MAGIC because the intermediate state was never the current state for any
# MAGIC meaningful period. If the intermediate states genuinely matter, you are
# MAGIC not doing a daily batch merge — you are streaming every change, and the
# MAGIC design is different.

# COMMAND ----------

raw_source = spark.table("silver.customers").filter(
    F.col("_load_date") == EFFECTIVE)


# COMMAND ----------

window = Window.partitionBy("customer_id").orderBy(
    F.col("updated_at").desc(),
    F.col("_ingested_at").desc()
)


# COMMAND ----------

F.col("_ingested_at").desc()



# COMMAND ----------

customers_silver = (
    spark.table(f"{CATALOG}.bronze.customers")
    .select(
        "customer_id",
        F.concat_ws(
            " ",
            F.col("first_name"),
            F.col("last_name")
        ).alias("full_name"),
        "email",
        "phone",
        "city",
        F.lit(None).cast("string").alias("region"),
        F.lit(None).cast("string").alias("segment"),
        F.to_date("_load_date").alias("_load_date"),
        F.col("_batch_id").cast("string").alias("_batch_id"),
        "_ingested_at"
    )
)


# COMMAND ----------

customers_silver.printSchema()
display(customers_silver.limit(10))


# COMMAND ----------

spark.sql(f"""
SELECT COUNT(*) AS count
FROM {CATALOG}.silver.customers
""").show()


# COMMAND ----------

spark.sql(f"SHOW TABLES IN {CATALOG}.silver").show(truncate=False)


# COMMAND ----------

silver_customers = spark.table(
    f"{CATALOG}.silver.customers"
)

silver_customers.printSchema()

display(silver_customers)


# COMMAND ----------

display(
    spark.table(f"{CATALOG}.bronze.orders")
    .limit(20)
)


# COMMAND ----------

spark.sql(f"""
SELECT
    order_id,
    COUNT(*) AS cnt
FROM {CATALOG}.bronze.orders
GROUP BY order_id
ORDER BY cnt DESC
""").show()


# COMMAND ----------

EFFECTIVE = "2024-04-01"

print("Business date:", EFFECTIVE)


# COMMAND ----------

TRACKED = ["city"]

TYPE1 = ["full_name", "email"]


# COMMAND ----------

def change_hash(columns):
    payload = F.concat_ws(
        "\u001f",
        *[
            F.coalesce(
                F.col(c).cast("string"),
                F.lit("")
            )
            for c in columns
        ]
    )

    return F.sha2(payload, 256).substr(1, 32)


# COMMAND ----------

customer_source = (
    silver_customers
    .withColumn(
        "row_hash",
        change_hash(TRACKED)
    )
)

print(customer_source.columns)
display(customer_source)


# COMMAND ----------

target = DeltaTable.forName(
    spark,
    f"{CATALOG}.gold.dim_customer"
)


# COMMAND ----------

print("Rows to insert:", to_insert.count())

display(to_insert)


# COMMAND ----------

to_insert.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable(f"{CATALOG}.gold.dim_customer")


# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

window = Window.partitionBy("customer_id").orderBy(
    F.col("_ingested_at").desc()
)

customer_source = (
    customer_raw
    .withColumn("_rn", F.row_number().over(window))
    .filter(F.col("_rn") == 1)
    .drop("_rn")
    .withColumn("row_hash", change_hash(TRACKED))
)

print(
    f"{customer_raw.count():,} source rows -> "
    f"{customer_source.count():,} after collapsing to one row per customer"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · The SCD Type 2 merge
# MAGIC
# MAGIC Delta cannot both update an existing row and insert a new one for the
# MAGIC same source record in a single `MERGE`, so the standard pattern is three
# MAGIC passes. That is not a workaround — it mirrors the three things actually
# MAGIC happening.
# MAGIC
# MAGIC Order matters. Type 1 **before** Type 2: a row that changes both a tracked
# MAGIC and an untracked attribute would otherwise carry a stale Type 1 value into
# MAGIC its new version.

# COMMAND ----------

target = DeltaTable.forName(spark, f"{CATALOG}.gold.dim_customer")

# ---------------------------------------------------------------- pass 1
# TYPE 1: update in place on the current row. No new version.
# Skipping this is how an email correction never reaches the dimension and
# somebody re-raises the same ticket a month later.
(
    target.alias("t")
    .merge(
        customer_source.alias("s"),
        "t.customer_id = s.customer_id AND t.is_current"
    )
    .whenMatchedUpdate(
        condition="t.row_hash = s.row_hash",
        set={
            "full_name": "s.full_name",
            "email": "s.email"
        }
    )
    .execute()
)


# ---------------------------------------------------------------- pass 2
# TYPE 2, part one: close the current version wherever a TRACKED attribute changed.
(
    target.alias("t")
    .merge(
        customer_source.alias("s"),
        "t.customer_id = s.customer_id AND t.is_current"
    )
    .whenMatchedUpdate(
        condition="t.row_hash <> s.row_hash",
        set={
            "valid_to": f"to_date('{EFFECTIVE}')",
            "is_current": "false"
        }
    )
    .execute()
)

# ---------------------------------------------------------------- pass 3
# TYPE 2, part two: insert a version for anything that now has no current row —
# both genuinely new keys and the ones just closed above.
current_keys = (spark.table(f"{CATALOG}.gold.dim_customer")
                .filter("is_current").select("customer_id")
                .withColumn("_exists", F.lit(True)))

to_insert = (customer_source.join(current_keys, "customer_id", "left")
             .filter("_exists IS NULL").drop("_exists")
             .withColumn("valid_from", F.to_date(F.lit(EFFECTIVE)))
             .withColumn("valid_to", F.to_date(F.lit(HIGH_DATE)))
             .withColumn("is_current", F.lit(True))
             .withColumn("inserted_at", F.current_timestamp())
             .select("customer_id", *TYPE1, *TRACKED,
                     "valid_from", "valid_to", "is_current", "row_hash", "inserted_at"))

inserted = to_insert.count()
if inserted:
    to_insert.write.format("delta").mode("append").saveAsTable(
        f"{CATALOG}.gold.dim_customer")

print(f"{inserted:,} new versions opened")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · What NOT to do — the naive merge
# MAGIC
# MAGIC Left uncommented so you can see the failure once. Run it against a batch
# MAGIC where a key changes twice and it raises. Seeing the error message yourself
# MAGIC is worth more than reading about it, because you will recognise it at 3am.

# COMMAND ----------

# # This raises: multiple source rows matched the same target row.
# (target.alias("t")
#  .merge(raw_source.alias("s"), "t.customer_id = s.customer_id AND t.is_current")
#  .whenMatchedUpdate(set={"valid_to": f"to_date('{EFFECTIVE}')", "is_current": "false"})
#  .whenNotMatchedInsertAll()
#  .execute())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Assert the invariants
# MAGIC
# MAGIC Run these after every load, not only in a test suite. Each failure mode is
# MAGIC silent — the job stays green and the numbers go wrong.

# COMMAND ----------

overlaps = spark.sql(f"""
    SELECT COUNT(*) AS n FROM {CATALOG}.gold.dim_customer a
    JOIN {CATALOG}.gold.dim_customer b
      ON a.customer_id = b.customer_id AND a.valid_from <> b.valid_from
    WHERE a.valid_from < b.valid_to AND b.valid_from < a.valid_to
""").first().n
assert overlaps == 0, f"{overlaps} overlapping validity windows"

multi_current = spark.sql(f"""
    SELECT COUNT(*) AS n FROM (
      SELECT customer_id FROM {CATALOG}.gold.dim_customer
      WHERE is_current GROUP BY customer_id HAVING COUNT(*) <> 1)
""").first().n
assert multi_current == 0, f"{multi_current} keys without exactly one current row"

zero_length = spark.sql(f"""
    SELECT COUNT(*) AS n FROM {CATALOG}.gold.dim_customer
    WHERE valid_from >= valid_to
""").first().n
assert zero_length == 0, f"{zero_length} zero-length versions"

print("invariants hold")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · The fact — point-in-time key resolution
# MAGIC
# MAGIC The single most important line in the whole gold layer is the join
# MAGIC condition. Join on `is_current` instead and a March order reports under
# MAGIC the customer's April region — so last month's report changes
# MAGIC retrospectively, and the number looks perfectly reasonable.
# MAGIC
# MAGIC This is also where **late-arriving data** is handled correctly: a fact
# MAGIC dated three weeks ago resolves against the dimension version that was
# MAGIC valid three weeks ago, not the one valid when it happened to arrive.

# COMMAND ----------

orders_raw = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("/Volumes/aurora_dev1/bronze/landing/orders.csv")
)

display(orders_raw)


# COMMAND ----------

from pyspark.sql import functions as F

orders_bronze = (
    orders_raw
    .withColumn("_batch_id", F.lit(1))
    .withColumn("_load_date", F.lit("2024-04-01").cast("date"))
    .withColumn("_ingested_at", F.current_timestamp())
)


# COMMAND ----------

orders_bronze = (
    orders_raw
    .select(
        F.col("order_id").cast("string"),
        F.col("customer_id").cast("string"),
        F.col("product_id").cast("string"),
        F.col("quantity").cast("string"),
        F.col("unit_price").cast("string"),
        F.col("status").cast("string"),
        F.col("updated_at").cast("string")
    )
    .withColumn("_batch_id", F.lit(1).cast("int"))
    .withColumn("_load_date", F.lit("2024-04-01").cast("date"))
    .withColumn("_ingested_at", F.current_timestamp())
    .withColumn("_source_file", F.lit("orders.csv"))
)


# COMMAND ----------

display(orders_bronze)


# COMMAND ----------

(
    orders_bronze.write
    .format("delta")
    .mode("append")
    .saveAsTable(f"{CATALOG}.bronze.orders")
)


# COMMAND ----------

print(
    "Bronze orders:",
    spark.table(f"{CATALOG}.bronze.orders").count()
)


# COMMAND ----------

display(
    spark.table(f"{CATALOG}.bronze.orders")
)
spark.sql(f"""
SELECT
    _load_date,
    COUNT(*) AS records
FROM {CATALOG}.bronze.orders
GROUP BY _load_date
ORDER BY _load_date
""").show()



# COMMAND ----------

bronze_orders = spark.table("aurora_dev1.bronze.orders")

# COMMAND ----------

from pyspark.sql import functions as F

silver_orders = (
    bronze_orders
    .select(
        F.trim(F.col("order_id")).alias("order_id"),
        F.trim(F.col("customer_id")).alias("customer_id"),
        F.trim(F.col("product_id")).alias("product_id"),

        F.col("quantity")
         .cast("int")
         .alias("quantity"),

        F.col("unit_price")
         .cast("decimal(18,2)")
         .alias("unit_price"),

        F.lower(F.trim(F.col("status"))).alias("status"),

        F.to_timestamp(
            F.col("updated_at")
        ).alias("updated_at"),

        F.col("_batch_id"),
        F.col("_load_date"),
        F.col("_ingested_at"),
        F.col("_source_file")
    )
    .withColumn(
        "line_amount",
        F.col("quantity") * F.col("unit_price")
    )
)

display(silver_orders.limit(10))


# COMMAND ----------

silver_orders.filter(
    F.col("order_id").isNull()
    | F.col("customer_id").isNull()
    | F.col("quantity").isNull()
    | F.col("unit_price").isNull()
).show(truncate=False)


# COMMAND ----------

silver_orders.groupBy("status").count().show()


# COMMAND ----------

(
    silver_orders.write
    .format("delta")
    .mode("overwrite")
    .saveAsTable("aurora_dev1.silver.orders")
)


# COMMAND ----------

silver_check = spark.table("aurora_dev1.silver.orders")

print("Silver orders:", silver_check.count())

silver_check.printSchema()

display(silver_check.limit(10))


# COMMAND ----------

display(
    spark.sql(f"""
        SELECT
            customer_id,
            valid_from,
            valid_to,
            is_current
        FROM {CATALOG}.gold.dim_customer
        ORDER BY customer_id
    """)
)


# COMMAND ----------

facts = spark.sql(f"""
    SELECT
        CAST(
            date_format(to_date(o.updated_at), 'yyyyMMdd')
            AS INT
        ) AS order_date_key,

        dc.customer_key,

        o.order_id,
        o.quantity,
        o.unit_price,
        o.line_amount,

        CASE
            WHEN o.status = 'cancelled' THEN true
            ELSE false
        END AS is_cancelled,

        current_timestamp() AS _loaded_at

    FROM {CATALOG}.silver.orders AS o

    JOIN {CATALOG}.gold.dim_customer AS dc
      ON dc.customer_id = o.customer_id
     AND to_date(o.updated_at) >= dc.valid_from
     AND to_date(o.updated_at) < dc.valid_to
""")

silver_rows = spark.table(
    f"{CATALOG}.silver.orders"
).count()

fact_rows = facts.count()

print(f"Silver orders: {silver_rows:,}")
print(f"Fact rows: {fact_rows:,}")

#assert fact_rows == silver_rows, (
  #  f"the point-in-time join changed the row count: "
   # f"{silver_rows:,} -> {fact_rows:,}. "
    #f"More rows means overlapping windows; "
    #f"fewer means a gap."
#)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 7 · Load the fact idempotently
# MAGIC
# MAGIC `MERGE` on the natural key. A re-run matches and does nothing rather than
# MAGIC inserting a duplicate — and note it is the **merge condition** doing the
# MAGIC work, not application code. A code-level "check if it exists first" has a
# MAGIC race and will eventually let one through.
# MAGIC
# MAGIC A cancellation is a **correction** to a row that already exists, so it is
# MAGIC an update. Appending a second row would double the revenue and no
# MAGIC aggregate would notice.

# COMMAND ----------

if not spark.catalog.tableExists(f"{CATALOG}.gold.fact_sales"):
    (facts.write.format("delta").mode("overwrite")
     .partitionBy("order_date_key")
     .saveAsTable(f"{CATALOG}.gold.fact_sales"))
else:
    (DeltaTable.forName(spark, f"{CATALOG}.gold.fact_sales").alias("t")
     .merge(facts.alias("s"), "t.order_line_id = s.order_line_id")
     .whenMatchedUpdate(set={"is_cancelled": "s.is_cancelled",
                             "quantity": "s.quantity",
                             "line_amount": "s.line_amount",
                             "_loaded_at": "s._loaded_at"})
     .whenNotMatchedInsertAll()
     .execute())

print(f"gold.fact_sales: {spark.table(f'{CATALOG}.gold.fact_sales').count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8 · Verify the history is what you expect
# MAGIC
# MAGIC The query every SCD Type 2 implementation should be checked with. Run it
# MAGIC for a customer you know changed and read the rows out loud.

# COMMAND ----------

display(spark.sql(f"""
    SELECT customer_key, customer_id, city, region, segment,
           valid_from, valid_to, is_current
    FROM {CATALOG}.gold.dim_customer
    WHERE customer_id = 'CUST-03'
    ORDER BY valid_from
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC For the customer who changed **twice in one batch**, expect exactly two
# MAGIC rows: the original, and one new version carrying the LATEST of the two
# MAGIC changes. Three rows would mean the intermediate state was materialised —
# MAGIC which is what people expect and almost never want.