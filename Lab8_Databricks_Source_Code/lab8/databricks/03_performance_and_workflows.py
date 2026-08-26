# Databricks notebook source
# MAGIC %md
# MAGIC # Performance and orchestration
# MAGIC
# MAGIC Partitioning, file sizing, skew, broadcast joins, caching, AQE — then
# MAGIC Workflows and job clusters.
# MAGIC
# MAGIC The rule that governs all of it: **measure, change ONE thing, measure
# MAGIC again.** Raising four settings and declaring victory tells you nothing
# MAGIC about which one helped, or whether any of them did.

# COMMAND ----------
dbutils.widgets.text("catalog", "aurora_dev")
CATALOG = dbutils.widgets.get("catalog")
spark.sql(f"USE CATALOG {CATALOG}")

from pyspark.sql import functions as F
import time

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1 · Read the Spark UI before touching anything
# MAGIC
# MAGIC Three task-distribution signatures cover almost every slow job:
# MAGIC
# MAGIC | What you see in the stage | What it is | What to do |
# MAGIC |---|---|---|
# MAGIC | 197 tasks in 10s, 3 tasks in 15min | **Skew** — a few partitions hold most of the data | Fix the join key or let AQE split it. Adding workers does nothing; they sit idle |
# MAGIC | 40,000 tasks each 200ms | **Small files** — scheduling overhead exceeds the work | `OPTIMIZE`, or fix whatever writes thousands of tiny files |
# MAGIC | All tasks slow, CPU pinned | Genuinely under-sized | Now a bigger cluster helps |
# MAGIC
# MAGIC Doubling the cluster when three tasks are doing all the work is the most
# MAGIC common wrong first move, and it makes the bill worse without making the
# MAGIC job faster.

# COMMAND ----------
def timed(label, fn):
    """Best-of-three. The mean measures whatever else the cluster was doing."""
    best = float("inf")
    for _ in range(3):
        start = time.time()
        result = fn()
        best = min(best, time.time() - start)
    print(f"{label:<44}{best:6.2f}s")
    return result

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2 · Partitioning — the one people over-apply
# MAGIC
# MAGIC Partition on a column that queries actually **filter** on, with enough
# MAGIC rows per partition to be worth a file. The failure mode is
# MAGIC over-partitioning: partitioning by customer_id on 50,000 customers gives
# MAGIC 50,000 directories holding a few rows each, and every query then pays
# MAGIC listing costs that dwarf the scan it saved.
# MAGIC
# MAGIC Rough guide: aim for partitions of at least ~1 GB. Below that, do not
# MAGIC partition — use clustering instead.

# COMMAND ----------
display(spark.sql(f"""
    SELECT order_date_key, COUNT(*) AS rows,
           ROUND(COUNT(*) / SUM(COUNT(*)) OVER () * 100, 2) AS pct_of_table
    FROM {CATALOG}.gold.fact_sales
    GROUP BY order_date_key ORDER BY rows DESC LIMIT 20
"""))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3 · Liquid clustering — prefer it to Z-order on a recent runtime
# MAGIC
# MAGIC Z-order is a one-off physical sort you re-apply as data grows. Liquid
# MAGIC clustering is declared on the table and maintained incrementally, and the
# MAGIC clustering keys can be **changed** without rewriting everything — which
# MAGIC matters because query patterns change and your original guess will be
# MAGIC wrong within a year.

# COMMAND ----------
# spark.sql(f"ALTER TABLE {CATALOG}.gold.fact_sales CLUSTER BY (customer_key, order_date_key)")
# spark.sql(f"OPTIMIZE {CATALOG}.gold.fact_sales")

# The older approach, still correct on runtimes without liquid clustering:
# spark.sql(f"OPTIMIZE {CATALOG}.gold.fact_sales ZORDER BY (customer_key, order_date_key)")

display(spark.sql(f"DESCRIBE DETAIL {CATALOG}.gold.fact_sales")
        .select("numFiles", "sizeInBytes", "clusteringColumns", "partitionColumns"))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4 · File sizing — the small-file problem, measured
# MAGIC
# MAGIC Target roughly 128 MB to 1 GB per file. Thousands of tiny files means the
# MAGIC driver spends longer listing and scheduling than the executors spend
# MAGIC reading.

# COMMAND ----------
detail = spark.sql(f"DESCRIBE DETAIL {CATALOG}.gold.fact_sales").first()
avg_mb = (detail.sizeInBytes / max(detail.numFiles, 1)) / 1024 / 1024
print(f"{detail.numFiles:,} files, average {avg_mb:.1f} MB")
if avg_mb < 32:
    print("  -> small-file problem. Run OPTIMIZE and find out what is writing "
          "so many files (usually a stream with too short a trigger, or an "
          "over-partitioned write).")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5 · Skew — and why AQE usually handles it now
# MAGIC
# MAGIC Adaptive Query Execution is on by default and does three things that used
# MAGIC to be manual work: it coalesces shuffle partitions after seeing real
# MAGIC sizes, it converts a sort-merge join to a broadcast when the small side
# MAGIC turns out to be small, and it splits skewed partitions.
# MAGIC
# MAGIC It cannot fix skew caused by a **single dominant key** — a
# MAGIC customer_id representing 40% of rows. For that you need salting, or a
# MAGIC different join strategy, or to question whether that key should be joined
# MAGIC at all.

# COMMAND ----------
print("AQE:            ", spark.conf.get("spark.sql.adaptive.enabled"))
print("skew join:      ", spark.conf.get("spark.sql.adaptive.skewJoin.enabled"))
print("coalesce parts: ", spark.conf.get("spark.sql.adaptive.coalescePartitions.enabled"))
print("broadcast limit:", spark.conf.get("spark.sql.autoBroadcastJoinThreshold"))

# Find a dominant key before blaming the cluster.
display(spark.sql(f"""
    SELECT customer_key, COUNT(*) AS rows,
           ROUND(COUNT(*) / SUM(COUNT(*)) OVER () * 100, 2) AS pct
    FROM {CATALOG}.gold.fact_sales
    GROUP BY customer_key ORDER BY rows DESC LIMIT 10
"""))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6 · Broadcast joins
# MAGIC
# MAGIC Broadcasting sends the small side to every executor so no shuffle is
# MAGIC needed. AQE now decides this automatically in most cases, and an explicit
# MAGIC hint is worth adding only when you know something the optimiser does not —
# MAGIC typically because statistics are stale.
# MAGIC
# MAGIC Broadcasting something too large is worse than not broadcasting: it puts
# MAGIC the whole table in every executor's memory and you get an OOM instead of a
# MAGIC slow join.

# COMMAND ----------
facts = spark.table(f"{CATALOG}.gold.fact_sales")
dim = spark.table(f"{CATALOG}.gold.dim_customer").filter("is_current")

timed("join, optimiser decides", lambda:
      facts.join(dim, "customer_key").count())

timed("join, broadcast hint", lambda:
      facts.join(F.broadcast(dim), "customer_key").count())

# Read the plan rather than guessing which one happened.
facts.join(dim, "customer_key").explain("formatted")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7 · Caching — narrower than people think
# MAGIC
# MAGIC Cache when a DataFrame is used **several times** in one job AND it fits
# MAGIC comfortably in memory. Caching something read once is pure cost: you pay
# MAGIC the materialisation and get nothing.
# MAGIC
# MAGIC On Databricks the disk cache handles repeated reads of the same Parquet
# MAGIC files automatically, which removes most of the reasons people reach for
# MAGIC `.cache()` in the first place.

# COMMAND ----------
# reused = spark.table(f"{CATALOG}.silver.orders").filter("status = 'delivered'")
# reused.cache()
# reused.count()      # materialise once
# ... several uses ...
# reused.unpersist()  # and release it. A cached DataFrame nobody released is
#                     # memory the next stage needed.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8 · Compute types — choosing the right one
# MAGIC
# MAGIC | Type | Use it for | The catch |
# MAGIC |---|---|---|
# MAGIC | **All-purpose** | Interactive development, notebooks | Stays alive between uses. Most expensive per hour, and the bill people are surprised by |
# MAGIC | **Job cluster** | Scheduled jobs | Created for the run and terminated after. Cheaper per DBU; pay a few minutes' start-up |
# MAGIC | **Serverless** | Both, increasingly | No start-up wait, no sizing decisions. Less control, and cost is per query rather than per hour |
# MAGIC | **SQL warehouse** | BI tools and SQL analysts | Optimised for concurrent short queries. Serverless variant starts in seconds |
# MAGIC
# MAGIC The single biggest cost mistake in a new workspace is running scheduled
# MAGIC jobs on an all-purpose cluster because it was already there. Job clusters
# MAGIC exist for exactly this and usually cost less than half.
# MAGIC
# MAGIC ### Cluster policies
# MAGIC
# MAGIC A policy caps what people can create: maximum workers, allowed node types,
# MAGIC mandatory auto-termination, required tags. Without policies you will find
# MAGIC a 40-node cluster somebody spun up for a one-off query in March and never
# MAGIC stopped. Two settings do most of the work:
# MAGIC
# MAGIC * `autotermination_minutes` — mandatory, and low. 20 minutes is generous.
# MAGIC * required `custom_tags` for cost centre — so the bill can be attributed,
# MAGIC   which is what makes the conversation about it possible at all.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9 · Workflows — one task per notebook, dependencies as edges
# MAGIC
# MAGIC The dependency order is the same one the local orchestrator uses:
# MAGIC bronze → silver → dimensions → facts → reconcile. **Dimensions before
# MAGIC facts**, because a fact cannot resolve a surrogate key for a dimension
# MAGIC version that has not been loaded yet — that is the whole of orchestration
# MAGIC in one sentence.

# COMMAND ----------
WORKFLOW = {
    "name": "aurora_daily",
    "job_clusters": [{
        "job_cluster_key": "etl",
        "new_cluster": {
            "spark_version": "15.4.x-scala2.12",
            "node_type_id": "Standard_DS4_v2",
            # Autoscale, not a fixed size: the silver step needs far more than
            # the reconcile step, and paying for the peak all night is waste.
            "autoscale": {"min_workers": 2, "max_workers": 8},
            "data_security_mode": "SINGLE_USER",   # required for Unity Catalog
            "custom_tags": {"cost_centre": "data-platform", "env": "prod"},
        },
    }],
    "tasks": [
        {"task_key": "bronze",
         "job_cluster_key": "etl",
         "notebook_task": {"notebook_path": "/Repos/aurora/01_lab8_bronze_silver"},
         # Retry transient failures, but not many times: a job that retries five
         # times turns a 10-minute failure into an hour of not knowing.
         "max_retries": 2, "min_retry_interval_millis": 300000,
         "timeout_seconds": 3600},

        {"task_key": "gold_dims",
         "depends_on": [{"task_key": "bronze"}],
         "job_cluster_key": "etl",
         "notebook_task": {"notebook_path": "/Repos/aurora/02_gold_scd2_merge"},
         "max_retries": 2, "timeout_seconds": 3600},

        {"task_key": "reconcile",
         "depends_on": [{"task_key": "gold_dims"}],
         "job_cluster_key": "etl",
         "notebook_task": {"notebook_path": "/Repos/aurora/04_reconcile"},
         "max_retries": 0,          # a failed reconciliation must not be retried
                                    # into passing; it means investigate
         "timeout_seconds": 1800},
    ],
    "schedule": {"quartz_cron_expression": "0 0 2 * * ?",
                 "timezone_id": "UTC",
                 "pause_status": "PAUSED"},   # committed paused, like an ADF trigger
    "email_notifications": {"on_failure": ["data-platform@aurora.example"]},
    "health": {"rules": [{"metric": "RUN_DURATION_SECONDS",
                          "op": "GREATER_THAN", "value": 7200}]},
    "max_concurrent_runs": 1,       # a second run starting while the first is
                                    # mid-merge is how a dimension gets two
                                    # current versions
}

import json
print(json.dumps(WORKFLOW, indent=2))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10 · Governance — lineage and audit
# MAGIC
# MAGIC Lineage is captured automatically, but **only** when the read and the
# MAGIC write both go through Unity Catalog. A job reading a storage path directly
# MAGIC produces no lineage, and the graph then has a hole exactly where the
# MAGIC interesting transformation was.
# MAGIC
# MAGIC Two queries worth knowing before you need them.

# COMMAND ----------
# What feeds this table, and what reads it?
display(spark.sql(f"""
    SELECT source_table_full_name, target_table_full_name,
           entity_type, MAX(event_time) AS last_seen
    FROM system.access.table_lineage
    WHERE target_table_full_name = '{CATALOG}.gold.fact_sales'
       OR source_table_full_name = '{CATALOG}.gold.fact_sales'
    GROUP BY 1, 2, 3 ORDER BY last_seen DESC LIMIT 20
"""))

# COMMAND ----------
# Who read the customer dimension in the last week, and from where?
display(spark.sql("""
    SELECT event_time, user_identity.email AS who, action_name,
           request_params.full_name_arg AS object, source_ip_address
    FROM system.access.audit
    WHERE service_name = 'unityCatalog'
      AND action_name IN ('getTable', 'generateTemporaryTableCredential')
      AND request_params.full_name_arg LIKE '%dim_customer%'
      AND event_date >= current_date() - INTERVAL 7 DAYS
    ORDER BY event_time DESC LIMIT 50
"""))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11 · Cost, in one query
# MAGIC
# MAGIC The tagging in the job cluster above is what makes this answerable. Without
# MAGIC required tags on a cluster policy, this returns a number nobody can
# MAGIC attribute — and an unattributable bill never gets reduced.

# COMMAND ----------
display(spark.sql("""
    SELECT u.usage_metadata.job_id,
           u.custom_tags.cost_centre,
           SUM(u.usage_quantity) AS dbus,
           ROUND(SUM(u.usage_quantity * p.pricing.default), 2) AS approx_cost
    FROM system.billing.usage u
    JOIN system.billing.list_prices p
      ON u.sku_name = p.sku_name AND u.usage_end_time >= p.price_start_time
    WHERE u.usage_date >= current_date() - INTERVAL 30 DAYS
    GROUP BY 1, 2 ORDER BY approx_cost DESC LIMIT 20
"""))
