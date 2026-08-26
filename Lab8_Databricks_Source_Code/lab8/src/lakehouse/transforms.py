"""The lakehouse transforms, in PySpark, runnable locally.

The notebooks in databricks/ are the deliverable — they use Delta and Unity
Catalog and run on a workspace. This module holds the same LOGIC in plain
PySpark so it can be executed and tested on a laptop, which matters for the
three things that are genuinely easy to get wrong:

  1. Deduplication that survives a tie
  2. The SCD Type 2 merge when a key changes TWICE IN ONE BATCH
  3. Late-arriving data landing in a partition that already exists

Delta's MERGE will throw on case 2 rather than silently doing the wrong thing,
which is a good design — but only if you know why it throws. The dedup-before-
merge here is what the notebook does, and this is where it is proved.

    python -m lakehouse.pipeline demo
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (BooleanType, DateType, DecimalType, IntegerType,
                               StringType, StructField, StructType, TimestampType)

HIGH_DATE = "9999-12-31"


def build_session(app: str = "lakehouse", shuffle_partitions: int = 4) -> SparkSession:
    """A local session sized for a laptop.

    shuffle.partitions defaults to 200, which on a 500-row test means 200 tasks
    doing nothing. On a cluster you leave it alone (or let AQE coalesce it); here
    it is the difference between two seconds and forty.
    """
    session = (SparkSession.builder
               .master("local[2]")
               .appName(app)
               .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
               .config("spark.sql.adaptive.enabled", "true")
               .config("spark.ui.showConsoleProgress", "false")
               .getOrCreate())
    session.sparkContext.setLogLevel("ERROR")
    return session


# ============================================================== BRONZE
BRONZE_SCHEMA = StructType([
    StructField("order_id",     StringType(),  True),
    StructField("customer_id",  StringType(),  True),
    StructField("product_id",   StringType(),  True),
    StructField("quantity",     StringType(),  True),   # read as string on purpose
    StructField("unit_price",   StringType(),  True),   # see the note below
    StructField("order_ts",     StringType(),  True),
    StructField("status",       StringType(),  True),
    StructField("updated_at",   StringType(),  True),
])


def read_landing(spark: SparkSession, path: str) -> DataFrame:
    """Read what ADF landed.

    Every column comes in as a STRING deliberately. Letting the reader infer
    types means a file where one row has 'N/A' in a numeric column silently
    changes that column's type for the whole batch — and the schema of your
    bronze table then depends on the worst row in the file. Read as text, cast
    in silver, and keep the decision visible.
    """
    return (spark.read
            .option("header", "true")
            .schema(BRONZE_SCHEMA)
            .csv(path))


def to_bronze(df: DataFrame, batch_id: int, load_date: str) -> DataFrame:
    """Add lineage and write. Bronze rejects nothing and corrects nothing.

    If your bronze layer rejects a row you have written silver by accident and
    lost the ability to re-derive. The whole value of bronze is that it is a
    faithful copy you can reprocess from.
    """
    return (df
            .withColumn("_batch_id", F.lit(batch_id))
            .withColumn("_load_date", F.lit(load_date).cast(DateType()))
            .withColumn("_ingested_at", F.current_timestamp())
            .withColumn("_source_file", F.input_file_name()))


# ============================================================== SILVER
def deduplicate(df: DataFrame, keys: list[str], order_by: list[str]) -> DataFrame:
    """One row per business key, keeping the latest.

    row_number, not rank. Rank returns 1 for every tied row, so a tie hands back
    two rows and silently reintroduces the duplicate you were removing.

    The order_by list must end in something guaranteed unique. Without a
    deterministic tie-break, which row survives is arbitrary and the output
    changes between runs — so two people running the same code on the same data
    get different answers, and neither can reproduce the other's.
    """
    window = Window.partitionBy(*keys).orderBy(
        *[F.col(c).desc() for c in order_by])
    return (df.withColumn("_rn", F.row_number().over(window))
              .filter(F.col("_rn") == 1)
              .drop("_rn"))


# Quality rules as data. Each is a name, a predicate rows must SATISFY, and a
# severity. Expressing them declaratively means the rule set is reviewable by
# somebody who does not read PySpark, and a new rule is a row rather than a commit.
QUALITY_RULES: list[tuple[str, str, str]] = [
    ("order_id_present",     "order_id IS NOT NULL AND trim(order_id) <> ''",  "QUARANTINE"),
    ("customer_id_present",  "customer_id IS NOT NULL",                        "QUARANTINE"),
    ("quantity_is_integer",  "quantity RLIKE '^-?[0-9]+$'",                     "QUARANTINE"),
    ("quantity_positive",    "CAST(quantity AS INT) > 0",                       "QUARANTINE"),
    ("unit_price_numeric",   "unit_price RLIKE '^-?[0-9]+(\\\\.[0-9]+)?$'",     "QUARANTINE"),
    ("unit_price_not_negative", "CAST(unit_price AS DOUBLE) >= 0",              "QUARANTINE"),
    ("order_ts_parseable",   "to_timestamp(order_ts) IS NOT NULL",              "QUARANTINE"),
    ("status_known",         "status IN ('placed','shipped','delivered','cancelled')", "WARN"),
]


def apply_quality(df: DataFrame,
                  rules: list[tuple[str, str, str]] = QUALITY_RULES
                  ) -> tuple[DataFrame, DataFrame]:
    """Split into (passing, failing). Failing rows carry the FIRST rule they break.

    Naming one rule is more actionable than listing all of them: the person who
    owns the source system needs to know what to fix first, not everything that
    is wrong at once.

    WARN rules are recorded but do not quarantine. Severity being data rather
    than a code path is what lets a rule be tightened in production without a
    deployment.
    """
    blocking = [(name, expr) for name, expr, sev in rules if sev == "QUARANTINE"]

    reason = F.lit(None).cast(StringType())
    for name, expr in reversed(blocking):
        # NULL-SAFE, and this matters more than it looks.
        #
        # SQL predicates are three-valued. If unit_price is NULL then
        # `unit_price RLIKE '...'` evaluates to NULL, and NOT NULL is NULL —
        # which is not TRUE, so a plain `when(~expr)` never fires and the row
        # sails through into silver.
        #
        # A row with a missing price is exactly the row you wanted to catch, so
        # a rule that silently passes NULLs is worse than no rule at all. Wrap
        # the predicate so that "could not be evaluated" counts as a failure.
        failed = ~F.coalesce(F.expr(expr), F.lit(False))
        reason = F.when(failed, F.lit(f"failed rule: {name}")).otherwise(reason)

    tagged = df.withColumn("_reject_reason", reason)
    return (tagged.filter(F.col("_reject_reason").isNull()).drop("_reject_reason"),
            tagged.filter(F.col("_reject_reason").isNotNull()))


def to_silver(df: DataFrame) -> DataFrame:
    """Cast and conform. This is where the string columns become real types.

    Money is a DECIMAL, never a double. Binary floating point misrepresents
    currency, the error compounds through aggregation, and it surfaces as a
    reconciliation that is out by a few rupees which nobody can explain.
    """
    return (df
            .withColumn("order_id",    F.trim("order_id"))
            .withColumn("customer_id", F.trim("customer_id"))
            .withColumn("product_id",  F.trim("product_id"))
            .withColumn("quantity",    F.col("quantity").cast(IntegerType()))
            .withColumn("unit_price",  F.col("unit_price").cast(DecimalType(18, 2)))
            .withColumn("line_amount",
                        (F.col("unit_price") * F.col("quantity")).cast(DecimalType(18, 2)))
            .withColumn("order_ts",    F.to_timestamp("order_ts"))
            .withColumn("order_date",  F.to_date("order_ts"))
            .withColumn("status",      F.lower(F.trim("status")))
            .withColumn("updated_at",  F.to_timestamp("updated_at"))
            .withColumn("_processed_at", F.current_timestamp()))


# ======================================================== SCD TYPE 2
def prepare_scd2_source(df: DataFrame, natural_key: str, tracked: list[str],
                        order_by: list[str]) -> DataFrame:
    """Collapse a batch to ONE row per key before it reaches MERGE.

    This function exists because of the case that catches everyone.

    Delta's MERGE raises when a single target row matches MORE THAN ONE source
    row. If a customer changed region twice in one batch, the source has two
    rows for that key, and the merge fails with:

        Cannot perform Merge as multiple source rows matched and attempted to
        modify the same target row in the Delta table

    That failure is a GOOD design — the alternative is a silent non-deterministic
    update. But the pipeline still has to answer the question: which of the two
    changes is the truth?

    For an SCD Type 2 dimension the answer is almost always the LATEST, because
    the intermediate state was never the current state for any meaningful period.
    If the intermediate states genuinely matter, you are not doing a daily batch
    merge — you are streaming every change, and the design is different.

    So: deduplicate to the latest per key, then merge. One row per key means at
    most one target match, and the merge cannot throw.
    """
    window = Window.partitionBy(natural_key).orderBy(
        *[F.col(c).desc() for c in order_by])
    return (df.withColumn("_rn", F.row_number().over(window))
              .filter(F.col("_rn") == 1)
              .drop("_rn"))


def scd2_change_hash(tracked: list[str]):
    """A hash over the TRACKED attributes only.

    Include a Type 1 column here and every email change opens a pointless new
    version. Comparing a hash rather than each column also keeps the merge
    condition readable as the dimension grows.
    """
    payload = F.concat_ws("\u001f", *[F.coalesce(F.col(c).cast("string"), F.lit(""))
                                      for c in tracked])
    return F.sha2(payload, 256).substr(1, 32)


def scd2_merge_pure(current: DataFrame, incoming: DataFrame, *, natural_key: str,
                    tracked: list[str], type1: list[str], effective_date: str
                    ) -> tuple[DataFrame, dict[str, int]]:
    """A pure-PySpark SCD Type 2 merge, so the semantics can be tested locally.

    Delta does this in one MERGE statement (see databricks/03_gold_scd2.py). The
    longhand version is here because it is the version an interviewer asks you
    to explain, and because it can be asserted on without a cluster.

    Four outcomes per incoming row, and the whole of SCD2 is deciding which:

        new         no current version exists      -> insert version 1
        changed     a tracked attribute differs    -> close the old, open a new
        type1       only an untracked attr differs -> update in place, no version
        unchanged   nothing differs                -> write nothing at all

    The unchanged path is what makes the load idempotent: re-running a batch
    finds everything unchanged and the dimension does not grow a duplicate
    version every time somebody retries a failed job.

    valid_to is EXCLUSIVE. The old version closes exactly where the new one
    opens, so a point-in-time lookup matches exactly one version — inclusive
    matches two and the fact fans out; a gap matches none and the fact is dropped.
    """
    incoming = incoming.withColumn("_hash", scd2_change_hash(tracked))

    open_versions = current.filter(F.col("is_current"))
    closed_versions = current.filter(~F.col("is_current"))

    joined = open_versions.alias("t").join(
        incoming.alias("s"), F.col(f"t.{natural_key}") == F.col(f"s.{natural_key}"),
        "full_outer")

    # Classify every row once, so the four outcomes are visible in the data
    # rather than buried in control flow.
    classified = joined.withColumn(
        "_outcome",
        F.when(F.col(f"t.{natural_key}").isNull(), F.lit("new"))
         .when(F.col(f"s.{natural_key}").isNull(), F.lit("absent"))
         .when(F.col("t.row_hash") != F.col("s._hash"), F.lit("changed"))
         .otherwise(F.lit("unchanged")))

    counts = {row["_outcome"]: row["n"] for row in
              classified.groupBy("_outcome").agg(F.count("*").alias("n")).collect()}

    def source_columns(prefix: str, extra: dict | None = None) -> list:
        cols = [F.col(f"{prefix}.{natural_key}").alias(natural_key)]
        cols += [F.col(f"{prefix}.{c}").alias(c) for c in tracked + type1]
        for name, expression in (extra or {}).items():
            cols.append(expression.alias(name))
        return cols

    # ---- rows that keep their current version, with Type 1 refreshed --------
    # A Type 1 attribute changing must be applied even when nothing tracked
    # changed. Skipping this is how an email correction never reaches the
    # dimension and somebody re-raises the same ticket a month later.
    unchanged = (classified.filter(F.col("_outcome") == "unchanged")
                 .select(source_columns("s", {
                     "valid_from": F.col("t.valid_from"),
                     "valid_to": F.col("t.valid_to"),
                     "is_current": F.lit(True),
                     "row_hash": F.col("t.row_hash")})))

    # ---- rows present in the dimension but absent from this batch ----------
    # Absent is NOT deleted. An incremental batch only carries what changed, so
    # treating absence as a delete would close every version every night.
    absent = (classified.filter(F.col("_outcome") == "absent")
              .select(source_columns("t", {
                  "valid_from": F.col("t.valid_from"),
                  "valid_to": F.col("t.valid_to"),
                  "is_current": F.lit(True),
                  "row_hash": F.col("t.row_hash")})))

    # ---- the old version being closed --------------------------------------
    closing = (classified.filter(F.col("_outcome") == "changed")
               .select(source_columns("t", {
                   "valid_from": F.col("t.valid_from"),
                   "valid_to": F.lit(effective_date).cast(DateType()),
                   "is_current": F.lit(False),
                   "row_hash": F.col("t.row_hash")})))

    # ---- the new version, for both changed and new keys --------------------
    opening = (classified.filter(F.col("_outcome").isin("changed", "new"))
               .select(source_columns("s", {
                   "valid_from": F.lit(effective_date).cast(DateType()),
                   "valid_to": F.lit(HIGH_DATE).cast(DateType()),
                   "is_current": F.lit(True),
                   "row_hash": F.col("s._hash")})))

    result = (closed_versions.select(*[F.col(c) for c in unchanged.columns])
              .unionByName(unchanged)
              .unionByName(absent)
              .unionByName(closing)
              .unionByName(opening))

    return result, {
        "new": counts.get("new", 0),
        "changed": counts.get("changed", 0),
        "unchanged": counts.get("unchanged", 0),
        "absent": counts.get("absent", 0),
    }


def empty_dimension(spark: SparkSession, natural_key: str,
                    tracked: list[str], type1: list[str]) -> DataFrame:
    """An empty dimension with the right schema, for a first run."""
    fields = [StructField(natural_key, StringType(), False)]
    fields += [StructField(c, StringType(), True) for c in tracked + type1]
    fields += [
        StructField("valid_from", DateType(), False),
        StructField("valid_to", DateType(), False),
        StructField("is_current", BooleanType(), False),
        StructField("row_hash", StringType(), False),
    ]
    return spark.createDataFrame([], StructType(fields))


# ============================================== POINT-IN-TIME KEY LOOKUP
def resolve_as_of(facts: DataFrame, dimension: DataFrame, *, natural_key: str,
                  event_date_column: str, surrogate: str) -> DataFrame:
    """Join a fact to the dimension version valid on the EVENT date.

    Join to is_current instead and a March order reports under the customer's
    April region — so last month's report changes retrospectively and the number
    looks perfectly reasonable. That is the most expensive silent bug in
    dimensional modelling.

    valid_to exclusive means exactly one version matches. If this ever produces
    more rows than it started with, the validity windows overlap and the merge
    is broken.
    """
    return (facts.alias("f").join(
        dimension.alias("d"),
        (F.col(f"f.{natural_key}") == F.col(f"d.{natural_key}")) &
        (F.col(f"f.{event_date_column}") >= F.col("d.valid_from")) &
        (F.col(f"f.{event_date_column}") < F.col("d.valid_to")),
        "left"))
