"""Tests for the lakehouse transforms.

These run real PySpark on a laptop. They exist because three things in this lab
are genuinely easy to get wrong, and all three fail silently in production:

  1. Deduplication that is not deterministic — two people get different answers
  2. The SCD Type 2 merge when a key changes twice in one batch
  3. Point-in-time key resolution — a March fact reporting under an April region

Delta's MERGE throws on case 2 rather than doing the wrong thing quietly, which
is good design. But you still have to decide what the right answer is, and these
tests encode that decision.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest
from pyspark.sql import functions as F
from pyspark.sql.types import (BooleanType, DateType, StringType, StructField,
                               StructType)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lakehouse import fixtures                                    # noqa: E402
from lakehouse.transforms import (apply_quality, build_session,    # noqa: E402
                                  deduplicate, empty_dimension,
                                  prepare_scd2_source, read_landing,
                                  resolve_as_of, scd2_change_hash,
                                  scd2_merge_pure, to_bronze, to_silver)

TRACKED = ["city", "region", "segment"]
TYPE1 = ["full_name", "email"]


@pytest.fixture(scope="session")
def spark():
    session = build_session("lab8-tests")
    yield session
    session.stop()


@pytest.fixture(scope="session")
def landing(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("landing")
    for batch in (1, 2, 3):
        fixtures.write_landing(base, batch)
    return base


def customers_df(spark, batch: int):
    rows = {1: fixtures.CUSTOMERS_BATCH_1, 2: fixtures.CUSTOMERS_BATCH_2,
            3: fixtures.CUSTOMERS_BATCH_3}[batch]
    parsed = [tuple(line.split(",")) for line in rows]
    return spark.createDataFrame(
        parsed,
        "customer_id string, full_name string, email string, city string, "
        "region string, segment string, updated_at string")


# ==========================================================================
# BRONZE
# ==========================================================================

def test_bronze_reads_every_column_as_a_string(spark, landing):
    """Inferring types means one bad row changes the column type for the batch,
    so your bronze schema depends on the worst row in the file."""
    df = read_landing(spark, str(landing / "orders" / "load_date=2024-04-01"))
    types = dict(df.dtypes)
    assert types["quantity"] == "string"
    assert types["unit_price"] == "string"


def test_bronze_rejects_nothing(spark, landing):
    """Ten rows in the file, ten rows in bronze — including the four broken ones."""
    df = read_landing(spark, str(landing / "orders" / "load_date=2024-04-01"))
    bronze = to_bronze(df, batch_id=1, load_date="2024-04-01")
    assert bronze.count() == 10
    assert bronze.filter(F.col("quantity") == "-2").count() == 1, \
        "the negative-quantity row must survive into bronze"


def test_bronze_carries_lineage(spark, landing):
    df = read_landing(spark, str(landing / "orders" / "load_date=2024-04-01"))
    bronze = to_bronze(df, batch_id=1, load_date="2024-04-01")
    for column in ("_batch_id", "_load_date", "_ingested_at", "_source_file"):
        assert column in bronze.columns


# ==========================================================================
# DEDUPLICATION
# ==========================================================================

def test_deduplication_keeps_the_latest_version(spark, landing):
    """ORD-0002 appears twice; the later row says 'delivered'."""
    df = read_landing(spark, str(landing / "orders" / "load_date=2024-04-01"))
    deduped = deduplicate(df, keys=["order_id"], order_by=["updated_at", "order_id"])
    row = deduped.filter(F.col("order_id") == "ORD-0002").collect()
    assert len(row) == 1, "deduplication left a duplicate behind"
    assert row[0]["status"] == "delivered", "it kept the superseded version"


def test_deduplication_is_deterministic_under_a_tie(spark):
    """Without a unique tie-break the surviving row is arbitrary, so the same
    code on the same data gives different answers on different runs."""
    rows = [("K1", "A", "2024-04-01 10:00:00"),
            ("K1", "B", "2024-04-01 10:00:00")]     # identical timestamps
    df = spark.createDataFrame(rows, "k string, v string, updated_at string")

    first = deduplicate(df, keys=["k"], order_by=["updated_at", "v"]).collect()
    second = deduplicate(df, keys=["k"], order_by=["updated_at", "v"]).collect()
    assert first[0]["v"] == second[0]["v"] == "B", \
        "the tie-break must make the outcome reproducible"


def test_deduplication_yields_exactly_one_row_per_key(spark, landing):
    df = read_landing(spark, str(landing / "orders" / "load_date=2024-04-01"))
    deduped = deduplicate(df, keys=["order_id"], order_by=["updated_at", "order_id"])
    duplicates = (deduped.groupBy("order_id").count()
                  .filter(F.col("count") > 1).count())
    assert duplicates == 0


# ==========================================================================
# QUALITY RULES
# ==========================================================================

def test_quality_rules_quarantine_the_four_broken_rows(spark, landing):
    df = read_landing(spark, str(landing / "orders" / "load_date=2024-04-01"))
    deduped = deduplicate(df, keys=["order_id"], order_by=["updated_at", "order_id"])
    passing, failing = apply_quality(deduped)

    assert failing.count() == 4, "expected four broken rows"
    reasons = {r["_reject_reason"] for r in failing.collect()}
    assert any("order_id_present" in r for r in reasons)
    assert any("quantity_positive" in r for r in reasons)
    assert any("quantity_is_integer" in r for r in reasons)
    assert any("unit_price_numeric" in r for r in reasons)


def test_a_quarantine_reason_names_one_rule_not_all_of_them(spark, landing):
    """The source owner needs to know what to fix first, not everything at once."""
    df = read_landing(spark, str(landing / "orders" / "load_date=2024-04-01"))
    _, failing = apply_quality(df)
    for row in failing.collect():
        assert row["_reject_reason"].count("failed rule") == 1


def test_a_warn_rule_does_not_quarantine(spark):
    """Severity is data, not a code path — so a rule can be tightened in
    production without a deployment."""
    rows = [("ORD-1", "CUST-1", "PROD-1", "1", "10.00",
             "2024-04-01 10:00:00", "unknown_status", "2024-04-01 10:00:00")]
    df = spark.createDataFrame(rows, ", ".join(
        f"{c} string" for c in ["order_id", "customer_id", "product_id", "quantity",
                                "unit_price", "order_ts", "status", "updated_at"]))
    passing, failing = apply_quality(df)
    assert passing.count() == 1, "a WARN rule must not quarantine the row"
    assert failing.count() == 0


def test_nothing_is_lost_between_bronze_and_silver(spark, landing):
    """passing + failing must account for every row that entered."""
    df = read_landing(spark, str(landing / "orders" / "load_date=2024-04-01"))
    deduped = deduplicate(df, keys=["order_id"], order_by=["updated_at", "order_id"])
    passing, failing = apply_quality(deduped)
    assert passing.count() + failing.count() == deduped.count()


# ==========================================================================
# SILVER CASTING
# ==========================================================================

def test_money_is_decimal_never_double(spark, landing):
    """Binary floating point misrepresents currency and the error compounds
    through aggregation, surfacing as a reconciliation nobody can explain."""
    df = read_landing(spark, str(landing / "orders" / "load_date=2024-04-01"))
    passing, _ = apply_quality(deduplicate(df, ["order_id"], ["updated_at", "order_id"]))
    silver = to_silver(passing)
    types = dict(silver.dtypes)
    assert types["unit_price"].startswith("decimal"), types["unit_price"]
    assert types["line_amount"].startswith("decimal")


def test_line_amount_is_computed_not_trusted(spark, landing):
    df = read_landing(spark, str(landing / "orders" / "load_date=2024-04-01"))
    passing, _ = apply_quality(deduplicate(df, ["order_id"], ["updated_at", "order_id"]))
    row = to_silver(passing).filter(F.col("order_id") == "ORD-0001").first()
    assert float(row["line_amount"]) == pytest.approx(3 * 1200.50)


# ==========================================================================
# SCD TYPE 2 — including the case that catches everyone
# ==========================================================================

def test_first_load_creates_one_version_per_customer(spark):
    dimension = empty_dimension(spark, "customer_id", TRACKED, TYPE1)
    incoming = prepare_scd2_source(customers_df(spark, 1), "customer_id",
                                  TRACKED, ["updated_at"])
    result, counts = scd2_merge_pure(dimension, incoming, natural_key="customer_id",
                                     tracked=TRACKED, type1=TYPE1,
                                     effective_date="2024-04-01")
    assert counts["new"] == 5 and counts["changed"] == 0
    assert result.count() == 5
    assert result.filter(F.col("is_current")).count() == 5


def test_a_tracked_change_opens_a_new_version(spark):
    """CUST-01 moves South -> West. Two versions, and the old one closes exactly
    where the new one opens."""
    dimension = empty_dimension(spark, "customer_id", TRACKED, TYPE1)
    dimension, _ = scd2_merge_pure(
        dimension, prepare_scd2_source(customers_df(spark, 1), "customer_id",
                                       TRACKED, ["updated_at"]),
        natural_key="customer_id", tracked=TRACKED, type1=TYPE1,
        effective_date="2024-04-01")

    dimension, counts = scd2_merge_pure(
        dimension, prepare_scd2_source(customers_df(spark, 2), "customer_id",
                                       TRACKED, ["updated_at"]),
        natural_key="customer_id", tracked=TRACKED, type1=TYPE1,
        effective_date="2024-04-02")

    assert counts["changed"] == 1, "only CUST-01 changed a tracked attribute"
    versions = sorted(
        dimension.filter(F.col("customer_id") == "CUST-01").collect(),
        key=lambda r: r["valid_from"])
    assert len(versions) == 2
    old, new = versions
    assert old["region"] == "South" and new["region"] == "West"
    assert old["valid_to"] == new["valid_from"], \
        "the old version must close exactly where the new one opens"
    assert old["is_current"] is False and new["is_current"] is True


def test_a_type1_change_does_not_open_a_version(spark):
    """CUST-02 changes email only. Overwritten in place."""
    dimension = empty_dimension(spark, "customer_id", TRACKED, TYPE1)
    dimension, _ = scd2_merge_pure(
        dimension, prepare_scd2_source(customers_df(spark, 1), "customer_id",
                                       TRACKED, ["updated_at"]),
        natural_key="customer_id", tracked=TRACKED, type1=TYPE1,
        effective_date="2024-04-01")
    dimension, _ = scd2_merge_pure(
        dimension, prepare_scd2_source(customers_df(spark, 2), "customer_id",
                                       TRACKED, ["updated_at"]),
        natural_key="customer_id", tracked=TRACKED, type1=TYPE1,
        effective_date="2024-04-02")

    rows = dimension.filter(F.col("customer_id") == "CUST-02").collect()
    assert len(rows) == 1, "a Type 1 change must not create a new version"
    assert rows[0]["email"] == "priya.sharma@example.com", \
        "and it must actually be applied"


def test_two_updates_in_one_batch_collapse_to_the_latest(spark):
    """THE case. CUST-03 changes region twice in one file.

    A naive Delta MERGE raises here, because one target row matches two source
    rows. That is a good failure — the alternative is a silent non-deterministic
    update — but the pipeline still has to answer: which change is the truth?

    For an SCD Type 2 dimension the answer is the LATEST, because the
    intermediate state was never the current state for any meaningful period. If
    the intermediate states genuinely matter, you are not doing a daily batch
    merge; you are streaming every change and the design is different.
    """
    raw = customers_df(spark, 3)
    assert raw.count() == 2, "the fixture must contain two rows for one key"

    prepared = prepare_scd2_source(raw, "customer_id", TRACKED, ["updated_at"])
    assert prepared.count() == 1, \
        "the source must be one row per key BEFORE it reaches the merge"
    assert prepared.first()["region"] == "West", \
        "the surviving row must be the later of the two (16:00, not 09:00)"


def test_the_merge_produces_exactly_two_versions_not_three(spark):
    """Following on: after batch 3, CUST-03 has an original and ONE new version.

    Three versions would mean the intermediate state was materialised, which is
    the behaviour people expect and almost never want.
    """
    dimension = empty_dimension(spark, "customer_id", TRACKED, TYPE1)
    dimension, _ = scd2_merge_pure(
        dimension, prepare_scd2_source(customers_df(spark, 1), "customer_id",
                                       TRACKED, ["updated_at"]),
        natural_key="customer_id", tracked=TRACKED, type1=TYPE1,
        effective_date="2024-04-01")
    dimension, counts = scd2_merge_pure(
        dimension, prepare_scd2_source(customers_df(spark, 3), "customer_id",
                                       TRACKED, ["updated_at"]),
        natural_key="customer_id", tracked=TRACKED, type1=TYPE1,
        effective_date="2024-04-03")

    versions = dimension.filter(F.col("customer_id") == "CUST-03").collect()
    assert len(versions) == 2, f"expected 2 versions, got {len(versions)}"
    current = [v for v in versions if v["is_current"]]
    assert len(current) == 1
    assert current[0]["region"] == "West"
    assert current[0]["city"] == "Mumbai"


def test_rerunning_the_same_batch_changes_nothing(spark):
    """Idempotency. The unchanged path is what makes a retried job safe."""
    dimension = empty_dimension(spark, "customer_id", TRACKED, TYPE1)
    incoming = prepare_scd2_source(customers_df(spark, 1), "customer_id",
                                   TRACKED, ["updated_at"])
    dimension, _ = scd2_merge_pure(dimension, incoming, natural_key="customer_id",
                                   tracked=TRACKED, type1=TYPE1,
                                   effective_date="2024-04-01")
    before = dimension.count()

    dimension, counts = scd2_merge_pure(dimension, incoming, natural_key="customer_id",
                                        tracked=TRACKED, type1=TYPE1,
                                        effective_date="2024-04-01")
    assert dimension.count() == before, "a re-run grew the dimension"
    assert counts["changed"] == 0 and counts["new"] == 0
    assert counts["unchanged"] == 5


def test_a_customer_absent_from_the_batch_is_not_closed(spark):
    """An incremental batch only carries what changed. Treating absence as a
    delete would close every version every night."""
    dimension = empty_dimension(spark, "customer_id", TRACKED, TYPE1)
    dimension, _ = scd2_merge_pure(
        dimension, prepare_scd2_source(customers_df(spark, 1), "customer_id",
                                       TRACKED, ["updated_at"]),
        natural_key="customer_id", tracked=TRACKED, type1=TYPE1,
        effective_date="2024-04-01")
    dimension, counts = scd2_merge_pure(
        dimension, prepare_scd2_source(customers_df(spark, 2), "customer_id",
                                       TRACKED, ["updated_at"]),
        natural_key="customer_id", tracked=TRACKED, type1=TYPE1,
        effective_date="2024-04-02")

    assert counts["absent"] == 3, "three customers were not in batch 2"
    still_current = dimension.filter(
        (F.col("customer_id") == "CUST-04") & F.col("is_current")).count()
    assert still_current == 1, "an absent customer was wrongly closed"


def test_validity_windows_never_overlap(spark):
    """An overlap makes a point-in-time lookup match two rows and the fact doubles."""
    dimension = empty_dimension(spark, "customer_id", TRACKED, TYPE1)
    for batch, effective in ((1, "2024-04-01"), (2, "2024-04-02"), (3, "2024-04-03")):
        dimension, _ = scd2_merge_pure(
            dimension, prepare_scd2_source(customers_df(spark, batch), "customer_id",
                                           TRACKED, ["updated_at"]),
            natural_key="customer_id", tracked=TRACKED, type1=TYPE1,
            effective_date=effective)

    a = dimension.alias("a")
    b = dimension.alias("b")
    overlaps = a.join(b, (F.col("a.customer_id") == F.col("b.customer_id")) &
                         (F.col("a.valid_from") != F.col("b.valid_from")) &
                         (F.col("a.valid_from") < F.col("b.valid_to")) &
                         (F.col("b.valid_from") < F.col("a.valid_to"))).count()
    assert overlaps == 0


def test_no_zero_length_versions(spark):
    """A version valid from and to the same date matches nothing on that date."""
    dimension = empty_dimension(spark, "customer_id", TRACKED, TYPE1)
    for batch, effective in ((1, "2024-04-01"), (2, "2024-04-02")):
        dimension, _ = scd2_merge_pure(
            dimension, prepare_scd2_source(customers_df(spark, batch), "customer_id",
                                           TRACKED, ["updated_at"]),
            natural_key="customer_id", tracked=TRACKED, type1=TYPE1,
            effective_date=effective)
    zero = dimension.filter(F.col("valid_from") >= F.col("valid_to")).count()
    assert zero == 0


def test_the_change_hash_covers_tracked_attributes_only(spark):
    """Include a Type 1 column and every email change opens a pointless version."""
    rows = [("Chennai", "South", "Retail", "a@x.com"),
            ("Chennai", "South", "Retail", "b@x.com")]
    df = spark.createDataFrame(rows, "city string, region string, segment string, email string")
    hashed = df.withColumn("h", scd2_change_hash(TRACKED)).collect()
    assert hashed[0]["h"] == hashed[1]["h"], \
        "an email change must not alter the tracked-attribute hash"


# ==========================================================================
# POINT-IN-TIME RESOLUTION
# ==========================================================================

def test_a_fact_resolves_to_the_version_valid_at_the_event_date(spark):
    """The most expensive silent bug in dimensional modelling.

    Join to is_current instead and a 1 April order reports under the customer's
    2 April region, so last month's report changes retrospectively.
    """
    dimension = empty_dimension(spark, "customer_id", TRACKED, TYPE1)
    for batch, effective in ((1, "2024-04-01"), (2, "2024-04-02")):
        dimension, _ = scd2_merge_pure(
            dimension, prepare_scd2_source(customers_df(spark, batch), "customer_id",
                                           TRACKED, ["updated_at"]),
            natural_key="customer_id", tracked=TRACKED, type1=TYPE1,
            effective_date=effective)

    facts = spark.createDataFrame(
        [("ORD-A", "CUST-01", date(2024, 4, 1)),
         ("ORD-B", "CUST-01", date(2024, 4, 2))],
        StructType([StructField("order_id", StringType()),
                    StructField("customer_id", StringType()),
                    StructField("order_date", DateType())]))

    joined = resolve_as_of(facts, dimension, natural_key="customer_id",
                           event_date_column="order_date", surrogate="customer_id")
    result = {r["order_id"]: r["region"] for r in joined.collect()}
    assert result["ORD-A"] == "South", "the 1 April fact resolved to the wrong version"
    assert result["ORD-B"] == "West"


def test_point_in_time_join_does_not_multiply_facts(spark):
    """If this ever returns more rows than it started with, the validity windows
    overlap and the merge is broken."""
    dimension = empty_dimension(spark, "customer_id", TRACKED, TYPE1)
    for batch, effective in ((1, "2024-04-01"), (2, "2024-04-02"), (3, "2024-04-03")):
        dimension, _ = scd2_merge_pure(
            dimension, prepare_scd2_source(customers_df(spark, batch), "customer_id",
                                           TRACKED, ["updated_at"]),
            natural_key="customer_id", tracked=TRACKED, type1=TYPE1,
            effective_date=effective)

    facts = spark.createDataFrame(
        [(f"ORD-{i}", "CUST-01", date(2024, 4, 2)) for i in range(10)],
        StructType([StructField("order_id", StringType()),
                    StructField("customer_id", StringType()),
                    StructField("order_date", DateType())]))

    joined = resolve_as_of(facts, dimension, natural_key="customer_id",
                           event_date_column="order_date", surrogate="customer_id")
    assert joined.count() == facts.count()


# ==========================================================================
# LATE-ARRIVING DATA
# ==========================================================================

def test_a_late_arriving_order_keeps_its_original_date(spark, landing):
    """ORD-9001 was placed on 12 March and arrived on 3 April.

    Re-dating it to the load date corrupts every historical trend and makes
    reconciliation with the source impossible.
    """
    df = read_landing(spark, str(landing / "orders" / "load_date=2024-04-03"))
    passing, _ = apply_quality(df)
    silver = to_silver(passing)
    row = silver.filter(F.col("order_id") == "ORD-9001").first()
    assert row["order_date"] == date(2024, 3, 12)


def test_a_late_arriving_row_writes_into_an_old_partition(spark, landing, tmp_path):
    """Late data means old partitions are not immutable. That is a design
    constraint, not an exception — and replaceWhere is how Delta handles it."""
    df = read_landing(spark, str(landing / "orders" / "load_date=2024-04-03"))
    passing, _ = apply_quality(df)
    silver = to_silver(passing)
    partitions = [r["order_date"] for r in silver.select("order_date").distinct().collect()]
    assert date(2024, 3, 12) in partitions
