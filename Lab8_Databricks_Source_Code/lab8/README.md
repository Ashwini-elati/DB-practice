# Lab 8 — bronze and silver in Unity Catalog

Read the files ADF landed, write a bronze Delta table, then a silver table with
deduplication and quality rules applied — all registered in Unity Catalog with
grants set.

```bash
pip install pyspark
python -m pytest          # 26 tests, real PySpark, ~90 seconds
```

## Two versions of the same logic

| | `databricks/` | `src/lakehouse/` |
|---|---|---|
| Storage | Delta in Unity Catalog | plain Parquet / DataFrames |
| Runs on | a workspace | your laptop |
| Purpose | the deliverable | the executable specification |

The notebooks are what you deploy. The local module holds the same logic in
plain PySpark so it can be **run and tested in seconds** — which matters because
three things in this lab are genuinely easy to get wrong and all three fail
silently.

Read the tests first. They are the specification.

## The three hard parts

**1 · Deduplication that survives a tie**

`row_number`, never `rank`. Rank returns 1 for every tied row, so a tie hands
back two rows and reintroduces the duplicate you were removing. The `orderBy`
must end in something guaranteed unique, or which row survives is arbitrary and
two people running the same code get different answers.

**2 · SCD Type 2 when a key changes TWICE in one batch**

Delta's `MERGE` raises:

```
Cannot perform Merge as multiple source rows matched and attempted to
modify the same target row in the Delta table
```

That failure is **good design** — the alternative is a silent non-deterministic
update. But the pipeline still has to answer: which change is the truth?

For an SCD Type 2 dimension it is the **latest**, because the intermediate state
was never the current state for any meaningful period. So collapse the source to
one row per key *before* the merge. `test_two_updates_in_one_batch_collapse_to_the_latest`
proves it, and `test_the_merge_produces_exactly_two_versions_not_three` proves
the consequence.

**3 · Point-in-time key resolution**

```python
JOIN dim_customer dc ON dc.customer_id = o.customer_id
                    AND o.order_date  >= dc.valid_from
                    AND o.order_date  <  dc.valid_to     # EXCLUSIVE
```

Join on `is_current` instead and a March order reports under the customer's
April region — so last month's report changes retrospectively and the number
looks perfectly reasonable.

## The NULL trap — a real bug, caught by a test

The first version of the quality rules let a row with a missing `unit_price`
straight through into silver. SQL predicates are three-valued: `NULL RLIKE '...'`
is NULL, and `NOT NULL` is NULL — which is not TRUE, so `when(~expr)` never
fired.

A row with a missing price is *exactly* the row the rule existed to catch, so a
rule that silently passes NULLs is worse than no rule at all.

```python
failed = ~F.coalesce(F.expr(expr), F.lit(False))   # "could not evaluate" = failure
```

## Unity Catalog

`ddl/unity_catalog_setup.sql` sets up the catalog, schemas, volumes and grants,
with a comment on every decision. The two lines worth internalising:

```sql
GRANT USE CATALOG ON CATALOG aurora_dev TO `data-analysts`;  -- traversal
GRANT SELECT      ON SCHEMA  gold       TO `data-analysts`;  -- read, GOLD ONLY
```

`USE CATALOG` is traversal, not read — without it a user cannot see the object
exists, however many SELECTs you grant. That's the commonest "I granted it and
it still doesn't work".

And analysts get **gold only**. Analysts querying silver is how two versions of
a number start circulating: one from the governed model, one from a table with
no business rules applied. Nobody can then say which is right.

## Layout

```
src/lakehouse/
    transforms.py     bronze, dedup, quality, SCD2, point-in-time — all testable
    fixtures.py       three batches, including the two-updates-in-one-batch case
databricks/
    01_lab8_bronze_silver.py         the lab
    02_gold_scd2_merge.py            SCD2 MERGE, three passes, the hard case
    03_performance_and_workflows.py  Spark UI, layout, AQE, Workflows, cost
ddl/unity_catalog_setup.sql          catalog, schemas, volumes, grants, masking
tests/test_transforms.py             26 tests
```

## Definition of done

- [ ] 26 tests pass
- [ ] bronze, silver and quarantine tables exist in Unity Catalog
- [ ] `SHOW GRANTS` proves analysts have gold and **not** silver
- [ ] The reconciliation assertion passes: `deduped == silver + quarantined`
- [ ] You can explain what `USE CATALOG` does and why `SELECT` alone isn't enough
- [ ] You can explain why the merge collapses two updates to one, and what you'd do differently if the intermediate states mattered
- [ ] PR opened, reviewed, merged
