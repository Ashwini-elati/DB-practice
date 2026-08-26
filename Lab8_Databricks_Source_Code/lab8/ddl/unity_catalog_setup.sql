-- ============================================================================
-- Unity Catalog — the object model, and the grants
--
-- The hierarchy is four levels and every one is a security boundary:
--
--     metastore   one per region. Shared by every workspace attached to it.
--       catalog   the top-level container. Usually an ENVIRONMENT or a domain.
--         schema  a database. Usually a LAYER or a subject area.
--           table / view / volume / function / model
--
-- The most consequential decision is what a CATALOG means in your estate,
-- because it is the boundary you cannot easily change later. Two conventions
-- both work; mixing them does not:
--
--   BY ENVIRONMENT   dev / test / prod, each with bronze/silver/gold schemas.
--                    Isolation is obvious. Right when one team owns the platform.
--
--   BY DOMAIN        sales / finance / supply_chain, with environment in the
--                    workspace binding. Right when domains own their own data.
--
-- This lab uses BY ENVIRONMENT because it is harder to get wrong, and moving to
-- domains later is a rename rather than a redesign.
-- ============================================================================

CREATE CATALOG IF NOT EXISTS aurora_dev
  MANAGED LOCATION 'abfss://unity@stauroradev.dfs.core.windows.net/aurora_dev'
  COMMENT 'Aurora lakehouse - development.';

-- Bind the catalog to the workspaces allowed to see it. Without this, any
-- workspace on the metastore can reach it, which is usually not what a client
-- means by "dev is isolated".

USE CATALOG aurora_dev;

-- ------------------------------------------------------------------ schemas
-- One schema per LAYER. The layer is what people reason about and what they are
-- granted on, so it is the right level for the boundary.
CREATE SCHEMA IF NOT EXISTS bronze
  COMMENT 'Raw as landed, with lineage columns. Rejects nothing, corrects nothing.';
CREATE SCHEMA IF NOT EXISTS silver
  COMMENT 'Cleansed, conformed, deduplicated. Failures in silver.quarantine.';
CREATE SCHEMA IF NOT EXISTS gold
  COMMENT 'The dimensional model. Business-readable. What analysts query.';
CREATE SCHEMA IF NOT EXISTS ops
  COMMENT 'Run logs, reconciliation and data quality results.';

-- ------------------------------------------------------------------ volumes
-- A VOLUME governs NON-TABULAR data: the raw files ADF landed, plus checkpoints.
-- Before volumes this was mount points and a credential nobody could audit. A
-- volume is a first-class securable with its own grants.
CREATE EXTERNAL VOLUME IF NOT EXISTS bronze.landing
  LOCATION 'abfss://landing@stauroradev.dfs.core.windows.net/aurora'
  COMMENT 'What ADF lands. EXTERNAL because ADF owns the lifecycle, not Databricks.';

CREATE VOLUME IF NOT EXISTS ops.checkpoints
  COMMENT 'Auto Loader and streaming checkpoints. MANAGED: Databricks owns these.';

-- ============================================================== GRANTS
-- Grants are inherited DOWNWARDS. A grant on a catalog applies to every schema
-- and table inside it, now and in future. That is how people accidentally expose
-- everything: granting SELECT on the catalog to "the analysts" also grants it on
-- the bronze table holding raw personal data.
--
-- Grant at the narrowest level that satisfies the requirement, and grant to
-- GROUPS. A grant to a person is a grant you will still be unwinding two years
-- after they change teams.

-- USE CATALOG and USE SCHEMA are TRAVERSAL rights, not read rights. Without them
-- a user cannot see the object exists, however many SELECTs you grant. This is
-- the most common "I granted it and it still does not work".
GRANT USE CATALOG ON CATALOG aurora_dev TO `data-engineers`;
GRANT USE CATALOG ON CATALOG aurora_dev TO `data-analysts`;
GRANT USE CATALOG ON CATALOG aurora_dev TO `data-scientists`;

-- Engineers own the plumbing.
GRANT ALL PRIVILEGES ON SCHEMA bronze TO `data-engineers`;
GRANT ALL PRIVILEGES ON SCHEMA silver TO `data-engineers`;
GRANT ALL PRIVILEGES ON SCHEMA gold   TO `data-engineers`;
GRANT ALL PRIVILEGES ON SCHEMA ops    TO `data-engineers`;

-- ANALYSTS GET GOLD ONLY. This is the important line in the file.
-- Analysts querying silver is how two versions of a number start circulating:
-- one from the governed model, one from a table with no business rules applied.
-- Nobody can then say which is right, and both get defended.
GRANT USE SCHEMA ON SCHEMA gold TO `data-analysts`;
GRANT SELECT     ON SCHEMA gold TO `data-analysts`;

-- Read access to quality results, so an analyst can answer "why is this number
-- low" without needing an engineer.
GRANT USE SCHEMA ON SCHEMA ops TO `data-analysts`;
GRANT SELECT ON TABLE ops.dq_result      TO `data-analysts`;
GRANT SELECT ON TABLE ops.reconciliation TO `data-analysts`;

-- Scientists get silver too, because feature engineering legitimately needs the
-- conformed-but-not-modelled layer. A deliberate exception with a reason, not a
-- default.
GRANT USE SCHEMA ON SCHEMA silver TO `data-scientists`;
GRANT SELECT     ON SCHEMA silver TO `data-scientists`;
GRANT USE SCHEMA ON SCHEMA gold   TO `data-scientists`;
GRANT SELECT     ON SCHEMA gold   TO `data-scientists`;

GRANT READ VOLUME  ON VOLUME bronze.landing   TO `data-engineers`;
GRANT WRITE VOLUME ON VOLUME ops.checkpoints  TO `data-engineers`;
-- Analysts get NO volume access. Raw files have no row filters and no column
-- masks, so everything below is bypassed the moment somebody reads the file.

-- Jobs run as a SERVICE PRINCIPAL, never as a person. A pipeline running as an
-- individual breaks the day they leave, and the audit log then attributes
-- machine actions to a human.
GRANT USE CATALOG ON CATALOG aurora_dev TO `sp-aurora-jobs`;
GRANT ALL PRIVILEGES ON SCHEMA bronze TO `sp-aurora-jobs`;
GRANT ALL PRIVILEGES ON SCHEMA silver TO `sp-aurora-jobs`;
GRANT ALL PRIVILEGES ON SCHEMA gold   TO `sp-aurora-jobs`;

-- ============================================== ROW AND COLUMN SECURITY
-- The function runs for every reader. Members of the group see the real value;
-- everyone else sees the mask. Applied to the TABLE, so every query path is
-- covered - SQL, notebooks, Power BI - rather than a view somebody can route
-- around.
CREATE OR REPLACE FUNCTION gold.mask_email(email STRING)
RETURN CASE
  WHEN is_account_group_member('pii-readers') THEN email
  ELSE regexp_replace(email, '^[^@]+', '***')
END;

ALTER TABLE gold.dim_customer
  ALTER COLUMN email SET MASK gold.mask_email;

CREATE OR REPLACE FUNCTION gold.region_filter(region STRING)
RETURN is_account_group_member('data-admins')
    OR region = current_user_region();

-- ALTER TABLE gold.fact_sales SET ROW FILTER gold.region_filter ON (region);

-- ============================================================= LINEAGE
-- Lineage is captured automatically, but only when the read AND the write both
-- go through Unity Catalog. A job that reads a path directly with
-- spark.read.parquet() produces no lineage, and the graph then has a hole
-- exactly where the interesting transformation was.
--
--   SELECT * FROM system.access.table_lineage
--    WHERE target_table_full_name = 'aurora_dev.gold.fact_sales'
--    ORDER BY event_time DESC;
--
--   SELECT * FROM system.access.audit
--    WHERE service_name = 'unityCatalog' AND action_name = 'getTable';
