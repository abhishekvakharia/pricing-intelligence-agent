"""
bq/queries.py
All BigQuery query functions for the Pricing Intelligence Agent.

Every query uses BQ_FULL_TABLE from config.py.

NOTE on schema:
  - price_fact_us has NO SCD columns (db_rec_del_flag, db_rec_close_date,
    db_rec_begin_date, db_prcsd_dttm, batch_number) — all rows are valid.
  - Flag columns (cost_override_flag, special_price_override_flag,
    allow_floor_flag, skip_bump_flag, etc.) are BOOL — compare with = TRUE.
  - final_cost and currency_exchange_rate are stored as STRING — CAST to
    FLOAT64 before arithmetic.
  - REGION, VENDOR_LOB_LVL2_DES, VENDOR_LOB_LVL3_DES, SoW_final_org_scale,
    price_fact_key are NOT in the table — never reference them.
"""

import logging
import re

import pandas as pd
from google.cloud import bigquery

from config import BQ_FULL_TABLE, GCP_PROJECT_ID

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_client() -> bigquery.Client:
    """Return a BigQuery client scoped to the configured project."""
    return bigquery.Client(project=GCP_PROJECT_ID)


def _run(sql: str) -> pd.DataFrame:
    """Execute *sql* and return a DataFrame."""
    client = _get_client()
    return client.query(sql).to_dataframe()


def _get_date_filter(
    date_from: str | None = None,
    date_to: str | None = None,
) -> str:
    """
    Build a TIMESTAMP-safe SQL date-range fragment for the `created` column.

    Uses explicit TIMESTAMP() casting so that BigQuery TIMESTAMP columns are
    compared correctly regardless of BQ version.  `date_to` is made fully
    inclusive by using TIMESTAMP_ADD(... INTERVAL 1 DAY), which captures all
    records through 23:59:59 UTC on that day.

    Returns an AND-prefixed clause (or empty string when no bounds are given),
    ready to drop directly inside a WHERE block.

    Defaults (when both args are None): no filter applied.
    Typical defaults used by callers: date_from='2025-12-25', date_to='2025-12-31'.
    """
    if date_from and date_to:
        return (
            f"AND created >= TIMESTAMP('{date_from}') "
            f"AND created < TIMESTAMP_ADD(TIMESTAMP('{date_to}'), INTERVAL 1 DAY)"
        )
    elif date_from:
        return f"AND created >= TIMESTAMP('{date_from}')"
    elif date_to:
        return f"AND created < TIMESTAMP_ADD(TIMESTAMP('{date_to}'), INTERVAL 1 DAY)"
    return ""


# ---------------------------------------------------------------------------
# Active-record filter — no SCD columns exist in this table
# ---------------------------------------------------------------------------

def get_active_filter() -> str:
    """
    The price_fact_us table does not have SCD columns (db_rec_del_flag,
    db_rec_close_date).  No active-record filter is needed — all rows
    are valid.
    """
    logging.info(
        "[BQ] No SCD filter applied — price_fact_us has no soft-delete columns"
    )
    return "1=1"


# ---------------------------------------------------------------------------
# Diagnostic query — basic health check
# ---------------------------------------------------------------------------

def run_diagnostic_query() -> dict:
    """
    Run unfiltered counts against the pricing table for a quick health check.

    Returns a dict with row counts and date range.
    """
    sql = f"""
    SELECT
        COUNT(*) AS total_rows,
        COUNT(DISTINCT country_code) AS distinct_countries,
        COUNT(DISTINCT pricing_rule) AS distinct_rules,
        COUNT(DISTINCT price_rule_source) AS distinct_rule_sources,
        COUNTIF(order_key IS NOT NULL) AS orders,
        COUNTIF(order_key IS NULL) AS quotes_only,
        MIN(created) AS earliest_record,
        MAX(created) AS latest_record
    FROM `{BQ_FULL_TABLE}`
    """
    logging.info("[BQ] Running diagnostic query on %s", BQ_FULL_TABLE)
    client = _get_client()
    row = list(client.query(sql).result())[0]
    result = dict(row)
    logging.info("[BQ] Diagnostics: %s", result)
    return result


# ---------------------------------------------------------------------------
# 1. Rule utilisation
# ---------------------------------------------------------------------------

def fetch_rule_utilization(
    limit: int = 100_000,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """
    Returns per-rule-source / rule / country / company breakdown with record
    count, avg margin, avg calculated price, and conversion rate.

    Parameters
    ----------
    limit : int
        Maximum rows to return (default 100,000).
    date_from : str | None
        Optional lower date bound for the `created` column (YYYY-MM-DD).
    date_to : str | None
        Optional upper date bound for the `created` column (YYYY-MM-DD).
    """
    date_filter = _get_date_filter(date_from, date_to)
    sql = f"""
    SELECT
        price_rule_source,
        pricing_rule,
        country_code,
        company_code,
        COUNT(*) AS record_count,
        ROUND(AVG(final_price_margin), 4) AS avg_final_price_margin,
        ROUND(
            AVG(calculated_price * CAST(COALESCE(NULLIF(currency_exchange_rate, ''), '1') AS FLOAT64)),
            4
        ) AS avg_calculated_price_base_ccy,
        ROUND(
            SAFE_DIVIDE(
                COUNTIF(order_key IS NOT NULL),
                COUNT(*)
            ) * 100, 2
        ) AS conversion_rate_pct,
        COUNTIF(cost_override_flag = TRUE) AS cost_override_count,
        COUNTIF(special_price_override_flag = TRUE) AS special_override_count
    FROM `{BQ_FULL_TABLE}`
    WHERE 1=1
    {date_filter}
    GROUP BY
        price_rule_source,
        pricing_rule,
        country_code,
        company_code
    ORDER BY record_count DESC
    LIMIT {limit}
    """
    logging.info(
        "[BQ] Executing fetch_rule_utilization (dates: %s → %s)",
        date_from or "unbounded", date_to or "unbounded",
    )
    df = _run(sql)
    logging.info("[BQ] fetch_rule_utilization returned %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 2. Country / company breakdown
# ---------------------------------------------------------------------------

def fetch_country_breakdown(
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """
    Returns per-country/company/rule-source aggregates including fallback rate.

    Parameters
    ----------
    date_from : str | None
        Optional lower date bound for the `created` column (YYYY-MM-DD).
    date_to : str | None
        Optional upper date bound for the `created` column (YYYY-MM-DD).
    """
    date_filter = _get_date_filter(date_from, date_to)
    sql = f"""
    SELECT
        country_code,
        company_code,
        price_rule_source,
        COUNT(*) AS record_count,
        ROUND(AVG(final_price_margin), 4) AS avg_final_price_margin,
        ROUND(
            SAFE_DIVIDE(
                COUNTIF(UPPER(price_rule_source) = 'DEFAULT'),
                COUNT(*)
            ) * 100, 2
        ) AS fallback_rate_pct,
        ROUND(
            SAFE_DIVIDE(
                COUNTIF(order_key IS NOT NULL),
                COUNT(*)
            ) * 100, 2
        ) AS conversion_rate_pct
    FROM `{BQ_FULL_TABLE}`
    WHERE 1=1
    {date_filter}
    GROUP BY
        country_code,
        company_code,
        price_rule_source
    ORDER BY record_count DESC
    """
    logging.info(
        "[BQ] Executing fetch_country_breakdown (dates: %s → %s)",
        date_from or "unbounded", date_to or "unbounded",
    )
    df = _run(sql)
    logging.info("[BQ] fetch_country_breakdown returned %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 3. Leakage candidates
# ---------------------------------------------------------------------------

def fetch_leakage_candidates(
    margin_threshold: float = 0.10,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """
    Returns records exhibiting one or more leakage signals:
      • margin_pct < margin_threshold  (low-margin)
      • calculated_price = engine_floor_price  (floor-hit)
      • cost_override_flag = TRUE or special_price_override_flag = TRUE (override)

    Parameters
    ----------
    margin_threshold : float
        Margin fraction below which a record is flagged as low-margin.
    date_from : str | None
        Optional lower date bound for the `created` column (YYYY-MM-DD).
    date_to : str | None
        Optional upper date bound for the `created` column (YYYY-MM-DD).
    """
    date_filter = _get_date_filter(date_from, date_to)
    sql = f"""
    SELECT
        sku_number,
        CAST(customer_number AS STRING) AS customer_number,
        country_code,
        pricing_rule,
        price_rule_source,
        calculated_price,
        CAST(final_cost AS FLOAT64) AS final_cost,
        final_price_margin,
        overriden_price,
        engine_floor_price,
        cost_override_flag,
        special_price_override_flag,
        ROUND(
            SAFE_DIVIDE(final_price_margin, NULLIF(calculated_price, 0)) * 100, 2
        ) AS margin_pct,
        CASE
            WHEN SAFE_DIVIDE(final_price_margin, NULLIF(calculated_price, 0)) < {margin_threshold}
                THEN 'low_margin'
            WHEN calculated_price = engine_floor_price
                THEN 'floor_hit'
            WHEN cost_override_flag = TRUE OR special_price_override_flag = TRUE
                THEN 'override'
            ELSE 'other'
        END AS leakage_type,
        created
    FROM `{BQ_FULL_TABLE}`
    WHERE (
        SAFE_DIVIDE(final_price_margin, NULLIF(calculated_price, 0)) < {margin_threshold}
        OR calculated_price = engine_floor_price
        OR cost_override_flag = TRUE
        OR special_price_override_flag = TRUE
    )
    {date_filter}
    ORDER BY final_price_margin ASC
    LIMIT 10000
    """
    logging.info(
        "[BQ] Executing fetch_leakage_candidates (threshold=%s, dates: %s → %s)",
        margin_threshold, date_from or "unbounded", date_to or "unbounded",
    )
    df = _run(sql)
    logging.info("[BQ] fetch_leakage_candidates returned %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 4. Revenue uplift opportunity
# ---------------------------------------------------------------------------

def fetch_revenue_opportunity(
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """
    For every SKU + country using the DEFAULT rule, compare avg margin against
    non-DEFAULT records on the same SKU/country and estimate the revenue uplift.

    Parameters
    ----------
    date_from : str | None
        Optional lower date bound for the `created` column (YYYY-MM-DD).
    date_to : str | None
        Optional upper date bound for the `created` column (YYYY-MM-DD).
    """
    date_filter = _get_date_filter(date_from, date_to)
    sql = f"""
    WITH base AS (
        SELECT
            sku_number,
            country_code,
            price_rule_source,
            final_price_margin,
            calculated_price
        FROM `{BQ_FULL_TABLE}`
        WHERE 1=1
        {date_filter}
          AND calculated_price > 0
    ),
    default_stats AS (
        SELECT
            sku_number,
            country_code,
            COUNT(*)                                    AS default_record_count,
            AVG(final_price_margin)                     AS default_avg_margin
        FROM base
        WHERE UPPER(price_rule_source) = 'DEFAULT'
        GROUP BY sku_number, country_code
    ),
    non_default_stats AS (
        SELECT
            sku_number,
            country_code,
            AVG(final_price_margin)                     AS non_default_avg_margin
        FROM base
        WHERE UPPER(price_rule_source) != 'DEFAULT'
        GROUP BY sku_number, country_code
    )
    SELECT
        d.sku_number,
        d.country_code,
        d.default_record_count,
        ROUND(d.default_avg_margin, 4)                  AS default_avg_margin,
        ROUND(n.non_default_avg_margin, 4)              AS non_default_avg_margin,
        ROUND(
            (n.non_default_avg_margin - d.default_avg_margin)
            * d.default_record_count, 2
        )                                               AS estimated_uplift
    FROM default_stats d
    JOIN non_default_stats n
      ON d.sku_number    = n.sku_number
     AND d.country_code  = n.country_code
    WHERE n.non_default_avg_margin > d.default_avg_margin
    ORDER BY estimated_uplift DESC
    LIMIT 200
    """
    logging.info(
        "[BQ] Executing fetch_revenue_opportunity (dates: %s → %s)",
        date_from or "unbounded", date_to or "unbounded",
    )
    df = _run(sql)
    logging.info("[BQ] fetch_revenue_opportunity returned %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 5. Schema introspection
# ---------------------------------------------------------------------------

def fetch_schema_info() -> pd.DataFrame:
    """
    Returns column names and data types for the configured pricing table
    via INFORMATION_SCHEMA.COLUMNS.
    """
    project, dataset, table = BQ_FULL_TABLE.split(".")
    sql = f"""
    SELECT
        column_name,
        data_type,
        is_nullable,
        ordinal_position
    FROM `{project}.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table}'
    ORDER BY ordinal_position
    """
    logging.info("[BQ] Executing fetch_schema_info for table %s", table)
    df = _run(sql)
    logging.info("[BQ] fetch_schema_info returned %d columns", len(df))
    return df


# ---------------------------------------------------------------------------
# 6. Ad-hoc / custom query
# ---------------------------------------------------------------------------

_MUTATING_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|MERGE|TRUNCATE|CREATE|ALTER|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def run_raw_query(sql: str) -> pd.DataFrame:
    """
    Execute a user-supplied SQL string against BigQuery.
    Raises ValueError if the SQL contains any mutating keyword.
    """
    match = _MUTATING_KEYWORDS.search(sql)
    if match:
        raise ValueError(
            f"Unsafe SQL detected — mutating keyword '{match.group()}' is not allowed. "
            "This agent is read-only."
        )
    logging.info("[BQ] Executing custom query: %.200s", sql.strip())
    df = _run(sql)
    logging.info("[BQ] Custom query returned %d rows", len(df))
    return df
