"""
bq/queries.py
All BigQuery query functions for the Pricing Intelligence Agent.
Every query uses BQ_FULL_TABLE from config.py and applies the SCD active-record
filter (db_rec_del_flag != 'Y' AND db_rec_close_date IS NULL) by default.
"""

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


# Base WHERE clause applied to every analytical query
_BASE_FILTER = """
    db_rec_del_flag != 'Y'
    AND db_rec_close_date IS NULL
"""

# ---------------------------------------------------------------------------
# 1. Rule utilisation
# ---------------------------------------------------------------------------

def fetch_rule_utilization(limit: int = 100_000) -> pd.DataFrame:
    """
    Returns per-rule-source / rule / country / region / vendor-LOB breakdown
    with record count, avg margin, avg calculated price, and conversion rate.

    Parameters
    ----------
    limit : int
        Maximum rows returned (default 100 000).
    """
    sql = f"""
    SELECT
        price_rule_source,
        pricing_rule,
        country_code,
        REGION,
        VENDOR_LOB_LVL2_DES,
        COUNT(*)                                                        AS record_count,
        ROUND(AVG(final_price_margin), 4)                               AS avg_final_price_margin,
        ROUND(AVG(calculated_price * COALESCE(currency_exchange_rate, 1)), 4)
                                                                        AS avg_calculated_price_base_ccy,
        ROUND(
            SAFE_DIVIDE(
                COUNTIF(order_key IS NOT NULL),
                COUNT(*)
            ) * 100, 2
        )                                                               AS conversion_rate_pct
    FROM `{BQ_FULL_TABLE}`
    WHERE {_BASE_FILTER}
    GROUP BY
        price_rule_source,
        pricing_rule,
        country_code,
        REGION,
        VENDOR_LOB_LVL2_DES
    ORDER BY record_count DESC
    LIMIT {limit}
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# 2. Country / region breakdown
# ---------------------------------------------------------------------------

def fetch_country_breakdown() -> pd.DataFrame:
    """
    Returns per-country/region/rule-source aggregates including fallback rate
    (percentage of records where price_rule_source = 'DEFAULT').
    """
    sql = f"""
    SELECT
        country_code,
        REGION,
        price_rule_source,
        COUNT(*)                                                        AS record_count,
        ROUND(AVG(final_price_margin), 4)                               AS avg_final_price_margin,
        ROUND(
            SAFE_DIVIDE(
                COUNTIF(UPPER(price_rule_source) = 'DEFAULT'),
                COUNT(*)
            ) * 100, 2
        )                                                               AS fallback_rate_pct
    FROM `{BQ_FULL_TABLE}`
    WHERE {_BASE_FILTER}
    GROUP BY
        country_code,
        REGION,
        price_rule_source
    ORDER BY record_count DESC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# 3. Leakage candidates
# ---------------------------------------------------------------------------

def fetch_leakage_candidates(margin_threshold: float = 0.10) -> pd.DataFrame:
    """
    Returns records that exhibit one or more leakage signals:
      • margin_pct < margin_threshold  (low-margin leakage)
      • calculated_price = engine_floor_price  (floor-hit leakage)
      • overriden_price IS NOT NULL  (manual-override leakage)

    Parameters
    ----------
    margin_threshold : float
        Margin fraction below which a record is considered low-margin.
        Default is 0.10 (10 %).
    """
    sql = f"""
    SELECT
        sku_number,
        customer_number,
        country_code,
        pricing_rule,
        price_rule_source,
        calculated_price,
        final_cost,
        final_price_margin,
        overriden_price,
        engine_floor_price,
        ROUND(
            SAFE_DIVIDE(final_price_margin, calculated_price) * 100, 2
        )                                                               AS margin_pct,
        CASE
            WHEN SAFE_DIVIDE(final_price_margin, calculated_price) < {margin_threshold}
                THEN 'low_margin'
            ELSE NULL
        END                                                             AS low_margin_flag,
        CASE
            WHEN calculated_price = engine_floor_price
                THEN 'floor_hit'
            ELSE NULL
        END                                                             AS floor_hit_flag,
        CASE
            WHEN overriden_price IS NOT NULL
                THEN 'manual_override'
            ELSE NULL
        END                                                             AS override_flag,
        db_rec_begin_date,
        created
    FROM `{BQ_FULL_TABLE}`
    WHERE {_BASE_FILTER}
      AND (
            SAFE_DIVIDE(final_price_margin, calculated_price) < {margin_threshold}
         OR calculated_price = engine_floor_price
         OR overriden_price IS NOT NULL
      )
    ORDER BY final_price_margin ASC
    """
    return _run(sql)


# ---------------------------------------------------------------------------
# 4. Revenue uplift opportunity
# ---------------------------------------------------------------------------

def fetch_revenue_opportunity() -> pd.DataFrame:
    """
    For every SKU + country combination that uses the DEFAULT rule, compare its
    average margin against non-DEFAULT records on the same SKU/country.

    Estimated uplift = (non_default_avg_margin − default_avg_margin)
                       × default_record_count

    Returns the top opportunities sorted by estimated_uplift descending.
    """
    sql = f"""
    WITH base AS (
        SELECT
            sku_number,
            country_code,
            price_rule_source,
            final_price_margin,
            calculated_price
        FROM `{BQ_FULL_TABLE}`
        WHERE {_BASE_FILTER}
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
    return _run(sql)


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
    return _run(sql)


# ---------------------------------------------------------------------------
# 6. Ad-hoc / custom query
# ---------------------------------------------------------------------------

# Keywords that would mutate the dataset — block them
_MUTATING_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|MERGE|TRUNCATE|CREATE|ALTER|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def run_raw_query(sql: str) -> pd.DataFrame:
    """
    Execute a user-supplied SQL string against BigQuery.

    Safety check: raises ValueError if the SQL contains any mutating keyword
    (INSERT, UPDATE, DELETE, DROP, MERGE, TRUNCATE, CREATE, ALTER, GRANT, REVOKE).

    Parameters
    ----------
    sql : str
        Raw SQL string to execute.

    Returns
    -------
    pd.DataFrame
        Query results (up to BigQuery's default row limit).
    """
    match = _MUTATING_KEYWORDS.search(sql)
    if match:
        raise ValueError(
            f"Unsafe SQL detected — mutating keyword '{match.group()}' is not allowed. "
            "This agent is read-only."
        )
    return _run(sql)
