"""
bq/queries.py
All BigQuery query functions for the Pricing Intelligence Agent.

Every query uses BQ_FULL_TABLE from config.py.  The active-record filter
is now *adaptive*: on first call, run_diagnostic_query() inspects the actual
data and chooses the least-restrictive filter that still returns rows.  This
prevents the common "0 rows" issue when db_rec_close_date is not populated.
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
# Diagnostic query — run once at startup, no active-record filter
# ---------------------------------------------------------------------------

def run_diagnostic_query() -> dict:
    """
    Run raw, unfiltered counts against the pricing table to diagnose why
    active-record filters may return 0 rows.

    Returns a dict with counts broken down by filter conditions so callers
    can decide which filter to apply.
    """
    sql = f"""
    SELECT
        COUNT(*)                                                           AS total_rows,
        COUNTIF(db_rec_del_flag = 'Y')                                     AS del_flag_y,
        COUNTIF(db_rec_del_flag != 'Y')                                    AS del_flag_not_y,
        COUNTIF(db_rec_del_flag IS NULL)                                   AS del_flag_null,
        COUNTIF(db_rec_close_date IS NULL)                                 AS close_date_null,
        COUNTIF(db_rec_close_date IS NOT NULL)                             AS close_date_not_null,
        COUNTIF(db_rec_del_flag != 'Y' AND db_rec_close_date IS NULL)      AS strict_active,
        COUNTIF(db_rec_del_flag IS NULL OR db_rec_del_flag != 'Y')         AS relaxed_active,
        COUNT(DISTINCT price_rule_source)                                  AS distinct_rule_sources,
        COUNT(DISTINCT country_code)                                       AS distinct_countries,
        MIN(created)                                                       AS earliest_record,
        MAX(created)                                                       AS latest_record
    FROM `{BQ_FULL_TABLE}`
    """
    logging.info("[BQ] Running diagnostic query on %s", BQ_FULL_TABLE)
    client = _get_client()
    row = list(client.query(sql).result())[0]
    result = dict(row)
    logging.info("[BQ] Diagnostics: %s", result)
    return result


# ---------------------------------------------------------------------------
# Adaptive active-record filter — module-level cache
# ---------------------------------------------------------------------------

_diag_cache: dict | None = None
_active_filter_cache: str | None = None


def _get_active_filter(diag: dict) -> str:
    """
    Choose the right WHERE clause based on what the data actually contains.

    Priority:
      1. Strict SCD filter (both columns populated and meaningful)
      2. Relaxed filter (only del_flag checked; close_date ignored)
      3. No filter at all (last resort — includes all rows)
    """
    if diag["strict_active"] > 0:
        logging.info("[BQ] Strict SCD filter viable (%d rows).", diag["strict_active"])
        return "db_rec_del_flag != 'Y' AND db_rec_close_date IS NULL"
    elif diag["relaxed_active"] > 0:
        logging.warning(
            "[BQ] Strict SCD filter returns 0 rows (db_rec_close_date may be unpopulated). "
            "Falling back to relaxed filter (del_flag only). %d rows available.",
            diag["relaxed_active"],
        )
        return "(db_rec_del_flag IS NULL OR db_rec_del_flag != 'Y')"
    else:
        logging.warning(
            "[BQ] Both strict and relaxed filters return 0 rows. "
            "Removing active-record filter entirely — all %d rows will be included.",
            diag["total_rows"],
        )
        return "1=1"


def get_active_filter() -> str:
    """
    Return the appropriate active-record WHERE clause, running the diagnostic
    query on first call and caching the result for the rest of the session.
    """
    global _diag_cache, _active_filter_cache
    if _active_filter_cache is None:
        _diag_cache = run_diagnostic_query()
        _active_filter_cache = _get_active_filter(_diag_cache)
        logging.info("[BQ] Active filter selected: %s", _active_filter_cache)
    return _active_filter_cache


# ---------------------------------------------------------------------------
# 1. Rule utilisation
# ---------------------------------------------------------------------------

def fetch_rule_utilization(
    limit: int = 100_000,
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """
    Returns per-rule-source / rule / country / region / vendor-LOB breakdown
    with record count, avg margin, avg calculated price, and conversion rate.

    Parameters
    ----------
    limit : int
        Maximum rows to return (default 100,000).
    date_from : str | None
        Optional lower date bound for the `created` column (YYYY-MM-DD).
    date_to : str | None
        Optional upper date bound for the `created` column (YYYY-MM-DD).
    """
    active_filter = get_active_filter()
    date_filter = _get_date_filter(date_from, date_to)
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
    WHERE {active_filter}
    {date_filter}
    GROUP BY
        price_rule_source,
        pricing_rule,
        country_code,
        REGION,
        VENDOR_LOB_LVL2_DES
    ORDER BY record_count DESC
    LIMIT {limit}
    """
    logging.info(
        "[BQ] Executing fetch_rule_utilization (filter: %s, dates: %s → %s)",
        active_filter, date_from or "unbounded", date_to or "unbounded",
    )
    df = _run(sql)
    logging.info("[BQ] fetch_rule_utilization returned %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# 2. Country / region breakdown
# ---------------------------------------------------------------------------

def fetch_country_breakdown(
    date_from: str | None = None,
    date_to: str | None = None,
) -> pd.DataFrame:
    """
    Returns per-country/region/rule-source aggregates including fallback rate.

    Parameters
    ----------
    date_from : str | None
        Optional lower date bound for the `created` column (YYYY-MM-DD).
    date_to : str | None
        Optional upper date bound for the `created` column (YYYY-MM-DD).
    """
    active_filter = get_active_filter()
    date_filter = _get_date_filter(date_from, date_to)
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
    WHERE {active_filter}
    {date_filter}
    GROUP BY
        country_code,
        REGION,
        price_rule_source
    ORDER BY record_count DESC
    """
    logging.info(
        "[BQ] Executing fetch_country_breakdown (filter: %s, dates: %s → %s)",
        active_filter, date_from or "unbounded", date_to or "unbounded",
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
      • overriden_price IS NOT NULL  (manual override)

    Parameters
    ----------
    margin_threshold : float
        Margin fraction below which a record is flagged as low-margin.
    date_from : str | None
        Optional lower date bound for the `created` column (YYYY-MM-DD).
    date_to : str | None
        Optional upper date bound for the `created` column (YYYY-MM-DD).
    """
    active_filter = get_active_filter()
    date_filter = _get_date_filter(date_from, date_to)
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
    WHERE {active_filter}
    {date_filter}
      AND (
            SAFE_DIVIDE(final_price_margin, calculated_price) < {margin_threshold}
         OR calculated_price = engine_floor_price
         OR overriden_price IS NOT NULL
      )
    ORDER BY final_price_margin ASC
    """
    logging.info(
        "[BQ] Executing fetch_leakage_candidates (threshold=%s, filter: %s, dates: %s → %s)",
        margin_threshold, active_filter, date_from or "unbounded", date_to or "unbounded",
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
    active_filter = get_active_filter()
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
        WHERE {active_filter}
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
        "[BQ] Executing fetch_revenue_opportunity (filter: %s, dates: %s → %s)",
        active_filter, date_from or "unbounded", date_to or "unbounded",
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
