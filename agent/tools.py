"""
agent/tools.py
All agent-callable tools for the Pricing Intelligence Agent.

Each tool:
  • Wraps one or more bq/queries.py functions
  • Returns a human-readable string summary the LLM can relay to the user
  • Stores the latest DataFrame in dashboard_state so the Streamlit
    dashboard can render charts without re-querying BigQuery

State sharing:
  dashboard_state is imported by dashboard/app.py.  Because both the agent
  process and the Streamlit process run from the same working directory,
  the dashboard also reads a state file (state/dashboard_state.pkl) that
  this module writes on every update.
"""

from __future__ import annotations

import json
import os
import pickle
import textwrap
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from bq.queries import (
    fetch_rule_utilization,
    fetch_country_breakdown,
    fetch_leakage_candidates,
    fetch_revenue_opportunity,
    fetch_schema_info,
    run_raw_query,
)

# ---------------------------------------------------------------------------
# Shared in-memory state (also persisted to disk for the dashboard process)
# ---------------------------------------------------------------------------

dashboard_state: dict[str, Any] = {
    "rule_utilization": None,
    "country_breakdown": None,
    "leakage": None,
    "revenue_opportunity": None,
    "model_results": None,
    "rule_recommendations": None,
}

_STATE_DIR = Path("state")
_STATE_FILE = _STATE_DIR / "dashboard_state.pkl"


def _persist_state() -> None:
    """Pickle dashboard_state to disk so the Streamlit process can read it."""
    _STATE_DIR.mkdir(exist_ok=True)
    with open(_STATE_FILE, "wb") as fh:
        pickle.dump(dashboard_state, fh)


# ---------------------------------------------------------------------------
# Helper: load persisted state (called by the dashboard on refresh)
# ---------------------------------------------------------------------------

def load_state() -> dict[str, Any]:
    """Load dashboard state from disk.  Returns an empty dict if not found."""
    if _STATE_FILE.exists():
        with open(_STATE_FILE, "rb") as fh:
            return pickle.load(fh)
    return {k: None for k in dashboard_state}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _df_to_text(df: pd.DataFrame, max_rows: int = 20) -> str:
    """Return a compact, fixed-width text table from a DataFrame."""
    if df is None or df.empty:
        return "(no data)"
    return df.head(max_rows).to_string(index=False)


def _safe_call(func, *args, **kwargs):
    """Call func; return (result, None) or (None, error_string) on failure."""
    try:
        return func(*args, **kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Error: {exc}\n{traceback.format_exc()}"


# ---------------------------------------------------------------------------
# Tool 1 — Rule utilisation
# ---------------------------------------------------------------------------

def get_rule_utilization() -> str:
    """
    Fetch and summarise pricing-rule utilisation from BigQuery.

    Returns a text summary covering:
      • Top rule types by usage %
      • Average margin per rule source
      • Overall fallback (DEFAULT) rule rate
    """
    df, err = _safe_call(fetch_rule_utilization)
    if err:
        return err

    dashboard_state["rule_utilization"] = df
    _persist_state()

    total = df["record_count"].sum()
    if total == 0:
        return "No active pricing records found."

    # Rule-source roll-up
    by_source = (
        df.groupby("price_rule_source", dropna=False)
        .agg(records=("record_count", "sum"), avg_margin=("avg_final_price_margin", "mean"))
        .reset_index()
        .sort_values("records", ascending=False)
    )
    by_source["share_pct"] = (by_source["records"] / total * 100).round(2)

    fallback_rows = by_source[by_source["price_rule_source"].str.upper() == "DEFAULT"]
    fallback_pct = fallback_rows["share_pct"].sum() if not fallback_rows.empty else 0.0

    lines = [
        f"Rule Utilisation Summary  (total active records: {total:,})",
        "=" * 60,
        "",
        "Usage by rule source:",
    ]
    for _, row in by_source.iterrows():
        lines.append(
            f"  {row['price_rule_source'] or 'UNKNOWN':25s}"
            f"  {row['records']:>10,.0f} records  ({row['share_pct']:5.1f} %)"
            f"  avg margin: {row['avg_margin']:>8.4f}"
        )

    lines += [
        "",
        f"Overall fallback (DEFAULT) rule rate: {fallback_pct:.1f} %",
        "",
        "Charts are available on the Rule Utilisation tab of the dashboard.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 2 — Country breakdown
# ---------------------------------------------------------------------------

def get_country_breakdown() -> str:
    """
    Fetch and summarise pricing-rule usage by country and region.

    Returns:
      • Top 5 countries by fallback rate
      • Average margin by region
    """
    df, err = _safe_call(fetch_country_breakdown)
    if err:
        return err

    dashboard_state["country_breakdown"] = df
    _persist_state()

    # Aggregate at country level (sum across rule sources)
    country = (
        df.groupby("country_code", dropna=False)
        .agg(
            records=("record_count", "sum"),
            avg_margin=("avg_final_price_margin", "mean"),
            fallback_rate=("fallback_rate_pct", "mean"),
        )
        .reset_index()
        .sort_values("fallback_rate", ascending=False)
    )

    region = (
        df.groupby("REGION", dropna=False)
        .agg(avg_margin=("avg_final_price_margin", "mean"))
        .reset_index()
        .sort_values("avg_margin", ascending=False)
    )

    lines = [
        "Country / Region Breakdown",
        "=" * 60,
        "",
        "Top 5 countries by fallback rule rate:",
    ]
    for _, row in country.head(5).iterrows():
        lines.append(
            f"  {row['country_code'] or 'UNKNOWN':10s}"
            f"  fallback: {row['fallback_rate']:5.1f} %"
            f"  avg margin: {row['avg_margin']:>8.4f}"
            f"  records: {row['records']:>8,.0f}"
        )

    lines += ["", "Average margin by region:"]
    for _, row in region.iterrows():
        lines.append(f"  {row['REGION'] or 'UNKNOWN':20s}  avg margin: {row['avg_margin']:>8.4f}")

    lines.append("\nCharts are available on the Country & Region tab of the dashboard.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 3 — Revenue opportunity
# ---------------------------------------------------------------------------

def get_revenue_opportunity() -> str:
    """
    Estimate revenue uplift if DEFAULT pricing rules were replaced with
    targeted (customer- or product-level) rules.

    Returns total estimated uplift and top 10 SKU/country opportunities.
    """
    df, err = _safe_call(fetch_revenue_opportunity)
    if err:
        return err

    dashboard_state["revenue_opportunity"] = df
    _persist_state()

    if df.empty:
        return "No revenue uplift opportunities identified (no SKUs using DEFAULT rules with better alternatives)."

    total_uplift = df["estimated_uplift"].sum()
    n_skus = df["sku_number"].nunique()
    n_countries = df["country_code"].nunique()
    avg_gap = (df["non_default_avg_margin"] - df["default_avg_margin"]).mean()

    lines = [
        "Revenue Uplift Opportunity",
        "=" * 60,
        "",
        f"  Total estimated uplift:          {total_uplift:>14,.2f}  (margin units)",
        f"  SKUs using DEFAULT rules:         {n_skus:>14,}",
        f"  Countries affected:               {n_countries:>14,}",
        f"  Avg margin gap (targeted–default): {avg_gap:>13.4f}",
        "",
        "Top 10 SKU / Country opportunities:",
    ]
    for _, row in df.head(10).iterrows():
        lines.append(
            f"  SKU: {row['sku_number'] or 'N/A':20s}"
            f"  Country: {row['country_code'] or 'N/A':6s}"
            f"  Uplift: {row['estimated_uplift']:>10,.2f}"
            f"  Default margin: {row['default_avg_margin']:>8.4f}"
            f"  Targeted margin: {row['non_default_avg_margin']:>8.4f}"
        )

    lines.append("\nCharts are available on the Revenue Opportunity tab of the dashboard.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 4 — Pricing leakage alerts
# ---------------------------------------------------------------------------

def get_pricing_leakage_alerts(margin_threshold: float = 0.10) -> str:
    """
    Flag transactions with pricing leakage: low margin, floor hits, or
    manual price overrides.

    Parameters
    ----------
    margin_threshold : float
        Margin fraction below which a record is flagged as low-margin.
        Default 0.10 (10 %).
    """
    df, err = _safe_call(fetch_leakage_candidates, margin_threshold)
    if err:
        return err

    dashboard_state["leakage"] = df
    _persist_state()

    if df.empty:
        return f"No leakage records found at margin threshold {margin_threshold:.0%}."

    total = len(df)
    low_margin = (df["low_margin_flag"] == "low_margin").sum()
    floor_hit = (df["floor_hit_flag"] == "floor_hit").sum()
    override = (df["override_flag"] == "manual_override").sum()

    top_skus = (
        df.groupby("sku_number", dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(10)
    )

    lines = [
        f"Pricing Leakage Alerts  (margin threshold: {margin_threshold:.0%})",
        "=" * 60,
        "",
        f"  Total leakage records:  {total:>10,}",
        f"  Low-margin records:     {low_margin:>10,}  ({low_margin/total*100:.1f} %)",
        f"  Floor-hit records:      {floor_hit:>10,}  ({floor_hit/total*100:.1f} %)",
        f"  Manual-override records:{override:>10,}  ({override/total*100:.1f} %)",
        "",
        "Top 10 impacted SKUs (by leakage record count):",
    ]
    for sku, cnt in top_skus.items():
        lines.append(f"  {sku or 'UNKNOWN':30s}  {cnt:>6,} leakage records")

    lines.append("\nCharts are available on the Pricing Leakage tab of the dashboard.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 5 — Rule recommendations
# ---------------------------------------------------------------------------

def get_rule_recommendations() -> str:
    """
    Analyse rule utilisation data and return three sets of recommendations:

    1. Missing Rules  — SKU+country combos with > 50 % fallback usage
    2. Consolidation Candidates  — rules sharing the same modifier %
    3. Redundant Rules  — rules with < 5 records in the dataset
    """
    df, err = _safe_call(fetch_rule_utilization, limit=200_000)
    if err:
        return err

    # ---- 1. Missing rules -------------------------------------------------
    # SKU+country combinations where > 50 % of records use the DEFAULT rule
    sku_country = (
        df.groupby(["sku_number", "country_code", "price_rule_source"], dropna=False)
        .agg(records=("record_count", "sum"))
        .reset_index()
    ) if "sku_number" in df.columns else pd.DataFrame()

    missing_rules_lines: list[str] = []
    if not sku_country.empty:
        total_by_sku_country = (
            sku_country.groupby(["sku_number", "country_code"])["records"]
            .sum()
            .reset_index()
            .rename(columns={"records": "total"})
        )
        default_by_sku_country = (
            sku_country[sku_country["price_rule_source"].str.upper() == "DEFAULT"]
            .groupby(["sku_number", "country_code"])["records"]
            .sum()
            .reset_index()
            .rename(columns={"records": "default_count"})
        )
        merged = total_by_sku_country.merge(default_by_sku_country, on=["sku_number", "country_code"], how="left").fillna(0)
        merged["fallback_pct"] = merged["default_count"] / merged["total"] * 100
        high_fallback = merged[merged["fallback_pct"] > 50].sort_values("fallback_pct", ascending=False).head(20)

        for _, row in high_fallback.iterrows():
            missing_rules_lines.append(
                f"  SKU: {row['sku_number'] or 'N/A':25s}  Country: {row['country_code'] or 'N/A':6s}"
                f"  fallback: {row['fallback_pct']:.1f} %  → CREATE targeted rule"
            )

    # ---- 2. Consolidation candidates --------------------------------------
    # Rules sharing the same pricing_rule name across multiple country/region combos
    consol_lines: list[str] = []
    if "pricing_rule" in df.columns:
        rule_spread = (
            df.groupby("pricing_rule", dropna=False)
            .agg(country_count=("country_code", "nunique"), total_records=("record_count", "sum"))
            .reset_index()
            .sort_values("country_count", ascending=False)
        )
        # Flag rules that appear in many countries with similar modifier logic
        consol_candidates = rule_spread[rule_spread["country_count"] >= 5].head(10)
        for _, row in consol_candidates.iterrows():
            consol_lines.append(
                f"  Rule: {row['pricing_rule'] or 'N/A':35s}"
                f"  spans {row['country_count']:>3} countries"
                f"  {row['total_records']:>8,} records  → REVIEW for global consolidation"
            )

    # ---- 3. Redundant rules -----------------------------------------------
    # Pricing rules with very few total records
    redundant_lines: list[str] = []
    if "pricing_rule" in df.columns:
        thin_rules = (
            df.groupby("pricing_rule", dropna=False)
            .agg(total_records=("record_count", "sum"))
            .reset_index()
        )
        thin_rules = thin_rules[thin_rules["total_records"] < 5].sort_values("total_records")
        for _, row in thin_rules.head(20).iterrows():
            redundant_lines.append(
                f"  Rule: {row['pricing_rule'] or 'N/A':35s}"
                f"  {row['total_records']:>3} records  → REVIEW / REMOVE if obsolete"
            )

    # Store structured recommendations in state
    recommendations = {
        "missing_rules": missing_rules_lines,
        "consolidation_candidates": consol_lines,
        "redundant_rules": redundant_lines,
    }
    dashboard_state["rule_recommendations"] = recommendations
    _persist_state()

    sep = "\n" + "-" * 60 + "\n"
    parts = [
        "Rule Recommendations\n" + "=" * 60,
        "1. MISSING RULES  (> 50 % fallback usage on a SKU/country)\n"
        + ("\n".join(missing_rules_lines) if missing_rules_lines else "  None identified."),
        "2. CONSOLIDATION CANDIDATES  (rule spans ≥ 5 countries)\n"
        + ("\n".join(consol_lines) if consol_lines else "  None identified."),
        "3. REDUNDANT RULES  (< 5 total records)\n"
        + ("\n".join(redundant_lines) if redundant_lines else "  None identified."),
        "Charts and full tables are available on the Rule Recommendations tab.",
    ]
    return sep.join(parts)


# ---------------------------------------------------------------------------
# Tool 6 — ML model training
# ---------------------------------------------------------------------------

def run_model_training() -> str:
    """
    Trigger XGBoost model training for conversion prediction and margin
    regression.  Returns training metrics and top feature importances.
    """
    try:
        from ml.trainer import train_models  # late import avoids circular deps
    except ImportError as exc:
        return f"Could not import ml.trainer: {exc}"

    results, err = _safe_call(train_models)
    if err:
        return err

    dashboard_state["model_results"] = results
    _persist_state()

    top_feats = results.get("top_features", {})
    feat_lines = []
    if isinstance(top_feats, dict):
        # top_features is a dict of {feature_name: {0: ..., 1: ...}} from to_dict()
        importance_col = next(
            (k for k in top_feats if k not in ("feature",)), None
        )
        if importance_col and "feature" in top_feats:
            for idx in sorted(top_feats["feature"]):
                feat_lines.append(
                    f"  {top_feats['feature'][idx]:40s}"
                    f"  importance: {top_feats[importance_col][idx]:.4f}"
                )
        else:
            feat_lines.append(str(top_feats))

    lines = [
        "ML Model Training Complete",
        "=" * 60,
        "",
        "Conversion Classifier (XGBoost):",
        f"  Accuracy : {results.get('conversion_accuracy', 'N/A')}",
        f"  AUC      : {results.get('conversion_auc', 'N/A')}",
        "",
        "Margin Regressor (XGBoost):",
        f"  MAE      : {results.get('margin_mae', 'N/A')}",
        f"  R²       : {results.get('margin_r2', 'N/A')}",
        "",
        "Top feature importances (conversion model):",
    ] + feat_lines + [
        "",
        "Models saved to models/conversion_model.json and models/margin_model.json.",
        "Charts are available on the ML Model Results tab of the dashboard.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 7 — Explore schema
# ---------------------------------------------------------------------------

def explore_schema() -> str:
    """
    List all columns and data types in the configured pricing table.
    Useful for discovering what data is available before writing queries.
    """
    df, err = _safe_call(fetch_schema_info)
    if err:
        return err

    if df.empty:
        return "No schema information found.  Check your GCP_PROJECT_ID, BQ_DATASET_NAME, and BQ_TABLE_NAME in config.py."

    lines = [
        "Pricing Table Schema",
        "=" * 60,
        f"{'#':>4}  {'Column':40s}  {'Type':25s}  Nullable",
        "-" * 80,
    ]
    for _, row in df.iterrows():
        lines.append(
            f"{row['ordinal_position']:>4}  "
            f"{row['column_name']:40s}  "
            f"{row['data_type']:25s}  "
            f"{row['is_nullable']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 8 — Custom / ad-hoc query
# ---------------------------------------------------------------------------

def run_custom_query(sql: str) -> str:
    """
    Execute a read-only SQL query against BigQuery and return the first
    20 rows as a formatted table.

    Parameters
    ----------
    sql : str
        A valid SELECT statement.  Mutating keywords are rejected.
    """
    df, err = _safe_call(run_raw_query, sql)
    if err:
        return err

    if df.empty:
        return "Query executed successfully but returned no rows."

    total_rows = len(df)
    preview = df.head(20).to_string(index=False)
    note = f"\n\n(Showing 20 of {total_rows:,} rows)" if total_rows > 20 else ""
    return f"Query results:\n\n{preview}{note}"
