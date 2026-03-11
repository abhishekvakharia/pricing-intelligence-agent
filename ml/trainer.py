"""
ml/trainer.py
Vertex AI AutoML Tabular training pipeline for the Pricing Intelligence Agent.

All training runs entirely on GCP — no local ML libraries are required.

Steps executed by train_models():
  1. Validate row count for the given date range
  2. Clean up existing BQ view, Vertex AI models, datasets, and endpoints
  3. Create fresh BigQuery ML view with derived feature columns
  4. Register the view as a Vertex AI Managed TabularDataset
  5. Launch AutoML Tabular training for:
       • Model 1  — Conversion Classifier  (target: quote_accepted)
       • Model 2  — Margin Regressor       (target: margin_pct)
  6. Fetch model-level evaluation metrics
  7. Deploy both models to Vertex AI Endpoints
  8. Persist endpoint resource names to endpoints.json
  9. Return a results dict consumed by agent/tools.py and the dashboard

Schema notes for price_fact_us:
  - No SCD columns (db_rec_del_flag, db_rec_close_date, REGION, etc.)
  - BOOL flags: use as-is — AutoML handles BOOL natively
  - final_cost, currency_exchange_rate: stored as STRING — CAST to FLOAT64
  - customer_number, customer_branch_number: stored as INT64 — CAST to STRING
"""

from __future__ import annotations

import json
import logging
from typing import Any

from google.cloud import aiplatform, bigquery
from google.cloud.aiplatform import AutoMLTabularTrainingJob

from config import (
    GCP_PROJECT_ID,
    BQ_DATASET_NAME,
    BQ_FULL_TABLE,
    VERTEX_REGION,
    VERTEX_STAGING_BUCKET,
    CONVERSION_MODEL_DISPLAY_NAME,
    MARGIN_MODEL_DISPLAY_NAME,
)

# ---------------------------------------------------------------------------
# BigQuery ML view constants
# ---------------------------------------------------------------------------

_ML_VIEW_NAME = "pricing_ml_view"
_ML_VIEW_FULL = f"{GCP_PROJECT_ID}.{BQ_DATASET_NAME}.{_ML_VIEW_NAME}"


# ---------------------------------------------------------------------------
# BigQuery ML view DDL
# ---------------------------------------------------------------------------

def build_ml_view_sql(date_from: str, date_to: str) -> str:
    """
    Build the CREATE OR REPLACE VIEW DDL for the Vertex AI training dataset.

    Removes all columns not present in price_fact_us (REGION,
    VENDOR_LOB_LVL2_DES, VENDOR_LOB_LVL3_DES, SoW_final_org_scale,
    price_fact_key, db_rec_* columns).

    Casts STRING-stored numeric columns (final_cost, currency_exchange_rate)
    to FLOAT64 for arithmetic.  BOOL flag columns are used as-is.

    Parameters
    ----------
    date_from : str
        Lower bound date string (YYYY-MM-DD), inclusive.
    date_to : str
        Upper bound date string (YYYY-MM-DD), inclusive.

    Returns
    -------
    str
        A complete CREATE OR REPLACE VIEW SQL statement.
    """
    return f"""
CREATE OR REPLACE VIEW `{_ML_VIEW_FULL}` AS
SELECT
    -- Identifiers (excluded from model features but kept for tracing)
    quote_key,
    order_key,
    price_fact_guid,

    -- Product
    sku_number,
    mpn,
    digital_sku_number,

    -- Customer & Org (INT64 → STRING for categorical encoding)
    CAST(customer_number AS STRING) AS customer_number,
    CAST(customer_branch_number AS STRING) AS customer_branch_number,
    vendor_number,
    company_code,
    country_code,
    calling_system,

    -- Pricing Rule
    pricing_rule,
    price_rule_source,
    price_rule_subsource,
    price_rule_cost_basis,
    price_rule_starting_amount,
    price_rule_currency,

    -- Price Values
    sku_price,
    base_price,
    calculated_price,
    overriden_price,
    cost_basis,
    final_price_margin,
    CAST(final_cost AS FLOAT64) AS final_cost,

    -- Modifiers
    modifier_percent,
    price_modifier_percent,
    price_modifier_delta,
    customer_bump_percent,
    total_acop_delta,
    total_margin_bump,
    total_base_price_bump,

    -- Floor Rule
    engine_floor_price,
    floor_rule_percent,
    floor_rule_cost_value,
    floor_rule_initial_amount,

    -- Currency (STRING → FLOAT64 for arithmetic)
    CAST(currency_exchange_rate AS FLOAT64) AS currency_exchange_rate,
    currency_code_from,
    currency_code_to,

    -- BOOL flags — AutoML handles BOOL natively
    reciprocal_flag,
    allow_quantity_break_flag,
    allow_floor_flag,
    cost_override_flag,
    special_price_override_flag,
    skip_additional_cost_flag,
    skip_bump_flag,
    skip_gpe_discount_flag,
    floor_override_price_rule,

    -- Derived target and feature columns
    CASE WHEN order_key IS NOT NULL THEN 1 ELSE 0 END
        AS quote_accepted,
    SAFE_DIVIDE(final_price_margin, NULLIF(calculated_price, 0)) * 100
        AS margin_pct,
    CASE WHEN overriden_price IS NOT NULL THEN 1 ELSE 0 END
        AS was_price_overridden,
    CASE WHEN calculated_price = engine_floor_price THEN 1 ELSE 0 END
        AS hit_floor_price

FROM `{BQ_FULL_TABLE}`
WHERE created >= TIMESTAMP('{date_from}')
  AND created < TIMESTAMP_ADD(TIMESTAMP('{date_to}'), INTERVAL 1 DAY)
"""


# ---------------------------------------------------------------------------
# Column transformation specs for AutoML
# ---------------------------------------------------------------------------

# Numeric columns — both models
NUMERIC_COLS = [
    "calculated_price",
    "base_price",
    "cost_basis",
    "final_cost",
    "final_price_margin",
    "modifier_percent",
    "customer_bump_percent",
    "floor_rule_percent",
    "engine_floor_price",
    "currency_exchange_rate",
    "total_acop_delta",
    "price_modifier_percent",
    "price_modifier_delta",
    "floor_rule_cost_value",
    "floor_rule_initial_amount",
    "price_rule_starting_amount",
    "sku_price",
    "margin_pct",  # excluded from margin model's transformations (it's the target)
]

# Categorical string columns
CATEGORICAL_COLS = [
    "sku_number",
    "mpn",
    "pricing_rule",
    "price_rule_source",
    "price_rule_subsource",
    "price_rule_cost_basis",
    "price_rule_currency",
    "country_code",
    "company_code",
    "vendor_number",
    "calling_system",
    "currency_code_from",
    "currency_code_to",
]

# BOOL columns — declared as categorical so AutoML handles them correctly
BOOL_COLS = [
    "reciprocal_flag",
    "allow_quantity_break_flag",
    "allow_floor_flag",
    "cost_override_flag",
    "special_price_override_flag",
    "skip_additional_cost_flag",
    "skip_bump_flag",
    "skip_gpe_discount_flag",
    "floor_override_price_rule",
    "was_price_overridden",
    "hit_floor_price",
]

# Full transformation list (used by conversion model)
COLUMN_TRANSFORMATIONS: list[dict] = (
    [{"numeric":     {"column_name": c}} for c in NUMERIC_COLS]
    + [{"categorical": {"column_name": c}} for c in CATEGORICAL_COLS]
    + [{"categorical": {"column_name": c}} for c in BOOL_COLS]
)

# Margin model transformation list — exclude margin_pct (it's the target)
MARGIN_COLUMN_TRANSFORMATIONS: list[dict] = [
    t for t in COLUMN_TRANSFORMATIONS
    if t.get("numeric", {}).get("column_name") != "margin_pct"
]


# ---------------------------------------------------------------------------
# Cleanup helpers — idempotent, errors are non-fatal
# ---------------------------------------------------------------------------

def cleanup_existing_resources(client: bigquery.Client) -> None:
    """
    Deletes the pricing_ml_view in BigQuery and any existing Vertex AI models,
    datasets, and endpoints with matching display names before recreating them.

    Errors are caught and logged — cleanup failures do not abort training.
    """
    # ── Delete BQ view ──────────────────────────────────────────────────────
    try:
        client.delete_table(_ML_VIEW_FULL, not_found_ok=True)
        logging.info("[TRAINER] Deleted BQ view (if existed): %s", _ML_VIEW_FULL)
    except Exception as e:
        logging.warning("[TRAINER] Could not delete BQ view %s: %s", _ML_VIEW_FULL, e)

    # ── Delete existing Vertex AI models by display name ────────────────────
    try:
        for display_name in [CONVERSION_MODEL_DISPLAY_NAME, MARGIN_MODEL_DISPLAY_NAME]:
            existing_models = aiplatform.Model.list(
                filter=f'display_name="{display_name}"',
                order_by="create_time desc",
            )
            for model in existing_models:
                logging.info(
                    "[TRAINER] Deleting existing Vertex AI model: %s (%s)",
                    model.display_name, model.resource_name,
                )
                try:
                    model.delete()
                    logging.info("[TRAINER] Deleted model: %s", display_name)
                except Exception as me:
                    logging.warning(
                        "[TRAINER] Could not delete model %s: %s", display_name, me
                    )
    except Exception as e:
        logging.warning("[TRAINER] Vertex AI model cleanup failed (non-fatal): %s", e)

    # ── Delete existing Vertex AI datasets by display name ──────────────────
    try:
        existing_datasets = aiplatform.TabularDataset.list(
            filter='display_name="pricing-ml-dataset"',
            order_by="create_time desc",
        )
        for ds in existing_datasets:
            logging.info(
                "[TRAINER] Deleting existing Vertex AI dataset: %s", ds.resource_name
            )
            try:
                ds.delete()
            except Exception as de:
                logging.warning("[TRAINER] Could not delete dataset: %s", de)
    except Exception as e:
        logging.warning(
            "[TRAINER] Vertex AI dataset cleanup failed (non-fatal): %s", e
        )


def _delete_endpoint_if_exists(display_name: str) -> None:
    """
    Undeploys all models and deletes a Vertex AI endpoint by display name if
    it exists.  Errors are caught and logged — non-fatal.
    """
    try:
        existing = aiplatform.Endpoint.list(
            filter=f'display_name="{display_name}"',
            order_by="create_time desc",
        )
        for ep in existing:
            logging.info(
                "[TRAINER] Undeploying all models from endpoint: %s", ep.resource_name
            )
            try:
                ep.undeploy_all()
                ep.delete()
                logging.info("[TRAINER] Deleted endpoint: %s", display_name)
            except Exception as e:
                logging.warning(
                    "[TRAINER] Could not delete endpoint %s: %s", display_name, e
                )
    except Exception as e:
        logging.warning(
            "[TRAINER] Endpoint cleanup failed (non-fatal): %s", e
        )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_models(date_from: str, date_to: str) -> dict[str, Any]:
    """
    Run the full Vertex AI AutoML Tabular training pipeline for a date range.

    1. Validate that records exist for the given date range
    2. Clean up existing BQ view and Vertex AI models / datasets / endpoints
    3. Create fresh BigQuery ML view (filtered to date range)
    4. Register view as a Vertex AI Managed TabularDataset
    5. Train conversion classifier (quote_accepted)
    6. Train margin regressor (margin_pct)
    7. Fetch evaluation metrics from Vertex AI
    8. Deploy both models to Vertex AI Endpoints
    9. Persist endpoint resource names to endpoints.json

    Parameters
    ----------
    date_from : str
        Lower bound for training data (YYYY-MM-DD), inclusive.
    date_to : str
        Upper bound for training data (YYYY-MM-DD), inclusive.

    Returns
    -------
    dict
        conversion_auc, conversion_log_loss,
        margin_rmse, margin_r2, margin_mae,
        conversion_endpoint (resource name),
        margin_endpoint (resource name)

    Raises
    ------
    ValueError
        If no records are found for the specified date range.
    """
    logging.info("[TRAINER] Training on date range: %s → %s", date_from, date_to)

    bq_client = bigquery.Client(project=GCP_PROJECT_ID)

    # ---- Step 1: Validate row count ----------------------------------------
    count_sql = f"""
        SELECT COUNT(*) AS cnt
        FROM `{BQ_FULL_TABLE}`
        WHERE created >= TIMESTAMP('{date_from}')
          AND created < TIMESTAMP_ADD(TIMESTAMP('{date_to}'), INTERVAL 1 DAY)
    """
    logging.info("[TRAINER] Row count query:\n%s", count_sql)
    count_row = list(bq_client.query(count_sql).result())[0]
    row_count = count_row["cnt"]
    logging.info("[TRAINER] Rows found: %d", row_count)
    if row_count == 0:
        raise ValueError(
            f"No records found in `{BQ_FULL_TABLE}` "
            f"for the date range {date_from} → {date_to}. "
            "Adjust the date range before training."
        )
    print(f"Row count for {date_from} → {date_to}: {row_count:,}")

    # ---- Step 2: Clean up existing resources --------------------------------
    logging.info(
        "[TRAINER] Cleaning up existing BQ view and Vertex AI models/datasets/endpoints…"
    )
    # Initialise Vertex AI SDK first so cleanup calls work
    print(
        f"Initialising Vertex AI (project={GCP_PROJECT_ID}, region={VERTEX_REGION}) …"
    )
    aiplatform.init(
        project=GCP_PROJECT_ID,
        location=VERTEX_REGION,
        staging_bucket=VERTEX_STAGING_BUCKET,
    )

    cleanup_existing_resources(bq_client)

    # ---- Step 3: Create fresh BigQuery ML view ------------------------------
    print(
        f"Creating BigQuery ML view `{_ML_VIEW_FULL}` "
        f"(dates: {date_from} → {date_to}) …"
    )
    view_sql = build_ml_view_sql(date_from, date_to)
    logging.info("[TRAINER] Creating pricing_ml_view…")
    bq_client.query(view_sql).result()
    logging.info("[TRAINER] pricing_ml_view created successfully")
    print("  View created successfully.")

    # ---- Step 4: Register Vertex AI Managed TabularDataset -----------------
    print("Registering Vertex AI Managed Dataset from BigQuery view …")
    dataset = aiplatform.TabularDataset.create(
        display_name="pricing-ml-dataset",
        bq_source=f"bq://{_ML_VIEW_FULL}",
    )
    print(f"  Dataset resource: {dataset.resource_name}")

    # ---- Step 5: Train conversion classifier --------------------------------
    print(f"Launching AutoML training job: {CONVERSION_MODEL_DISPLAY_NAME} …")
    conversion_job = AutoMLTabularTrainingJob(
        display_name=CONVERSION_MODEL_DISPLAY_NAME,
        optimization_prediction_type="classification",
        optimization_objective="minimize-log-loss",
        column_transformations=COLUMN_TRANSFORMATIONS,
    )
    model_conversion = conversion_job.run(
        dataset=dataset,
        target_column="quote_accepted",
        budget_milli_node_hours=1000,
        model_display_name=CONVERSION_MODEL_DISPLAY_NAME,
        sync=True,
    )
    print(f"  Conversion model resource: {model_conversion.resource_name}")

    # ---- Step 6: Train margin regressor -------------------------------------
    print(f"Launching AutoML training job: {MARGIN_MODEL_DISPLAY_NAME} …")
    margin_job = AutoMLTabularTrainingJob(
        display_name=MARGIN_MODEL_DISPLAY_NAME,
        optimization_prediction_type="regression",
        optimization_objective="minimize-rmse",
        column_transformations=MARGIN_COLUMN_TRANSFORMATIONS,
    )
    model_margin = margin_job.run(
        dataset=dataset,
        target_column="margin_pct",
        budget_milli_node_hours=1000,
        model_display_name=MARGIN_MODEL_DISPLAY_NAME,
        sync=True,
    )
    print(f"  Margin model resource: {model_margin.resource_name}")

    # ---- Step 7: Fetch evaluation metrics -----------------------------------
    print("Fetching model evaluation metrics …")

    conv_evaluations = model_conversion.list_model_evaluations()
    conv_metrics: dict = conv_evaluations[0].metrics if conv_evaluations else {}

    margin_evaluations = model_margin.list_model_evaluations()
    margin_metrics: dict = margin_evaluations[0].metrics if margin_evaluations else {}

    print(
        f"  Conversion — AUC: {conv_metrics.get('auRoc')}  "
        f"LogLoss: {conv_metrics.get('logLoss')}"
    )
    print(
        f"  Margin     — RMSE: {margin_metrics.get('rootMeanSquaredError')}  "
        f"R²: {margin_metrics.get('rSquared')}  "
        f"MAE: {margin_metrics.get('meanAbsoluteError')}"
    )

    # ---- Step 8: Deploy both models to Vertex AI Endpoints -----------------
    conv_endpoint_display_name = f"{CONVERSION_MODEL_DISPLAY_NAME}-endpoint"
    print(f"Deploying conversion model to Vertex AI Endpoint ({conv_endpoint_display_name}) …")
    _delete_endpoint_if_exists(conv_endpoint_display_name)
    conv_endpoint = model_conversion.deploy(
        deployed_model_display_name=conv_endpoint_display_name,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=2,
        sync=True,
    )
    print(f"  Conversion endpoint: {conv_endpoint.resource_name}")

    margin_endpoint_display_name = f"{MARGIN_MODEL_DISPLAY_NAME}-endpoint"
    print(f"Deploying margin model to Vertex AI Endpoint ({margin_endpoint_display_name}) …")
    _delete_endpoint_if_exists(margin_endpoint_display_name)
    margin_endpoint = model_margin.deploy(
        deployed_model_display_name=margin_endpoint_display_name,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=2,
        sync=True,
    )
    print(f"  Margin endpoint: {margin_endpoint.resource_name}")

    # ---- Step 9: Persist endpoint resource names ---------------------------
    endpoints_payload = {
        "conversion_endpoint": conv_endpoint.resource_name,
        "margin_endpoint":     margin_endpoint.resource_name,
    }
    with open("endpoints.json", "w") as fh:
        json.dump(endpoints_payload, fh, indent=2)
    print("Endpoint resource names saved to endpoints.json")

    # ---- Return results dict -----------------------------------------------
    return {
        "conversion_auc":      conv_metrics.get("auRoc"),
        "conversion_log_loss": conv_metrics.get("logLoss"),
        "margin_rmse":         margin_metrics.get("rootMeanSquaredError"),
        "margin_r2":           margin_metrics.get("rSquared"),
        "margin_mae":          margin_metrics.get("meanAbsoluteError"),
        "conversion_endpoint": conv_endpoint.resource_name,
        "margin_endpoint":     margin_endpoint.resource_name,
    }
