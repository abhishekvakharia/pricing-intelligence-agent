"""
ml/trainer.py
Vertex AI AutoML Tabular training pipeline for the Pricing Intelligence Agent.

All training runs entirely on GCP — no local ML libraries are required.

Steps executed by train_models():
  1. Initialise Vertex AI SDK
  2. Create (or replace) a BigQuery ML view with derived feature columns
  3. Register the view as a Vertex AI Managed TabularDataset
  4. Launch AutoML Tabular training for:
       • Model 1  — Conversion Classifier  (target: quote_accepted)
       • Model 2  — Margin Regressor       (target: margin_pct)
  5. Fetch model-level evaluation metrics
  6. Deploy both models to Vertex AI Endpoints
  7. Persist endpoint resource names to endpoints.json
  8. Return a results dict consumed by agent/tools.py and the dashboard
"""

from __future__ import annotations

import json
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
# BigQuery ML view DDL
# ---------------------------------------------------------------------------

# The view materialises all raw columns plus four derived ML features so that
# Vertex AI AutoML can read them directly from BigQuery.
_ML_VIEW_NAME = "pricing_ml_view"
_ML_VIEW_FULL = f"{GCP_PROJECT_ID}.{BQ_DATASET_NAME}.{_ML_VIEW_NAME}"

CREATE_VIEW_SQL = f"""
CREATE OR REPLACE VIEW `{_ML_VIEW_FULL}` AS
SELECT
    sku_number,
    mpn,
    customer_number,
    vendor_number,
    company_code,
    country_code,
    REGION,
    pricing_rule,
    price_rule_source,
    price_rule_subsource,
    calculated_price,
    base_price,
    sku_price,
    cost_basis,
    final_cost,
    final_price_margin,
    modifier_percent,
    customer_bump_percent,
    floor_rule_percent,
    engine_floor_price,
    total_acop_delta,
    currency_exchange_rate,
    VENDOR_LOB_LVL2_DES,
    VENDOR_LOB_LVL3_DES,
    allow_quantity_break_flag,
    allow_floor_flag,
    allow_acop_flag,
    cost_override_flag,
    skip_bump_flag,
    skip_gpe_discount_flag,
    -- Derived target + feature columns
    CASE WHEN order_key IS NOT NULL THEN 1 ELSE 0 END
        AS quote_accepted,
    SAFE_DIVIDE(final_price_margin, NULLIF(calculated_price, 0)) * 100
        AS margin_pct,
    CASE WHEN overriden_price IS NOT NULL THEN 1 ELSE 0 END
        AS was_price_overridden,
    CASE WHEN calculated_price = engine_floor_price THEN 1 ELSE 0 END
        AS hit_floor_price
FROM `{BQ_FULL_TABLE}`
WHERE db_rec_del_flag != 'Y'
  AND db_rec_close_date IS NULL
"""

# ---------------------------------------------------------------------------
# Column transformation specs for AutoML
# ---------------------------------------------------------------------------

# Numeric columns fed to both models
_NUMERIC_COLS = [
    "calculated_price",
    "base_price",
    "sku_price",
    "cost_basis",
    "final_cost",
    "final_price_margin",
    "modifier_percent",
    "customer_bump_percent",
    "floor_rule_percent",
    "engine_floor_price",
    "currency_exchange_rate",
    "total_acop_delta",
    "margin_pct",        # excluded from margin model's transformations (it's the target)
]

# Categorical columns fed to both models
_CATEGORICAL_COLS = [
    "sku_number",
    "pricing_rule",
    "price_rule_source",
    "price_rule_subsource",
    "country_code",
    "REGION",
    "company_code",
    "VENDOR_LOB_LVL2_DES",
    "VENDOR_LOB_LVL3_DES",
    "allow_quantity_break_flag",
    "allow_floor_flag",
    "allow_acop_flag",
    "cost_override_flag",
    "skip_bump_flag",
    "skip_gpe_discount_flag",
    "was_price_overridden",   # derived binary, treated as categorical
    "hit_floor_price",        # derived binary, treated as categorical
]

# Full transformation list (used by conversion model)
COLUMN_TRANSFORMATIONS: list[dict] = (
    [{"numeric":     {"column_name": c}} for c in _NUMERIC_COLS]
    + [{"categorical": {"column_name": c}} for c in _CATEGORICAL_COLS]
)

# Margin model transformation list — exclude margin_pct (it's the target)
MARGIN_COLUMN_TRANSFORMATIONS: list[dict] = [
    t for t in COLUMN_TRANSFORMATIONS
    if t.get("numeric", {}).get("column_name") != "margin_pct"
]

# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_models() -> dict[str, Any]:
    """
    Run the full Vertex AI AutoML Tabular training pipeline.

    1. Initialise Vertex AI SDK
    2. Create/replace BigQuery ML view
    3. Register view as a Vertex AI Managed TabularDataset
    4. Train conversion classifier (quote_accepted)
    5. Train margin regressor (margin_pct)
    6. Fetch evaluation metrics from Vertex AI
    7. Deploy both models to Vertex AI Endpoints
    8. Persist endpoint resource names to endpoints.json

    Returns
    -------
    dict
        conversion_auc, conversion_log_loss,
        margin_rmse, margin_r2, margin_mae,
        conversion_endpoint (resource name),
        margin_endpoint (resource name)
    """
    # ---- Step 1: Initialise Vertex AI SDK ----------------------------------
    print(f"Initialising Vertex AI (project={GCP_PROJECT_ID}, region={VERTEX_REGION}) …")
    aiplatform.init(
        project=GCP_PROJECT_ID,
        location=VERTEX_REGION,
        staging_bucket=VERTEX_STAGING_BUCKET,
    )

    # ---- Step 2: Create BigQuery ML view -----------------------------------
    print(f"Creating BigQuery ML view `{_ML_VIEW_FULL}` …")
    bq_client = bigquery.Client(project=GCP_PROJECT_ID)
    bq_client.query(CREATE_VIEW_SQL).result()
    print("  View created successfully.")

    # ---- Step 3: Register Vertex AI Managed TabularDataset -----------------
    print("Registering Vertex AI Managed Dataset from BigQuery view …")
    dataset = aiplatform.TabularDataset.create(
        display_name="pricing-ml-dataset",
        bq_source=f"bq://{_ML_VIEW_FULL}",
    )
    print(f"  Dataset resource: {dataset.resource_name}")

    # ---- Step 4: Train conversion classifier -------------------------------
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
        sync=True,   # wait for completion before proceeding
    )
    print(f"  Conversion model resource: {model_conversion.resource_name}")

    # ---- Step 5: Train margin regressor ------------------------------------
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

    # ---- Step 6: Fetch evaluation metrics ----------------------------------
    print("Fetching model evaluation metrics …")

    conv_evaluations = model_conversion.list_model_evaluations()
    conv_metrics: dict = conv_evaluations[0].metrics if conv_evaluations else {}

    margin_evaluations = model_margin.list_model_evaluations()
    margin_metrics: dict = margin_evaluations[0].metrics if margin_evaluations else {}

    print(f"  Conversion — AUC: {conv_metrics.get('auRoc')}  LogLoss: {conv_metrics.get('logLoss')}")
    print(f"  Margin     — RMSE: {margin_metrics.get('rootMeanSquaredError')}  R²: {margin_metrics.get('rSquared')}  MAE: {margin_metrics.get('meanAbsoluteError')}")

    # ---- Step 7: Deploy both models to Vertex AI Endpoints -----------------
    print("Deploying conversion model to Vertex AI Endpoint …")
    conv_endpoint = model_conversion.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=2,
        sync=True,
    )
    print(f"  Conversion endpoint: {conv_endpoint.resource_name}")

    print("Deploying margin model to Vertex AI Endpoint …")
    margin_endpoint = model_margin.deploy(
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=2,
        sync=True,
    )
    print(f"  Margin endpoint: {margin_endpoint.resource_name}")

    # ---- Step 8: Persist endpoint resource names ---------------------------
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
