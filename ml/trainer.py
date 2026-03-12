"""
ml/trainer.py
Vertex AI AutoML Tabular training pipeline for the Pricing Intelligence Agent.

All training runs entirely on GCP -- no local ML libraries are required.

Steps executed by train_models():
  1. Validate row count for the given date range
  2. Clean up existing BQ view, Vertex AI models, datasets, and endpoints
  3. Create fresh BigQuery ML view with derived feature columns
  4. Verify the view was created and has rows
  5. Register the view as a Vertex AI Managed TabularDataset
  6. Launch AutoML Tabular training for:
       * Model 1  -- Conversion Classifier  (target: quote_accepted)
       * Model 2  -- Margin Regressor       (target: margin_pct)
  7. Fetch model-level evaluation metrics
  8. Deploy both models to Vertex AI Endpoints
  9. Persist endpoint resource names to endpoints.json
 10. Return a results dict consumed by agent/tools.py and the dashboard

Schema notes for price_fact_us:
  - No SCD columns (db_rec_del_flag, db_rec_close_date, REGION, etc.)
  - BOOL flags: use as-is -- AutoML handles BOOL natively
  - final_cost, currency_exchange_rate: stored as STRING -- CAST to FLOAT64
  - customer_number, customer_branch_number: stored as INT64 -- CAST to STRING
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
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
    DEFAULT_MARGIN_MODEL_RESOURCE,
    DEFAULT_CONVERSION_MODEL_RESOURCE,
)

# ---------------------------------------------------------------------------
# BigQuery ML view constants
# ---------------------------------------------------------------------------

_ML_VIEW_NAME = "pricing_ml_view"
_ML_VIEW_FULL = f"{GCP_PROJECT_ID}.{BQ_DATASET_NAME}.{_ML_VIEW_NAME}"

# Path for persisting endpoint resource names (relative to project root)
_ENDPOINTS_FILE = Path(__file__).resolve().parents[1] / "endpoints.json"


def get_active_endpoints() -> dict:
    """
    Return the active model/endpoint resource names.

    Priority:
      1. endpoints.json written by a completed training run (user-trained model)
      2. DEFAULT_*_MODEL_RESOURCE values from config (pre-trained baseline)

    Returns a dict with keys:
      margin_endpoint     -- resource name of the margin model or endpoint
      conversion_endpoint -- resource name of the conversion model/endpoint (may be None)
      source              -- "user_trained" | "default"
    """
    if _ENDPOINTS_FILE.exists():
        try:
            data = json.loads(_ENDPOINTS_FILE.read_text(encoding="utf-8"))
            if data.get("margin_endpoint"):
                data["source"] = "user_trained"
                return data
        except Exception:
            pass
    # Fall back to pre-trained defaults
    return {
        "margin_endpoint":     DEFAULT_MARGIN_MODEL_RESOURCE,
        "conversion_endpoint": DEFAULT_CONVERSION_MODEL_RESOURCE,
        "source":              "default",
    }

# Shared training-status file (same file the dashboard polls)
_TRAINING_STATUS_FILE = Path(__file__).resolve().parents[1] / "training_status.json"

# Exclusive lock file created by train_models() itself -- separate from the
# status file so the dashboard can freely write 'running' without triggering
# the guard.
_TRAINING_LOCK_FILE = Path(__file__).resolve().parents[1] / "training.lock"


def _acquire_training_lock() -> bool:
    """Create the lock file atomically. Returns True if lock was acquired, False if already locked."""
    try:
        _TRAINING_LOCK_FILE.touch(exist_ok=False)  # fails if already exists
        return True
    except FileExistsError:
        return False
    except Exception:
        return True  # if we can't create it, allow training (non-fatal)


def _release_training_lock() -> None:
    """Remove the lock file; errors are silently ignored."""
    try:
        _TRAINING_LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# BigQuery ML view DDL
# ---------------------------------------------------------------------------

def _table_has_column(bq_client: bigquery.Client, column_name: str) -> bool:
    """Return True if *column_name* exists in the source BQ table schema."""
    try:
        table = bq_client.get_table(BQ_FULL_TABLE)
        return any(f.name.lower() == column_name.lower() for f in table.schema)
    except Exception:
        return False


def build_ml_view_sql(date_from: str, date_to: str, has_sow_scale_col: bool = False) -> str:
    """
    Build the CREATE OR REPLACE VIEW DDL for the Vertex AI training dataset.

    Parameters
    ----------
    date_from : str
        Lower bound date string (YYYY-MM-DD), inclusive.
    date_to : str
        Upper bound date string (YYYY-MM-DD), inclusive.
    has_sow_scale_col : bool
        Whether SoW_final_org_scale exists in the source table. If True,
        rows with NULL in that column are excluded.

    Returns
    -------
    str
        A complete CREATE OR REPLACE VIEW SQL statement.
    """
    _sow_filter = "\n  AND SoW_final_org_scale IS NOT NULL" if has_sow_scale_col else ""
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

    -- Customer & Org (INT64 -> STRING for categorical encoding)
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

    -- Floor Rule
    engine_floor_price,
    floor_rule_percent,
    floor_rule_cost_value,
    floor_rule_initial_amount,

    -- Currency (STRING -> FLOAT64 for arithmetic)
    CAST(currency_exchange_rate AS FLOAT64) AS currency_exchange_rate,
    currency_code_from,
    currency_code_to,

    -- BOOL flags -- AutoML handles BOOL natively
    reciprocal_flag,
    allow_quantity_break_flag,
    allow_floor_flag,
    cost_override_flag,
    special_price_override_flag,
    skip_additional_cost_flag,
    skip_bump_flag,
    skip_gpe_discount_flag,

    -- Derived target and feature columns
    -- quote_accepted cast to STRING so AutoML treats it as a categorical label
    CAST(CASE WHEN order_key IS NOT NULL THEN 1 ELSE 0 END AS STRING)
        AS quote_accepted,
    final_price_margin
        AS margin_pct,
    CAST(CASE WHEN overriden_price IS NOT NULL THEN 1 ELSE 0 END AS STRING)
        AS was_price_overridden,
    CAST(CASE WHEN calculated_price = engine_floor_price THEN 1 ELSE 0 END AS STRING)
        AS hit_floor_price

FROM `{BQ_FULL_TABLE}`
WHERE created >= TIMESTAMP('{date_from}')
  AND created < TIMESTAMP_ADD(TIMESTAMP('{date_to}'), INTERVAL 1 DAY)
  AND final_price_margin > 0
  AND calculated_price IS NOT NULL
  AND calculated_price != 0{_sow_filter}
"""


# ---------------------------------------------------------------------------
# Column transformation specs for AutoML
# ---------------------------------------------------------------------------

# Numeric columns -- both models
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

# BOOL columns -- declared as categorical so AutoML handles them correctly
BOOL_COLS = [
    "reciprocal_flag",
    "allow_quantity_break_flag",
    "allow_floor_flag",
    "cost_override_flag",
    "special_price_override_flag",
    "skip_additional_cost_flag",
    "skip_bump_flag",
    "skip_gpe_discount_flag",
    "was_price_overridden",
    "hit_floor_price",
]

# Full transformation list (used by conversion model)
COLUMN_TRANSFORMATIONS: list[dict] = (
    [{"numeric":     {"column_name": c}} for c in NUMERIC_COLS]
    + [{"categorical": {"column_name": c}} for c in CATEGORICAL_COLS]
    + [{"categorical": {"column_name": c}} for c in BOOL_COLS]
)

# Margin model transformation list -- exclude margin_pct (it's the target)
MARGIN_COLUMN_TRANSFORMATIONS: list[dict] = [
    t for t in COLUMN_TRANSFORMATIONS
    if t.get("numeric", {}).get("column_name") != "margin_pct"
]


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def _step(n: int, total: int, msg: str) -> None:
    """Log and print a numbered step banner."""
    banner = f"[TRAINER] Step {n}/{total}: {msg}"
    logging.info(banner)
    print(banner)


def _verify_view_exists(bq_client: bigquery.Client) -> int:
    """
    Verify the ML view was created and has rows.

    Returns the row count from the view.

    Raises
    ------
    RuntimeError
        If the view has 0 rows (DDL succeeded but the source table returned
        nothing - likely a bad date filter).
    """
    count_sql = f"SELECT COUNT(*) AS cnt FROM `{_ML_VIEW_FULL}`"
    logging.info("[TRAINER] Verifying view: %s", count_sql)
    rows = list(bq_client.query(count_sql).result())
    view_cnt = rows[0]["cnt"]
    logging.info("[TRAINER] View row count: %d", view_cnt)
    if view_cnt == 0:
        raise RuntimeError(
            f"BigQuery ML view `{_ML_VIEW_FULL}` was created but contains 0 rows. "
            "Check the date filter -- the source table may have no data for this range."
        )
    return view_cnt


# ---------------------------------------------------------------------------
# Cleanup helpers -- idempotent, errors are non-fatal
# ---------------------------------------------------------------------------

def cleanup_existing_resources(bq_client: bigquery.Client) -> None:
    """
    Deletes existing Vertex AI models, datasets, and endpoints with matching
    display names before recreating them.

    The BQ view is intentionally NOT deleted here -- the DDL uses
    CREATE OR REPLACE VIEW, so it is safe to overwrite in place.  Deleting
    it before training completes would break any in-flight Vertex AI job that
    still holds a reference to the view.

    Errors are caught and logged -- cleanup failures do not abort training.
    """
    # ── Delete existing Vertex AI endpoints first (must undeploy before model delete) ──
    for display_name in [
        f"{CONVERSION_MODEL_DISPLAY_NAME}-endpoint",
        f"{MARGIN_MODEL_DISPLAY_NAME}-endpoint",
    ]:
        _delete_endpoint_if_exists(display_name)

    # ── Delete existing Vertex AI models by display name ────────────────────
    # Models can only be deleted after all endpoints referencing them are gone.
    for display_name in [CONVERSION_MODEL_DISPLAY_NAME, MARGIN_MODEL_DISPLAY_NAME]:
        try:
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
                    model.delete(sync=True)  # block until deletion confirmed
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
                ds.delete(sync=True)  # block until deletion confirmed
            except Exception as de:
                logging.warning("[TRAINER] Could not delete dataset: %s", de)
    except Exception as e:
        logging.warning(
            "[TRAINER] Vertex AI dataset cleanup failed (non-fatal): %s", e
        )


def _delete_endpoint_if_exists(display_name: str) -> None:
    """
    Undeploys all models from a Vertex AI endpoint and deletes it, blocking
    until the operations complete so that the model can be safely deleted next.
    Errors are caught and logged -- non-fatal.
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
                ep.undeploy_all(sync=True)  # block -- model delete will fail if deployed
                ep.delete(sync=True)        # block until deletion confirmed
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

_TOTAL_STEPS = 10


def train_models(date_from: str, date_to: str) -> dict[str, Any]:
    """
    Run the full Vertex AI AutoML Tabular training pipeline for a date range.

    Steps
    -----
    1.  Validate that source table has rows for the date range
    2.  Initialise Vertex AI SDK
    3.  Clean up existing BQ view and Vertex AI resources
    4.  Create fresh BigQuery ML view
    5.  Verify view has rows
    6.  Register view as a Vertex AI Managed TabularDataset
    7.  Train conversion classifier (quote_accepted)
    8.  Train margin regressor (margin_pct)
    9.  Fetch evaluation metrics
    10. Deploy both models and save endpoints.json

    Parameters
    ----------
    date_from : str
        Lower bound for training data (YYYY-MM-DD), inclusive.
    date_to : str
        Upper bound for training data (YYYY-MM-DD), inclusive.

    Returns
    -------
    dict
        Keys: conversion_auc, conversion_log_loss,
              margin_rmse, margin_r2, margin_mae,
              conversion_endpoint, margin_endpoint

    Raises
    ------
    ValueError
        If no records found in the source table for the given date range.
    RuntimeError
        If the BQ view is created but contains 0 rows, or if a Vertex job
        reports errors after completing.
    Exception
        Re-raised from Vertex AI SDK on training/deployment failures.
    """
    # Guard: refuse to start if another train_models() call is already running.
    # Uses a dedicated lock file (training.lock) rather than training_status.json
    # to avoid the race where the dashboard pre-writes 'running' before spawning
    # the thread, which would make train_models() think it is already in progress.
    if not _acquire_training_lock():
        try:
            _existing = json.loads(_TRAINING_STATUS_FILE.read_text(encoding="utf-8"))
            _d_from = _existing.get("date_from", "?")
            _d_to   = _existing.get("date_to",   "?")
            _since  = _existing.get("updated_at", "")[:19].replace("T", " ")
        except Exception:
            _d_from = _d_to = "?"
            _since = ""
        raise RuntimeError(
            f"A training job is already in progress (date range {_d_from} to {_d_to}, "
            f"started ~{_since} UTC). "
            "Wait for it to complete or cancel it in the Vertex AI console before "
            "starting a new run."
        )

    logging.info(
        "[TRAINER] ===== Training pipeline START  %s to %s =====",
        date_from, date_to,
    )

    bq_client = bigquery.Client(project=GCP_PROJECT_ID)

    # ── Step 1: Validate source row count ────────────────────────────────────
    _step(1, _TOTAL_STEPS, f"Validating source row count for {date_from} to {date_to}")
    count_sql = f"""
        SELECT COUNT(*) AS cnt
        FROM `{BQ_FULL_TABLE}`
        WHERE created >= TIMESTAMP('{date_from}')
          AND created < TIMESTAMP_ADD(TIMESTAMP('{date_to}'), INTERVAL 1 DAY)
    """
    logging.info("[TRAINER] Row count SQL:\n%s", count_sql)
    count_job = bq_client.query(count_sql)
    count_rows = list(count_job.result())  # blocks until complete
    if count_job.errors:
        raise RuntimeError(
            f"Row count query failed with errors: {count_job.errors}"
        )
    row_count = count_rows[0]["cnt"]
    logging.info("[TRAINER] Source rows in range: %d", row_count)
    print(f"  Source rows found: {row_count:,}")
    if row_count == 0:
        raise ValueError(
            f"No records found in `{BQ_FULL_TABLE}` "
            f"for the date range {date_from} to {date_to}. "
            "Adjust the date range before training."
        )

    # Check quote_accepted class balance -- AutoML classification requires >= 2 classes
    balance_sql = f"""
        SELECT
            COUNTIF(order_key IS NOT NULL) AS accepted,
            COUNTIF(order_key IS NULL)     AS not_accepted
        FROM `{BQ_FULL_TABLE}`
        WHERE created >= TIMESTAMP('{date_from}')
          AND created < TIMESTAMP_ADD(TIMESTAMP('{date_to}'), INTERVAL 1 DAY)
    """
    bal_rows = list(bq_client.query(balance_sql).result())
    accepted_count = int(bal_rows[0]["accepted"])
    not_accepted_count = int(bal_rows[0]["not_accepted"])
    logging.info(
        "[TRAINER] Class balance: accepted=%d  not_accepted=%d",
        accepted_count, not_accepted_count,
    )
    print(f"  Class balance: accepted={accepted_count:,}  not_accepted={not_accepted_count:,}")
    train_conversion = accepted_count > 0 and not_accepted_count > 0
    if not train_conversion:
        logging.warning(
            "[TRAINER] quote_accepted has only one class (%d accepted, %d not_accepted) "
            "for this date range. Conversion classifier will be skipped.",
            accepted_count, not_accepted_count,
        )
        print(
            "  WARNING: quote_accepted is single-class for this date range "
            "-- conversion model will be skipped. Only the margin regressor will be trained."
        )

    # ── Step 2: Initialise Vertex AI SDK ─────────────────────────────────────
    _step(2, _TOTAL_STEPS, f"Initialising Vertex AI (project={GCP_PROJECT_ID}, region={VERTEX_REGION})")
    aiplatform.init(
        project=GCP_PROJECT_ID,
        location=VERTEX_REGION,
        staging_bucket=VERTEX_STAGING_BUCKET,
    )
    logging.info("[TRAINER] Vertex AI SDK initialised")

    # ── Step 3: Cleanup existing resources ────────────────────────────────────
    _step(3, _TOTAL_STEPS, "Cleaning up existing BQ view and Vertex AI resources")
    cleanup_existing_resources(bq_client)

    # ── Step 4: Create BigQuery ML view ───────────────────────────────────────
    _step(4, _TOTAL_STEPS, f"Creating BigQuery ML view `{_ML_VIEW_FULL}`")
    _has_sow = _table_has_column(bq_client, "SoW_final_org_scale")
    logging.info("[TRAINER] SoW_final_org_scale present in source table: %s", _has_sow)
    view_sql = build_ml_view_sql(date_from, date_to, has_sow_scale_col=_has_sow)
    logging.info("[TRAINER] ML view DDL:\n%s", view_sql)
    view_job = bq_client.query(view_sql)
    view_job.result()  # blocks until complete
    if view_job.errors:
        raise RuntimeError(
            f"ML view DDL failed with BigQuery errors: {view_job.errors}"
        )
    logging.info("[TRAINER] ML view created: %s", _ML_VIEW_FULL)
    print(f"  View created: {_ML_VIEW_FULL}")

    # ── Step 5: Verify view has rows ─────────────────────────────────────────
    _step(5, _TOTAL_STEPS, "Verifying ML view row count")
    view_row_count = _verify_view_exists(bq_client)
    print(f"  View rows: {view_row_count:,}")

    # ── Step 6: Register Vertex AI Managed TabularDataset ────────────────────
    _step(6, _TOTAL_STEPS, "Registering Vertex AI Managed TabularDataset")
    dataset = aiplatform.TabularDataset.create(
        display_name="pricing-ml-dataset",
        bq_source=f"bq://{_ML_VIEW_FULL}",
    )
    logging.info("[TRAINER] Dataset registered: %s", dataset.resource_name)
    print(f"  Dataset: {dataset.resource_name}")

    # ── Step 7: Train conversion classifier ──────────────────────────────────
    model_conversion = None
    if train_conversion:
        _step(7, _TOTAL_STEPS, f"Training conversion classifier: {CONVERSION_MODEL_DISPLAY_NAME}")
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
            training_fraction_split=0.8,
            validation_fraction_split=0.1,
            test_fraction_split=0.1,
            sync=True,  # block until job finishes -- surface any errors immediately
        )
        if conversion_job.has_failed:
            raise RuntimeError(
                f"Conversion classifier training failed. "
                f"State: {conversion_job.state}. "
                f"Check Vertex AI console for pipeline details."
            )
        logging.info(
            "[TRAINER] Conversion model trained: %s", model_conversion.resource_name
        )
        print(f"  Conversion model: {model_conversion.resource_name}")
    else:
        _step(7, _TOTAL_STEPS, "Skipping conversion classifier (single-class label in date range)")

    # ── Step 8: Train margin regressor ───────────────────────────────────────
    _step(8, _TOTAL_STEPS, f"Training margin regressor: {MARGIN_MODEL_DISPLAY_NAME}")
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
        training_fraction_split=0.8,
        validation_fraction_split=0.1,
        test_fraction_split=0.1,
        sync=True,  # block until job finishes -- surface any errors immediately
    )
    if margin_job.has_failed:
        raise RuntimeError(
            f"Margin regressor training failed. "
            f"State: {margin_job.state}. "
            f"Check Vertex AI console for pipeline details."
        )
    logging.info(
        "[TRAINER] Margin model trained: %s", model_margin.resource_name
    )
    print(f"  Margin model: {model_margin.resource_name}")

    # ── Step 9: Fetch evaluation metrics ─────────────────────────────────────
    _step(9, _TOTAL_STEPS, "Fetching evaluation metrics from Vertex AI")

    conv_metrics: dict = {}
    if model_conversion is not None:
        conv_evaluations = model_conversion.list_model_evaluations()
        if conv_evaluations:
            conv_metrics = dict(conv_evaluations[0].metrics)
        else:
            logging.warning("[TRAINER] No evaluation metrics returned for conversion model")
    else:
        logging.info("[TRAINER] Conversion model was skipped -- no metrics to fetch")

    margin_evaluations = model_margin.list_model_evaluations()
    margin_metrics: dict = {}
    if margin_evaluations:
        margin_metrics = dict(margin_evaluations[0].metrics)
    else:
        logging.warning("[TRAINER] No evaluation metrics returned for margin model")

    logging.info(
        "[TRAINER] Conversion metrics -- AUC: %s  LogLoss: %s",
        conv_metrics.get("auRoc"), conv_metrics.get("logLoss"),
    )
    logging.info(
        "[TRAINER] Margin metrics - RMSE: %s  R2: %s  MAE: %s",
        margin_metrics.get("rootMeanSquaredError"),
        margin_metrics.get("rSquared"),
        margin_metrics.get("meanAbsoluteError"),
    )
    print(
        f"  Conversion - AUC: {conv_metrics.get('auRoc')}  "
        f"LogLoss: {conv_metrics.get('logLoss')}"
    )
    print(
        f"  Margin     - RMSE: {margin_metrics.get('rootMeanSquaredError')}  "
        f"R2: {margin_metrics.get('rSquared')}  "
        f"MAE: {margin_metrics.get('meanAbsoluteError')}"
    )

    # ── Step 10: Deploy both models + save endpoints.json ────────────────────
    _step(10, _TOTAL_STEPS, "Deploying models to Vertex AI Endpoints")

    conv_endpoint = None
    if model_conversion is not None:
        conv_endpoint_display_name = f"{CONVERSION_MODEL_DISPLAY_NAME}-endpoint"
        logging.info("[TRAINER] Deploying conversion model -> %s", conv_endpoint_display_name)
        _delete_endpoint_if_exists(conv_endpoint_display_name)
        conv_endpoint = model_conversion.deploy(
            deployed_model_display_name=conv_endpoint_display_name,
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=2,
            sync=True,
        )
        logging.info("[TRAINER] Conversion endpoint: %s", conv_endpoint.resource_name)
        print(f"  Conversion endpoint: {conv_endpoint.resource_name}")
    else:
        logging.info("[TRAINER] Conversion model was skipped -- no endpoint to deploy")
        print("  Conversion endpoint: skipped (single-class label)")

    margin_endpoint_display_name = f"{MARGIN_MODEL_DISPLAY_NAME}-endpoint"
    logging.info("[TRAINER] Deploying margin model -> %s", margin_endpoint_display_name)
    _delete_endpoint_if_exists(margin_endpoint_display_name)
    margin_endpoint = model_margin.deploy(
        deployed_model_display_name=margin_endpoint_display_name,
        machine_type="n1-standard-4",
        min_replica_count=1,
        max_replica_count=2,
        sync=True,
    )
    logging.info("[TRAINER] Margin endpoint: %s", margin_endpoint.resource_name)
    print(f"  Margin endpoint: {margin_endpoint.resource_name}")

    # Persist endpoint resource names to disk
    endpoints_payload = {
        "conversion_endpoint": conv_endpoint.resource_name if conv_endpoint else None,
        "margin_endpoint":     margin_endpoint.resource_name,
    }
    _ENDPOINTS_FILE.write_text(json.dumps(endpoints_payload, indent=2), encoding="utf-8")
    logging.info("[TRAINER] Endpoints saved to %s", _ENDPOINTS_FILE)
    print(f"  Endpoints written to {_ENDPOINTS_FILE}")

    results = {
        "conversion_auc":      conv_metrics.get("auRoc"),
        "conversion_log_loss": conv_metrics.get("logLoss"),
        "margin_rmse":         margin_metrics.get("rootMeanSquaredError"),
        "margin_r2":           margin_metrics.get("rSquared"),
        "margin_mae":          margin_metrics.get("meanAbsoluteError"),
        "conversion_endpoint": conv_endpoint.resource_name if conv_endpoint else None,
        "margin_endpoint":     margin_endpoint.resource_name,
        "conversion_skipped":  not train_conversion,
    }
    logging.info(
        "[TRAINER] ===== Training pipeline COMPLETE  results=%s =====", results
    )
    return results
