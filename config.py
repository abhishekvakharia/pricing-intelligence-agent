# config.py — Fill in before running

GCP_PROJECT_ID = "YOUR_GCP_PROJECT_ID"
BQ_DATASET_NAME = "YOUR_DATASET_NAME"
BQ_TABLE_NAME   = "YOUR_TABLE_NAME"

# Derived full table reference — do not edit
BQ_FULL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET_NAME}.{BQ_TABLE_NAME}"

# Vertex AI — fill in before running model training
VERTEX_REGION                 = "YOUR_VERTEX_REGION"         # e.g. "us-central1"
VERTEX_STAGING_BUCKET         = "gs://YOUR_STAGING_BUCKET"   # must exist in GCP_PROJECT_ID
CONVERSION_MODEL_DISPLAY_NAME = "pricing-conversion-model"
MARGIN_MODEL_DISPLAY_NAME     = "pricing-margin-model"
