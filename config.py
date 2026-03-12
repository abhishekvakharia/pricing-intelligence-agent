# config.py — Fill in before running

GCP_PROJECT_ID = "imgcp-20260224090629"
BQ_DATASET_NAME = "Pricing_Dataset"
BQ_TABLE_NAME   = "price_facts"

# Derived full table reference — do not edit
BQ_FULL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET_NAME}.{BQ_TABLE_NAME}"

# Vertex AI — fill in before running model training
VERTEX_REGION                 = "us-central1"         # e.g. "us-central1"
VERTEX_STAGING_BUCKET         = "gs://imgcp-20260224090629-vertex-staging"   # must exist in GCP_PROJECT_ID
CONVERSION_MODEL_DISPLAY_NAME = "pricing-conversion-model"
MARGIN_MODEL_DISPLAY_NAME     = "pricing-margin-model"
 
