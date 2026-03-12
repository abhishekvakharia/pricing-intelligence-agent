# config.py -- Fill in before running

GCP_PROJECT_ID = "imgcp-20260224090629"
BQ_DATASET_NAME = "Pricing_Dataset"
BQ_TABLE_NAME   = "price_fact"

# Derived full table reference -- do not edit
BQ_FULL_TABLE = f"{GCP_PROJECT_ID}.{BQ_DATASET_NAME}.{BQ_TABLE_NAME}"

# Vertex AI -- fill in before running model training
VERTEX_REGION                 = "us-central1"
VERTEX_STAGING_BUCKET         = "gs://imgcp-20260224090629-vertex-staging"
CONVERSION_MODEL_DISPLAY_NAME = "pricing-conversion-model"
MARGIN_MODEL_DISPLAY_NAME     = "pricing-margin-model"

# Default pre-trained model resource names (price_fact_dataset baseline).
# These are used when no user-initiated training has completed yet.
# After a successful training run, endpoints.json overrides these values.
DEFAULT_MARGIN_MODEL_RESOURCE     = (
    "projects/212065201761/locations/us-central1"
    "/endpoints/4314378624032571392"
)
DEFAULT_CONVERSION_MODEL_RESOURCE = None  # no default conversion model

