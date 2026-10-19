"""Project-wide constants: canonical schema, column aliases, session keys, seed."""

RANDOM_SEED = 42

# Canonical business schema. First three are required; rest are optional.
REQUIRED_FIELDS = ("customer_id", "transaction_date", "amount_spent")
OPTIONAL_FIELDS = ("gender", "age", "city", "state", "country", "region")
CANONICAL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

# Likely source-column variations for automatic mapping (spec §Phase 1).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "customer_id": ("customer_id", "customerid", "cust_id", "client_id", "member_id", "user_id"),
    "transaction_date": (
        "transaction_date",
        "purchase_date",
        "order_date",
        "invoice_date",
        "txn_date",
        "date",
    ),
    "amount_spent": (
        "amount_spent",
        "amount",
        "sales",
        "revenue",
        "spend",
        "transaction_amount",
        "total_amount",
    ),
    "gender": ("gender", "sex"),
    "age": ("age", "customer_age"),
    "city": ("city", "town"),
    "state": ("state", "province"),
    "country": ("country", "nation"),
    "region": ("region", "zone", "territory"),
}

# CSV encoding fallback order (spec §Phase 1).
CSV_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

SUPPORTED_UPLOAD_SUFFIXES = (".csv", ".xlsx", ".xls", ".parquet")

# Streamlit session-state contract (spec §6). Owned by services/state_service.py.
SESSION_KEYS = (
    "raw_data",
    "clean_data",
    "column_mapping",
    "cleaning_config",
    "cleaning_audit",
    "data_dictionary",
    "customer_features",
    "rfm_results",
    "lrfm_results",
    "demographic_profile",
    "segmentation_runs",
    "selected_segmentation_model",
    "segmented_customers",
    "churn_dataset",
    "churn_model",
    "churn_model_metadata",
    "churn_scores",
    "retention_results",
    "segment_history",
    "segment_history_labels",
    "global_filters",
)
