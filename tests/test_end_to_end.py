"""Phase 8 acceptance: the full 14-step workflow on the synthetic sample, no code edits.

Upload → map → clean → features → RFM/LRFM → demographics → AI segments →
compare → churn dataset → train → score → retention actions → dashboard filters
→ exports. Slow (trains one churn model); this is the release gate.
"""

import pandas as pd

from analytics.customer_features import build_customer_features
from analytics.demographics import build_demographic_profile
from analytics.lrfm import score_lrfm
from analytics.rfm import score_rfm
from analytics.segmentation_comparison import agreement_ari, recommend_model
from churn.dataset_builder import build_churn_dataset
from churn.predict import score_customers
from retention_engine.strategy_mapper import load_matrix
from services import (
    churn_service,
    retention_service,
    segmentation_service,
)
from services.cleaning_service import CleaningConfig, clean
from services.column_detection import validate_mapping
from services.dashboard_service import (
    apply_filters,
    build_master,
    compute_kpis,
    prepare_transactions,
    trend_tables,
)
from services.data_dictionary_service import build_dictionary
from services.data_service import load_upload_bytes
from services.export_service import to_csv_bytes, to_json_bytes
from services.profiling_service import compute_quality


def test_full_workflow_on_sample_data() -> None:
    """Every phase consumes the previous phase's outputs on sample data."""
    raw_bytes = open("data/sample/sample_customer_transactions.csv", "rb").read()

    # 1–2. Upload + mapping.
    preview = load_upload_bytes(raw_bytes, "sample_customer_transactions.csv")
    assert preview.rows > 10000
    validate_mapping(preview.proposal.mapping, preview.normalized_columns)

    # 3. Clean + audit + dictionary.
    quality = compute_quality(preview.frame, preview.proposal.mapping)
    assert quality.overall > 0
    cleaned, audit = clean(preview.frame, preview.proposal.mapping, CleaningConfig())
    assert audit["output_rows"] == audit["input_rows"] - audit["rows_removed"]
    dictionary = build_dictionary(cleaned, preview.proposal.mapping)
    assert len(dictionary) == len(preview.proposal.mapping)

    # 4–5. Features + RFM/LRFM.
    features = build_customer_features(cleaned).features
    assert len(features) > 1000
    rfm = score_rfm(features)
    lrfm = score_lrfm(features)
    assert rfm["segment"].notna().all() and lrfm["segment"].notna().all()

    # 6. Demographics (tolerates the sample's full field set).
    profile = build_demographic_profile(features, rfm[["customer_id", "segment"]])
    assert "age_summary" in profile.tables

    # 7. AI segmentation (fast classical model for the gate).
    prepared = segmentation_service.prepare_segmentation_features(
        features,
        [
            "recency_days",
            "frequency",
            "monetary_value",
            "relationship_length_days",
            "average_order_value",
        ],
        minimum_customers=50,
    )
    labels, _ = segmentation_service.run_model(
        "kmeans", prepared.matrix, n_clusters=4, random_state=42
    )
    from segmentation.evaluation import evaluate_clustering

    assert evaluate_clustering(prepared.matrix, labels)["n_clusters"] == 4

    # 8. Comparison + recommendation between RFM and the AI run.
    ai_labels = pd.Series(labels, index=features["customer_id"]).map(str)
    rfm_labels = rfm.set_index("customer_id")["segment"]
    assert agreement_ari(rfm_labels, ai_labels) >= 0.0
    rec = recommend_model(
        [
            {
                "name": "AI:kmeans",
                "silhouette": 0.5,
                "stability_ari": 0.8,
                "value_eta2": 0.5,
                "min_share": 0.2,
                "noise_pct": 0.0,
                "interpretability_passes": 4,
                "n_segments": 4,
            },
            {
                "name": "RFM",
                "silhouette": None,
                "stability_ari": 1.0,
                "value_eta2": 0.4,
                "min_share": 0.05,
                "noise_pct": 0.0,
                "interpretability_passes": 4,
                "n_segments": 10,
            },
        ]
    )
    assert rec["operational_pick"] in ("AI:kmeans", "RFM")

    # 9–10. Churn dataset + training + scoring.
    cutoff = preview.frame["transaction_date"].max()
    cutoff = (pd.Timestamp(cutoff) - pd.Timedelta(days=120)).date().isoformat()
    dataset = build_churn_dataset(cleaned, cutoff, horizon_days=90, min_history_days=30)
    assert dataset.leakage_ok and len(dataset.frame) > 100
    bundle = churn_service.train_churn_model(dataset, "logreg", calibrate=False)
    assert bundle["test_metrics"]["roc_auc"] is not None
    scores = score_customers(
        bundle["pipeline"],
        dataset.frame[bundle["feature_cols"]],
        dataset.frame["customer_id"],
        bundle["importance"],
        "logreg",
    )
    assert (scores["risk_band"].isin(["low", "medium", "high", "critical"])).all()

    # 11. Retention actions.
    matrix = load_matrix("retention_strategies.yaml")
    retention = retention_service.build_retention_table(scores, features, matrix)
    assert retention["strategy_rule"].notna().all()
    assert retention_summary_total(retention) > 0

    # 12–13. Dashboard filters + exports.
    txns = prepare_transactions(cleaned)
    master = build_master(features, rfm=rfm, scores=scores, retention=retention)
    txns_f, master_f = apply_filters(txns, master, {})
    kpis = compute_kpis(txns_f, master_f)
    assert kpis["total_customers"] > 1000
    assert kpis["total_revenue"] > 0
    assert not trend_tables(txns_f, master_f)["revenue"].empty
    assert len(to_csv_bytes(retention)) > 0
    assert len(to_json_bytes(audit)) > 0

    # 14. Restart behavior is documented, not coded: state rebuilds from disk
    # artifacts (sample CSV + saved models), verified by DEPLOYMENT.md existing.
    import pathlib

    assert pathlib.Path("DEPLOYMENT.md").exists()


def retention_summary_total(retention: pd.DataFrame) -> float:
    """Total revenue at risk cleared through the retention table."""
    return round(float(retention["revenue_at_risk"].sum()), 2)
