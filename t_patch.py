from services import feature_service
from tests.test_customer_features import _history_txns
import pandas as pd

orig = feature_service.period_end_dates


def patched(max_date, granularity, periods):
    res = orig(max_date, granularity, periods)
    if granularity == "monthly":
        max_ts = pd.Timestamp(max_date)
        if max_ts == pd.Timestamp("2024-06-15"):
            res = sorted(set(list(res) + [max_ts]))
    return res


feature_service.period_end_dates = patched
history, trans, labels = feature_service.run_segment_trends_with_labels(_history_txns(), granularity="monthly", quantiles=3)
print(labels["customer_id"].nunique())
feature_service.period_end_dates = orig
