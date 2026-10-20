from services.feature_service import run_segment_history, run_segment_trends_with_labels
from tests.test_customer_features import _history_txns
history = run_segment_history(_history_txns(), granularity='monthly', quantiles=3)
print(list(history['period'].unique()))
print(history.groupby('period')['customers'].sum().to_dict())
PY
