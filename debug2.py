from services.feature_service import run_segment_trends_with_labels, run_segment_history
from tests.test_customer_features import _history_txns
history, transitions, labels = run_segment_trends_with_labels(_history_txns(), granularity='monthly', quantiles=3)
print('periods:', list(history['period'].unique()))
print(history.groupby('period')['customers'].sum().to_dict())
PY
