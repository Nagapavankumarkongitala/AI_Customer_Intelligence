"""Navigation source of truth: ordered (label, path) pairs for every view.

``NAV_PAGES`` is the single nav definition — the sidebar renders it and the
order test pins it to the ``pages/`` directory.
"""

from __future__ import annotations

NAV_PAGES: list[tuple[str, str]] = [
    ("Home", "app.py"),
    ("Upload", "pages/01_Data_Upload_and_Mapping.py"),
    ("Quality", "pages/02_Data_Quality_and_Cleaning.py"),
    ("RFM", "pages/03_RFM_LRFM_Analysis.py"),
    ("Demographics", "pages/04_Demographic_Profiling.py"),
    ("Segments", "pages/05_Advanced_Segmentation.py"),
    ("Compare", "pages/06_Segmentation_Comparison.py"),
    ("Churn", "pages/07_Churn_Prediction.py"),
    ("Retention", "pages/08_Retention_Strategies.py"),
    ("Dashboard", "pages/09_Executive_Dashboard.py"),
]
