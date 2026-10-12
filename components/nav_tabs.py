"""Top tab bar (web-style navigation) shared by the landing page and all 9 pages.

Replaces the Streamlit sidebar nav (hidden via theme CSS). Tabs call
st.switch_page, so per-page URLs and deep links keep working. The `active`
argument highlights the current view; pass the page's own path.
"""

import streamlit as st

NAV_PAGES: list[tuple[str, str]] = [
    ("Home", "app.py"),
    ("01 Upload", "pages/01_Data_Upload_and_Mapping.py"),
    ("02 Quality", "pages/02_Data_Quality_and_Cleaning.py"),
    ("03 RFM", "pages/03_RFM_LRFM_Analysis.py"),
    ("04 Demographics", "pages/04_Demographic_Profiling.py"),
    ("05 Segments", "pages/05_Advanced_Segmentation.py"),
    ("06 Compare", "pages/06_Segmentation_Comparison.py"),
    ("07 Churn", "pages/07_Churn_Prediction.py"),
    ("08 Retention", "pages/08_Retention_Strategies.py"),
    ("09 Dashboard", "pages/09_Executive_Dashboard.py"),
]


def render_nav_tabs(active: str) -> None:
    """Render the tab row; the active view renders highlighted and non-clickable."""
    cols = st.columns(len(NAV_PAGES))
    for col, (label, path) in zip(cols, NAV_PAGES, strict=True):
        with col:
            if path == active:
                st.markdown(
                    f"<div class='nav-tab nav-active'>{label}</div>",
                    unsafe_allow_html=True,
                )
            elif st.button(label, key=f"nav_{label}", use_container_width=True):
                st.switch_page(path)
