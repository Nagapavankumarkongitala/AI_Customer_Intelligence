"""Sidebar tests: theme resolution, status curation, and nav wiring."""

from __future__ import annotations

import pandas as pd

from components.sidebar import collect_status, sidebar_status_html
from components.theme import THEME_KEY, THEME_OPTIONS, resolve_theme_mode
from visualization import tokens


def test_theme_options_default_to_light() -> None:
    """No stored preference (or garbage) resolves to the light default."""
    assert THEME_OPTIONS[0] == "Light"
    assert resolve_theme_mode({}) == tokens.MODE_LIGHT
    assert resolve_theme_mode({THEME_KEY: "nonsense"}) == tokens.MODE_LIGHT
    assert resolve_theme_mode({THEME_KEY: None}) == tokens.MODE_LIGHT


def test_theme_resolution_accepts_labels_and_modes() -> None:
    """Radio labels and raw mode strings both resolve."""
    assert resolve_theme_mode({THEME_KEY: "Light"}) == tokens.MODE_LIGHT
    assert resolve_theme_mode({THEME_KEY: "Dark"}) == tokens.MODE_DARK
    assert resolve_theme_mode({THEME_KEY: "light"}) == tokens.MODE_LIGHT
    assert resolve_theme_mode({THEME_KEY: "dark"}) == tokens.MODE_DARK


def test_collect_status_curates_session_facts() -> None:
    """Empty state degrades to dashes; loaded frames report row counts."""
    assert collect_status({}) == {
        "Transactions": "–",
        "Customers": "–",
        "Segments": "–",
        "Churn model": "–",
    }
    state = {
        "raw_data": pd.DataFrame({"a": [1, 2, 3]}),
        "customer_features": pd.DataFrame({"a": range(50)}),
        "segmented_customers": pd.DataFrame({"a": [1]}),
        "selected_segmentation_model": "kmeans",
        "churn_model_metadata": {"model": "random_forest"},
    }
    status = collect_status(state)
    assert status["Transactions"] == "3 rows"
    assert status["Customers"] == "50 rows"
    assert status["Segments"] == "kmeans"
    assert status["Churn model"] == "random_forest"


def test_sidebar_status_html_escapes() -> None:
    """Status lines render label/value pairs with content escaped."""
    out = sidebar_status_html({"A<B": "1 & 2"})
    assert "A&lt;B" in out and "1 &amp; 2" in out


def test_theme_preference_survives_outside_the_contract() -> None:
    """The preference key is session UI state, not pipeline state."""
    from services.state_service import _DEFAULTS

    assert THEME_KEY not in _DEFAULTS
