"""Persistent left sidebar: navigation, appearance, and data status.

The single nav source for the app — page order and labels come from
``components.nav_tabs.NAV_PAGES`` so the sidebar can never disagree with page
routing. ``render_sidebar`` applies the stored theme first (so the toggle takes
effect on the rerun it triggers), then renders nav links, the appearance radio,
and a compact data-status block. Pure builders stay testable without a session.
"""

from __future__ import annotations

import html
from collections.abc import Mapping

import pandas as pd
import streamlit as st

from components.nav_tabs import NAV_PAGES
from components.theme import ensure_theme, theme_toggle
from visualization import tokens


def sidebar_status_html(status: Mapping[str, str]) -> str:
    """Compact data-status block: one muted line per fact (pure)."""
    rows = "".join(
        "<div style='display:flex;justify-content:space-between;gap:0.5rem;"
        "font-size:0.78rem;padding:0.1rem 0;'>"
        f"<span style='color:{tokens.color('muted')};'>{html.escape(str(label))}</span>"
        f"<span style='font-variant-numeric:tabular-nums;'>{html.escape(str(value))}</span>"
        "</div>"
        for label, value in status.items()
    )
    return rows


def collect_status(state: Mapping) -> dict[str, str]:
    """Curate the session facts worth one glance in the sidebar (pure)."""

    def _rows(value: object) -> str:
        if isinstance(value, pd.DataFrame) and not value.empty:
            return f"{len(value):,} rows"
        return "–"

    return {
        "Transactions": _rows(state.get("raw_data")),
        "Customers": _rows(state.get("customer_features")),
        "Segments": (
            str(state.get("selected_segmentation_model") or "–")
            if state.get("segmented_customers") is not None
            else "–"
        ),
        "Churn model": str((state.get("churn_model_metadata") or {}).get("model", "–")),
    }


def render_sidebar(active: str) -> None:
    """Render the full sidebar. ``active`` is kept for call-site stability."""
    ensure_theme()
    with st.sidebar:
        st.markdown("**AI Customer Intelligence**")
        for label, path in NAV_PAGES:
            st.page_link(path, label=label)
        st.divider()
        theme_toggle()
        st.caption("Dark is the heritage console skin; native controls stay light.")
        st.divider()
        st.markdown(sidebar_status_html(collect_status(st.session_state)), unsafe_allow_html=True)
