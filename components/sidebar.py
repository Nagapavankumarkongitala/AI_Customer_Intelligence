"""Persistent left sidebar: navigation, page slicers, appearance, data status.

The single nav source for the app — page order and labels come from
``components.nav_tabs.NAV_PAGES`` so the sidebar can never disagree with page
routing. Views compose the rail in order:

1. ``render_sidebar_nav`` first (theme, title, page links, appearance toggle),
2. ``sidebar_filters`` sections for the page's view slicers,
3. ``render_sidebar_footer`` last (data status).

Compute-driving configuration stays in-page next to the analysis it drives;
only view slicers move to the rail. Pure builders stay testable without a
session.
"""

from __future__ import annotations

import html
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

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


def render_sidebar_nav(active: str) -> None:
    """Rail head: stored theme, title, page links, appearance toggle.

    ``active`` is kept for call-site stability; link highlighting is automatic.
    """
    ensure_theme()
    with st.sidebar:
        st.markdown("**AI Customer Intelligence**")
        for label, path in NAV_PAGES:
            st.page_link(path, label=label)
        st.divider()
        theme_toggle()
        st.caption("Dark is the heritage console skin; native controls stay light.")


@contextmanager
def sidebar_filters(title: str = "Filters", expanded: bool = False) -> Iterator[None]:
    """Collapsed rail section holding a page's view slicers."""
    with st.sidebar:
        with st.expander(title, expanded=expanded):
            yield


def render_sidebar_footer() -> None:
    """Rail foot: data-status block. Called last so slicers sit above it."""
    with st.sidebar:
        st.divider()
        st.markdown(sidebar_status_html(collect_status(st.session_state)), unsafe_allow_html=True)
