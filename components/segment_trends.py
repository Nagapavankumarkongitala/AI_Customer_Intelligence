"""Shared segment-trend compositions: line+Sankey and a compact Sankey pair.

The dashboard and page 03 show the same "Customer segments over time" story.
Keeping the two-column compositions here means both render the same heights,
empty states, and shared guidance text, and neither page has to know how a
stored history dict is shaped.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from components.chart_export import plotly_chart
from visualization.rfm_charts import segment_transition_sankey
from visualization.theme import apply_chart_theme, without_title

TRANSITION_CAPTION = (
    "Transitions count customers present at both consecutive period ends, by the "
    "segment they held at each. Customers who joined or left between snapshots "
    "are not flows."
)

DEFAULT_EMPTY_MESSAGE = (
    "No transition data yet. Rebuild segment trends to compare consecutive periods."
)

#: Two compact Sankeys side by side need less height than a full-width one;
#: this keeps eight quarter columns legible without dominating the overview.
COMPACT_TRANSITION_HEIGHT = 340


def transition_frame(hist_state: dict | None) -> pd.DataFrame:
    """Transitions stored beside the segment-history table (empty frame if absent)."""
    rows = (hist_state or {}).get("transitions")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def _transition_figure(flows: pd.DataFrame, height: int) -> Any:
    """One compact, title-free transition Sankey."""
    fig = apply_chart_theme(segment_transition_sankey(flows), height=height, compact=True)
    fig = without_title(fig, compact=True)
    fig.update_layout(height=height, margin={"l": 8, "r": 8, "t": 20, "b": 58})
    return fig


def render_sankey_pair(
    left_transitions: pd.DataFrame,
    right_transitions: pd.DataFrame,
    *,
    left_title: str,
    right_title: str,
    key_prefix: str,
    height: int = COMPACT_TRANSITION_HEIGHT,
    left_empty_message: str = DEFAULT_EMPTY_MESSAGE,
    right_empty_message: str = DEFAULT_EMPTY_MESSAGE,
) -> None:
    """Draw two compact transition Sankeys side by side.

    The overview uses this for the saved AI labeling on the left and the
    re-scored RFM moves on the right. Each side degrades to its own guidance
    when no flows exist, so one missing source never removes the other.
    """
    left, right = st.columns(2)
    with left:
        st.markdown(f"**{left_title}**")
        if left_transitions.empty:
            st.info(left_empty_message)
        else:
            plotly_chart(
                _transition_figure(left_transitions, height),
                title=left_title,
                width="stretch",
                key=f"{key_prefix}_left",
            )
    with right:
        st.markdown(f"**{right_title}**")
        if right_transitions.empty:
            st.info(right_empty_message)
        else:
            plotly_chart(
                _transition_figure(right_transitions, height),
                title=right_title,
                width="stretch",
                key=f"{key_prefix}_right",
            )


def render_trend_pair(
    left_fig: Any,
    transitions: pd.DataFrame,
    *,
    model: str | None,
    key_prefix: str,
    left_title: str = "Customer segment trend",
    right_title: str = "Segment transitions",
    height: int = 460,
    empty_message: str = DEFAULT_EMPTY_MESSAGE,
) -> None:
    """Draw a trend figure and its model's transition Sankey in one row.

    ``left_fig`` is an already-themed factory figure; the caller decides which
    trend view it shows. ``left_title`` / ``right_title`` name the two PNG
    exports, since these charts have no HTML section heading of their own. The
    right figure is skipped with guidance when the history predates transition
    capture, when no customers were present at both ends of a pair, or when the
    selected source has no re-scored transitions (``model=None``, used for fixed
    AI labelings).
    """
    flows = transitions
    if model is None:
        flows = transitions.iloc[0:0]
    elif "model" in transitions.columns:
        flows = transitions[transitions["model"] == model]
    left, right = st.columns(2)
    with left:
        left_fig.update_layout(height=height)
        plotly_chart(left_fig, title=left_title, width="stretch", key=f"{key_prefix}_trend")
    with right:
        if flows.empty:
            st.info(empty_message)
        else:
            fig = without_title(segment_transition_sankey(flows))
            fig.update_layout(height=height, margin={"b": 64})
            plotly_chart(fig, title=right_title, width="stretch", key=f"{key_prefix}_sankey")
    st.caption(TRANSITION_CAPTION)
