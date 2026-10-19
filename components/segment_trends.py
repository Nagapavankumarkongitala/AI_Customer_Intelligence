"""Shared trend pair: segment lines beside a customer-transition Sankey.

The dashboard and page 03 show the same "Customer segments over time" story.
Keeping the two-column composition here means both render the same heights,
empty states, and honesty caption, and neither page has to know how a stored
history dict is shaped.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from visualization.rfm_charts import segment_transition_sankey
from visualization.theme import without_title

TRANSITION_CAPTION = (
    "Transitions count customers present at both consecutive period ends, by the "
    "segment they held at each. Customers who joined or left between snapshots "
    "are not flows."
)

DEFAULT_EMPTY_MESSAGE = (
    "No transition data yet. Rebuild segment trends to compare consecutive periods."
)


def transition_frame(hist_state: dict | None) -> pd.DataFrame:
    """Transitions stored beside the segment-history table (empty frame if absent)."""
    rows = (hist_state or {}).get("transitions")
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def render_trend_pair(
    left_fig: Any,
    transitions: pd.DataFrame,
    *,
    model: str | None,
    key_prefix: str,
    height: int = 460,
    empty_message: str = DEFAULT_EMPTY_MESSAGE,
) -> None:
    """Draw a trend figure and its model's transition Sankey in one row.

    ``left_fig`` is an already-themed factory figure; the caller decides which
    trend view it shows. The right figure is skipped with guidance when the
    history predates transition capture, when no customers were present at both
    ends of a pair, or when the selected source has no re-scored transitions
    (``model=None``, used for fixed AI labelings).
    """
    flows = transitions
    if model is None:
        flows = transitions.iloc[0:0]
    elif "model" in transitions.columns:
        flows = transitions[transitions["model"] == model]
    left, right = st.columns(2)
    with left:
        left_fig.update_layout(height=height)
        st.plotly_chart(left_fig, width="stretch", key=f"{key_prefix}_trend")
    with right:
        if flows.empty:
            st.info(empty_message)
        else:
            fig = without_title(segment_transition_sankey(flows))
            fig.update_layout(height=height, margin={"b": 64})
            st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_sankey")
    st.caption(TRANSITION_CAPTION)
