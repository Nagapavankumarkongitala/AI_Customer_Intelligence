"""Tables: formatted frames with conditional risk/value formatting.

Numbers are formatted through ``utils.formatting`` before display; risk cells
take their badge backgrounds from tokens so tables, badges, and charts agree.
Built on ``Styler.map`` (pandas 3 removed ``applymap``) and rendered with
``st.dataframe``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import formatting
from visualization import tokens


def format_frame(
    df: pd.DataFrame,
    money_cols: tuple[str, ...] = (),
    pct_cols: tuple[str, ...] = (),
    int_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Return a display copy with numeric columns rendered as strings (pure)."""
    if df.empty:
        return df.copy()
    view = df.copy()
    for column in money_cols:
        if column in view.columns:
            view[column] = [formatting.fmt_money(v) for v in view[column]]
    for column in pct_cols:
        if column in view.columns:
            view[column] = [formatting.fmt_pct(v) for v in view[column]]
    for column in int_cols:
        if column in view.columns:
            view[column] = [formatting.fmt_int(v) for v in view[column]]
    return view


def _risk_cell_style(value: object) -> str:
    band = str(value).strip().lower()
    pairs = tokens.RISK_BADGE[tokens.active_mode()]
    if band not in pairs:
        return ""
    bg, fg = pairs[band]
    return f"background-color: {bg}; color: {fg}; font-weight: 600;"


def style_table(
    df: pd.DataFrame,
    risk_cols: tuple[str, ...] = (),
    money_cols: tuple[str, ...] = (),
    pct_cols: tuple[str, ...] = (),
    int_cols: tuple[str, ...] = (),
) -> pd.io.formats.style.Styler:
    """Format numerics and paint risk-band cells; returns a Styler (pure)."""
    view = format_frame(df, money_cols, pct_cols, int_cols)
    styler = view.style.map(_risk_cell_style, subset=[c for c in risk_cols if c in view.columns])
    return styler.set_table_styles(
        [{"selector": "td, th", "props": "font-variant-numeric: tabular-nums;"}]
    )


def render_table(
    df: pd.DataFrame,
    risk_cols: tuple[str, ...] = (),
    money_cols: tuple[str, ...] = (),
    pct_cols: tuple[str, ...] = (),
    int_cols: tuple[str, ...] = (),
    caption: str | None = None,
) -> None:
    """Render a formatted, conditionally painted table full-width."""
    if caption:
        st.caption(caption)
    st.dataframe(
        style_table(df, risk_cols, money_cols, pct_cols, int_cols),
        use_container_width=True,
    )
