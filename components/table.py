"""Tables: formatted frames with conditional risk/value formatting.

Numbers are formatted through ``utils.formatting`` before display; risk cells
take their badge backgrounds from tokens so tables, badges, and charts agree.
With the optional ``jinja2`` package the Styler path keeps dataframe
interactivity; without it the standalone HTML table keeps the same values and
risk colors instead of crashing.
"""

from __future__ import annotations

import html
from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

from utils import formatting
from visualization import tokens

if TYPE_CHECKING:  # never imported at runtime: pandas hard-requires jinja2 here
    from pandas.io.formats.style import Styler

try:
    import jinja2  # noqa: F401

    _JINJA2_IMPORTED = True
except ImportError:
    _JINJA2_IMPORTED = False


def _jinja2_usable() -> bool:
    """True when the Styler backend is usable: jinja2>=3.1.5 per pandas.

    Presence alone is not enough — older installs pass the import and then
    fail at first Styler access, which is exactly the crash this guards.
    """
    if not _JINJA2_IMPORTED:
        return False
    try:
        from importlib.metadata import version

        major, minor, patch, *_ = (int(part) for part in version("jinja2").split("."))
    except (ImportError, ValueError):
        return False
    return (major, minor, patch) >= (3, 1, 5)


_HAS_JINJA2 = _jinja2_usable()


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


def html_table(df: pd.DataFrame, risk_cols: tuple[str, ...] = ()) -> str:
    """Standalone HTML table with inline risk-cell painting (pure, no jinja2).

    Used when the Styler backend is unavailable; keeps the conditional colors
    instead of silently dropping them. Values must already be display strings
    (see ``format_frame``) — everything is HTML-escaped here.
    """
    border = tokens.color("border")
    muted = tokens.color("muted")
    text = tokens.color("text")
    head = "".join(
        f"<th style='text-align:left;padding:6px 8px;color:{muted};font-size:0.78rem;"
        f"text-transform:uppercase;letter-spacing:0.04em;border-bottom:2px solid {border};'>"
        f"{html.escape(str(column))}</th>"
        for column in df.columns
    )
    rows = []
    for _, record in df.iterrows():
        cells = "".join(
            f"<td style='padding:6px 8px;border-top:1px solid {border};color:{text};"
            f"{_risk_cell_style(record[column]) if column in risk_cols else ''}'>"
            f"{html.escape(str(record[column]))}</td>"
            for column in df.columns
        )
        rows.append(f"<tr>{cells}</tr>")
    return (
        "<div style='overflow-x:auto;border:1px solid "
        f"{border};border-radius:12px;'>"
        "<table style='width:100%;border-collapse:collapse;font-size:0.85rem;"
        "font-variant-numeric:tabular-nums;'>"
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def style_table(
    df: pd.DataFrame,
    risk_cols: tuple[str, ...] = (),
    money_cols: tuple[str, ...] = (),
    pct_cols: tuple[str, ...] = (),
    int_cols: tuple[str, ...] = (),
) -> Styler | pd.DataFrame:
    """Format numerics and paint risk-band cells.

    Returns a Styler when ``jinja2`` is installed, otherwise the formatted
    frame itself (values formatted, painting skipped). ``st.dataframe``
    renders both, so callers never branch.
    """
    view = format_frame(df, money_cols, pct_cols, int_cols)
    if not _HAS_JINJA2:
        return view
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
    """Render a formatted, conditionally painted table full-width.

    With jinja2 the Styler path keeps dataframe interactivity; without it the
    standalone HTML table keeps the same values and risk colors.
    """
    if caption:
        st.caption(caption)
    if _HAS_JINJA2:
        st.dataframe(
            style_table(df, risk_cols, money_cols, pct_cols, int_cols),
            width="stretch",
        )
    else:
        st.markdown(
            html_table(format_frame(df, money_cols, pct_cols, int_cols), risk_cols),
            unsafe_allow_html=True,
        )
