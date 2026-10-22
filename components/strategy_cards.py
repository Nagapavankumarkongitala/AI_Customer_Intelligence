"""Strategy playbook cards: one card per retention rule with its live footprint.

The strategy matrix (``retention_strategies.yaml``) is the business owner's
vocabulary; these cards render it back as an at-a-glance playbook, joined to the
retention table so each rule shows how many customers it currently owns and how
much revenue sits at risk. Rendering reuses the shared card chrome (tokens) and
``utils.formatting`` so numbers never drift between pages.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from retention_engine.offer_recommender import describe_offer
from retention_engine.strategy_mapper import StrategyMatrix
from utils import formatting
from visualization import tokens

#: Cards per row: three keeps the value column readable on wide screens.
DEFAULT_COLUMNS = 3


@dataclass(frozen=True)
class StrategyCard:
    """One strategy card: what it is, when it fires, and its live footprint."""

    name: str
    action: str
    value_tiers: tuple[str, ...]
    risk_bands: tuple[str, ...]
    channel: str
    offer_type: str
    reason: str
    customers: int
    revenue_at_risk: float


def _render_reason(
    template: str, *, value: float | None, bands: tuple[str, ...], probability: float | None
) -> str:
    """Fill a rule's reason template with the rule's aggregate stats.

    Placeholders are replaced by string substitution (not ``str.format``) so a
    matrix that adds a placeholder cannot crash the page; unknown ones are left
    for the caller to see.
    """
    mapping = {
        "value": formatting.fmt_money(value) if value is not None else "—",
        "band": "/".join(bands) or "band",
        "probability": formatting.fmt_float(probability, 3) if probability is not None else "—",
    }
    rendered = template
    for key, text in mapping.items():
        rendered = rendered.replace("{" + key + "}", text)
    return rendered


def _group_stats(table: pd.DataFrame, key_col: str, value_col: str | None) -> dict[str, dict]:
    """Per-rule footprint: customer count, revenue at risk, mean value/probability."""
    grouped: dict[str, dict] = {}
    for key, segment in table.groupby(key_col):
        at_risk = (
            pd.to_numeric(segment["revenue_at_risk"], errors="coerce").fillna(0.0).sum()
            if "revenue_at_risk" in segment.columns
            else 0.0
        )
        probability = (
            pd.to_numeric(segment["churn_probability"], errors="coerce").mean()
            if "churn_probability" in segment.columns
            else None
        )
        value = (
            pd.to_numeric(segment[value_col], errors="coerce").mean()
            if value_col and value_col in segment.columns
            else None
        )
        grouped[str(key)] = {
            "customers": int(len(segment)),
            "revenue_at_risk": float(at_risk),
            "probability": None if pd.isna(probability) else float(probability),
            "value": None if pd.isna(value) else float(value),
        }
    return grouped


def build_strategy_cards(table: pd.DataFrame, matrix: StrategyMatrix) -> list[StrategyCard]:
    """Aggregate the retention table into one card per rule, in rule order.

    Rules with no matching customers still get a card (zero footprint) so the
    playbook always shows the full strategy vocabulary.
    """
    if table is None or table.empty:
        return []
    key_col = "strategy_rule" if "strategy_rule" in table.columns else "recommended_action"
    value_col = (
        "expected_value_proxy_historical"
        if "expected_value_proxy_historical" in table.columns
        else "monetary_value" if "monetary_value" in table.columns else None
    )
    grouped = _group_stats(table, key_col, value_col)
    cards: list[StrategyCard] = []
    for rule in matrix.rules:
        stats = grouped.get(rule.name) or grouped.get(rule.action) or {}
        cards.append(
            StrategyCard(
                name=rule.name,
                action=rule.action,
                value_tiers=tuple(rule.value_tiers),
                risk_bands=tuple(rule.risk_bands),
                channel=rule.channel,
                offer_type=rule.offer_type,
                reason=_render_reason(
                    rule.reason,
                    value=stats.get("value"),
                    bands=tuple(rule.risk_bands),
                    probability=stats.get("probability"),
                ),
                customers=int(stats.get("customers", 0)),
                revenue_at_risk=float(stats.get("revenue_at_risk", 0.0)),
            )
        )
    return cards


def strategy_card_html(card: StrategyCard) -> str:
    """Card markup: rule identity, trigger, offer, reason, and live footprint."""
    accent = tokens.action_color(card.action)
    text = tokens.color("text")
    muted = tokens.color("muted")
    border = tokens.color("border")
    surface = tokens.color("surface")
    tiers = " / ".join(card.value_tiers)
    bands = " / ".join(card.risk_bands)
    return (
        "<div style='"
        f"background:{surface};border:1px solid {border};border-top:3px solid {accent};"
        f"border-radius:{tokens.RADIUS['lg']};box-shadow:{tokens.shadow('card')};"
        "padding:0.85rem 0.95rem;display:flex;flex-direction:column;gap:0.4rem;height:100%;'>"
        f"<div style='font-weight:700;font-size:0.98rem;color:{text};overflow-wrap:break-word;'>"
        f"{html.escape(card.action)}</div>"
        f"<div style='font-size:0.70rem;font-weight:600;letter-spacing:0.03em;"
        f"text-transform:uppercase;color:{muted};'>"
        f"{html.escape(tiers)} value · {html.escape(bands)} risk</div>"
        f"<div style='font-size:0.82rem;color:{muted};overflow-wrap:break-word;'>"
        f"<strong style='color:{text};'>Channel</strong> {html.escape(card.channel)}"
        f" &nbsp;·&nbsp; <strong style='color:{text};'>Offer</strong> {html.escape(card.offer_type)}</div>"
        f"<div style='font-size:0.78rem;color:{muted};border-top:1px solid {border};"
        f"padding-top:0.4rem;overflow-wrap:break-word;'>{html.escape(card.reason)}</div>"
        f"<div style='font-size:0.72rem;color:{muted};overflow-wrap:break-word;'>"
        f"{html.escape(describe_offer(card.offer_type))}</div>"
        "<div style='margin-top:auto;display:flex;justify-content:space-between;gap:0.5rem;"
        f"border-top:1px solid {border};padding-top:0.45rem;'>"
        "<div><div style='font-size:0.66rem;text-transform:uppercase;letter-spacing:0.03em;"
        f"color:{muted};'>Customers</div>"
        f"<div style='font-size:1.12rem;font-weight:700;color:{text};"
        "font-variant-numeric:tabular-nums;'>"
        f"{html.escape(formatting.fmt_int(card.customers))}</div></div>"
        "<div style='text-align:right;'><div style='font-size:0.66rem;text-transform:uppercase;"
        f"letter-spacing:0.03em;color:{muted};'>At risk</div>"
        f"<div style='font-size:1.12rem;font-weight:700;color:{accent};"
        "font-variant-numeric:tabular-nums;'>"
        f"{html.escape(formatting.fmt_compact(card.revenue_at_risk))}</div></div>"
        "</div></div>"
    )


def render_strategy_playbook(
    table: pd.DataFrame,
    matrix: StrategyMatrix,
    width: int = DEFAULT_COLUMNS,
) -> None:
    """Render the strategy playbook as rows of cards (Streamlit sink)."""
    cards = build_strategy_cards(table, matrix)
    if not cards:
        return
    for start in range(0, len(cards), width):
        row = cards[start : start + width]
        columns = st.columns(len(row))
        for column, card in zip(columns, row, strict=True):
            with column:
                st.markdown(strategy_card_html(card), unsafe_allow_html=True)
