"""Display formatting: one spelling for every number a page shows.

Values arrive preformatted at every card, table, and chart label, so ``₹``,
thousands separators, and percent scales cannot drift between pages. Pure
functions — no Streamlit, no session state.
"""

from __future__ import annotations

import datetime as _datetime
from functools import lru_cache

import yaml

_FALLBACK_CURRENCY = "₹"


@lru_cache(maxsize=1)
def currency_symbol(path: str = "config.yaml") -> str:
    """Currency symbol from config, cached for the process lifetime.

    Defensive on purpose: formatting must never crash a page over config, so
    anything unreadable falls back to the configured default symbol.
    """
    try:
        cfg = yaml.safe_load(open(path, encoding="utf-8"))
        symbol = (cfg.get("app") or {}).get("currency_symbol")
    except (OSError, ValueError, AttributeError):
        return _FALLBACK_CURRENCY
    return str(symbol) if symbol else _FALLBACK_CURRENCY


def fmt_int(value: float | int | None, placeholder: str = "–") -> str:
    """Whole number with thousands separators (1,234,567)."""
    if value is None:
        return placeholder
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return placeholder


def fmt_money(value: float | int | None, decimals: int = 0, placeholder: str = "–") -> str:
    """Money with the configured currency symbol and thousands grouping."""
    if value is None:
        return placeholder
    try:
        return f"{currency_symbol()}{float(value):,.{decimals}f}"
    except (TypeError, ValueError):
        return placeholder


def fmt_pct(fraction: float | int | None, decimals: int = 1, placeholder: str = "–") -> str:
    """Fraction rendered as percent (0.123 → '12.3%')."""
    if fraction is None:
        return placeholder
    try:
        return f"{100 * float(fraction):,.{decimals}f}%"
    except (TypeError, ValueError):
        return placeholder


def fmt_compact(value: float | int | None, placeholder: str = "–") -> str:
    """Short scale for cramped card slots (1,500 → '1.5K', 2,300,000 → '2.3M')."""
    if value is None:
        return placeholder
    try:
        number = float(value)
    except (TypeError, ValueError):
        return placeholder
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs(number) >= threshold:
            scaled = number / threshold
            text = f"{scaled:.1f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return fmt_int(number, placeholder)


def fmt_date(
    value: _datetime.date | _datetime.datetime | str | None, placeholder: str = "–"
) -> str:
    """ISO date for headers and captions (2026-03-14)."""
    if value is None:
        return placeholder
    if isinstance(value, (_datetime.datetime, _datetime.date)):
        return value.strftime("%Y-%m-%d")
    try:
        return _datetime.date.fromisoformat(str(value)[:10]).strftime("%Y-%m-%d")
    except ValueError:
        return str(value)
