"""Display formatting: one spelling for every number a page shows.

Values arrive preformatted at every card, table, and chart label, so ``₹``,
digit grouping, and percent scales cannot drift between pages. Pure functions —
no Streamlit, no session state.

Money has two notations, switchable at runtime through :func:`set_notation`:

* **Indian** (default) groups as ``12,34,567`` and abbreviates lakh/crore —
  ``₹1.5L``, ``₹5.83Cr``. This is how the amounts read to the operators the
  report is written for, and it is the default.
* **International** groups as ``1,234,567`` and abbreviates thousand/million/
  billion — ``₹1.5K``, ``₹5.8B``.

The active notation lives in one module-level variable rather than in session
state, because formatting is called from figure factories and export builders
that never see Streamlit. Pages call :func:`set_notation` once per run, so a
single switch re-spells every card, table, chart label, and export at once.
"""

from __future__ import annotations

import datetime as _datetime
from functools import lru_cache

import yaml

_FALLBACK_CURRENCY = "₹"

#: The two supported money notations. Indian is the default (see module docstring).
NOTATION_INDIAN = "indian"
NOTATION_INTERNATIONAL = "international"
NOTATIONS = (NOTATION_INDIAN, NOTATION_INTERNATIONAL)

#: Active notation for the current run; pages set it via ``set_notation``.
_notation = NOTATION_INDIAN


def set_notation(notation: str) -> str:
    """Set the money notation for this run; unknown values fall back to Indian."""
    global _notation
    _notation = notation if notation in NOTATIONS else NOTATION_INDIAN
    return _notation


def get_notation() -> str:
    """The active money notation."""
    return _notation


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


def _group_indian(digits: str) -> str:
    """Group digits the Indian way: last three, then pairs (1234567 → 12,34,567)."""
    if len(digits) <= 3:
        return digits
    head, tail = digits[:-3], digits[-3:]
    chunks: list[str] = []
    while len(head) > 2:
        chunks.insert(0, head[-2:])
        head = head[:-2]
    if head:
        chunks.insert(0, head)
    return ",".join([*chunks, tail])


def group_digits(value: float | int, decimals: int = 0) -> str:
    """Group a number's digits in the active notation, keeping its sign.

    ``1234567`` → ``1,234,567`` (international) or ``12,34,567`` (Indian). Rounds
    to ``decimals`` first so the grouping always matches what is displayed.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{abs(number):.{max(decimals, 0)}f}"
    whole, _, fraction = text.partition(".")
    grouped = _group_indian(whole) if _notation == NOTATION_INDIAN else f"{int(whole):,}"
    sign = "-" if number < 0 else ""
    return f"{sign}{grouped}.{fraction}" if fraction else f"{sign}{grouped}"


def _compact_steps() -> tuple[tuple[float, str], ...]:
    """Scale thresholds and suffixes for the active notation, largest first.

    Indian notation stops at crore rather than continuing to "1,000 Cr": past
    that point the crore figure itself stays readable and an invented unit would
    only confuse a reader.
    """
    if _notation == NOTATION_INDIAN:
        return ((10_000_000.0, "Cr"), (100_000.0, "L"), (1_000.0, "K"))
    return ((1_000_000_000.0, "B"), (1_000_000.0, "M"), (1_000.0, "K"))


def _scale_text(scaled: float, decimals: int = 1) -> str:
    """Trim a scaled amount to ``decimals`` places, dropping a trailing '.0'."""
    text = f"{scaled:.{max(decimals, 1)}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-") else "0"


def fmt_int(value: float | int | None, placeholder: str = "–") -> str:
    """Whole number, grouped in the active notation (1234567 → 12,34,567).

    Counts follow the money notation on purpose: a page that writes ₹12,34,567
    and 1,234,567 in the same table looks like two different systems.
    """
    if value is None:
        return placeholder
    try:
        return group_digits(int(value))
    except (TypeError, ValueError):
        return placeholder


def fmt_money(value: float | int | None, decimals: int = 0, placeholder: str = "–") -> str:
    """Money with the configured currency symbol and notation-aware grouping.

    ``1234567`` → ``₹12,34,567`` (Indian, default) or ``₹1,234,567``
    (international).
    """
    if value is None:
        return placeholder
    try:
        number = float(value)
    except (TypeError, ValueError):
        return placeholder
    return f"{currency_symbol()}{group_digits(number, decimals)}"


def fmt_money_compact(value: float | int | None, decimals: int = 1, placeholder: str = "–") -> str:
    """Abbreviated money at a compact scale, in the active notation.

    Indian: ``₹1.5L`` (lakh), ``₹5.83Cr`` (crore). International: ``₹1.5K``,
    ``₹5.8B``. Use it wherever a cramped slot would otherwise show a full
    nine-digit amount next to a short one, so money reads at one scale on a page.

    ``decimals`` controls precision for the crore/billion step, which is the one
    that carries the biggest number and so needs the most digits to stay honest.
    """
    if value is None:
        return placeholder
    try:
        number = float(value)
    except (TypeError, ValueError):
        return placeholder
    steps = _compact_steps()
    for index, (threshold, suffix) in enumerate(steps):
        if abs(number) >= threshold:
            # The largest step keeps extra precision: ₹5.8Cr and ₹5.83Cr read
            # very differently, and the top step is the one executives quote.
            places = decimals + 1 if index == 0 else decimals
            scaled = _scale_text(round(number / threshold, places), places)
            return f"{currency_symbol()}{scaled}{suffix}"
    return fmt_money(number, placeholder=placeholder)


def fmt_pct(fraction: float | int | None, decimals: int = 1, placeholder: str = "–") -> str:
    """Fraction rendered as percent (0.123 → '12.3%').

    The integer part groups in the active notation, so a percent stays consistent
    with the money and counts beside it.
    """
    if fraction is None:
        return placeholder
    try:
        return f"{group_digits(100 * float(fraction), decimals)}%"
    except (TypeError, ValueError):
        return placeholder


def fmt_pct100(value: float | int | None, decimals: int = 1, placeholder: str = "–") -> str:
    """Percent already on a 0–100 scale (12.3 → '12.3%').

    The counterpart to :func:`fmt_pct` for the scores and shares that services
    compute pre-scaled, so pages never write ``f"{value}%"`` and drift on
    decimals between tabs.
    """
    if value is None:
        return placeholder
    try:
        return f"{group_digits(float(value), decimals)}%"
    except (TypeError, ValueError):
        return placeholder


def fmt_float(value: float | int | None, decimals: int = 2, placeholder: str = "–") -> str:
    """Fixed-decimal number for metric cards and score tables (0.8231 → '0.82').

    Carries the same thousands grouping in the integer part as the other
    helpers, so a metric never looks like it came from another app.
    """
    if value is None:
        return placeholder
    try:
        number = float(value)
    except (TypeError, ValueError):
        return placeholder
    return group_digits(number, max(decimals, 0))


def fmt_compact(value: float | int | None, placeholder: str = "–") -> str:
    """Short scale for cramped card slots, following the active notation.

    ``1500`` → ``1.5K``; ``2300000`` → ``23L`` (Indian) or ``2.3M``
    (international).
    """
    if value is None:
        return placeholder
    try:
        number = float(value)
    except (TypeError, ValueError):
        return placeholder
    steps = _compact_steps()
    for index, (threshold, suffix) in enumerate(steps):
        if abs(number) >= threshold:
            places = 1 + (1 if index == 0 else 0)
            return f"{_scale_text(round(number / threshold, places), places)}{suffix}"
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
