"""Money and number formatting: the two notations, grouping, and the toggle."""

from __future__ import annotations

import pytest

from utils import formatting
from utils.formatting import (
    NOTATION_INDIAN,
    NOTATION_INTERNATIONAL,
    group_digits,
    set_notation,
)


@pytest.fixture(autouse=True)
def _restore_notation():
    """Notation is module-level state; never leak one test's choice into the next."""
    original = formatting.get_notation()
    yield
    formatting.set_notation(original)


def test_indian_is_the_default() -> None:
    """A fresh process formats in lakh/crore, not K/M/B."""
    assert formatting.get_notation() == NOTATION_INDIAN


def test_indian_grouping_places_commas_by_scale() -> None:
    """Indian grouping: last three digits, then pairs — 12,34,567 not 1,234,567."""
    set_notation(NOTATION_INDIAN)
    assert group_digits(999) == "999"
    assert group_digits(1234) == "1,234"
    assert group_digits(100000) == "1,00,000"
    assert group_digits(1234567) == "12,34,567"
    assert group_digits(5830732) == "58,30,732"
    assert group_digits(-1234567) == "-12,34,567"


def test_international_grouping_stays_thousands() -> None:
    """The international notation groups in threes, sign preserved."""
    set_notation(NOTATION_INTERNATIONAL)
    assert group_digits(1234567) == "1,234,567"
    assert group_digits(-4500) == "-4,500"


def test_compact_money_switches_scale_and_suffix() -> None:
    """The same amount reads lakh/crore or thousand/million/billion."""
    set_notation(NOTATION_INDIAN)
    assert formatting.fmt_money_compact(1_500) == "₹1.5K"
    assert formatting.fmt_money_compact(150_000) == "₹1.5L"
    assert formatting.fmt_money_compact(2_300_000) == "₹23L"
    # The top step keeps an extra decimal: ₹5.83Cr is a different claim to ₹5.8Cr.
    assert formatting.fmt_money_compact(5_830_732_765) == "₹583.07Cr"

    set_notation(NOTATION_INTERNATIONAL)
    assert formatting.fmt_money_compact(1_500) == "₹1.5K"
    assert formatting.fmt_money_compact(2_300_000) == "₹2.3M"
    assert formatting.fmt_money_compact(5_830_732_765) == "₹5.83B"


def test_compact_money_keeps_a_small_amount_unscaled() -> None:
    """Below the smallest step the value falls back to full money."""
    set_notation(NOTATION_INDIAN)
    assert formatting.fmt_money_compact(999) == "₹999"
    assert formatting.fmt_money_compact(0) == "₹0"


def test_fmt_money_uses_the_active_grouping() -> None:
    """Full money carries the symbol and the notation's digit grouping."""
    set_notation(NOTATION_INDIAN)
    assert formatting.fmt_money(1_234_567) == "₹12,34,567"
    set_notation(NOTATION_INTERNATIONAL)
    assert formatting.fmt_money(1_234_567) == "₹1,234,567"


def test_fmt_int_follows_the_money_notation() -> None:
    """Counts group like money, so one table never shows two number systems."""
    set_notation(NOTATION_INDIAN)
    assert formatting.fmt_int(39903) == "39,903"
    assert formatting.fmt_int(1234567) == "12,34,567"
    set_notation(NOTATION_INTERNATIONAL)
    assert formatting.fmt_int(1234567) == "1,234,567"


def test_unknown_notation_falls_back_to_indian() -> None:
    """Garbage never leaves the module in an unformattable state."""
    assert set_notation("martian") == NOTATION_INDIAN
    assert set_notation("") == NOTATION_INDIAN
    assert formatting.get_notation() == NOTATION_INDIAN


def test_fmt_pct100_treats_input_as_already_scaled() -> None:
    """fmt_pct100 is for 0-100 scores; fmt_pct is for 0-1 fractions."""
    assert formatting.fmt_pct100(51.04) == "51.0%"
    assert formatting.fmt_pct100(51.04, 2) == "51.04%"
    assert formatting.fmt_pct(0.5104) == "51.0%"


def test_fmt_float_groups_integer_part() -> None:
    """Decimals group like the rest of the app and honour the precision asked for."""
    set_notation(NOTATION_INDIAN)
    assert formatting.fmt_float(0.8231) == "0.82"
    assert formatting.fmt_float(1.2345, 3) == "1.234"
    assert formatting.fmt_float(12345.678, 2) == "12,345.68"


def test_placeholders_are_never_crashes() -> None:
    """None and junk render the placeholder rather than raising on a page."""
    assert formatting.fmt_money(None) == "–"
    assert formatting.fmt_int(None) == "–"
    assert formatting.fmt_pct(None) == "–"
    assert formatting.fmt_pct100(None) == "–"
    assert formatting.fmt_float(None) == "–"
    assert formatting.fmt_money("junk") == "–"
