"""Component tests: formatting and every pure HTML/Styler builder.

Builders stay pure (``*_html``) with thin ``st`` wrappers so the whole library
is assertable without a Streamlit session.
"""

from __future__ import annotations

import pandas as pd
import pytest

from components.empty_state import chart_section_html, empty_state_html
from components.insight_box import insight_box_html
from components.metric_card import (
    MAX_CARDS_PER_ROW,
    Metric,
    chunk_metrics,
    metric_card_html,
)
from components.page_header import page_header_html, section_header_html
from components.panels import active_filter_chips_html
from components.status_badge import risk_badge_html, status_badge_html
from components.table import _risk_cell_style, format_frame, style_table
from utils import formatting
from visualization import tokens

# --------------------------------------------------------------------------- #
# Formatting.
# --------------------------------------------------------------------------- #


def test_formatting_primitives() -> None:
    """One spelling for ints, money, percents, compact scale, and dates."""
    assert formatting.fmt_int(1234567) == "1,234,567"
    assert formatting.fmt_int(None) == "–"
    assert formatting.fmt_money(123456) == f"{formatting.currency_symbol()}123,456"
    assert formatting.fmt_money(12.345, decimals=2).endswith("12.35")
    assert formatting.fmt_pct(0.123) == "12.3%"
    assert formatting.fmt_pct(None) == "–"
    assert formatting.fmt_compact(1500) == "1.5K"
    assert formatting.fmt_compact(2_300_000) == "2.3M"
    assert formatting.fmt_compact(999) == "999"
    assert formatting.fmt_compact(None) == "–"


def test_formatting_degrades_on_bad_input() -> None:
    """Unparseable values render the placeholder instead of raising mid-page."""
    assert formatting.fmt_int("not-a-number") == "–"
    assert formatting.fmt_money("not-a-number") == "–"
    assert formatting.fmt_pct("not-a-number") == "–"
    assert formatting.fmt_compact("not-a-number") == "–"
    assert formatting.fmt_date(None) == "–"
    assert formatting.fmt_date("not-a-date") == "not-a-date"


def test_formatting_dates() -> None:
    """Datetime, date, and ISO strings converge on one ISO spelling."""
    import datetime as dt

    assert formatting.fmt_date(dt.datetime(2026, 3, 14, 9, 30)) == "2026-03-14"
    assert formatting.fmt_date(dt.date(2026, 3, 14)) == "2026-03-14"
    assert formatting.fmt_date("2026-03-14") == "2026-03-14"


def test_currency_symbol_falls_back_off_config() -> None:
    """Formatting never crashes a page over an unreadable config."""
    assert formatting.currency_symbol("definitely-not-a-config.yaml") == "₹"
    assert formatting.currency_symbol() == "₹"


# --------------------------------------------------------------------------- #
# Headers and cards.
# --------------------------------------------------------------------------- #


def test_page_header_hierarchy_and_escaping() -> None:
    """Badge eyebrow, title, subtitle — with user content escaped."""
    out = page_header_html("Churn <b>Prediction</b>", "Subtitle & more", badge="Step 7 of 9")
    assert "<h1" in out and "Churn &lt;b&gt;Prediction&lt;/b&gt;" in out
    assert "Subtitle &amp; more" in out and "Step 7 of 9" in out
    assert "font-size:1.4rem" in out
    minimal = page_header_html("Title")
    assert "<h1" in minimal and "Subtitle" not in minimal


def test_section_header() -> None:
    """Compact section title with an optional muted subtitle."""
    out = section_header_html("Trends", "Last 90 days")
    assert "<h2" in out and "Last 90 days" in out
    assert "90 days" not in section_header_html("Trends")


def test_metric_cards_and_chunking() -> None:
    """Cards carry label/value/delta/tone; rows never exceed five."""
    assert MAX_CARDS_PER_ROW == 5
    card = metric_card_html(Metric("Revenue", "₹1.2M", "+4.2%", tone="positive"))
    assert "Revenue" in card and "₹1.2M" in card and "+4.2%" in card
    assert tokens.LIGHT["positive"] in card

    neutral = metric_card_html(Metric("Customers", "1,024", tone="mystery"))
    assert tokens.LIGHT["muted"] in neutral  # unknown tones degrade to neutral

    metrics = [Metric(f"K{i}", str(i)) for i in range(12)]
    rows = chunk_metrics(metrics)
    assert [len(row) for row in rows] == [5, 5, 2]
    assert chunk_metrics([]) == []
    try:
        chunk_metrics(metrics, width=0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("width=0 did not raise")


# --------------------------------------------------------------------------- #
# Badges, insights, empty states.
# --------------------------------------------------------------------------- #


def test_status_and_risk_badges_use_tokens() -> None:
    """Pills resolve through the token tables, so hues cannot drift."""
    for band in ("low", "medium", "high", "critical"):
        out = risk_badge_html(band)
        bg, fg = tokens.risk_badge(band)
        assert "status-badge" in out and bg in out and fg in out

    info = status_badge_html("Ready", tone="positive")
    bg, _ = tokens.tone_badge("positive")
    assert bg in info
    fallback = status_badge_html("Ready", tone="mystery")
    assert tokens.tone_badge("info")[0] in fallback


def test_insight_box_carries_one_state() -> None:
    """Background, text, and border resolve to a single tone pair."""
    out = insight_box_html(
        "Concentration risk", "Top decile drives 41% of revenue.", tone="warning"
    )
    bg, fg = tokens.tone_badge("warning")
    assert "insight-box" in out and bg in out and f"border-left-color:{fg}" in out
    assert "Top decile" in out


def test_empty_state_and_chart_section() -> None:
    """Empty states name the next step; sections frame figures with captions."""
    out = empty_state_html("No data yet", "Upload a file to continue.")
    assert "No data yet" in out and "Upload a file" in out
    section = chart_section_html("Revenue trend", "Trailing 12 weeks")
    assert "Revenue trend" in section and "Trailing 12 weeks" in section
    assert "Trailing" not in chart_section_html("Revenue trend")


def test_active_filter_chips() -> None:
    """No active filters render nothing; values are escaped."""
    assert active_filter_chips_html({}) == ""
    out = active_filter_chips_html({"Segment": "Champions", "Risk": "<all>"})
    assert "Segment" in out and "Champions" in out and "&lt;all&gt;" in out


# --------------------------------------------------------------------------- #
# Tables.
# --------------------------------------------------------------------------- #


def test_format_frame_and_risk_cells() -> None:
    """Numerics format per column role; risk cells paint from the badge table."""
    df = pd.DataFrame(
        {
            "customer": ["a", "b"],
            "risk_band": ["high", "low"],
            "value": [1234.5, 678.9],
            "rate": [0.125, 0.034],
            "count": [10, 20],
        }
    )
    view = format_frame(df, money_cols=("value",), pct_cols=("rate",), int_cols=("count",))
    assert view["value"].tolist()[0].startswith("₹")
    assert view["rate"].tolist()[0] == "12.5%"
    assert view["count"].tolist() == ["10", "20"]
    assert view["customer"].tolist() == ["a", "b"]

    assert format_frame(df.iloc[0:0]).empty
    assert "missing" not in str(format_frame(df, money_cols=("missing",)).columns)

    bg, fg = tokens.risk_badge("high")
    painted = _risk_cell_style("high")
    assert bg in painted and fg in painted
    assert _risk_cell_style("whatever") == ""

    styler = style_table(df, risk_cols=("risk_band",), money_cols=("value",))
    import components.table as table_module

    if table_module._HAS_JINJA2:
        from pandas.io.formats.style import Styler

        assert isinstance(styler, Styler)
    else:
        # jinja2 present but below pandas' floor: the formatted frame is the fallback.
        assert isinstance(styler, pd.DataFrame)


def test_style_table_falls_back_without_jinja2(monkeypatch) -> None:
    """Missing jinja2 degrades to formatted values, never a page crash."""
    import components.table as table_module

    monkeypatch.setattr(table_module, "_HAS_JINJA2", False)
    df = pd.DataFrame({"risk_band": ["high"], "value": [1234.5]})
    fallback = table_module.style_table(df, risk_cols=("risk_band",), money_cols=("value",))
    assert isinstance(fallback, pd.DataFrame)
    assert fallback["value"].tolist()[0].startswith("₹")


def test_jinja2_usability_checks_version_not_presence(monkeypatch) -> None:
    """Version, not presence, decides: 3.1.4 falls back, 3.1.5 is usable."""
    pytest.importorskip("jinja2")
    import importlib.metadata

    import components.table as table_module

    monkeypatch.setattr(importlib.metadata, "version", lambda name: "3.1.4")
    assert table_module._jinja2_usable() is False
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "3.1.5")
    assert table_module._jinja2_usable() is True


def test_html_table_paints_without_jinja2() -> None:
    """The standalone table keeps values, escaping, and risk colors."""
    from components.table import html_table

    df = pd.DataFrame(
        {"risk_band": ["high", "low"], "value": ["₹1,235", "₹679"], "note": ["<b>x</b>", "ok"]}
    )
    out = html_table(df, risk_cols=("risk_band",))
    assert "<table" in out and out.count("<tr>") == 3  # header + 2 rows
    assert "&lt;b&gt;x&lt;/b&gt;" in out
    bg, fg = tokens.risk_badge("high")
    assert bg in out and fg in out
    assert tokens.risk_badge("low")[0] in out


def test_html_table_height_caps_scroll_region() -> None:
    """A height caps the scroll div; without it the table sizes naturally."""
    from components.table import html_table

    df = pd.DataFrame({"a": ["x"]})
    assert "max-height:320px" in html_table(df, height=320)
    assert "max-height" not in html_table(df)


def test_builders_emit_active_mode_ink() -> None:
    """No inherited root ink: text nodes carry the active mode's color.

    The app root follows the static light config, so any text without an
    explicit color renders light ink on dark surfaces (invisible KPI values).
    """
    from components import sidebar as sidebar_module

    original = tokens.active_mode()
    try:
        tokens.set_active_mode(tokens.MODE_DARK)
        dark_text = tokens.DARK["text"]
        assert dark_text in metric_card_html(Metric("K", "1"))
        assert dark_text in empty_state_html("T", "B")
        assert dark_text in chart_section_html("T")
        assert dark_text in sidebar_module.sidebar_status_html({"A": "b"})
    finally:
        tokens.set_active_mode(original)
    assert tokens.LIGHT["text"] in metric_card_html(Metric("K", "1"))


# --------------------------------------------------------------------------- #
# Strategy playbook cards.
# --------------------------------------------------------------------------- #


def _strategy_matrix():
    """Three-rule matrix mirroring the YAML schema (one rule owns no customers)."""
    from retention_engine.strategy_mapper import StrategyMatrix, StrategyRule

    return StrategyMatrix(
        rules=[
            StrategyRule(
                name="vip_save",
                value_tiers=["High"],
                risk_bands=["high", "critical"],
                action="Personal outreach",
                channel="Phone & email",
                offer_type="Loyalty recovery",
                reason="High value ({value}) with {band} risk (p={probability}).",
            ),
            StrategyRule(
                name="nurture",
                value_tiers=["Low"],
                risk_bands=["low"],
                action="Routine nurture",
                channel="Email",
                offer_type="None",
                reason="Low value ({value}) with low risk.",
            ),
            StrategyRule(
                name="idle",
                value_tiers=["Medium"],
                risk_bands=["medium"],
                action="Limited nudge",
                channel="Email",
                offer_type="Limited incentive",
                reason="Medium value ({value}) with medium risk.",
            ),
        ]
    )


def test_strategy_cards_aggregate_rule_footprints() -> None:
    """Each card carries its own customer count, revenue at risk, and means."""
    from components.strategy_cards import build_strategy_cards

    table = pd.DataFrame(
        {
            "customer_id": ["a", "b", "c"],
            "strategy_rule": ["vip_save", "vip_save", "nurture"],
            "revenue_at_risk": [100.0, 300.0, 50.0],
            "churn_probability": [0.8, 0.6, 0.1],
            "expected_value_proxy_historical": [1000.0, 2000.0, 100.0],
        }
    )
    cards = build_strategy_cards(table, _strategy_matrix())

    assert [c.name for c in cards] == ["vip_save", "nurture", "idle"]
    vip, nurture, idle = cards
    assert (vip.customers, nurture.customers) == (2, 1)
    assert vip.revenue_at_risk == pytest.approx(400.0)
    assert idle.customers == 0 and idle.revenue_at_risk == 0.0
    # Reason is rendered from the rule's own averages, not one sample customer.
    assert "0.700" in vip.reason and formatting.fmt_money(1500.0) in vip.reason
    # A rule with no customers cannot invent a value or a probability.
    assert "0.000" not in idle.reason and "—" in idle.reason


def test_strategy_cards_handle_empty_and_unlabelled_tables() -> None:
    """No table and no rule column both degrade cleanly instead of raising."""
    from components.strategy_cards import build_strategy_cards

    assert build_strategy_cards(pd.DataFrame(), _strategy_matrix()) == []

    table = pd.DataFrame(
        {
            "customer_id": ["a"],
            "recommended_action": ["Routine nurture"],
            "revenue_at_risk": [10.0],
        }
    )
    cards = build_strategy_cards(table, _strategy_matrix())
    assert [c.customers for c in cards] == [0, 1, 0]


def test_strategy_card_html_escapes_and_uses_action_accent() -> None:
    """Card markup escapes user content and paints the action's own hue."""
    from components.strategy_cards import StrategyCard, strategy_card_html

    card = StrategyCard(
        name="x",
        action="Personal <b>outreach</b>",
        value_tiers=("High",),
        risk_bands=("high", "critical"),
        channel="Phone & email",
        offer_type="Loyalty recovery",
        reason="High value (₹1) with high risk.",
        customers=1234,
        revenue_at_risk=12_345.0,
    )
    out = strategy_card_html(card)

    assert "Personal &lt;b&gt;outreach&lt;/b&gt;" in out
    assert "Phone &amp; email" in out
    assert "High value · high / critical risk" in out
    assert tokens.action_color("Personal <b>outreach</b>") in out
    assert formatting.fmt_int(1234) in out and formatting.fmt_compact(12_345.0) in out
