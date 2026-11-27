"""Tests for titled PNG export on every Plotly chart."""

from __future__ import annotations

import json

import plotly.graph_objects as go
import pytest

from components import chart_export, empty_state
from components.chart_export import build_export_config, download_png_button, slugify_filename
from components.empty_state import chart_title


def test_slugify_filename() -> None:
    """Titles become readable, filesystem-safe PNG stems."""
    assert slugify_filename("Customers per segment") == "customers-per-segment"
    assert slugify_filename("Revenue (₹) / 2025!") == "revenue-2025"
    assert slugify_filename("   ") == "chart"


def test_resolve_title_precedence() -> None:
    """Explicit title wins, then the section heading, then the figure title."""
    figure = go.Figure()
    figure.update_layout(title="Factory title")
    token = chart_title.set("Section heading")
    try:
        assert chart_export._resolve_title(figure, "Explicit title") == "Explicit title"
        assert chart_export._resolve_title(figure, None) == "Section heading"
    finally:
        chart_title.reset(token)

    token = chart_title.set(None)
    try:
        assert chart_export._resolve_title(figure, None) == "Factory title"
        figure.update_layout(title="")
        assert chart_export._resolve_title(figure, None) == "Chart"
    finally:
        chart_title.reset(token)


def test_build_export_config_carries_title_and_export_geometry() -> None:
    """The component payload is JSON-safe and sets a roomy titled canvas."""
    figure = go.Figure(data=go.Bar(x=["A"], y=[1]))
    figure.update_layout(height=340)

    payload = build_export_config(figure, "Revenue per segment", "revenue.png", chart_height=340)

    assert json.loads(payload["figure"])["data"][0]["type"] == "bar"
    assert payload["title"] == "Revenue per segment"
    assert payload["filename"] == "revenue.png"
    assert payload["chartHeight"] == 340
    assert (payload["width"], payload["height"], payload["scale"]) == (1000, 600, 2)
    assert payload["margin"]["t"] == 95
    assert payload["margin"]["b"] == 110
    assert payload["legend"]["y"] == -0.12
    assert payload["titleSpec"]["font"]["size"] == 20
    assert payload["axisFontSize"] >= 14
    assert payload["textFontSize"] >= 14
    assert payload["legendFontSize"] >= 14


def test_plotly_chart_mounts_component_with_section_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper mounts one component with the resolved heading and filename."""
    captured: dict = {}

    def fake_component(**kwargs: object) -> str:
        captured.update(kwargs)
        return "component-result"

    monkeypatch.setattr(chart_export, "_TITLED_PLOT", fake_component)
    mounted_toolbar: list[dict] = []
    monkeypatch.setattr(
        chart_export,
        "_mount_download_toolbar",
        lambda payload, key: mounted_toolbar.append(payload),
    )
    figure = go.Figure(data=go.Bar(x=["A"], y=[1]))
    figure.update_layout(height=300)
    token = chart_title.set("Customers per segment")
    try:
        result = chart_export.plotly_chart(figure, width="stretch", key="segment_mix")
    finally:
        chart_title.reset(token)

    assert result == "component-result"
    assert captured["key"] == "segment_mix"
    assert captured["width"] == "stretch"
    assert captured["height"] == 316
    payload = captured["data"]
    assert payload["title"] == "Customers per segment"
    assert payload["filename"] == "customers-per-segment.png"


def test_toolbar_chart_hides_its_own_button(monkeypatch: pytest.MonkeyPatch) -> None:
    """A chart in a toolbar leaves the download action to the icon button."""
    captured: dict = {}

    def fake_component(**kwargs: object) -> str:
        captured.update(kwargs)
        return "chart"

    monkeypatch.setattr(chart_export, "_TITLED_PLOT", fake_component)
    toolbar_calls: list = []
    monkeypatch.setattr(
        chart_export,
        "_mount_download_toolbar",
        lambda payload, key: toolbar_calls.append(key),
    )
    figure = go.Figure(data=go.Bar(x=["A"], y=[1]))
    figure.update_layout(height=300)

    chart_export.plotly_chart(figure, key="toolbar_chart", show_download=False)

    assert captured["data"]["showDownload"] is False
    assert captured["height"] == 316
    assert toolbar_calls == []


def test_download_png_button_mounts_icon_component(monkeypatch: pytest.MonkeyPatch) -> None:
    """The standalone action carries the same export payload as the chart card."""
    captured: dict = {}

    def fake_component(**kwargs: object) -> str:
        captured.update(kwargs)
        return "icon"

    monkeypatch.setattr(chart_export, "_DOWNLOAD_ICON", fake_component)
    figure = go.Figure(data=go.Bar(x=["A"], y=[1]))

    result = download_png_button(figure, title="Revenue trend", key="revenue_icon")

    assert result == "icon"
    assert captured["key"] == "revenue_icon"
    assert captured["data"]["title"] == "Revenue trend"
    assert captured["data"]["filename"] == "revenue-trend.png"
    assert captured["width"] == "content"


def test_chart_section_publishes_heading_for_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    """chart_section hands its heading to download buttons and restores state."""
    monkeypatch.setattr(empty_state.st, "markdown", lambda *args, **kwargs: None)
    token = chart_title.set(None)
    try:
        with empty_state.chart_section("Revenue per segment", "Last 30 days"):
            assert chart_title.get() == "Revenue per segment"
        assert chart_title.get() is None
    finally:
        chart_title.reset(token)
