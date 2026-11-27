"""Titled, high-resolution PNG export for every Plotly chart.

The in-app figures are deliberately compact: headings live in HTML and most
charts are passed through ``without_title()`` so the card sits tight under its
heading. Plotly's mode-bar camera exports that compact figure as rendered —
which is why downloads come out small and title-free.

This module renders each chart through a small ``st.components.v2`` component.
The export action can either sit inside the chart card as a text button, or be
mounted on its own as an icon-only button so controls can share its toolbar row.
Both paths clone the figure in the browser, restore the section heading as the
title, give legends and labels room to breathe, and call Plotly's own client-side
``toImage``. The mode-bar camera is removed so the titled action is the only PNG
download path. Nothing is rendered on the server, so no optional image
dependency is required and no work happens until the user actually clicks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import plotly
import plotly.io as pio
import streamlit as st

from components.empty_state import chart_title as current_chart_title
from visualization import tokens

__all__ = ["build_export_config", "download_png_button", "plotly_chart", "slugify_filename"]

#: Export canvas. At 2x scale this produces a 2000x1200 PNG. Keeping the layout
#: width close to a typical review window matters: a much wider canvas makes
#: axis ticks and value labels shrink to a few pixels when the PNG is viewed
#: fit-to-window, which reads as "the labels are missing".
EXPORT_WIDTH = 1000
EXPORT_HEIGHT = 600
EXPORT_SCALE = 2
DEFAULT_CHART_HEIGHT = 300

#: Export-only type sizes. The in-app chart keeps the theme's compact 11-12px
#: type; the download bumps every text element so it survives a scaled-down
#: image viewer without changing what operators see on screen.
EXPORT_BASE_FONT_SIZE = 16
EXPORT_AXIS_FONT_SIZE = 15
EXPORT_TEXT_FONT_SIZE = 14
EXPORT_LEGEND_FONT_SIZE = 14
EXPORT_TITLE_FONT_SIZE = 20

_EXPORT_MARGIN = {"l": 90, "r": 90, "t": 95, "b": 110}
_EXPORT_LEGEND = {
    "orientation": "h",
    "yanchor": "top",
    "y": -0.12,
    "xanchor": "center",
    "x": 0.5,
    "title": None,
    "bgcolor": "rgba(0,0,0,0)",
}

_COMPONENT_HTML = """
<div class="ci-export-card">
  <div class="ci-export-chart"></div>
  <div class="ci-export-actions">
    <button type="button" class="ci-export-button">Download PNG</button>
  </div>
</div>
"""

_ICON_SVG = """
<svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
  <path d="M12 3v11" />
  <path d="m7.5 10 4.5 4.5 4.5-4.5" />
  <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
</svg>
"""

_ICON_COMPONENT_HTML = (
    """
<div class="ci-export-icon-wrap">
  <button type="button" class="ci-export-icon" title="Download PNG" aria-label="Download PNG">"""
    + _ICON_SVG
    + """</button>
</div>
"""
)

_COMPONENT_CSS = """
.ci-export-card { padding: 8px; margin: 0 0 0.5rem 0; }
.ci-export-chart { width: 100%; }
.ci-export-actions { display: flex; justify-content: flex-end; padding: 6px 4px 2px; }
.ci-export-button, .ci-export-icon {
    cursor: pointer;
    background: transparent;
    border: 1px solid;
    border-radius: 8px;
}
.ci-export-button { font-size: 0.8rem; padding: 0.3rem 0.7rem; }
.ci-export-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 34px;
    height: 34px;
    padding: 0;
}
.ci-export-icon svg {
    width: 18px;
    height: 18px;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.8;
    stroke-linecap: round;
    stroke-linejoin: round;
}
.ci-export-button:disabled, .ci-export-icon:disabled { opacity: 0.6; cursor: wait; }
"""

# Plotly's own bundle is trusted, local code. Loading it with the component
# (instead of from a CDN) keeps exports working in offline deployments.
_PLOTLY_JS = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(
    encoding="utf-8"
)

# Shared by the chart card and the standalone icon button. Both buttons keep
# their original markup while exporting, so swapping text for an SVG needs no
# second implementation.
_EXPORT_BUTTON_JS = """
function ciAttachDownload(button, data) {
    button.onclick = async () => {
        const originalMarkup = button.innerHTML;
        const originalTitle = button.getAttribute('title') || '';
        let hidden = null;
        button.disabled = true;
        button.setAttribute('title', 'Preparing PNG…');
        try {
            const spec = JSON.parse(data.figure);
            const layout = spec.layout || {};
            layout.font = Object.assign({}, layout.font || {}, { size: data.baseFontSize });
            Object.keys(layout).forEach((key) => {
                if (/^[xy]axis[0-9]*$/.test(key) && layout[key]) {
                    layout[key].tickfont = Object.assign(
                        {}, layout[key].tickfont || {}, { size: data.axisFontSize }
                    );
                }
            });
            (spec.data || []).forEach((trace) => {
                if (trace.textfont) {
                    trace.textfont = Object.assign({}, trace.textfont, { size: data.textFontSize });
                }
            });
            layout.title = Object.assign({}, data.titleSpec, { text: data.title });
            layout.margin = data.margin;
            layout.width = data.width;
            layout.height = data.height;
            layout.autosize = false;
            if (spec.data && spec.data.length > 1) {
                layout.showlegend = true;
                layout.legend = Object.assign({}, data.legend, {
                    font: Object.assign({}, data.legend.font || {}, {
                        size: data.legendFontSize,
                        color: layout.font.color,
                    }),
                });
            }
            hidden = document.createElement('div');
            hidden.style.position = 'fixed';
            hidden.style.left = '-10000px';
            hidden.style.top = '0';
            hidden.style.width = data.width + 'px';
            hidden.style.height = data.height + 'px';
            hidden.style.pointerEvents = 'none';
            document.body.appendChild(hidden);
            await Plotly.newPlot(hidden, spec.data, layout, {
                displayModeBar: false,
                staticPlot: true,
            });
            const url = await Plotly.toImage(hidden, {
                format: 'png',
                width: data.width,
                height: data.height,
                scale: data.scale,
            });
            const link = document.createElement('a');
            link.href = url;
            link.download = data.filename;
            document.body.appendChild(link);
            link.click();
            link.remove();
        } catch (error) {
            console.error('PNG export failed', error);
            button.setAttribute('title', 'Export failed');
        } finally {
            if (hidden && hidden.parentNode) hidden.remove();
            if (button.getAttribute('title') !== 'Export failed') {
                button.innerHTML = originalMarkup;
                button.setAttribute('title', originalTitle || 'Download PNG');
                button.disabled = false;
            }
        }
    };
}
"""

_COMPONENT_JS = _PLOTLY_JS + _EXPORT_BUTTON_JS + """
export default function(component) {
    const { data, parentElement } = component;
    const root = parentElement.querySelector('.ci-export-card');
    if (!root) return;

    root.style.background = data.surface;
    root.style.border = '1px solid ' + data.border;
    root.style.borderRadius = data.radius;
    const button = root.querySelector('.ci-export-button');
    button.style.color = data.text;
    button.style.borderColor = data.border;
    if (data.showDownload === false) {
        const actions = root.querySelector('.ci-export-actions');
        if (actions) actions.remove();
    } else {
        ciAttachDownload(button, data);
    }

    const chart = root.querySelector('.ci-export-chart');
    chart.style.height = data.chartHeight + 'px';
    const figure = JSON.parse(data.figure);
    figure.layout.height = data.chartHeight;
    Plotly.newPlot(chart, figure.data, figure.layout, {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
    }).catch(() => {});
}
"""

_ICON_COMPONENT_JS = _PLOTLY_JS + _EXPORT_BUTTON_JS + """
export default function(component) {
    const { data, parentElement } = component;
    const button = parentElement.querySelector('.ci-export-icon');
    if (!button) return;
    button.style.color = data.text;
    button.style.borderColor = data.border;
    ciAttachDownload(button, data);
}
"""

_TITLED_PLOT = st.components.v2.component(
    "ci_titled_plotly",
    html=_COMPONENT_HTML,
    css=_COMPONENT_CSS,
    js=_COMPONENT_JS,
    isolate_styles=False,
)

_DOWNLOAD_ICON = st.components.v2.component(
    "ci_download_icon",
    html=_ICON_COMPONENT_HTML,
    css=_COMPONENT_CSS,
    js=_ICON_COMPONENT_JS,
    isolate_styles=False,
)


def slugify_filename(title: str) -> str:
    """Turn a human chart title into a safe, readable PNG filename stem."""
    stem = re.sub(r"[^a-z0-9]+", "-", str(title).lower()).strip("-")
    return stem or "chart"


def _resolve_title(figure: Any, title: str | None) -> str:
    """Pick the export title: explicit, then section heading, then figure title."""
    for candidate in (
        title,
        current_chart_title.get(),
        getattr(figure.layout.title, "text", None) if hasattr(figure, "layout") else None,
    ):
        if candidate is not None and str(candidate).strip():
            return str(candidate).strip()
    return "Chart"


def build_export_config(
    figure: Any,
    title: str,
    filename: str,
    *,
    chart_height: int = DEFAULT_CHART_HEIGHT,
) -> dict[str, Any]:
    """Build the JSON-safe payload consumed by the export components."""
    ink = tokens.color("text")
    return {
        "figure": pio.to_json(figure),
        "title": title,
        "filename": filename,
        "chartHeight": chart_height,
        "surface": tokens.color("surface"),
        "border": tokens.color("border"),
        "radius": tokens.RADIUS["md"],
        "text": ink,
        "width": EXPORT_WIDTH,
        "height": EXPORT_HEIGHT,
        "scale": EXPORT_SCALE,
        "baseFontSize": EXPORT_BASE_FONT_SIZE,
        "axisFontSize": EXPORT_AXIS_FONT_SIZE,
        "textFontSize": EXPORT_TEXT_FONT_SIZE,
        "legendFontSize": EXPORT_LEGEND_FONT_SIZE,
        "margin": dict(_EXPORT_MARGIN),
        "legend": dict(_EXPORT_LEGEND),
        "titleSpec": {
            "x": 0.02,
            "xanchor": "left",
            "y": 0.96,
            "yanchor": "top",
            "font": {"size": EXPORT_TITLE_FONT_SIZE, "color": ink},
        },
    }


def _mount_icon(payload: dict[str, Any], *, key: str | None = None) -> Any:
    """Mount the icon-only export component from a prepared payload."""
    return _DOWNLOAD_ICON(data=payload, key=key, width="content", height=44)


def _mount_download_toolbar(payload: dict[str, Any], key: str | None) -> Any:
    """Mount the default toolbar: a right-aligned download icon under the chart."""
    with st.container(
        horizontal=True,
        wrap=False,
        horizontal_alignment="right",
        vertical_alignment="center",
    ):
        return _mount_icon(payload, key=f"{key}_download" if key else None)


def plotly_chart(
    figure: Any,
    *,
    title: str | None = None,
    filename: str | None = None,
    key: str | None = None,
    show_download: bool = True,
    **kwargs: Any,
) -> Any:
    """Render a Plotly chart with a titled, high-resolution PNG download icon.

    Args:
        figure: Any Plotly figure accepted by ``plotly.io.to_json``.
        title: Export title. Defaults to the enclosing ``chart_section`` heading.
        filename: Optional PNG filename; otherwise derived from the title.
        key: Optional Streamlit widget key for this chart instance.
        show_download: Mount the default icon toolbar. Set ``False`` when the
            page mounts :func:`download_png_button` beside its own controls.
        **kwargs: ``width`` / ``height`` are passed to the component; other
            options are accepted and ignored for compatibility.

    Returns:
        The chart component result, so the call can still be used as an element.
    """
    resolved_title = _resolve_title(figure, title)
    chart_height = int(getattr(figure.layout, "height", None) or DEFAULT_CHART_HEIGHT)
    payload = build_export_config(
        figure,
        resolved_title,
        filename or f"{slugify_filename(resolved_title)}.png",
        chart_height=chart_height,
    )
    # The button now lives in the toolbar below the chart for every graph.
    payload["showDownload"] = False
    result = _TITLED_PLOT(
        data=payload,
        key=key,
        width=kwargs.get("width", "stretch"),
        height=kwargs.get("height", chart_height + 16),
    )
    if show_download:
        _mount_download_toolbar(payload, key)
    return result


def download_png_button(
    figure: Any,
    *,
    title: str | None = None,
    filename: str | None = None,
    key: str | None = None,
) -> Any:
    """Mount an icon-only PNG download inside a page-managed toolbar.

    Use this when chart-specific controls share the row: mount the chart with
    ``plotly_chart(..., show_download=False)``, then call this beside those
    controls. It carries the same export title and figure as the default toolbar.
    """
    resolved_title = _resolve_title(figure, title)
    payload = build_export_config(
        figure,
        resolved_title,
        filename or f"{slugify_filename(resolved_title)}.png",
    )
    return _mount_icon(payload, key=key)
