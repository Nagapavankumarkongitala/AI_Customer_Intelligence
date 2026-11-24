"""Titled, high-resolution PNG export for every Plotly chart.

The in-app figures are deliberately compact: headings live in HTML and most
charts are passed through ``without_title()`` so the card sits tight under its
heading. Plotly's mode-bar camera exports that compact figure as rendered —
which is why downloads come out small and title-free.

This module renders each chart through a small ``st.components.v2`` component
with a companion **Download PNG** button. The button clones the figure in the
browser, restores the section heading as the title, gives legends and labels
room to breathe, and calls Plotly's own client-side ``toImage``. The mode-bar
camera is removed so the titled button is the only PNG download path. Nothing
is rendered on the server, so no optional image dependency is required and no
work happens until the user actually clicks.
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

__all__ = ["build_export_config", "plotly_chart", "slugify_filename"]

#: Export canvas. At 2x scale this produces a 2800x1700 PNG — sharp enough for
#: slides and reports while staying a reasonable file size.
EXPORT_WIDTH = 1400
EXPORT_HEIGHT = 850
EXPORT_SCALE = 2
DEFAULT_CHART_HEIGHT = 300

_EXPORT_MARGIN = {"l": 70, "r": 50, "t": 120, "b": 140}
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

_COMPONENT_CSS = """
.ci-export-card { padding: 8px; margin: 0 0 0.5rem 0; }
.ci-export-chart { width: 100%; }
.ci-export-actions { display: flex; justify-content: flex-end; padding: 6px 4px 2px; }
.ci-export-button {
    cursor: pointer;
    background: transparent;
    border: 1px solid;
    border-radius: 8px;
    font-size: 0.8rem;
    padding: 0.3rem 0.7rem;
}
.ci-export-button:disabled { opacity: 0.6; cursor: wait; }
"""

# Plotly's own bundle is trusted, local code. Loading it with the component
# (instead of from a CDN) keeps exports working in offline deployments.
_PLOTLY_JS = (Path(plotly.__file__).parent / "package_data" / "plotly.min.js").read_text(
    encoding="utf-8"
)

_COMPONENT_JS = _PLOTLY_JS + """
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

    const chart = root.querySelector('.ci-export-chart');
    chart.style.height = data.chartHeight + 'px';
    const figure = JSON.parse(data.figure);
    figure.layout.height = data.chartHeight;
    Plotly.newPlot(chart, figure.data, figure.layout, {
        responsive: true,
        displaylogo: false,
        modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
    }).catch(() => {});

    button.onclick = async () => {
        const originalLabel = 'Download PNG';
        let hidden = null;
        button.disabled = true;
        button.textContent = 'Preparing…';
        try {
            const spec = JSON.parse(data.figure);
            const layout = spec.layout || {};
            layout.title = Object.assign({}, data.titleSpec, { text: data.title });
            layout.margin = data.margin;
            layout.width = data.width;
            layout.height = data.height;
            layout.autosize = false;
            if (spec.data && spec.data.length > 1) {
                layout.showlegend = true;
                layout.legend = data.legend;
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
            button.textContent = 'Export failed';
        } finally {
            if (hidden && hidden.parentNode) hidden.remove();
            if (button.textContent !== 'Export failed') {
                button.disabled = false;
                button.textContent = originalLabel;
            }
        }
    };
}
"""

_TITLED_PLOT = st.components.v2.component(
    "ci_titled_plotly",
    html=_COMPONENT_HTML,
    css=_COMPONENT_CSS,
    js=_COMPONENT_JS,
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
    """Build the JSON-safe payload consumed by the export component."""
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
        "margin": dict(_EXPORT_MARGIN),
        "legend": dict(_EXPORT_LEGEND),
        "titleSpec": {
            "x": 0.02,
            "xanchor": "left",
            "y": 0.96,
            "yanchor": "top",
            "font": {"size": 20, "color": ink},
        },
    }


def plotly_chart(
    figure: Any,
    *,
    title: str | None = None,
    filename: str | None = None,
    key: str | None = None,
    **kwargs: Any,
) -> Any:
    """Render a Plotly chart with a titled, high-resolution PNG download button.

    Args:
        figure: Any Plotly figure accepted by ``plotly.io.to_json``.
        title: Export title. Defaults to the enclosing ``chart_section`` heading.
        filename: Optional PNG filename; otherwise derived from the title.
        key: Optional Streamlit widget key for this chart instance.
        **kwargs: ``width`` / ``height`` are passed to the component; other
            options are accepted and ignored for compatibility.

    Returns:
        The component result, so the call can still be used as a chart element.
    """
    resolved_title = _resolve_title(figure, title)
    chart_height = int(getattr(figure.layout, "height", None) or DEFAULT_CHART_HEIGHT)
    payload = build_export_config(
        figure,
        resolved_title,
        filename or f"{slugify_filename(resolved_title)}.png",
        chart_height=chart_height,
    )
    return _TITLED_PLOT(
        data=payload,
        key=key,
        width=kwargs.get("width", "stretch"),
        height=kwargs.get("height", chart_height + 64),
    )
