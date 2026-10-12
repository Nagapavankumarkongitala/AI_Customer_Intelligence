"""Terminal theme: obsidian black + electric volt lime shared by every page.

Pages call inject_terminal_css() once at the top; every Plotly factory routes its
figure through style_plotly_fig() before returning. Risk semantics keep
red/green hues (labels always carry the meaning — never color alone).
"""

OBSIDIAN = "#050505"
CARD = "#141414"
CARD_BORDER = "#222222"
VOLT = "#CCFF00"
VOLT_DIM = "#9DB800"
MUTED = "#7A7A7A"  # secondary text floor: ≥ 4.5:1 on obsidian (computed in tests)
RISK_RED = "#FF4D4D"
RISK_GREEN = "#3DDC84"
RISK_AMBER = "#E8A838"

# Glassmorphism tokens (fallback defaults; skill design-system query wins on conflict).
GLASS_BG = "rgba(20,20,20,0.55)"
GLASS_BORDER = "rgba(204,255,0,0.14)"
GLASS_BLUR = "blur(14px)"
GLASS_RADIUS = "16px"
GLASS_SHADOW = "0 8px 32px rgba(0,0,0,0.45)"
CANVAS_GLOW_TOP = "radial-gradient(1200px 600px at 15% -5%, rgba(204,255,0,0.06), transparent)"
CANVAS_GLOW_BOTTOM = "radial-gradient(1000px 700px at 90% 110%, rgba(61,220,132,0.05), transparent)"

TERMINAL_CSS = """
<style>
:root { color-scheme: dark; }
html, body { touch-action: manipulation; }
/* Layered volt glow canvas over obsidian */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1200px 600px at 15% -5%, rgba(204,255,0,0.06), transparent),
        radial-gradient(1000px 700px at 90% 110%, rgba(61,220,132,0.05), transparent),
        #050505;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
/* Frosted-glass cards (blur only over the gradient canvas, never on controls) */
.dashboard-card {
    background-color: rgba(20,20,20,0.55);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(204,255,0,0.14);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45);
}
[data-testid="stSidebar"] {
    background-color: #050505;
}
/* Native selects need explicit colors on Windows dark mode */
[data-testid="stSidebar"] select, div[data-baseweb="select"] {
    background-color: #141414;
    color: #E8E8E8;
}
[data-testid="stDataFrame"] {
    max-height: 420px;
    overflow-y: auto;
    font-variant-numeric: tabular-nums;
}
[data-testid="stMetricValue"] {
    color: #CCFF00;
    font-variant-numeric: tabular-nums;
}
</style>
"""


def inject_terminal_css() -> None:
    """Inject the terminal density/card CSS. Call once per page, top of file."""
    import streamlit as st

    st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


def style_plotly_fig(fig, height: int = 380):
    """Apply the dark terminal look to any Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": MUTED, "size": 11},
        margin={"l": 15, "r": 15, "t": 35, "b": 15},
        height=height,
    )
    for trace in fig.data:
        marker = getattr(trace, "marker", None)
        if marker is None:
            continue
        try:
            marker.update(line_color=OBSIDIAN, line_width=1.2)
        except (ValueError, AttributeError):
            continue  # trace types without marker lines (heatmaps, Sankeys, ...)
    return fig


VOLT_SCALE = ["#1A1A00", "#4D5C00", "#9DB800", "#CCFF00"]
VOLT_DISCRETE = [VOLT, VOLT_DIM, MUTED, "#E8E8E8", RISK_AMBER]


def styled(fn=None, *, height: int = 380):
    """Decorator routing a Plotly factory's figure through style_plotly_fig."""

    def wrap(factory):
        import functools

        @functools.wraps(factory)
        def inner(*args, **kwargs):
            return style_plotly_fig(factory(*args, **kwargs), height=height)

        return inner

    return wrap(fn) if fn is not None else wrap
