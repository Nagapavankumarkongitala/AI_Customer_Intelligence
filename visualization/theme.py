"""Terminal theme: obsidian black + electric volt lime shared by every page.

Pages call inject_terminal_css() once at the top; every Plotly factory routes its
figure through style_plotly_fig() before returning. Risk semantics keep
red/green hues (labels always carry the meaning — never color alone).
"""

CANVAS = "#070B14"
CARD = "#0D1930"
CARD_BORDER = "#222222"
PRIMARY = "#22D3EE"
PRIMARY_DIM = "#0E7490"
PRIMARY_PALE = "#A5F3FC"
TEXT = "#E8EEF6"
MUTED = "#8B98AD"  # secondary text floor: ≥ 4.5:1 on canvas (computed in tests)
RISK_RED = "#FF4D4D"
RISK_GREEN = "#3DDC84"
RISK_AMBER = "#E8A838"

# Glassmorphism tokens (fallback defaults; skill design-system query wins on conflict).
GLASS_BG = "rgba(13,25,48,0.55)"
GLASS_BORDER = "rgba(255,255,255,0.12)"
GLASS_BLUR = "blur(12px)"
GLASS_RADIUS = "16px"
GLASS_SHADOW = "0 8px 32px rgba(0,0,0,0.45), 0 0 24px rgba(34,211,238,0.07)"
CANVAS_GLOW_TOP = "radial-gradient(1200px 600px at 15% -5%, rgba(34,211,238,0.07), transparent)"
CANVAS_GLOW_BOTTOM = "radial-gradient(1000px 700px at 90% 110%, rgba(61,220,132,0.05), transparent)"

TERMINAL_CSS = """
<style>
:root { color-scheme: dark; }
html, body { touch-action: manipulation; }
/* Layered volt glow canvas over obsidian */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1200px 600px at 15% -5%, rgba(34,211,238,0.07), transparent),
        radial-gradient(1000px 700px at 90% 110%, rgba(61,220,132,0.05), transparent),
        #070B14;
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 110rem;
}
/* Sidebar removed: navigation is the top tab bar, filters live in expanders */
[data-testid="stSidebar"], [data-testid="stSidebarNav"] {
    display: none;
}
/* Top tab bar */
.nav-tab {
    text-align: center;
    padding: 0.45rem 0.2rem;
    font-size: 0.82rem;
    color: #E8EEF6;
}
.nav-active {
    color: #22D3EE;
    border-bottom: 2px solid #22D3EE;
    font-weight: 600;
}
/* Frosted-glass cards (blur only over the gradient canvas, never on controls) */
.dashboard-card {
    background-color: rgba(13,25,48,0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45), 0 0 24px rgba(34,211,238,0.07);
}
/* KPI metrics double as glass cards */
[data-testid="stMetric"] {
    background-color: rgba(13,25,48,0.55);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 1rem 1.25rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.45), 0 0 24px rgba(34,211,238,0.07);
}
[data-testid="stSidebar"] {
    background-color: #070B14;
}
/* Native selects need explicit colors on Windows dark mode */
[data-testid="stSidebar"] select, div[data-baseweb="select"] {
    background-color: #0D1930;
    color: #E8EEF6;
}
[data-testid="stDataFrame"] {
    max-height: 420px;
    overflow-y: auto;
    font-variant-numeric: tabular-nums;
}
[data-testid="stMetricValue"] {
    color: #22D3EE;
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
            marker.update(line_color=CANVAS, line_width=1.2)
        except (ValueError, AttributeError):
            continue  # trace types without marker lines (heatmaps, Sankeys, ...)
    return fig


PRIMARY_SCALE = ["#082F3A", "#0E7490", "#22D3EE", "#A5F3FC"]
PRIMARY_DISCRETE = [PRIMARY, PRIMARY_DIM, MUTED, TEXT, RISK_AMBER]


def styled(fn=None, *, height: int = 380):
    """Decorator routing a Plotly factory's figure through style_plotly_fig."""

    def wrap(factory):
        import functools

        @functools.wraps(factory)
        def inner(*args, **kwargs):
            return style_plotly_fig(factory(*args, **kwargs), height=height)

        return inner

    return wrap(fn) if fn is not None else wrap
