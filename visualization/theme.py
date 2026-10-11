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
MUTED = "#737373"
RISK_RED = "#FF4D4D"
RISK_GREEN = "#3DDC84"
RISK_AMBER = "#E8A838"

TERMINAL_CSS = """
<style>
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
.dashboard-card {
    background-color: #141414;
    border: 1px solid #222222;
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
}
[data-testid="stSidebar"] {
    background-color: #050505;
}
[data-testid="stDataFrame"] {
    max-height: 420px;
    overflow-y: auto;
}
[data-testid="stMetricValue"] {
    color: #CCFF00;
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
