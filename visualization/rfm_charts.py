"""Plotly figure factories for RFM/LRFM pages. Pure functions (no Streamlit calls)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import formatting
from visualization import tokens
from visualization.theme import PRIMARY_DISCRETE, PRIMARY_SCALE, empty_figure, styled


def _segment_map_or_sequence(values: object) -> dict:
    """Keyword args painting a segment column with stable hues (pure helper)."""
    mapped = tokens.segment_map(values)
    if mapped:
        return {"color_discrete_map": mapped}
    return {"color_discrete_sequence": PRIMARY_DISCRETE}


def _rgba(hex_color: str, alpha: float) -> str:
    """Translucent form of a palette hex for Sankey links (non-hex passthrough)."""
    value = str(hex_color).lstrip("#")
    if len(value) != 6:
        return str(hex_color)
    red, green, blue = (int(value[index : index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


@styled
def segment_distribution_bar(profile: pd.DataFrame, segment_col: str = "segment") -> px.Figure:
    """Customer counts per segment, largest first, in segment hues.

    Values ride on the bars and the legend names every segment, so both axes
    stay clean: no y-axis, no x tick labels. The shared theme still supplies
    fonts, grid, and margins.
    """
    if profile.empty:
        return empty_figure("Customers per segment")
    col = segment_col if segment_col in profile.columns else profile.columns[0]
    fig = px.bar(
        profile,
        x=col,
        y="customers",
        color=col,
        text="customers",
        **_segment_map_or_sequence(profile[col]),
        title="Customers per segment",
        labels={col: "Segment", "customers": "Customers"},
    )
    fig.update_traces(texttemplate="%{y:,}", textposition="outside", textfont_size=13)
    top = float(profile["customers"].max())
    fig.update_yaxes(visible=False, range=[0, top * 1.12 if top > 0 else 1])
    fig.update_xaxes(title_text="", showticklabels=False)
    fig.update_layout(bargap=0.1)
    return fig


@styled
def revenue_by_segment_bar(profile: pd.DataFrame, segment_col: str = "segment") -> px.Figure:
    """Revenue contribution per segment, in segment hues.

    Values ride on the bars and the legend names every segment, so both axes
    stay clean: no y-axis, no x tick labels. The shared theme still supplies
    fonts, grid, and margins.
    """
    if profile.empty:
        return empty_figure("Revenue per segment")
    col = segment_col if segment_col in profile.columns else profile.columns[0]
    fig = px.bar(
        profile,
        x=col,
        y="revenue",
        color=col,
        text="revenue",
        **_segment_map_or_sequence(profile[col]),
        title="Revenue per segment",
        labels={col: "Segment", "revenue": "Revenue"},
    )
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside", textfont_size=13)
    top = float(profile["revenue"].max())
    fig.update_yaxes(visible=False, range=[0, top * 1.12 if top > 0 else 1])
    fig.update_xaxes(title_text="", showticklabels=False)
    fig.update_layout(bargap=0.1)
    return fig


@styled
def segment_share_donut(profile: pd.DataFrame, segment_col: str = "segment") -> px.Figure:
    """Customer share per segment as a donut, in segment hues.

    Slices under a few percent stay thin but legible through hover values;
    the hole centers the total customer count. Segment names ride on the
    slices themselves so the figure stays legible even when a ten-entry
    legend would otherwise wrap and squeeze the ring into slivers.
    """
    if profile.empty:
        return empty_figure("Customers per segment")
    col = segment_col if segment_col in profile.columns else profile.columns[0]
    fig = px.pie(
        profile,
        names=col,
        values="customers",
        hole=0.45,
        color=col,
        **_segment_map_or_sequence(profile[col]),
        title="Customers per segment",
        labels={col: "Segment", "customers": "Customers"},
    )
    # Names ride on the slices and the legend is suppressed: with ten long
    # segment names a side/top legend wraps across the compact card and
    # squeezes the ring into unreadable slivers.
    fig.update_traces(textinfo="label+percent")
    fig.update_layout(showlegend=False)
    total = int(profile["customers"].sum())
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=f"{total:,}<br>customers",
        showarrow=False,
        font={"size": 14},
    )
    return fig


@styled(height=480)
def rfm_scatter_matrix(
    features: pd.DataFrame, color_col: str = "segment", max_points: int = 2000
) -> px.Figure:
    """Pairwise R/F/M relationships as a scatter matrix, colored by segment.

    Points are a deterministic random sample past ``max_points`` so 25k-row
    frames stay interactive; the page captions the sampling when it applies.
    """
    dims = [c for c in ("recency_days", "frequency", "monetary_value") if c in features.columns]
    if features.empty or not dims:
        return empty_figure("Pairwise RFM relationships")
    data = features.copy()
    if len(data) > max_points:
        data = data.sample(n=max_points, random_state=42).reset_index(drop=True)
    color = color_col if color_col in data.columns else None
    color_kwargs = (
        _segment_map_or_sequence(data[color])
        if color
        else {"color_discrete_sequence": PRIMARY_DISCRETE}
    )
    hover = [c for c in ("customer_id",) if c in data.columns] or None
    return px.scatter_matrix(
        data,
        dimensions=dims,
        color=color,
        **color_kwargs,
        hover_data=hover,
        opacity=0.6,
        title="Pairwise RFM relationships",
        labels={
            "recency_days": "Recency (days)",
            "frequency": "Frequency",
            "monetary_value": "Monetary value",
        },
    )


@styled(height=480)
def rfm_scatter_3d(
    features: pd.DataFrame, color_col: str = "segment", max_points: int = 2000
) -> go.Figure:
    """Recency × frequency × monetary value as a 3D scatter, colored by segment.

    Alternative to the scatter matrix: all three axes in one rotatable view, so
    value concentration across R/F/M reads as a spatial cluster instead of three
    pairwise panels. Points are a deterministic sample past ``max_points`` so
    large frames stay interactive.
    """
    dims = [c for c in ("recency_days", "frequency", "monetary_value") if c in features.columns]
    if features.empty or not dims or len(dims) < 3:
        return empty_figure("RFM relationships (3D)")
    data = features.copy()
    if len(data) > max_points:
        data = data.sample(n=max_points, random_state=42).reset_index(drop=True)
    color = color_col if color_col in data.columns else None
    color_kwargs = (
        _segment_map_or_sequence(data[color])
        if color
        else {"color_discrete_sequence": PRIMARY_DISCRETE}
    )
    hover = [c for c in ("customer_id",) if c in data.columns] or None
    return px.scatter_3d(
        data,
        x="recency_days",
        y="frequency",
        z="monetary_value",
        color=color,
        **color_kwargs,
        hover_data=hover,
        opacity=0.85,
        title="RFM relationships (3D)",
        labels={
            "recency_days": "Recency (days)",
            "frequency": "Frequency",
            "monetary_value": "Monetary value",
        },
    )


@styled
def segment_trend_ribbon(history: pd.DataFrame, segment_col: str = "segment") -> px.Figure:
    """Stacked customer counts per segment over periods, in segment hues."""
    if history.empty:
        return empty_figure("Customer segments over time")
    col = segment_col if segment_col in history.columns else history.columns[0]
    return px.area(
        history,
        x="period",
        y="customers",
        color=col,
        **_segment_map_or_sequence(history[col]),
        title="Customer segments over time",
        labels={"period": "Period", "customers": "Customers", col: "Segment"},
    )


@styled
def segment_trend_lines(
    history: pd.DataFrame, segment_col: str = "segment", share: bool = False
) -> px.Figure:
    """One line per segment over periods, in segment hues.

    Alternative to the stacked ribbon: lines keep each cohort's own baseline, so
    a single segment's rise or fall reads directly instead of being masked by the
    stack. Best when segment counts are modest; the stacked ribbon stays the
    default for very many segments.

    With ``share=True`` every period is re-based to 100% of its own total, which
    is the honest view when the customer base itself is growing: a segment rising
    in absolute terms can still be losing ground, and only the normalised line
    shows which ones are. The share is computed here (Plotly's ``groupnorm`` does
    not normalise line traces) so a test can see it.
    """
    if history.empty:
        return empty_figure("Segment trends")
    col = segment_col if segment_col in history.columns else history.columns[0]
    frame = history
    value, title, axis_label = "customers", "Segment trends", "Customers"
    if share:
        frame = history.copy()
        totals = frame.groupby("period", observed=True)["customers"].transform("sum")
        frame["share"] = 100 * frame["customers"] / totals.where(totals > 0, 1)
        value, title, axis_label = "share", "Segment share over time", "Share of base"
    fig = px.line(
        frame,
        x="period",
        y=value,
        color=col,
        markers=True,
        **_segment_map_or_sequence(frame[col]),
        title=title,
        labels={"period": "Period", value: axis_label, col: "Segment"},
    )
    if share:
        fig.update_traces(hovertemplate="%{y:.1f}%<extra>%{fullData.name}</extra>")
        fig.update_yaxes(ticksuffix="%")
    return fig


def biggest_segment_shifts(
    history: pd.DataFrame, segment_col: str = "segment", top_n: int = 3
) -> pd.DataFrame:
    """Period-over-period moves worth calling out, largest change first.

    For every segment the change between consecutive periods is computed from the
    stored counts (``share`` optional). Returns one row per move with
    ``from_period``, ``to_period``, ``segment``, ``delta``, and ``share_delta``
    (percentage-point change in the segment's share of the base), so a page can
    annotate the periods where a cohort jumped or collapsed — a spike in "Lost
    Customers", say — instead of leaving the reader to spot it.

    Empty when there are fewer than two periods or no segment appears in both.
    """
    cols = ["from_period", "to_period", "segment", "delta", "share_delta"]
    needed = {"period", "customers", segment_col}
    if history.empty or not needed.issubset(history.columns):
        return pd.DataFrame(columns=cols)
    frame = history.copy()
    frame["period"] = frame["period"].astype(str)
    periods = sorted(frame["period"].unique())
    if len(periods) < 2:
        return pd.DataFrame(columns=cols)
    totals = frame.groupby("period", observed=True)["customers"].sum()
    wide = (
        frame.pivot_table(
            index=segment_col, columns="period", values="customers", aggfunc="sum", fill_value=0
        )
        .reindex(columns=periods, fill_value=0)
        .astype(float)
    )
    rows: list[dict] = []
    for previous, current in zip(periods, periods[1:], strict=False):
        base_prev = float(totals.get(previous, 0)) or 1.0
        base_curr = float(totals.get(current, 0)) or 1.0
        for segment in wide.index:
            before, after = wide.at[segment, previous], wide.at[segment, current]
            delta = after - before
            if delta == 0:
                continue
            rows.append(
                {
                    "from_period": previous,
                    "to_period": current,
                    "segment": str(segment),
                    "delta": float(delta),
                    "share_delta": 100 * after / base_curr - 100 * before / base_prev,
                }
            )
    if not rows:
        return pd.DataFrame(columns=cols)
    ranked = pd.DataFrame(rows)
    # Rank by the size of the move on a comparable scale: absolute-count moves for
    # the count view, percentage-point moves when shares are what matters. Both
    # are kept so the caller can label with either.
    ranked["_rank"] = ranked["delta"].abs()
    ranked = ranked.sort_values("_rank", ascending=False).drop(columns="_rank")
    return ranked.head(top_n).reset_index(drop=True)


#: Transition Sankey height: taller than a single trend line so eight quarter
#: columns and their segment bars stay separable.
TRANSITION_HEIGHT = 460


@styled(height=TRANSITION_HEIGHT)
def segment_transition_sankey(
    transitions: pd.DataFrame, title: str = "Customer transitions"
) -> go.Figure:
    """Customer moves between consecutive period ends, as a multi-stage Sankey.

    ``transitions`` holds one row per (period_from, period_to, segment_from,
    segment_to, customers). Nodes are (period, segment) pairs; links carry
    customers who held a label at both ends, so widths are real moves, never
    inferred from aggregate counts. Interior stage labels are blank: eight
    quarter columns of repeated names would collide, while the adjacent line
    chart already names every hue. Stage labels sit under the columns and hover
    names each move.
    """
    columns = ["period_from", "period_to", "segment_from", "segment_to", "customers"]
    if transitions.empty or not set(columns).issubset(transitions.columns):
        return empty_figure(title)
    frame = transitions.loc[transitions["customers"] > 0, columns].copy()
    if frame.empty:
        return empty_figure(title)
    frame["period_from"] = frame["period_from"].astype(str)
    frame["period_to"] = frame["period_to"].astype(str)
    periods = sorted(set(frame["period_from"]) | set(frame["period_to"]))
    totals = pd.concat([frame["segment_from"], frame["segment_to"]]).value_counts()
    segments = sorted(totals.index, key=lambda name: (-int(totals[name]), str(name)))
    present = {period: set() for period in periods}
    for row in frame.itertuples(index=False):
        present[row.period_from].add(row.segment_from)
        present[row.period_to].add(row.segment_to)
    positions = {period: index for index, period in enumerate(periods)}
    last_stage = len(periods) - 1

    labels: list[str] = []
    node_x: list[float] = []
    node_colors: list[str] = []
    node_hover: list[str] = []
    index: dict[tuple[str, str], int] = {}
    for period in periods:
        for segment in segments:
            if segment not in present[period]:
                continue
            index[(period, segment)] = len(labels)
            labels.append(segment if period in (periods[0], periods[-1]) else "")
            node_x.append(positions[period] / last_stage if last_stage else 0.5)
            node_colors.append(tokens.segment_color(segment))
            node_hover.append(f"{period} · {segment}")

    source: list[int] = []
    target: list[int] = []
    values: list[int] = []
    link_colors: list[str] = []
    link_hover: list[str] = []
    for row in frame.itertuples(index=False):
        source.append(index[(row.period_from, row.segment_from)])
        target.append(index[(row.period_to, row.segment_to)])
        values.append(int(row.customers))
        link_colors.append(_rgba(tokens.segment_color(row.segment_from), 0.3))
        link_hover.append(
            f"{row.segment_from} → {row.segment_to}<br>{row.period_from} → {row.period_to}"
        )

    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node={
                "label": labels,
                "x": node_x,
                "color": node_colors,
                "pad": 14,
                "thickness": 14,
                "line": {"width": 0},
                "customdata": node_hover,
                "hovertemplate": "%{customdata}<extra></extra>",
            },
            link={
                "source": source,
                "target": target,
                "value": values,
                "color": link_colors,
                "customdata": link_hover,
                "hovertemplate": "%{customdata}<br>%{value:,} customers<extra></extra>",
            },
            textfont={"size": 10},
        )
    )
    fig.update_layout(title_text=title)
    for period in periods:
        fig.add_annotation(
            x=positions[period] / last_stage if last_stage else 0.5,
            y=-0.07,
            xref="paper",
            yref="paper",
            text=period,
            showarrow=False,
            yanchor="top",
            font={"size": 10, "color": tokens.color("muted")},
        )
    return fig


#: Panels per row in the small-multiples view (four keeps even 12 clusters to
#: three rows, which is what a 420px grid can show without crushing the panels).
FACET_COLUMNS = 4

#: Facet grids need more height than a single chart: up to three panel rows.
FACET_HEIGHT = 420


@styled
def segment_share_bar(
    profile: pd.DataFrame, segment_col: str = "segment", value: str = "customers"
) -> px.Figure:
    """Sorted horizontal bars per segment — the comparable alternative to a donut.

    With ten segments a donut makes mid-sized slices impossible to rank; a sorted
    bar chart puts the largest on top and every bar on a shared baseline, so
    "Needs Attention (11.1%)" versus "Cannot Lose Them (11.0%)" reads instantly.
    ``value`` selects the measure ("customers" or "revenue") so the caller can
    show count and revenue as two panels of the same shape.
    """
    if profile.empty or value not in profile.columns:
        return empty_figure("Customers per segment")
    col = segment_col if segment_col in profile.columns else profile.columns[0]
    title = "Revenue per segment" if value == "revenue" else "Customers per segment"
    frame = profile.sort_values(value, ascending=True).copy()
    total = float(frame[value].sum())
    if value == "customers":
        labels = [
            (
                f"{formatting.fmt_int(v)}  ({formatting.fmt_pct100(100 * v / total)})"
                if total
                else formatting.fmt_int(v)
            )
            for v in frame[value]
        ]
    else:
        labels = [
            (
                f"{formatting.fmt_money_compact(v)}  ({formatting.fmt_pct100(100 * v / total)})"
                if total
                else formatting.fmt_money_compact(v)
            )
            for v in frame[value]
        ]
    fig = px.bar(
        frame,
        x=value,
        y=col,
        orientation="h",
        color=col,
        text=labels,
        **_segment_map_or_sequence(frame[col]),
        title=title,
        labels={value: title.split()[0], col: "Segment"},
    )
    fig.update_traces(textposition="outside", textfont_size=11, cliponaxis=False)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(title_text="")
    return fig


#: Segments below this share of customers fold into an "Other" bucket when the
#: caller opts in, so a long ten-segment palette does not drown the few that
#: matter. Only used when the caller asks for grouping.
SEGMENT_OTHER_THRESHOLD = 5.0

#: Label for the folded tail of a grouped segment chart.
SEGMENT_OTHER_LABEL = "Other segments"


def group_small_segments(
    profile: pd.DataFrame, segment_col: str = "segment", min_share: float = SEGMENT_OTHER_THRESHOLD
) -> pd.DataFrame:
    """Fold segments under ``min_share`` percent of customers into one "Other" row.

    Purely a display transform: customer and revenue totals are preserved, so the
    grouped frame sums to the same base as the input. Returns the input unchanged
    when nothing falls below the threshold (or when folding would not reduce the
    count). Callers offer the ungrouped table next to the chart for drill-down.
    """
    if profile.empty or segment_col not in profile.columns or "customers" not in profile.columns:
        return profile
    total = float(profile["customers"].sum())
    if not total:
        return profile
    small = profile[profile["customers"] / total * 100 < min_share]
    if len(small) < 2:  # nothing meaningful to fold
        return profile
    big = profile[~profile.index.isin(small.index)]
    folded = {col: small[col].sum() for col in profile.columns if col != segment_col}
    folded[segment_col] = SEGMENT_OTHER_LABEL
    grouped = pd.concat([big, pd.DataFrame([folded])], ignore_index=True)
    if "customer_share" in grouped.columns:
        grouped["customer_share"] = (grouped["customers"] / total * 100).round(2)
    if "revenue_share" in grouped.columns and "revenue" in grouped.columns:
        rev_total = float(grouped["revenue"].sum())
        grouped["revenue_share"] = (
            (grouped["revenue"] / rev_total * 100).round(2) if rev_total else 0.0
        )
    return grouped.sort_values("customers", ascending=False).reset_index(drop=True)


@styled
def segment_trend_share(history: pd.DataFrame, segment_col: str = "segment") -> px.Figure:
    """Composition over time: every period re-based to 100% of its own total.

    The stacked area answers "how many"; this answers "what share". It is the
    more explainable view whenever the base itself grows, because segments rising
    in absolute terms can still be losing ground — only the normalised view shows
    which ones are.

    Plotly's ``groupnorm`` does not normalise area traces (verified on 7.1), so
    the share is computed here, where a test can also see it. The y-axis carries
    a percent suffix because the shared theme drops axis titles.
    """
    if history.empty:
        return empty_figure("Segment share over time")
    col = segment_col if segment_col in history.columns else history.columns[0]
    frame = history.copy()
    totals = frame.groupby("period", observed=True)["customers"].transform("sum")
    frame["share"] = 100 * frame["customers"] / totals.where(totals > 0, 1)
    fig = px.area(
        frame,
        x="period",
        y="share",
        color=col,
        **_segment_map_or_sequence(frame[col]),
        title="Segment share over time",
        labels={"period": "Period", "share": "Share of base", col: "Segment"},
    )
    fig.update_traces(hovertemplate="%{y:.1f}%<extra>%{fullData.name}</extra>")
    fig.update_yaxes(ticksuffix="%")
    return fig


@styled(height=FACET_HEIGHT)
def segment_trend_facets(history: pd.DataFrame, segment_col: str = "segment") -> px.Figure:
    """Small multiples: one panel per segment, each with its own y-axis.

    The most explainable trend view — no segment hides behind another and none is
    flattened by a bigger neighbour's scale. Panels are titled with the segment
    name, so the shared legend is redundant and dropped.
    """
    if history.empty:
        return empty_figure("Segment trends (one panel per segment)")
    col = segment_col if segment_col in history.columns else history.columns[0]
    fig = px.area(
        history,
        x="period",
        y="customers",
        color=col,
        facet_col=col,
        facet_col_wrap=FACET_COLUMNS,
        **_segment_map_or_sequence(history[col]),
        title="Segment trends (one panel per segment)",
        labels={"period": "Period", "customers": "Customers", col: "Segment"},
    )
    fig.for_each_annotation(
        lambda annotation: annotation.update(text=annotation.text.split("=")[-1])
    )
    fig.update_yaxes(matches=None)
    fig.update_layout(showlegend=False)
    return fig


@styled
def segment_trend_heatmap(history: pd.DataFrame, segment_col: str = "segment") -> px.Figure:
    """Period × segment grid: colour reads the count, so growth shows as a band.

    Every period and segment fits at once, and rows are ordered by total size so
    the segments that matter sit on top. Best for "which segment changed" rather
    than for reading exact values.
    """
    if history.empty:
        return empty_figure("Segment activity by period")
    col = segment_col if segment_col in history.columns else history.columns[0]
    pivot = history.pivot_table(
        index=col, columns="period", values="customers", aggfunc="sum", fill_value=0
    )
    if pivot.empty:
        return empty_figure("Segment activity by period")
    pivot = pivot.reindex(pivot.sum(axis=1).sort_values(ascending=False).index)
    fig = px.imshow(
        pivot.to_numpy().tolist(),
        x=[str(period) for period in pivot.columns],
        y=[str(segment) for segment in pivot.index],
        color_continuous_scale=tokens.sequential_scale(),
        text_auto=True,
        aspect="auto",
        title="Segment activity by period",
        labels={"x": "Period", "y": "Segment", "color": "Customers"},
    )
    fig.update_traces(textfont_size=10)
    return fig


@styled
def rfm_parallel_plot(features: pd.DataFrame) -> go.Figure:
    """Recency, frequency, and monetary value as parallel coordinates.

    One quiet line per customer, colored by monetary value: value
    concentration across recency and frequency reads as color bands.
    Replaces the two overplotted 25k-point scatters with a single view.
    """
    dims = [c for c in ("recency_days", "frequency", "monetary_value") if c in features.columns]
    if features.empty or not dims:
        return empty_figure("Recency, frequency and value")
    data = features[dims].apply(pd.to_numeric, errors="coerce").dropna()
    if data.empty:
        return empty_figure("Recency, frequency and value")
    color = data["monetary_value"] if "monetary_value" in data.columns else data[dims[-1]]
    fig = go.Figure(
        go.Parcoords(
            dimensions=[{"label": c.replace("_", " "), "values": data[c]} for c in dims],
            line={
                "color": color,
                "colorscale": PRIMARY_SCALE,
                "showscale": True,
                "colorbar": {"title": "Value"},
            },
            labelangle=0,
        )
    )
    fig.update_layout(title="Recency, frequency and value")
    return fig
