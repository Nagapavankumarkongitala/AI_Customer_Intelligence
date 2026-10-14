"""Design-token invariants: palette shape, semantic coverage, and contrast.

These are the guards that make "the same thing is always the same colour" a
testable property rather than a convention: every segment, risk band, and
retention action the pipeline can emit is checked against the code that emits it.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from analytics.rfm import default_naming_rules
from retention_engine.strategy_mapper import load_matrix
from segmentation.segment_naming import GRID
from visualization import tokens

HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")

#: The palette the spec brief prescribes for the default light surface (§3).
DOCX_LIGHT = {
    "primary": "#2563EB",
    "secondary": "#06B6D4",
    "positive": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "critical": "#991B1B",
    "purple": "#8B5CF6",
    "background": "#F6F8FC",
    "surface": "#FFFFFF",
    "border": "#E2E8F0",
    "text": "#0F172A",
    "muted": "#64748B",
}


def _is_hex(value: str) -> bool:
    return bool(HEX.match(value))


# --------------------------------------------------------------------------- #
# Palette shape.
# --------------------------------------------------------------------------- #


def test_palettes_share_keys_with_unique_values() -> None:
    """Identical key sets and unique values per palette.

    Uniqueness is what lets the dark-mode re-map be a well-defined lookup: one
    colour value maps to exactly one semantic name, so nothing maps ambiguously.
    """
    assert set(tokens.LIGHT) == set(tokens.DARK)
    light_colors = {v for k, v in tokens.LIGHT.items() if not k.startswith("shadow_")}
    dark_colors = {v for k, v in tokens.DARK.items() if not k.startswith("shadow_")}
    assert not (light_colors & dark_colors), "palettes share a colour value"
    segments = {v.upper() for v in tokens.SEGMENT_COLORS.values()}
    assert not (segments & {v.upper() for v in (*light_colors, *dark_colors)}), (
        "segment hues must stay disjoint from palette values so the dark-mode "
        "re-map never rewrites a segment identity"
    )
    dark_segments = {v.upper() for v in tokens.SEGMENT_COLORS_DARK.values()}
    assert set(tokens.SEGMENT_COLORS_DARK) == set(tokens.SEGMENT_COLORS)
    assert len(dark_segments) == len(tokens.SEGMENT_COLORS_DARK), "dark hues collide"
    assert not (dark_segments & {v.upper() for v in (*light_colors, *dark_colors)}), (
        "dark segment hues must stay disjoint from palette values so the "
        "reverse re-map stays a well-defined lookup"
    )
    for mode, palette in tokens.PALETTES.items():
        colors = [v for k, v in palette.items() if not k.startswith("shadow_")]
        assert len(set(colors)) == len(colors), f"{mode} reuses a colour value"
        for key, value in palette.items():
            if not key.startswith("shadow_"):
                assert _is_hex(value), f"{mode}.{key} is not a #RRGGBB colour: {value!r}"


def test_light_palette_matches_brief() -> None:
    """The light palette carries the exact hexes the spec brief recommends."""
    for key, value in DOCX_LIGHT.items():
        assert tokens.LIGHT[key].upper() == value.upper(), f"light.{key} drifted"


def test_mode_state_round_trips() -> None:
    """The active mode switches and rejects nonsense back to the default."""
    original = tokens.active_mode()
    try:
        assert tokens.set_active_mode(tokens.MODE_DARK) == tokens.MODE_DARK
        assert tokens.palette() is tokens.DARK
        assert tokens.set_active_mode("chartreuse") == tokens.DEFAULT_MODE
        assert tokens.active_mode() == tokens.DEFAULT_MODE
    finally:
        tokens.set_active_mode(original)
    assert tokens.DEFAULT_MODE == tokens.MODE_LIGHT


def test_resolvers_reject_unknown_keys() -> None:
    """Unknown token lookups raise rather than silently inventing a colour."""
    for call in (lambda: tokens.color("nope"), lambda: tokens.shadow("nope")):
        try:
            call()
        except KeyError:
            continue
        raise AssertionError("unknown token did not raise")


# --------------------------------------------------------------------------- #
# Semantic coverage.
# --------------------------------------------------------------------------- #


def test_segment_colors_cover_every_pipeline_name() -> None:
    """RFM/LRFM vocabulary, AI cluster grid, and the unassigned fallback all map."""
    rfm_names = {name for name, _ in default_naming_rules()}
    grid_names = set(GRID.values())
    vocabulary = rfm_names | grid_names | {"Unsegmented"}
    missing = vocabulary - set(tokens.SEGMENT_COLORS)
    assert not missing, f"segments without a colour: {sorted(missing)}"

    values = list(tokens.SEGMENT_COLORS.values())
    assert len(set(values)) == len(values), "two segments share a colour"
    for name in vocabulary:
        assert tokens.segment_color(name) == tokens.SEGMENT_COLORS[name]


def test_segment_color_handles_collision_suffix_and_unknown() -> None:
    """Letter-suffixed collisions resolve; unknown labels degrade to neutral.

    ``propose_names`` appends ' A'..' Z' when two clusters land on the same grid
    cell, so the colour lookup must survive that without a per-page special case.
    """
    assert tokens.segment_color("Premium Active B") == tokens.SEGMENT_COLORS["Premium Active"]
    assert tokens.segment_color("  Champions  ") == tokens.SEGMENT_COLORS["Champions"]
    assert tokens.segment_color("Not A Real Segment") == tokens.LIGHT["muted_soft"]
    assert tokens.segment_color("Champions", tokens.MODE_DARK) == (
        tokens.SEGMENT_COLORS_DARK["Champions"]
    )
    assert tokens.segment_color("Premium Active B", tokens.MODE_DARK) == (
        tokens.SEGMENT_COLORS_DARK["Premium Active"]
    )
    assert tokens.segment_color("Champions", tokens.MODE_LIGHT) == (
        tokens.SEGMENT_COLORS["Champions"]
    )


def test_segment_and_action_maps_are_all_or_nothing() -> None:
    """Maps resolve fully or refuse — partial maps would leak a third palette."""
    assert tokens.segment_map(["Champions", "Lost Customers"]) == {
        "Champions": tokens.SEGMENT_COLORS["Champions"],
        "Lost Customers": tokens.SEGMENT_COLORS["Lost Customers"],
    }
    assert tokens.segment_map(["Premium Active B"]) == {
        "Premium Active B": tokens.SEGMENT_COLORS["Premium Active"]
    }
    assert tokens.segment_map(["Champions", "Mystery"]) is None
    assert tokens.segment_map([]) is None
    assert tokens.segment_map(None) is None

    assert tokens.action_map(["Monitor"]) == {"Monitor": tokens.color("muted_soft")}
    assert tokens.action_map(["Monitor", "Nope"]) is None
    assert tokens.action_map([]) is None
    dark_map = tokens.segment_map(["Champions", "Lost Customers"], tokens.MODE_DARK)
    assert dark_map == {
        "Champions": tokens.SEGMENT_COLORS_DARK["Champions"],
        "Lost Customers": tokens.SEGMENT_COLORS_DARK["Lost Customers"],
    }
    assert tokens.segment_map(5) is None
    assert tokens.action_map("Monitor") == {"Monitor": tokens.color("muted_soft")}


def test_readable_ink_and_bad_hex() -> None:
    """Ink picks the legible side; malformed hex raises instead of guessing."""
    assert tokens.readable_ink("#FFFFFF") == tokens.LIGHT["text"]
    assert tokens.readable_ink("#000000") == tokens.LIGHT["surface"]
    for bad in ("zzz", "#FFF", "#GGGGGG"):
        try:
            tokens.contrast_ratio(bad, "#FFFFFF")
        except ValueError:
            continue
        raise AssertionError(f"{bad} did not raise")


def test_risk_colors_cover_configured_bands() -> None:
    """Every band in config.yaml resolves to a hue and a badge pair in both modes."""
    bands = set(
        yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))["churn"]["risk_bands"]
    )
    assert bands == set(tokens.RISK_COLOR_KEYS), "risk colour map drifted from config.yaml"
    assert bands == set(tokens.RISK_BADGE[tokens.MODE_LIGHT])
    assert bands == set(tokens.RISK_BADGE[tokens.MODE_DARK])
    for band in bands:
        assert _is_hex(tokens.risk_color(band))
        bg, fg = tokens.risk_badge(band)
        assert _is_hex(bg) and _is_hex(fg)
    # Unknown bands degrade instead of raising mid-render.
    assert _is_hex(tokens.risk_color("whatever"))


def test_action_colors_cover_strategy_matrix() -> None:
    """Every strategy action, including the mapper's fallback, has a colour."""
    matrix = load_matrix(Path("retention_strategies.yaml"))
    actions = {rule.action for rule in matrix.rules} | {"Monitor"}
    assert actions == set(tokens.ACTION_COLOR_KEYS), "action colour map drifted from the matrix"
    for action in actions:
        assert _is_hex(tokens.action_color(action))
    assert _is_hex(tokens.action_color("Unmapped action"))


def test_scales_and_colorway_only_use_palette_values() -> None:
    """Sequences, scales, and the colourway are built from palette keys, not literals."""
    palette_values = set(tokens.LIGHT.values()) | set(tokens.DARK.values())
    for mode in tokens.MODES:
        for color in (
            *tokens.colorway(mode),
            *tokens.discrete_colors(mode),
            *tokens.sequential_scale(mode),
        ):
            assert color in palette_values, f"{color} is not a palette value"


def test_tone_badges_cover_insight_tone() -> None:
    """Insight-box tones resolve to legible pairs in both modes."""
    tones = ("info", "positive", "warning", "critical")
    for mode in tokens.MODES:
        assert tones == tuple(tokens.TONE_BADGE[mode])
        for tone in tones:
            bg, fg = tokens.tone_badge(tone, mode)
            assert _is_hex(bg) and _is_hex(fg)


# --------------------------------------------------------------------------- #
# Contrast.
# --------------------------------------------------------------------------- #


def test_light_contrast_meets_floor() -> None:
    """Computed WCAG ratios, not eyeballed: body ink ≥ 4.5:1, data hues ≥ 3:1."""
    surface = tokens.LIGHT["surface"]
    assert tokens.contrast_ratio(tokens.LIGHT["text"], surface) >= 7.0
    assert tokens.contrast_ratio(tokens.LIGHT["muted"], surface) >= 4.5
    assert tokens.contrast_ratio(tokens.LIGHT["primary"], surface) >= 4.5

    weak = {
        name: round(tokens.contrast_ratio(hex_color, surface), 2)
        for name, hex_color in tokens.SEGMENT_COLORS.items()
        if tokens.contrast_ratio(hex_color, surface) < 3.0
    }
    assert not weak, f"segment hues too faint on the light surface: {weak}"


def test_dark_contrast_meets_floor() -> None:
    """The dark skin clears the same bars against its own surface."""
    surface = tokens.DARK["surface"]
    assert tokens.contrast_ratio(tokens.DARK["text"], surface) >= 7.0
    assert tokens.contrast_ratio(tokens.DARK["muted"], surface) >= 4.5
    assert tokens.contrast_ratio(tokens.DARK["muted_soft"], surface) >= 4.5
    assert tokens.contrast_ratio(tokens.DARK["primary"], surface) >= 4.5
    assert tokens.contrast_ratio(tokens.DARK["critical"], surface) >= 4.5

    weak = {
        name: round(tokens.contrast_ratio(hex_color, surface), 2)
        for name, hex_color in tokens.SEGMENT_COLORS_DARK.items()
        if tokens.contrast_ratio(hex_color, surface) < 3.0
    }
    assert not weak, f"segment hues too faint on the dark surface: {weak}"


def test_segment_remap_covers_every_name_both_ways() -> None:
    """The mode re-map carries each segment hue to its counterpart and back."""
    from visualization import theme

    forward = theme._remap_table(tokens.MODE_DARK)
    for name, light_hex in tokens.SEGMENT_COLORS.items():
        dark_hex = tokens.SEGMENT_COLORS_DARK[name]
        if light_hex == dark_hex:
            continue
        assert forward[light_hex.upper()] == dark_hex, name

    backward = theme._remap_table(tokens.MODE_LIGHT)
    for name, dark_hex in tokens.SEGMENT_COLORS_DARK.items():
        light_hex = tokens.SEGMENT_COLORS[name]
        if light_hex == dark_hex:
            continue
        assert backward[dark_hex.upper()] == light_hex, name


def test_risk_badges_are_legible() -> None:
    """Badge foregrounds clear 4.5:1 against their own backgrounds."""
    for mode in tokens.MODES:
        for band, (bg, fg) in tokens.RISK_BADGE[mode].items():
            ratio = tokens.contrast_ratio(fg, bg)
            assert ratio >= 4.5, f"{mode}/{band} badge contrast {ratio:.2f} is too low"


def test_with_alpha_builds_rgba_from_tokens() -> None:
    """Translucent fills are derived, so rgba literals cannot sneak back in."""
    assert tokens.with_alpha("#2563EB", 0.25) == "rgba(37,99,235,0.25)"
    assert tokens.with_alpha("#FFFFFF", 1) == "rgba(255,255,255,1)"
    try:
        tokens.with_alpha("blue", 0.5)
    except ValueError:
        return
    raise AssertionError("non-hex input did not raise")


def test_contrast_maths_matches_wcag_reference() -> None:
    """Spot-check the luminance implementation against known WCAG values."""
    assert tokens.contrast_ratio("#FFFFFF", "#000000") == 21.0
    assert round(tokens.relative_luminance("#FFFFFF"), 3) == 1.0
    assert round(tokens.relative_luminance("#000000"), 3) == 0.0
    assert tokens.contrast_ratio("#000000", "#000000") == 1.0
