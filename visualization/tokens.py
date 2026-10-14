"""Design tokens: one source of truth for colour, type, space, and depth.

Two palettes ship side by side so the console can be re-skinned without editing a
page or a figure factory:

* ``LIGHT`` — the default professional analytics surface (spec §3).
* ``DARK`` — the retained navy/cyan console, kept selectable.

Rules this module exists to enforce:

1. ``LIGHT`` and ``DARK`` expose an identical key set with unique values, so a
   colour can never have two definitions and a value can always be traced back
   to exactly one semantic name (asserted by ``tests/test_tokens.py``).
2. Semantic maps reference palette *keys*, so a risk band or retention action
   follows the active accent system instead of being re-picked per page.
3. ``SEGMENT_COLORS`` / ``SEGMENT_COLORS_DARK`` are literal hue tables covering
   **every** segment name the pipeline can emit — RFM/LRFM share one vocabulary
   (``analytics/rfm.py``) and AI clusters another (``segmentation/segment_naming.py``)
   — so a "Champions" segment is the same colour on every page. The dark table
   carries the same hue families lightened onto navy (identities survive the
   re-skin; per-mode hues are the documented exception to literal invariance).
4. Nothing here imports Streamlit. The active mode is module-level state, which
   keeps figure factories free of session state (spec §17).
"""

from __future__ import annotations

import re
from typing import Final

MODE_LIGHT: Final = "light"
MODE_DARK: Final = "dark"
MODES: Final[tuple[str, ...]] = (MODE_LIGHT, MODE_DARK)
DEFAULT_MODE: Final = MODE_LIGHT

# --------------------------------------------------------------------------- #
# Palettes. Identical keys, unique values (tests enforce both).
# --------------------------------------------------------------------------- #

LIGHT: Final[dict[str, str]] = {
    # Accent system.
    "primary": "#2563EB",
    "primary_100": "#DBEAFE",
    "primary_300": "#93C5FD",
    "primary_500": "#3B82F6",
    "primary_700": "#1D4ED8",
    "primary_900": "#1E3A8A",
    "secondary": "#06B6D4",
    "secondary_dim": "#0E7490",
    "positive": "#10B981",
    "warning": "#F59E0B",
    "danger": "#EF4444",
    "critical": "#991B1B",
    "purple": "#8B5CF6",
    "teal": "#0F766E",
    # Ink.
    "text": "#0F172A",
    "muted": "#64748B",
    "muted_soft": "#94A3B8",
    # Surfaces and chrome.
    "background": "#F6F8FC",
    "surface": "#FFFFFF",
    "surface_alt": "#F8FAFC",
    "sidebar": "#E9EFFB",
    "hover": "#EFF6FF",
    "border": "#E2E8F0",
    "grid": "#EEF2F7",
    # Depth.
    "shadow_card": "0 4px 18px rgba(15, 23, 42, 0.06)",
    "shadow_raised": "0 8px 24px rgba(15, 23, 42, 0.10)",
    "shadow_hover": "0 10px 28px rgba(15, 23, 42, 0.12)",
}

DARK: Final[dict[str, str]] = {
    # Accent system: deliberately brighter than the light accents, because
    # dark surfaces need more luminance for the same contrast ratios.
    "primary": "#22D3EE",
    "primary_100": "#1E537F",
    "primary_300": "#2A6A9C",
    "primary_500": "#0891B2",
    "primary_700": "#38BDF8",
    "primary_900": "#A5F3FC",
    "secondary": "#7DD3FC",
    "secondary_dim": "#0284C7",
    "positive": "#3DDC84",
    "warning": "#E8A838",
    "danger": "#FF4D4D",
    "critical": "#F25555",
    "purple": "#A78BFA",
    "teal": "#2DD4BF",
    # Ink.
    "text": "#E8EEF6",
    "muted": "#8B98AD",
    "muted_soft": "#7588A3",
    # Surfaces and chrome.
    "background": "#070B14",
    "surface": "#0D1930",
    "surface_alt": "#0B1426",
    "sidebar": "#08101F",
    "hover": "#12203A",
    "border": "#1F2A44",
    "grid": "#16213A",
    # Depth.
    "shadow_card": "0 8px 32px rgba(0, 0, 0, 0.45)",
    "shadow_raised": "0 12px 40px rgba(0, 0, 0, 0.55)",
    "shadow_hover": "0 14px 44px rgba(0, 0, 0, 0.60)",
}

PALETTES: Final[dict[str, dict[str, str]]] = {MODE_LIGHT: LIGHT, MODE_DARK: DARK}

#: Colour-bearing keys only; "shadow_*" values are box-shadow recipes, not colours.
COLOR_KEYS: Final[tuple[str, ...]] = tuple(k for k in LIGHT if not k.startswith("shadow_"))

# --------------------------------------------------------------------------- #
# Type, space, and form.
# --------------------------------------------------------------------------- #

FONT_STACK: Final = "Inter, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"

TYPE_SCALE: Final[dict[str, str]] = {
    "display": "2.0rem",
    "h1": "1.6rem",
    "h2": "1.25rem",
    "h3": "1.05rem",
    "body": "0.9375rem",
    "small": "0.8125rem",
    "micro": "0.75rem",
}

WEIGHTS: Final[dict[str, int]] = {"regular": 400, "medium": 500, "semibold": 600, "bold": 700}

SPACING: Final[dict[str, str]] = {
    "xs": "0.25rem",
    "sm": "0.5rem",
    "md": "1rem",
    "lg": "1.5rem",
    "xl": "2rem",
    "xxl": "3rem",
}

RADIUS: Final[dict[str, str]] = {"sm": "8px", "md": "12px", "lg": "16px", "pill": "999px"}

#: Sidebar rail width. Streamlit pins the rail with an inline ``width: 300px``
#: (which beats every stylesheet rule), so this ships ``!important`` like the
#: standard sidebar-narrowing recipe. Trade-off: the native drag-resize handle
#: can no longer stretch the rail at runtime.
SIDEBAR_WIDTH: Final = "15rem"

#: Content cap for the page canvas; wide enough for a 5-card KPI row.
CONTENT_MAX_WIDTH: Final = "110rem"

#: Chart height everywhere. Compact enough for dense tabbed layouts while
#: keeping legends, titles, and axis labels legible (spec §6). The ``compact``
#: theme flag additionally tightens margins and type for single-screen groups.
CHART_HEIGHT: Final = 300

# --------------------------------------------------------------------------- #
# Semantic maps.
# --------------------------------------------------------------------------- #

#: Segment hues are literal and mode-invariant so an identity survives re-skins.
#: Two families: value-blue/teal for the RFM/LRFM vocabulary, a violet ramp for
#: AI clusters (purple signals model output, spec §3). Lapsed segments go grey,
#: which reads as inactive rather than as a fifth competing hue.
SEGMENT_COLORS: Final[dict[str, str]] = {
    # RFM / LRFM business segments (analytics/rfm.py::default_naming_rules).
    "Champions": "#1E40AF",
    "Loyal Customers": "#1D64D8",
    "Potential Loyalists": "#2F7DE1",
    "New Customers": "#0B8AC0",
    "Promising": "#0E7C8C",
    "Needs Attention": "#7C3AED",
    "At Risk": "#C026D3",
    "Cannot Lose Them": "#A21CAF",
    "Hibernating": "#54657A",
    "Lost Customers": "#3B4A5E",
    # AI cluster names (segmentation/segment_naming.py::GRID) — a violet ramp
    # whose darkness tracks the value tier and whose step tracks activity.
    "Premium Active": "#4C1D95",
    "Premium Steady": "#5B21B6",
    "Premium Dormant": "#6D28D9",
    "Core Active": "#7E22CE",
    "Core Steady": "#9333EA",
    "Core Dormant": "#9C36E8",
    "Light Active": "#A855F7",
    "Light Steady": "#B453EA",
    "Light Dormant": "#B968F0",
    # Fallback for rows the pipeline could not assign.
    "Unsegmented": "#7C8AA0",
}

#: Segment hues for the dark surface: the same hue families lightened onto
#: navy, tier ordering preserved (Premium darkest, Light lightest). Every
#: entry clears 3:1 against the dark surface; the identical key set keeps the
#: light→dark re-map well-defined.
SEGMENT_COLORS_DARK: Final[dict[str, str]] = {
    "Champions": "#556FC2",
    "Loyal Customers": "#2C6FDB",
    "Potential Loyalists": "#2F7DE1",
    "New Customers": "#0B8AC0",
    "Promising": "#127E8E",
    "Needs Attention": "#8A4EEF",
    "At Risk": "#C026D3",
    "Cannot Lose Them": "#B242BD",
    "Hibernating": "#657487",
    "Lost Customers": "#687483",
    "Premium Active": "#8363B6",
    "Premium Steady": "#875CC9",
    "Premium Dormant": "#8A53E1",
    "Core Active": "#984ED8",
    "Core Steady": "#9B43EC",
    "Core Dormant": "#A13FE9",
    "Light Active": "#A855F7",
    "Light Steady": "#B453EA",
    "Light Dormant": "#B968F0",
    "Unsegmented": "#7C8AA0",
}

#: Palette keys per risk band, so band colour follows the accent system.
RISK_COLOR_KEYS: Final[dict[str, str]] = {
    "low": "positive",
    "medium": "warning",
    "high": "danger",
    "critical": "critical",
}

#: Palette keys per retention action (retention_strategies.yaml). Urgent actions
#: take risk hues, cooperative actions take healthy hues, routine takes grey.
ACTION_COLOR_KEYS: Final[dict[str, str]] = {
    "Personal outreach (VIP intervention)": "critical",
    "Personalized reactivation": "danger",
    "Automated low-cost re-engagement": "warning",
    "Recognition and referral invitation": "positive",
    "Cross-sell and early access": "primary",
    "Routine nurture and monitoring": "muted",
    # Fallback emitted by retention_engine.strategy_mapper for uncovered cells.
    "Monitor": "muted_soft",
}

#: Categorical sequence for unlabelled series (age bands, gender, ...). Deliberately
#: excludes red/amber/green: those carry risk meaning and must not be decorative.
PRIMARY_DISCRETE_KEYS: Final[tuple[str, ...]] = (
    "primary",
    "secondary",
    "teal",
    "purple",
    "secondary_dim",
    "primary_500",
    "muted",
    "muted_soft",
)

#: Sequential scale for heatmaps and magnitude bars.
PRIMARY_SCALE_KEYS: Final[tuple[str, ...]] = (
    "primary_100",
    "primary_300",
    "primary_500",
    "primary_700",
)

#: Default colourway for figures with no explicit colour mapping (spec §6).
PLOTLY_COLORWAY_KEYS: Final[tuple[str, ...]] = (
    "primary",
    "secondary",
    "positive",
    "warning",
    "purple",
    "danger",
    "teal",
    "muted",
)

#: Risk badge pairs (background, foreground) per mode — spec §14.
RISK_BADGE: Final[dict[str, dict[str, tuple[str, str]]]] = {
    MODE_LIGHT: {
        "low": ("#DCFCE7", "#166534"),
        "medium": ("#FEF3C7", "#92400E"),
        "high": ("#FEE2E2", "#991B1B"),
        "critical": ("#991B1B", "#FFFFFF"),
    },
    MODE_DARK: {
        "low": ("#07301F", "#6EE7B7"),
        "medium": ("#3A2A08", "#FCD34D"),
        "high": ("#3F1212", "#FCA5A5"),
        "critical": ("#7F1D1D", "#FEE2E2"),
    },
}

#: Insight-box tones (spec §13) as (background, foreground) per mode.
TONE_BADGE: Final[dict[str, dict[str, tuple[str, str]]]] = {
    MODE_LIGHT: {
        "info": ("#EFF6FF", "#1D4ED8"),
        "positive": ("#ECFDF5", "#065F46"),
        "warning": ("#FFFBEB", "#92400E"),
        "critical": ("#FEF2F2", "#991B1B"),
    },
    MODE_DARK: {
        "info": ("#0B2137", "#7DD3FC"),
        "positive": ("#052E22", "#6EE7B7"),
        "warning": ("#3A2A08", "#FCD34D"),
        "critical": ("#3F1212", "#FCA5A5"),
    },
}

# --------------------------------------------------------------------------- #
# Active-mode state.
# --------------------------------------------------------------------------- #

_ACTIVE_MODE: str = DEFAULT_MODE

#: Letter suffix appended by segmentation/segment_naming.py on name collisions.
_COLLISION_SUFFIX: Final = re.compile(r" [A-Z]$")


def palette(mode: str | None = None) -> dict[str, str]:
    """Token dict for ``mode`` (defaults to the active mode)."""
    return PALETTES[_resolve_mode(mode)]


def active_mode() -> str:
    """The mode figure factories and CSS currently render in."""
    return _ACTIVE_MODE


def set_active_mode(mode: str | None) -> str:
    """Switch the active mode. Unknown or missing values fall back to the default."""
    global _ACTIVE_MODE
    _ACTIVE_MODE = mode if mode in MODES else DEFAULT_MODE
    return _ACTIVE_MODE


def _resolve_mode(mode: str | None) -> str:
    return mode if mode in MODES else _ACTIVE_MODE


def color(key: str, mode: str | None = None) -> str:
    """Palette colour by key. Raises for unknown keys instead of guessing a hue."""
    tokens = palette(mode)
    if key not in tokens:
        raise KeyError(f"unknown colour token: {key!r}")
    return tokens[key]


def shadow(name: str = "card", mode: str | None = None) -> str:
    """Box-shadow recipe by name ('card', 'raised', 'hover')."""
    key = f"shadow_{name}"
    tokens = palette(mode)
    if key not in tokens:
        raise KeyError(f"unknown shadow token: {name!r}")
    return tokens[key]


# --------------------------------------------------------------------------- #
# Semantic resolvers.
# --------------------------------------------------------------------------- #


def _segment_table(mode: str | None) -> dict[str, str]:
    """Hue table for the mode (dark variants keep identities legible on navy)."""
    return SEGMENT_COLORS_DARK if _resolve_mode(mode) == MODE_DARK else SEGMENT_COLORS


def segment_color(name: str, mode: str | None = None) -> str:
    """Colour for any segment label, in the active mode's hue table.

    Handles the letter suffixes ``propose_names`` appends on collisions
    ("Premium Active B") and falls back to the neutral swatch for labels the
    pipeline did not produce, so an unexpected value degrades instead of crashing.
    """
    table = _segment_table(mode)
    label = str(name).strip()
    if label in table:
        return table[label]
    stripped = _COLLISION_SUFFIX.sub("", label)
    if stripped in table:
        return table[stripped]
    return color("muted_soft", mode)


def risk_color(band: str, mode: str | None = None) -> str:
    """Chart hue for a churn risk band (config.yaml ``churn.risk_bands``)."""
    key = RISK_COLOR_KEYS.get(str(band).strip().lower())
    return color(key, mode) if key else color("muted_soft", mode)


def risk_badge(band: str, mode: str | None = None) -> tuple[str, str]:
    """(background, foreground) pair for a risk badge."""
    resolved = _resolve_mode(mode)
    band_key = str(band).strip().lower()
    pairs = RISK_BADGE[resolved]
    return pairs.get(band_key, pairs["low"])


def tone_badge(tone: str, mode: str | None = None) -> tuple[str, str]:
    """(background, foreground) pair for an insight-box tone."""
    pairs = TONE_BADGE[_resolve_mode(mode)]
    return pairs.get(str(tone).strip().lower(), pairs["info"])


def action_color(action: str, mode: str | None = None) -> str:
    """Chart hue for a retention action from retention_strategies.yaml."""
    key = ACTION_COLOR_KEYS.get(str(action).strip())
    return color(key, mode) if key else color("muted_soft", mode)


def segment_map(values: object, mode: str | None = None) -> dict[str, str] | None:
    """Discrete colour map when every value is known segment vocabulary.

    Returns ``None`` unless the whole set resolves (collision suffixes
    included) — a partial map would let Plotly assign the leftovers from a
    third palette, which is exactly the drift this module exists to prevent.
    Callers fall back to the neutral categorical sequence on ``None``.
    """
    table = _segment_table(mode)
    unique = {str(value).strip() for value in _as_values(values)}
    if not unique:
        return None
    mapped: dict[str, str] = {}
    for label in unique:
        if label in table:
            mapped[label] = table[label]
            continue
        stripped = _COLLISION_SUFFIX.sub("", label)
        if stripped in table:
            mapped[label] = table[stripped]
            continue
        return None
    return mapped


def action_map(values: object, mode: str | None = None) -> dict[str, str] | None:
    """Discrete colour map when every value is a known retention action."""
    unique = {str(value).strip() for value in _as_values(values)}
    if not unique:
        return None
    if any(label not in ACTION_COLOR_KEYS for label in unique):
        return None
    return {label: color(ACTION_COLOR_KEYS[label], mode) for label in unique}


def _as_values(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    try:
        return list(values)  # type: ignore[arg-type]
    except TypeError:
        return [values]


def colorway(mode: str | None = None) -> list[str]:
    """Default Plotly colourway for the mode."""
    return [color(key, mode) for key in PLOTLY_COLORWAY_KEYS]


def discrete_colors(mode: str | None = None) -> list[str]:
    """Categorical sequence for unlabelled series."""
    return [color(key, mode) for key in PRIMARY_DISCRETE_KEYS]


def sequential_scale(mode: str | None = None) -> list[str]:
    """Sequential scale for heatmaps and magnitude bars."""
    return [color(key, mode) for key in PRIMARY_SCALE_KEYS]


# --------------------------------------------------------------------------- #
# Contrast maths (shared by tests and components; no third-party colour lib).
# --------------------------------------------------------------------------- #


def relative_luminance(hex_color: str) -> float:
    """WCAG relative luminance of a ``#RRGGBB`` colour."""
    text = hex_color.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected a #RRGGBB colour, got {hex_color!r}")
    channels = [int(text[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG contrast ratio between two ``#RRGGBB`` colours."""
    lighter, darker = sorted(
        (relative_luminance(foreground), relative_luminance(background)), reverse=True
    )
    return (lighter + 0.05) / (darker + 0.05)


def with_alpha(hex_color: str, alpha: float) -> str:
    """Translucent form of a palette colour, for fills and link ribbons.

    Figures that need translucency build it from a token instead of carrying a
    literal ``rgba(...)``, so the retired-volt rgba that once hid here cannot
    come back unnoticed.
    """
    text = hex_color.strip().lstrip("#")
    if len(text) != 6:
        raise ValueError(f"expected a #RRGGBB colour, got {hex_color!r}")
    red, green, blue = (int(text[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha:g})"


def readable_ink(background: str, mode: str | None = None) -> str:
    """Pick the text colour that reads on an arbitrary background."""
    tokens = palette(mode)
    return (
        tokens["text"] if contrast_ratio(tokens["text"], background) >= 4.5 else tokens["surface"]
    )
