"""Theme preference: resolve, apply, and toggle between the two skins.

The preference lives in ``st.session_state["theme_mode"]`` as the radio's
display value (``"Light"``/``"Dark"``). It is deliberately *outside* the data
pipeline contract in ``services/state_service.py``: ``initialize`` and
``reset`` only touch contract keys, so the appearance choice survives data
resets. ``resolve_theme_mode`` is pure over any mapping and fully tested.
"""

from __future__ import annotations

from collections.abc import Mapping

import streamlit as st

from visualization import tokens
from visualization.theme import load_theme

#: Session key holding the radio's display value.
THEME_KEY = "theme_mode"

#: Radio labels, in order. The first is the default before any interaction.
THEME_OPTIONS = ("Light", "Dark")

_OPTION_TO_MODE = {"Light": tokens.MODE_LIGHT, "Dark": tokens.MODE_DARK}


def resolve_theme_mode(state: Mapping) -> str:
    """Map stored preference (or garbage, or nothing) onto a valid mode."""
    stored = state.get(THEME_KEY, THEME_OPTIONS[0])
    if stored in _OPTION_TO_MODE:
        return _OPTION_TO_MODE[stored]
    if stored in tokens.MODES:
        return stored
    return tokens.DEFAULT_MODE


def ensure_theme() -> str:
    """Inject the stylesheet for the stored preference. Returns the mode."""
    return load_theme(resolve_theme_mode(st.session_state))


def theme_toggle() -> str:
    """Sidebar radio for the appearance. Returns the newly selected mode.

    The selection takes effect through ``ensure_theme`` at the top of the next
    run (Streamlit reruns on widget input), so this stays a pure widget.
    """
    st.radio("Appearance", THEME_OPTIONS, key=THEME_KEY, horizontal=True)
    return resolve_theme_mode(st.session_state)
