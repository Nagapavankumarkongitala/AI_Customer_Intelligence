"""Theme preference: resolve, apply, and toggle between the two skins.

The preference lives in ``st.session_state["theme_mode"]`` as the radio's
display value (``"Light"``/``"Dark"``). It is deliberately *outside* the data
pipeline contract in ``services/state_service.py``: ``initialize`` and
``reset`` only touch contract keys, so the appearance choice survives data
resets. ``resolve_theme_mode`` is pure over any mapping and fully tested.

The radio carries ``key=`` **and** ``persist_state="session"``. Streamlit
drops page-scoped widget values on every page switch (verified live: the
choice reset to Light on navigation without it); the session scope keeps the
value for the whole session. A keyless radio was tried and rejected — without
a key the framework cannot deliver the new value before the script runs.
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
_MODE_TO_OPTION = {mode: option for option, mode in _OPTION_TO_MODE.items()}


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

    Keyed with session persistence so the choice survives page switches; the
    display index follows the stored value so first paint is already correct.
    Application happens through ``ensure_theme`` at the rail head on the
    rerun the interaction triggers.
    """
    stored_label = _stored_label()
    st.radio(
        "Appearance",
        THEME_OPTIONS,
        index=THEME_OPTIONS.index(stored_label),
        key=THEME_KEY,
        horizontal=True,
        persist_state="session",
    )
    return resolve_theme_mode(st.session_state)


def _stored_label(state: Mapping | None = None) -> str:
    """Radio display value for the stored preference (defaults to Light)."""
    source = st.session_state if state is None else state
    stored = source.get(THEME_KEY, THEME_OPTIONS[0])
    if stored in THEME_OPTIONS:
        return stored
    return _MODE_TO_OPTION.get(stored, THEME_OPTIONS[0])
