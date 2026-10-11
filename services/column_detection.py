"""Automatic source-to-canonical column detection and mapping validation.

Pipeline: normalize headers -> disambiguate duplicates with numeric suffixes ->
exact alias match -> fuzzy match -> validation. Each source column maps to at most
one canonical field (required fields claim first).
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from utils.constants import CANONICAL_FIELDS, COLUMN_ALIASES, REQUIRED_FIELDS
from utils.exceptions import MappingError

FUZZY_CUTOFF = 0.8
_SUFFIX_RE = re.compile(r"_(\d+)$")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def normalize_header(name: str) -> str:
    """Lowercase, trim, turn separators into single underscores: 'Customer ID' -> 'customer_id'."""
    cleaned = _NON_ALNUM_RE.sub("_", str(name).strip().lower()).strip("_")
    return re.sub(r"_+", "_", cleaned)


def deduplicate_columns(columns: list[str]) -> list[str]:
    """Make normalized headers unique by appending _2, _3, ... to repeats."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        base = normalize_header(col)
        seen[base] = seen.get(base, 0) + 1
        out.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    return out


def _match_key(normalized: str) -> str:
    """Strip a deduplication suffix so 'amount_2' still matches the 'amount' alias."""
    return _SUFFIX_RE.sub("", normalized)


def auto_map(columns: list[str]) -> MappingProposal:
    """Propose {canonical_field: source_column} for the given (already normalized) columns."""
    alias_to_field: dict[str, str] = {}
    for canonical in CANONICAL_FIELDS:
        for alias in COLUMN_ALIASES[canonical]:
            alias_to_field.setdefault(alias, canonical)
    claimed: set[str] = set()
    mapping: dict[str, str] = {}
    # Required fields first so optional fields can never steal their columns.
    for canonical in (*REQUIRED_FIELDS, *[f for f in CANONICAL_FIELDS if f not in REQUIRED_FIELDS]):
        for col in columns:
            if col in claimed:
                continue
            key = _match_key(col)
            if key in COLUMN_ALIASES[canonical]:
                mapping[canonical] = col
                claimed.add(col)
                break
        else:
            matches = difflib.get_close_matches(
                canonical,
                [_match_key(c) for c in columns if c not in claimed],
                n=1,
                cutoff=FUZZY_CUTOFF,
            )
            if not matches:
                continue
            for col in columns:
                if col not in claimed and _match_key(col) == matches[0]:
                    mapping[canonical] = col
                    claimed.add(col)
                    break
    unmapped_sources = [c for c in columns if c not in claimed]
    unmapped_canonical = [f for f in CANONICAL_FIELDS if f not in mapping]
    return MappingProposal(
        mapping=mapping,
        unmapped_sources=unmapped_sources,
        unmapped_canonical=unmapped_canonical,
    )


@dataclass
class MappingProposal:
    """Result of auto_map with gaps explicitly listed."""

    mapping: dict[str, str] = field(default_factory=dict)
    unmapped_sources: list[str] = field(default_factory=list)
    unmapped_canonical: list[str] = field(default_factory=list)

    @property
    def missing_required(self) -> list[str]:
        """Required canonical fields with no mapped source column."""
        return [f for f in REQUIRED_FIELDS if f not in self.mapping]


def validate_mapping(mapping: dict[str, str], available_columns: list[str]) -> dict[str, str]:
    """Strict validation used before analysis. Raises MappingError on any problem.

    Returns per-field 'ok' statuses when valid.
    """
    problems: list[str] = []
    missing = [f for f in REQUIRED_FIELDS if f not in mapping]
    if missing:
        problems.append(
            "Required field(s) have no mapped column: "
            + ", ".join(missing)
            + ". Correct the mapping on the Data Upload page."
        )
    unknown = [c for c in mapping.values() if c not in available_columns]
    if unknown:
        problems.append("Mapped column(s) not present in the dataset: " + ", ".join(unknown) + ".")
    seen_sources: dict[str, str] = {}
    for canonical, source in mapping.items():
        if source in seen_sources:
            problems.append(
                f"Column '{source}' is mapped to both '{seen_sources[source]}' and "
                f"'{canonical}'. Each source column may map to only one field."
            )
        seen_sources[source] = canonical
    if problems:
        raise MappingError(" ".join(problems))
    return {canonical: "ok" for canonical in mapping}
