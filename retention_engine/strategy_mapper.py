"""Strategy matrix loader, validator, and matcher.

Matrices live outside code (YAML/JSON) so business owners can tune rules without
developers. validate_matrix proves full value-tier × risk-band coverage and
rejects ambiguous doubles only when they change outcomes — first match wins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

BANNED_CLAIM_PHRASES = ("guarantee", "will retain", "proven to retain", "will save")


@dataclass
class StrategyRule:
    """One first-match-wins rule with a fully rendered reason template."""

    name: str
    value_tiers: list[str] = field(default_factory=list)
    risk_bands: list[str] = field(default_factory=list)
    action: str = ""
    channel: str = ""
    offer_type: str = ""
    reason: str = ""

    def matches(self, tier: str, band: str) -> bool:
        """True when both the tier and the band fall in this rule's sets."""
        return tier in self.value_tiers and band in self.risk_bands


@dataclass
class StrategyMatrix:
    """Ordered rules plus the no-causal-claims disclaimer."""

    rules: list[StrategyRule] = field(default_factory=list)
    disclaimer: str = ""

    def match(self, tier: str, band: str) -> StrategyRule:
        """First matching rule; explicit fallback keeps every customer actionable."""
        for rule in self.rules:
            if rule.matches(tier, band):
                return rule
        return StrategyRule(
            name="monitor_fallback",
            value_tiers=[tier],
            risk_bands=[band],
            action="Monitor",
            channel="None",
            offer_type="None",
            reason="No strategy rule covers this combination; monitoring only.",
        )


def load_matrix(path: str | Path) -> StrategyMatrix:
    """Load a YAML or JSON strategy matrix with schema validation."""
    src = Path(path)
    if src.suffix.lower() == ".json":
        payload = json.loads(src.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(src.read_text(encoding="utf-8"))
    rules = []
    for i, raw in enumerate(payload.get("rules", [])):
        for key in (
            "name",
            "value_tiers",
            "risk_bands",
            "action",
            "channel",
            "offer_type",
            "reason",
        ):
            if key not in raw:
                raise ValueError(f"Rule #{i} is missing required key '{key}'.")
        lowered = str(raw["reason"]).lower()
        for phrase in BANNED_CLAIM_PHRASES:
            if phrase in lowered:
                raise ValueError(
                    f"Rule '{raw['name']}' claims causal impact ('{phrase}'). "
                    "Reasons must state risk/value facts only."
                )
        rules.append(
            StrategyRule(
                **{
                    k: raw[k]
                    for k in (
                        "name",
                        "value_tiers",
                        "risk_bands",
                        "action",
                        "channel",
                        "offer_type",
                        "reason",
                    )
                }
            )
        )
    if not rules:
        raise ValueError("Strategy matrix contains no rules.")
    return StrategyMatrix(rules=rules, disclaimer=str(payload.get("disclaimer", "")))


def validate_matrix(matrix: StrategyMatrix, tiers: list[str], bands: list[str]) -> dict[str, list]:
    """Report uncovered value×band cells (must be empty for production use)."""
    uncovered = [
        {"value_tier": tier, "risk_band": band}
        for tier in tiers
        for band in bands
        if not any(rule.matches(tier, band) for rule in matrix.rules)
    ]
    counts: dict[str, int] = {}
    for rule in matrix.rules:
        counts[rule.name] = sum(1 for tier in tiers for band in bands if rule.matches(tier, band))
    return {"uncovered": uncovered, "cells_per_rule": counts}
