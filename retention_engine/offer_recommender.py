"""Offer catalog: human-readable descriptions for each strategy offer type."""

from __future__ import annotations

OFFER_CATALOG = {
    "Loyalty recovery": "High-touch incentive for valuable at-risk customers (e.g. personal discount, concierge callback).",
    "Early access": "Recognition perk for healthy high-value customers (previews, referral bonuses).",
    "Limited incentive": "Time-boxed discount or bundle to prompt another purchase.",
    "Cross-sell": "Adjacent-category recommendations based on past purchases.",
    "Win-back": "Low-cost automated nudge (single small incentive at most).",
    "None": "No offer. Routine content only.",
}


def describe_offer(offer_type: str) -> str:
    """One-line explanation of what an offer type means in practice."""
    return OFFER_CATALOG.get(offer_type, "Custom offer configured in the strategy matrix.")
