"""Provider registry and selection.

`auto` picks whichever vendor has a key present. When both do, the order below
decides, and the one not chosen becomes the failover target — so a refusal or a
rate limit on one vendor is survivable without touching the heuristic path.
"""

from __future__ import annotations

import logging

from ...settings import TierCfg
from .anthropic_provider import AnthropicCurator
from .base import CallResult, Curator, estimate_cost, usage_from
from .openai_provider import OpenAICurator

log = logging.getLogger(__name__)

# Order used when provider: auto and both keys are present.
AUTO_ORDER = ("openai", "anthropic")


def build(tier: TierCfg, name: str) -> Curator:
    if name == "anthropic":
        return AnthropicCurator(tier.anthropic_model, tier.anthropic_effort)
    if name == "openai":
        return OpenAICurator(tier.openai_model)
    raise ValueError(f"unknown provider {name!r}")


def chain_for(tier: TierCfg) -> list[Curator]:
    """Providers to try, in order, for this tier.

    Returns an empty list when the tier is set to heuristic or nothing is
    configured — the caller treats that as 'skip this tier', not as an error.
    """
    if tier.provider == "heuristic":
        return []

    if tier.provider in ("anthropic", "openai"):
        candidate = build(tier, tier.provider)
        if candidate.is_available():
            # An explicitly chosen provider still gets a failover partner: being
            # specific about the preference should not mean losing the edition
            # when that vendor has a bad minute.
            other = "openai" if tier.provider == "anthropic" else "anthropic"
            fallback = build(tier, other)
            return [candidate, fallback] if fallback.is_available() else [candidate]
        log.warning("tier %s requests provider %s but it is unavailable",
                    tier.name, tier.provider)
        return []

    return [c for c in (build(tier, n) for n in AUTO_ORDER) if c.is_available()]


__all__ = ["Curator", "CallResult", "build", "chain_for", "estimate_cost",
           "usage_from", "AnthropicCurator", "OpenAICurator"]
