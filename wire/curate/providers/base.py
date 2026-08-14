"""The Curator protocol — the seam that keeps this model-agnostic.

Every provider implements the same three calls and returns the same plain dicts,
validated against the same schema. Nothing above this layer knows which vendor
ran, which is what makes swapping or adding one a single-file change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ...models import CurationUnavailable, TierUsage
from ...settings import Config, TierCfg


@dataclass
class CallResult:
    """A provider's answer plus what it cost."""

    data: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    cached_read_tokens: int = 0
    model: str = ""


class Curator(Protocol):
    name: str
    model: str

    def is_available(self) -> bool:
        """True when this provider has credentials and its SDK is importable."""
        ...

    def call(self, system: str, user: str, schema: dict[str, Any],
             schema_name: str, cfg: Config, tier: TierCfg) -> CallResult:
        """Run one structured-output request.

        Must raise CurationUnavailable (never a vendor exception) on any failure
        the pipeline should degrade past.
        """
        ...


# Published list prices, USD per million tokens. Used only for the cost ceiling
# and the reported spend estimate; billing is whatever the vendor says it is.
PRICES: dict[str, tuple[float, float]] = {
    "claude-fable-5": (10.0, 50.0),
    "claude-mythos-5": (10.0, 50.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

DEFAULT_PRICE = (5.0, 25.0)


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, DEFAULT_PRICE)
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def usage_from(tier: str, provider: str, results: list[CallResult]) -> TierUsage:
    if not results:
        return TierUsage(tier=tier, provider=provider, model="")
    model = results[0].model
    inp = sum(r.input_tokens for r in results)
    out = sum(r.output_tokens for r in results)
    return TierUsage(
        tier=tier,
        provider=provider,
        model=model,
        calls=len(results),
        input_tokens=inp,
        output_tokens=out,
        cached_read_tokens=sum(r.cached_read_tokens for r in results),
        cost_usd=round(estimate_cost(model, inp, out), 6),
    )


__all__ = ["Curator", "CallResult", "CurationUnavailable", "estimate_cost",
           "usage_from", "PRICES"]
