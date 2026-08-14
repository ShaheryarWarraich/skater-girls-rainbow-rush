"""Data shapes passed between pipeline stages.

Everything is frozen. The pipeline is a chain of pure-ish transforms, and making
the payloads immutable means a later stage cannot quietly mutate something an
earlier stage still holds a reference to.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class FeedResult:
    """One HTTP attempt against one feed URL."""

    feed_id: str
    source_id: str
    ok: bool
    status: int | None = None
    body: bytes | None = None
    etag: str | None = None
    last_modified: str | None = None
    error: str | None = None
    elapsed_ms: int = 0

    @property
    def not_modified(self) -> bool:
        return self.status == 304


@dataclass(frozen=True, slots=True)
class RawEntry:
    """A single <item>/<entry>, straight out of feedparser and not yet cleaned."""

    feed_id: str
    source_id: str
    title_raw: str
    link_raw: str
    summary_raw: str
    published_parsed: time.struct_time | None
    feed_categories: tuple[str, ...] = ()
    position: int = 0  # index within its feed; earlier usually means more prominent


@dataclass(frozen=True, slots=True)
class Article:
    """A normalised story reference.

    `dek` is the publisher's own blurb from the feed. It is MODEL INPUT ONLY and
    must never reach a template — see the never-republish rule. tests/ asserts
    that no template references it.
    """

    id: str  # sha1(canonical_url)[:16]
    title: str
    url: str  # canonicalised
    source_id: str
    source_name: str
    source_homepage: str
    source_weight: float
    source_owner: str
    source_tier: str
    feed_id: str
    dek: str
    published_at: datetime
    time_confidence: str  # "high" | "low"
    category_hint: str
    position: int = 0

    def redacted(self) -> "Article":
        """Copy with the publisher blurb stripped, for anything render-bound."""
        return replace(self, dek="")


@dataclass(frozen=True, slots=True)
class Cluster:
    """A set of articles believed to be the same event."""

    key: str
    members: tuple[Article, ...]

    @property
    def lead(self) -> Article:
        return self.members[0]

    @property
    def outlet_count(self) -> int:
        return len({m.source_id for m in self.members})

    @property
    def owner_count(self) -> int:
        """Distinct media groups. This is what corroboration actually means."""
        return len({m.source_owner for m in self.members})

    @property
    def earliest(self) -> datetime:
        return min(m.published_at for m in self.members)


@dataclass(frozen=True, slots=True)
class Story:
    """A curated, publishable item."""

    cluster_key: str
    headline: str  # the lead outlet's headline, verbatim and deliberately so
    summary: str  # our own words (AI) or composed metadata (heuristic)
    why_it_matters: str
    category: str
    score: float
    lead: Article
    also_reported_by: tuple[Article, ...]
    curated_by: str  # "editor:<model>" | "worker:<model>" | "heuristic"
    confidence: str = "medium"

    @property
    def all_sources(self) -> tuple[Article, ...]:
        return (self.lead, *self.also_reported_by)


@dataclass(frozen=True, slots=True)
class TierUsage:
    """Token accounting for one model tier, so the cost claim is measurable."""

    tier: str
    provider: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_read_tokens: int = 0
    cost_usd: float = 0.0


@dataclass(frozen=True, slots=True)
class EditionStats:
    feeds_total: int = 0
    feeds_ok: int = 0
    feeds_failed: int = 0
    feeds_disabled: int = 0
    articles_seen: int = 0
    articles_new: int = 0
    clusters: int = 0
    usage: tuple[TierUsage, ...] = ()
    plagiarism_rejections: int = 0

    @property
    def total_cost_usd(self) -> float:
        return round(sum(u.cost_usd for u in self.usage), 4)


@dataclass(frozen=True, slots=True)
class Edition:
    date: date
    generated_at: datetime
    stories: tuple[Story, ...]
    brief: str | None
    mode: str  # "ai" | "mixed" | "heuristic" | "empty"
    degrade_reason: str | None
    stats: EditionStats
    feed_health: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def is_degraded(self) -> bool:
        return self.mode != "ai"


class CurationUnavailable(Exception):
    """Raised when a provider cannot produce a usable result.

    Always carries a machine-readable reason so the degrade path can be reported
    to the reader rather than failing silently.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
