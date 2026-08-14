"""Load and validate the three YAML config files into frozen objects.

Validation happens once, here, at startup. A typo in a weight should fail
immediately with a clear message rather than silently scoring everything zero at
2am.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CATEGORY_SLUGS = ("politics", "economy", "security", "frontier")


class ConfigError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class FeedCfg:
    id: str
    url: str
    category_hint: str
    source_id: str


@dataclass(frozen=True, slots=True)
class SourceCfg:
    id: str
    name: str
    owner: str
    homepage: str
    weight: float
    tier: str
    enabled: bool
    pakistan_filter: bool
    title_suffix_re: re.Pattern[str] | None
    feeds: tuple[FeedCfg, ...]


@dataclass(frozen=True, slots=True)
class CategoryCfg:
    slug: str
    label: str
    weight: float
    keywords: tuple[str, ...]
    # Compiled once with word boundaries. Plain substring matching is a trap
    # here: short keywords like "us", "pm" and "eu" match inside "august",
    # "business" and "queue", which silently gave every story in August a
    # security-category hit.
    pattern: re.Pattern[str] | None = None

    def count_hits(self, haystack: str) -> int:
        if self.pattern is None:
            return 0
        return len(self.pattern.findall(haystack))


@dataclass(frozen=True, slots=True)
class TierCfg:
    name: str
    provider: str
    anthropic_model: str
    anthropic_effort: str
    openai_model: str
    batch_size: int
    parallel: int
    max_input_tokens: int


@dataclass(frozen=True, slots=True)
class Config:
    # site
    title: str
    tagline: str
    description: str
    base_url: str
    base_path: str
    timezone: str
    locale: str
    contact: str
    archive_retention_days: int
    # edition
    max_stories: int
    min_score: float
    max_per_category: int
    max_per_source_as_lead: int
    max_age_hours: int
    # curation
    categories: dict[str, CategoryCfg]
    signals: dict[str, float]
    worker: TierCfg
    editor: TierCfg
    max_output_tokens: int
    timeout_s: int
    cost_ceiling_usd: float
    summary_max_chars: int
    summary_min_chars: int
    use_publisher_dek_in_fallback: bool
    # feeds
    sources: tuple[SourceCfg, ...]
    pakistan_filter_keywords: tuple[str, ...]
    disable_after_failures: int
    probe_disabled_every_n_days: int
    reenable_after_successes: int

    @property
    def all_feeds(self) -> tuple[FeedCfg, ...]:
        return tuple(f for s in self.sources if s.enabled for f in s.feeds)

    def source(self, source_id: str) -> SourceCfg:
        for s in self.sources:
            if s.id == source_id:
                return s
        raise KeyError(source_id)

    def url_for(self, path: str) -> str:
        """Build a site-root-relative URL honouring base_path.

        Project Pages sites live under /<repo>/. Getting this wrong is the
        single most common way a published site renders as unstyled text.
        """
        base = self.base_path.rstrip("/")
        path = "/" + path.lstrip("/")
        return f"{base}{path}" if base else path


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"missing config file: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _flatten_keywords(raw: Any) -> tuple[str, ...]:
    """Accept both a plain list and the comma-packed list style used in the
    shipped config, so hand-edits in either style keep working."""
    out: list[str] = []
    for item in raw or []:
        for part in str(item).split(","):
            part = part.strip().lower()
            if part:
                out.append(part)
    return tuple(dict.fromkeys(out))


def _keyword_pattern(keywords: tuple[str, ...]) -> re.Pattern[str] | None:
    """Word-boundary alternation over the keyword list.

    Longest-first so "supreme court" wins over a bare "court", and \\b on both
    ends so short keywords cannot match inside longer words.
    """
    if not keywords:
        return None
    ordered = sorted(keywords, key=len, reverse=True)
    body = "|".join(re.escape(k) for k in ordered)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.I)


def _tier(name: str, raw: dict[str, Any]) -> TierCfg:
    provider = str(raw.get("provider", "auto")).lower()
    if provider not in ("auto", "anthropic", "openai", "heuristic"):
        raise ConfigError(
            f"tiers.{name}.provider must be auto|anthropic|openai|heuristic, "
            f"got {provider!r}"
        )
    anth = raw.get("anthropic") or {}
    oai = raw.get("openai") or {}
    return TierCfg(
        name=name,
        provider=provider,
        anthropic_model=str(anth.get("model", "claude-opus-5")),
        anthropic_effort=str(anth.get("effort", "medium")),
        openai_model=str(oai.get("model", "gpt-5")),
        batch_size=int(raw.get("batch_size", 30)),
        parallel=int(raw.get("parallel", 4)),
        max_input_tokens=int(raw.get("max_input_tokens", 0)),
    )


def load_config(config_dir: Path) -> Config:
    site = _load_yaml(config_dir / "site.yml")
    cur = _load_yaml(config_dir / "curation.yml")
    fee = _load_yaml(config_dir / "feeds.yml")

    # --- categories ---
    cats: dict[str, CategoryCfg] = {}
    for slug, body in (cur.get("categories") or {}).items():
        if slug not in CATEGORY_SLUGS:
            raise ConfigError(
                f"unknown category {slug!r}; expected one of {CATEGORY_SLUGS}. "
                "Adding a category also means updating the model schema."
            )
        keywords = _flatten_keywords(body.get("keywords"))
        cats[slug] = CategoryCfg(
            slug=slug,
            label=str(body.get("label", slug.title())),
            weight=float(body.get("weight", 1.0)),
            keywords=keywords,
            pattern=_keyword_pattern(keywords),
        )
    missing = set(CATEGORY_SLUGS) - set(cats)
    if missing:
        raise ConfigError(f"curation.yml is missing categories: {sorted(missing)}")
    if all(c.weight <= 0 for c in cats.values()):
        raise ConfigError("every category weight is 0 — nothing could ever publish")

    # --- sources ---
    defaults = fee.get("defaults") or {}
    sources: list[SourceCfg] = []
    seen_feed_ids: set[str] = set()
    for raw in fee.get("sources") or []:
        sid = str(raw["id"])
        suffix = raw.get("title_suffix_re")
        feeds: list[FeedCfg] = []
        for f in raw.get("feeds") or []:
            fid = str(f["id"])
            if fid in seen_feed_ids:
                raise ConfigError(f"duplicate feed id {fid!r} in feeds.yml")
            seen_feed_ids.add(fid)
            hint = str(f.get("category_hint", "politics"))
            if hint not in CATEGORY_SLUGS and hint != "general":
                raise ConfigError(
                    f"feed {fid!r} has category_hint {hint!r}; "
                    f"expected one of {CATEGORY_SLUGS} or 'general'"
                )
            feeds.append(FeedCfg(id=fid, url=str(f["url"]),
                                 category_hint=hint, source_id=sid))
        if not feeds:
            continue
        weight = float(raw.get("weight", defaults.get("weight", 1.0)))
        if not 0.0 <= weight <= 1.5:
            raise ConfigError(f"source {sid!r} weight {weight} outside 0.0-1.5")
        sources.append(SourceCfg(
            id=sid,
            name=str(raw.get("name", sid)),
            owner=str(raw.get("owner", sid)),
            homepage=str(raw.get("homepage", "")),
            weight=weight,
            tier=str(raw.get("tier", defaults.get("tier", "national"))),
            enabled=bool(raw.get("enabled", defaults.get("enabled", True))),
            pakistan_filter=bool(raw.get("pakistan_filter", False)),
            title_suffix_re=re.compile(suffix, re.I) if suffix else None,
            feeds=tuple(feeds),
        ))
    if not sources:
        raise ConfigError("feeds.yml defines no usable sources")

    ed = cur.get("edition") or {}
    health = fee.get("health") or {}
    summary = cur.get("summary") or {}
    model = cur.get("model") or {}
    tiers = cur.get("tiers") or {}

    return Config(
        title=str(site.get("title", "The Pakistan Wire")),
        tagline=str(site.get("tagline", "")),
        description=str(site.get("description", "")),
        base_url=str(site.get("base_url", "")).rstrip("/"),
        base_path=str(site.get("base_path", "")).rstrip("/"),
        timezone=str(site.get("timezone", "Asia/Karachi")),
        locale=str(site.get("locale", "en-PK")),
        contact=str(site.get("contact", "")),
        archive_retention_days=int(site.get("archive_retention_days", 365)),
        max_stories=int(ed.get("max_stories", 15)),
        min_score=float(ed.get("min_score", 35.0)),
        max_per_category=int(ed.get("max_per_category", 6)),
        max_per_source_as_lead=int(ed.get("max_per_source_as_lead", 4)),
        max_age_hours=int(ed.get("max_age_hours", 36)),
        categories=cats,
        signals={k: float(v) for k, v in (cur.get("signals") or {}).items()},
        worker=_tier("worker", tiers.get("worker") or {}),
        editor=_tier("editor", tiers.get("editor") or {}),
        max_output_tokens=int(model.get("max_output_tokens", 16000)),
        timeout_s=int(model.get("timeout_s", 600)),
        cost_ceiling_usd=float(model.get("cost_ceiling_usd", 1.5)),
        summary_max_chars=int(summary.get("max_chars", 320)),
        summary_min_chars=int(summary.get("min_chars", 80)),
        use_publisher_dek_in_fallback=bool(
            summary.get("use_publisher_dek_in_fallback", False)),
        sources=tuple(sources),
        pakistan_filter_keywords=_flatten_keywords(
            fee.get("pakistan_filter_keywords")),
        disable_after_failures=int(health.get("disable_after_consecutive_failures", 5)),
        probe_disabled_every_n_days=int(health.get("probe_disabled_every_n_days", 7)),
        reenable_after_successes=int(health.get("reenable_after_consecutive_successes", 2)),
    )
