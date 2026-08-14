"""RawEntry -> Article.

This is where duplicates are actually killed, via URL canonicalisation. Getting
this right matters more than the clustering step: Dawn and Tribune each publish
the same story under two or three URL variants (AMP, tracking parameters,
trailing slash), and without canonicalisation the site publishes the same item
several times regardless of how good the clustering is.
"""

from __future__ import annotations

import calendar
import logging
from datetime import datetime, timedelta, timezone

from .models import Article, RawEntry
from .settings import Config, FeedCfg
from .util import article_id, canonical_url, strip_html, truncate

log = logging.getLogger(__name__)

DEK_MAX_CHARS = 400
FUTURE_TOLERANCE = timedelta(hours=36)
PAST_TOLERANCE = timedelta(days=30)


def to_articles(
    entries: list[RawEntry], cfg: Config, feed: FeedCfg, now: datetime
) -> list[Article]:
    source = cfg.source(feed.source_id)
    out: list[Article] = []

    for entry in entries:
        url = canonical_url(entry.link_raw)
        if not url:
            continue

        title = strip_html(entry.title_raw)
        if source.title_suffix_re is not None:
            title = source.title_suffix_re.sub("", title).strip()
        if not title:
            continue

        # The publisher's blurb. Model input only — never rendered.
        dek = truncate(strip_html(entry.summary_raw), DEK_MAX_CHARS)

        published, confidence = _resolve_time(entry, now)

        article = Article(
            id=article_id(url),
            title=title,
            url=url,
            source_id=source.id,
            source_name=source.name,
            source_homepage=source.homepage,
            source_weight=source.weight,
            source_owner=source.owner,
            source_tier=source.tier,
            feed_id=feed.id,
            dek=dek,
            published_at=published,
            time_confidence=confidence,
            category_hint=feed.category_hint,
            position=entry.position,
        )

        if source.pakistan_filter and not _mentions_pakistan(article, cfg):
            # International wires are ~95% not about Pakistan. Without this gate
            # they would swamp the pool on volume alone.
            continue

        out.append(article)

    return out


def _resolve_time(entry: RawEntry, now: datetime) -> tuple[datetime, str]:
    """Best-effort publication time, with an honest confidence flag.

    A missing or absurd timestamp is common in these feeds. Rather than dropping
    the story we fall back to 'now' and mark it low-confidence, which costs it a
    small scoring penalty later.
    """
    if entry.published_parsed is not None:
        try:
            stamp = datetime.fromtimestamp(
                calendar.timegm(entry.published_parsed), tz=timezone.utc
            )
        except (ValueError, OverflowError, TypeError):
            return now, "low"
        if now - PAST_TOLERANCE <= stamp <= now + FUTURE_TOLERANCE:
            return stamp, "high"
        return now, "low"
    return now, "low"


def _mentions_pakistan(article: Article, cfg: Config) -> bool:
    haystack = f"{article.title} {article.dek}".lower()
    return any(kw in haystack for kw in cfg.pakistan_filter_keywords)


def dedupe(articles: list[Article]) -> list[Article]:
    """Collapse identical canonical URLs, keeping the earliest sighting.

    The same URL can arrive from two feeds of the same outlet (home and
    business, say). Keeping the earlier timestamp preserves 'when did this
    actually break', which the recency score depends on.
    """
    best: dict[str, Article] = {}
    for a in articles:
        prior = best.get(a.id)
        if prior is None or a.published_at < prior.published_at:
            best[a.id] = a
    return sorted(best.values(), key=lambda a: (a.published_at, a.id))


def drop_stale(articles: list[Article], now: datetime, max_age_hours: int) -> list[Article]:
    cutoff = now - timedelta(hours=max_age_hours)
    return [a for a in articles if a.published_at >= cutoff]
