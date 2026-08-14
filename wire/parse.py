"""Turn feed bytes into RawEntry records.

feedparser's `bozo` flag is deliberately ignored. Nearly every Pakistani news
feed sets it — unescaped ampersands, stray tags, non-standard date formats — and
still yields a perfectly good entry list. Treating bozo as fatal would drop most
of the sources. The real failure signal is an empty entry list.
"""

from __future__ import annotations

import logging
from typing import Any

from .models import FeedResult, RawEntry
from .settings import FeedCfg

log = logging.getLogger(__name__)


def parse(result: FeedResult, feed: FeedCfg) -> list[RawEntry]:
    if not result.ok or not result.body:
        return []

    try:
        import feedparser
    except ImportError:
        log.error("feedparser is not installed; cannot parse %s", feed.id)
        return []

    try:
        parsed = feedparser.parse(result.body)
    except Exception as exc:  # noqa: BLE001
        log.warning("feed %s could not be parsed at all: %s", feed.id, exc)
        return []

    entries = getattr(parsed, "entries", None) or []
    if not entries:
        log.warning("feed %s returned no entries (bozo=%s)", feed.id,
                    getattr(parsed, "bozo", "?"))
        return []

    out: list[RawEntry] = []
    for position, raw in enumerate(entries):
        # Per-entry isolation: one malformed <item> costs one story, not a feed.
        try:
            entry = _to_entry(raw, feed, position)
        except Exception as exc:  # noqa: BLE001
            log.debug("skipping malformed entry %d in %s: %s", position, feed.id, exc)
            continue
        if entry is not None:
            out.append(entry)
    return out


def _to_entry(raw: Any, feed: FeedCfg, position: int) -> RawEntry | None:
    title = (raw.get("title") or "").strip()
    link = (raw.get("link") or "").strip()
    if not link:
        # Atom sometimes puts the URL only in links[].href.
        for candidate in raw.get("links") or []:
            if candidate.get("rel") in (None, "alternate") and candidate.get("href"):
                link = candidate["href"].strip()
                break
    if not title or not link:
        return None

    summary = (
        raw.get("summary")
        or raw.get("description")
        or (raw.get("content") or [{}])[0].get("value", "")
        or ""
    )

    published = (
        raw.get("published_parsed")
        or raw.get("updated_parsed")
        or raw.get("created_parsed")
    )

    categories = tuple(
        (t.get("term") or "").strip().lower()
        for t in (raw.get("tags") or [])
        if (t.get("term") or "").strip()
    )

    return RawEntry(
        feed_id=feed.id,
        source_id=feed.source_id,
        title_raw=title,
        link_raw=link,
        summary_raw=summary,
        published_parsed=published,
        feed_categories=categories,
        position=position,
    )
