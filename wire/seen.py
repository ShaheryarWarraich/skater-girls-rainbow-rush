"""The no-repeats guarantee.

state/seen.json records every article we have already published, keyed by URL
hash. It lives in git rather than in actions/cache: the cache is evicted
silently after a week of non-use, and a silent eviction here would make the site
republish a week of old stories with no warning at all.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from .models import Article
from .util import read_json, write_json_atomic

FILENAME = "seen.json"
MIN_RETENTION_DAYS = 30


class SeenStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        raw = read_json(path, {}) or {}
        self.version: int = int(raw.get("version", 1))
        self.articles: dict[str, dict[str, Any]] = dict(raw.get("articles") or {})

    def __contains__(self, article_id: str) -> bool:
        return article_id in self.articles

    def is_new(self, article: Article) -> bool:
        return article.id not in self.articles

    def filter_new(self, articles: list[Article]) -> list[Article]:
        return [a for a in articles if a.id not in self.articles]

    def record(self, articles: list[Article], edition_date: date,
               cluster_keys: dict[str, str] | None = None) -> None:
        cluster_keys = cluster_keys or {}
        stamp = edition_date.isoformat()
        for a in articles:
            self.articles.setdefault(a.id, {
                "first_seen": stamp,
                "published_at": a.published_at.isoformat(),
                "source_id": a.source_id,
                "cluster_key": cluster_keys.get(a.id, ""),
                "published_in_edition": stamp,
            })

    def prune(self, now: datetime, max_age_hours: int) -> int:
        """Drop entries older than the retention window.

        Retention is generous relative to max_age_hours so a story cannot fall
        out of `seen` while still being inside the freshness window — that would
        let it be republished the day after it appeared.
        """
        days = max(MIN_RETENTION_DAYS, (max_age_hours // 24) + 7)
        cutoff = (now - timedelta(days=days)).date().isoformat()
        before = len(self.articles)
        self.articles = {
            k: v for k, v in self.articles.items()
            if str(v.get("first_seen", "")) >= cutoff
        }
        self._pruned_before = cutoff
        return before - len(self.articles)

    def save(self) -> None:
        write_json_atomic(self.path, {
            "version": self.version,
            "pruned_before": getattr(self, "_pruned_before", ""),
            "articles": self.articles,
        })

    def clear(self) -> None:
        self.articles = {}
