"""Feed health tracking, auto-quarantine, and reporting.

The feed URLs in config/feeds.yml could not be validated from the environment
they were written in, and news sites move their feeds without notice anyway. So
health is treated as a first-class, observable thing rather than an afterthought:
failures accumulate, persistently dead feeds are quarantined, quarantined feeds
are periodically re-probed, and the whole table is published on the site.

State lives in state/feeds_health.json. The bot never rewrites feeds.yml — a
config file that the automation edits behind your back is one you cannot trust.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .models import FeedResult
from .settings import Config, FeedCfg
from .util import read_json, write_json_atomic

FILENAME = "feeds_health.json"


def _blank() -> dict[str, Any]:
    return {
        "etag": None,
        "last_modified": None,
        "last_success": None,
        "last_attempt": None,
        "consecutive_failures": 0,
        "consecutive_successes": 0,
        "disabled": False,
        "disabled_at": None,
        "last_error": None,
        "entries_last_run": 0,
        "reported_issue": None,
    }


class HealthStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        raw = read_json(path, {}) or {}
        self.feeds: dict[str, dict[str, Any]] = dict(raw.get("feeds") or {})

    def get(self, feed_id: str) -> dict[str, Any]:
        return self.feeds.setdefault(feed_id, _blank())

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return self.feeds

    def due_feeds(self, cfg: Config, now: datetime) -> list[FeedCfg]:
        """Feeds to attempt this run.

        A quarantined feed is retried every `probe_disabled_every_n_days` — dead
        feeds sometimes come back, and a permanently skipped feed would need a
        human to notice and re-enable it by hand.
        """
        due: list[FeedCfg] = []
        for feed in cfg.all_feeds:
            state = self.feeds.get(feed.id)
            if not state or not state.get("disabled"):
                due.append(feed)
                continue
            last = _parse(state.get("last_attempt"))
            if last is None or now - last >= timedelta(days=cfg.probe_disabled_every_n_days):
                due.append(feed)
        return due

    def record(self, result: FeedResult, entry_count: int, cfg: Config,
               now: datetime) -> None:
        state = self.get(result.feed_id)
        state["last_attempt"] = now.isoformat()

        # A 200 with zero entries is how a silently relocated feed presents
        # itself — the server happily serves an empty channel or an HTML page.
        # Treating it as success would hide the outage indefinitely.
        healthy = result.ok and (result.not_modified or entry_count > 0)

        if healthy:
            state["last_success"] = now.isoformat()
            state["consecutive_failures"] = 0
            state["consecutive_successes"] = int(state.get("consecutive_successes", 0)) + 1
            state["last_error"] = None
            state["entries_last_run"] = entry_count
            if result.etag:
                state["etag"] = result.etag
            if result.last_modified:
                state["last_modified"] = result.last_modified
            if state.get("disabled") and \
                    state["consecutive_successes"] >= cfg.reenable_after_successes:
                state["disabled"] = False
                state["disabled_at"] = None
                state["reported_issue"] = None
        else:
            state["consecutive_successes"] = 0
            state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
            state["entries_last_run"] = 0
            state["last_error"] = result.error or (
                "no entries" if result.ok else f"http {result.status}")
            if state["consecutive_failures"] >= cfg.disable_after_failures \
                    and not state.get("disabled"):
                state["disabled"] = True
                state["disabled_at"] = now.isoformat()

    def newly_disabled(self) -> list[str]:
        """Quarantined feeds that have not yet been reported as an issue."""
        return [
            fid for fid, s in self.feeds.items()
            if s.get("disabled") and not s.get("reported_issue")
        ]

    def mark_reported(self, feed_ids: list[str], marker: str) -> None:
        for fid in feed_ids:
            if fid in self.feeds:
                self.feeds[fid]["reported_issue"] = marker

    def report_rows(self, cfg: Config) -> list[dict[str, Any]]:
        """Rows for docs/sources.html — the public transparency table."""
        rows: list[dict[str, Any]] = []
        for source in cfg.sources:
            for feed in source.feeds:
                s = self.feeds.get(feed.id, _blank())
                if s.get("disabled"):
                    status = "disabled"
                elif s.get("consecutive_failures", 0) > 0:
                    status = "failing"
                elif s.get("last_success"):
                    status = "ok"
                else:
                    status = "unknown"
                rows.append({
                    "source_id": source.id,
                    "source_name": source.name,
                    "homepage": source.homepage,
                    "feed_id": feed.id,
                    "url": feed.url,
                    "status": status,
                    # Parsed for the template's benefit; the on-disk state keeps
                    # the ISO string so the JSON stays diffable.
                    "last_success": _parse(s.get("last_success")),
                    "consecutive_failures": s.get("consecutive_failures", 0),
                    "last_error": s.get("last_error"),
                    "entries_last_run": s.get("entries_last_run", 0),
                })
        rows.sort(key=lambda r: ({"disabled": 0, "failing": 1, "unknown": 2, "ok": 3}[r["status"]],
                                 r["source_name"]))
        return rows

    def save(self) -> None:
        write_json_atomic(self.path, {"version": 1, "feeds": self.feeds})


def _parse(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
