"""Fetch feed bytes.

Two rules govern this module:

1. One dead feed must never break a run. Every exception is caught per feed and
   turned into a FeedResult with ok=False.
2. We are polite. Conditional GET via stored ETag/Last-Modified means a feed
   that has not changed costs the publisher a 304 instead of a full payload,
   and the User-Agent carries a real contact URL. These are small newsrooms;
   an anonymous hammering scraper gets IP-banned, and rightly.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .models import FeedResult
from .settings import Config, FeedCfg

log = logging.getLogger(__name__)

CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 15.0
MAX_ATTEMPTS = 2
BACKOFF_S = 1.5


def user_agent(cfg: Config) -> str:
    about = f"{cfg.base_url}{cfg.url_for('/about.html')}" if cfg.base_url else cfg.contact
    return f"ThePakistanWire/1.0 (+{about})"


def fetch_all(
    cfg: Config,
    feeds: list[FeedCfg],
    health: dict[str, dict[str, Any]],
    *,
    offline_dir: Path | None = None,
) -> list[FeedResult]:
    """Fetch every feed. Returns one FeedResult per feed, always."""
    if offline_dir is not None:
        return [_fetch_offline(f, offline_dir) for f in feeds]
    return _fetch_online(cfg, feeds, health)


def _fetch_offline(feed: FeedCfg, fixtures: Path) -> FeedResult:
    """Resolve a feed to fixtures/<feed_id>.xml.

    A missing fixture is reported as a 404 rather than an error, so the offline
    path exercises the dead-feed handling instead of routing around it.
    """
    path = fixtures / f"{feed.id}.xml"
    if not path.exists():
        return FeedResult(feed_id=feed.id, source_id=feed.source_id, ok=False,
                          status=404, error="fixture not found")
    try:
        body = path.read_bytes()
    except OSError as exc:
        return FeedResult(feed_id=feed.id, source_id=feed.source_id, ok=False,
                          error=f"fixture unreadable: {exc}")
    return FeedResult(feed_id=feed.id, source_id=feed.source_id, ok=True,
                      status=200, body=body)


def _fetch_online(
    cfg: Config, feeds: list[FeedCfg], health: dict[str, dict[str, Any]]
) -> list[FeedResult]:
    try:
        import httpx
    except ImportError:
        log.error("httpx is not installed; no feed can be fetched")
        return [
            FeedResult(feed_id=f.id, source_id=f.source_id, ok=False,
                       error="httpx not installed")
            for f in feeds
        ]

    timeout = httpx.Timeout(connect=CONNECT_TIMEOUT, read=READ_TIMEOUT,
                            write=5.0, pool=30.0)
    headers = {
        "User-Agent": user_agent(cfg),
        "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
    }
    results: list[FeedResult] = []
    with httpx.Client(timeout=timeout, follow_redirects=True, max_redirects=5,
                      headers=headers, http2=False) as client:
        workers = max(1, min(8, len(feeds)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_fetch_one, client, f, health.get(f.id, {}))
                for f in feeds
            ]
            for fut in futures:
                try:
                    results.append(fut.result())
                except Exception as exc:  # pragma: no cover - belt and braces
                    log.exception("unexpected fetch failure: %s", exc)
    return results


def _fetch_one(client: Any, feed: FeedCfg, prior: dict[str, Any]) -> FeedResult:
    import httpx

    headers: dict[str, str] = {}
    if prior.get("etag"):
        headers["If-None-Match"] = prior["etag"]
    if prior.get("last_modified"):
        headers["If-Modified-Since"] = prior["last_modified"]

    started = time.monotonic()
    last_error = "unknown"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client.get(feed.url, headers=headers)
        except httpx.HTTPError as exc:
            last_error = type(exc).__name__
            if attempt < MAX_ATTEMPTS:
                time.sleep(BACKOFF_S)
                continue
            break
        except Exception as exc:  # noqa: BLE001 - a feed must never raise upward
            last_error = f"{type(exc).__name__}: {exc}"
            break

        elapsed = int((time.monotonic() - started) * 1000)

        if resp.status_code == 304:
            return FeedResult(feed_id=feed.id, source_id=feed.source_id, ok=True,
                              status=304, etag=prior.get("etag"),
                              last_modified=prior.get("last_modified"),
                              elapsed_ms=elapsed)

        if resp.status_code >= 500 and attempt < MAX_ATTEMPTS:
            last_error = f"http {resp.status_code}"
            time.sleep(BACKOFF_S)
            continue

        # 4xx is never retried; it will not fix itself within a run.
        if resp.status_code >= 400:
            return FeedResult(feed_id=feed.id, source_id=feed.source_id, ok=False,
                              status=resp.status_code,
                              error=f"http {resp.status_code}", elapsed_ms=elapsed)

        return FeedResult(
            feed_id=feed.id, source_id=feed.source_id, ok=True,
            status=resp.status_code, body=resp.content,
            etag=resp.headers.get("etag"),
            last_modified=resp.headers.get("last-modified"),
            elapsed_ms=elapsed,
        )

    return FeedResult(feed_id=feed.id, source_id=feed.source_id, ok=False,
                      error=last_error,
                      elapsed_ms=int((time.monotonic() - started) * 1000))
