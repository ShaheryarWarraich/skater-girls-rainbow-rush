"""Command line entrypoint.

    python -m wire.cli doctor
    python -m wire.cli build --offline --fixtures fixtures --out /tmp/out
    python -m wire.cli report-health
    python -m wire.cli clean --older-than 0

The design rule for `build`: with no key, no network and no flags it must still
produce a viewable page rather than a stack trace. The first thing a new
developer runs should show them something.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from . import cluster as cluster_mod
from . import fetch, health, heuristic, normalize, parse, render, seen
from .curate import curate
from .models import Edition, EditionStats
from .settings import Config, ConfigError, load_config
from .util import has_key, utcnow

log = logging.getLogger("wire")

ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wire", description="The Pakistan Wire")
    ap.add_argument("--config", type=Path, default=ROOT / "config")
    ap.add_argument("--state", type=Path, default=ROOT / "state")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="fetch, curate and render an edition")
    b.add_argument("--out", type=Path, default=ROOT / "docs")
    b.add_argument("--templates", type=Path, default=ROOT / "templates")
    b.add_argument("--assets", type=Path, default=ROOT / "assets")
    b.add_argument("--offline", action="store_true",
                   help="read fixtures instead of the network")
    b.add_argument("--fixtures", type=Path, default=ROOT / "fixtures")
    b.add_argument("--record", action="store_true",
                   help="save fetched feeds into --fixtures (refresh test data "
                        "from a machine that can reach the sources)")
    b.add_argument("--force-heuristic", action="store_true",
                   help="skip every model call and use the deterministic ranker")
    b.add_argument("--dry-run", action="store_true",
                   help="print the write plan without touching disk")
    b.add_argument("--explain", action="store_true",
                   help="print the per-cluster score breakdown")
    b.add_argument("--explain-cost", action="store_true",
                   help="print the per-tier token and cost split")
    b.add_argument("--date", dest="pinned", default=None,
                   help="pin today's date (YYYY-MM-DD) for reproducible output")

    sub.add_parser("doctor", help="preflight checks")

    r = sub.add_parser("report-health", help="print feed health, exit 1 if degraded")
    r.add_argument("--github", action="store_true",
                   help="emit ::warning:: / ::error:: annotations")

    c = sub.add_parser("clean", help="prune or wipe the seen-articles store")
    c.add_argument("--older-than", type=int, default=None,
                   help="days to keep; 0 wipes everything and forces a republish")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "build":
        return cmd_build(args, cfg)
    if args.command == "doctor":
        return cmd_doctor(args, cfg)
    if args.command == "report-health":
        return cmd_report_health(args, cfg)
    if args.command == "clean":
        return cmd_clean(args, cfg)
    return 2


# ---------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace, cfg: Config) -> int:
    now = _now(args.pinned)
    today = _local_date(now, cfg)

    health_store = health.HealthStore(args.state / health.FILENAME)
    seen_store = seen.SeenStore(args.state / seen.FILENAME)

    due = health_store.due_feeds(cfg, now)
    total_feeds = len(cfg.all_feeds)
    log.info("fetching %d of %d feeds (%s)", len(due), total_feeds,
             "offline" if args.offline else "network")

    results = fetch.fetch_all(
        cfg, due, health_store.as_dict(),
        offline_dir=args.fixtures if args.offline else None,
    )

    if args.record and not args.offline:
        saved = _record_fixtures(results, args.fixtures)
        log.info("recorded %d feed(s) into %s", saved, args.fixtures)

    feeds_by_id = {f.id: f for f in cfg.all_feeds}
    articles = []
    ok = failed = 0
    for result in results:
        feed = feeds_by_id.get(result.feed_id)
        if feed is None:
            continue
        entries = parse.parse(result, feed)
        health_store.record(result, len(entries), cfg, now)
        if result.ok and (result.not_modified or entries):
            ok += 1
        else:
            failed += 1
        articles.extend(normalize.to_articles(entries, cfg, feed, now))

    articles = normalize.dedupe(articles)
    articles = normalize.drop_stale(articles, now, cfg.max_age_hours)
    seen_total = len(articles)

    fresh = seen_store.filter_new(articles)
    log.info("%d articles in window, %d new", seen_total, len(fresh))

    clusters = cluster_mod.precluster(fresh)
    log.info("%d clusters", len(clusters))

    if args.explain:
        for row in heuristic.explain(clusters, cfg, now):
            print(json.dumps(row, ensure_ascii=False))

    outcome = curate(clusters, cfg, now, force_heuristic=args.force_heuristic)

    if outcome.mode != "empty" and outcome.stories:
        log.info("edition: %d stories (%s)", len(outcome.stories), outcome.mode)
    else:
        log.warning("no publishable stories; keeping the previous edition")

    disabled = sum(1 for s in health_store.as_dict().values() if s.get("disabled"))
    stats = EditionStats(
        feeds_total=total_feeds, feeds_ok=ok, feeds_failed=failed,
        feeds_disabled=disabled, articles_seen=seen_total,
        articles_new=len(fresh), clusters=len(clusters),
        usage=tuple(outcome.usage),
        plagiarism_rejections=outcome.plagiarism_rejections,
    )

    if args.explain_cost:
        _print_cost(stats)

    # An empty edition must not create an archive page, a feed item, or mutate
    # the seen store — otherwise a bad-feed morning permanently consumes a date
    # and silently marks stories as published.
    if not outcome.stories:
        health_store.save() if not args.dry_run else None
        print(json.dumps({"mode": "empty", "stories": 0,
                          "feeds_ok": ok, "feeds_failed": failed}))
        _emit_github_output(today, "empty", 0)
        return 0

    archive_dates = _archive_dates(args.out, today, cfg)
    edition = Edition(
        date=today, generated_at=now, stories=tuple(outcome.stories),
        brief=outcome.brief, mode=outcome.mode,
        degrade_reason=outcome.reason, stats=stats,
        feed_health=tuple(health_store.report_rows(cfg)),
    )

    try:
        ops = render.render(edition, cfg, args.templates, archive_dates,
                            assets=args.assets)
    except Exception as exc:  # noqa: BLE001
        # Nothing has been written yet, so the previous docs/ is untouched.
        log.exception("render failed; previous site left intact")
        print(f"render error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        for op in ops:
            print(f"{len(op.data):>9}  {op.path}")
        return 0

    render.commit(ops, args.out)

    published = [a for s in outcome.stories for a in s.all_sources]
    keys = {a.id: s.cluster_key for s in outcome.stories for a in s.all_sources}
    seen_store.record(published, today, keys)
    removed = seen_store.prune(now, cfg.max_age_hours)
    seen_store.save()
    health_store.save()

    log.info("wrote %d files to %s (pruned %d seen entries)", len(ops), args.out, removed)
    print(json.dumps({
        "mode": edition.mode, "stories": len(edition.stories),
        "date": today.isoformat(), "feeds_ok": ok, "feeds_failed": failed,
        "cost_usd": stats.total_cost_usd,
    }))
    _emit_github_output(today, edition.mode, len(edition.stories))
    return 0


def cmd_doctor(args: argparse.Namespace, cfg: Config) -> int:
    rows: list[tuple[str, str]] = [
        ("python", sys.version.split()[0]),
        ("config", f"{len(cfg.sources)} sources, {len(cfg.all_feeds)} feeds"),
        ("categories", ", ".join(
            f"{s}={c.weight:g}" for s, c in cfg.categories.items())),
        ("edition", f"max {cfg.max_stories} stories, min score {cfg.min_score:g}"),
    ]

    for mod in ("feedparser", "httpx", "jinja2", "yaml", "pydantic",
                "anthropic", "openai"):
        try:
            __import__(mod)
            rows.append((mod, "installed"))
        except ImportError:
            rows.append((mod, "MISSING"))

    # Presence only — the value is never printed.
    rows.append(("ANTHROPIC_API_KEY", "present" if has_key("ANTHROPIC_API_KEY") else "absent"))
    rows.append(("OPENAI_API_KEY", "present" if has_key("OPENAI_API_KEY") else "absent"))
    rows.append(("worker tier", f"{cfg.worker.provider} "
                                f"({cfg.worker.anthropic_model} / {cfg.worker.openai_model})"))
    rows.append(("editor tier", f"{cfg.editor.provider} "
                                f"({cfg.editor.anthropic_model} / {cfg.editor.openai_model}), "
                                f"guard {cfg.editor.max_input_tokens} tok"))

    store = health.HealthStore(args.state / health.FILENAME)
    disabled = [f for f, s in store.as_dict().items() if s.get("disabled")]
    rows.append(("disabled feeds", str(len(disabled)) + (
        f" ({', '.join(sorted(disabled)[:5])})" if disabled else "")))

    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        print(f"{key.ljust(width)}  {value}")

    if not has_key("ANTHROPIC_API_KEY") and not has_key("OPENAI_API_KEY"):
        print("\nNo model API key found. Builds will use the deterministic "
              "ranker and publish a complete, clearly-labelled site.")
    if cfg.editor.anthropic_model in ("claude-fable-5", "claude-mythos-5"):
        print("\nNote: Claude Fable 5 is unavailable to organisations on "
              "zero-data-retention. If every request 400s with a valid payload, "
              "check the org's retention setting before debugging the request.")
    return 0


def cmd_report_health(args: argparse.Namespace, cfg: Config) -> int:
    store = health.HealthStore(args.state / health.FILENAME)
    rows = store.report_rows(cfg)
    bad = [r for r in rows if r["status"] in ("failing", "disabled")]

    for row in rows:
        print(f"{row['status']:<9} {row['feed_id']:<24} "
              f"{row['entries_last_run']:>4} entries  {row['last_error'] or ''}")

    if args.github and bad:
        for row in bad:
            level = "error" if row["status"] == "disabled" else "warning"
            print(f"::{level} title=Feed {row['status']}::"
                  f"{row['feed_id']} ({row['url']}) — {row['last_error'] or 'no entries'}")

    disabled = [r for r in rows if r["status"] == "disabled"]
    if disabled:
        print(f"\n{len(disabled)} feed(s) quarantined after repeated failure. "
              "Their URLs probably moved and need correcting in config/feeds.yml.")
        return 1
    return 0


def cmd_clean(args: argparse.Namespace, cfg: Config) -> int:
    store = seen.SeenStore(args.state / seen.FILENAME)
    before = len(store.articles)
    if args.older_than == 0:
        store.clear()
        print(f"cleared {before} seen entries; the next build will republish "
              "everything currently in the feeds")
    else:
        days = args.older_than if args.older_than is not None else 30
        removed = store.prune(utcnow(), days * 24)
        print(f"pruned {removed} of {before} seen entries")
    store.save()
    return 0


# ---------------------------------------------------------------------------


def _record_fixtures(results: list, fixtures: Path) -> int:
    """Save fetched feed bodies as offline fixtures.

    The environment this project was built in cannot reach Pakistani news
    domains, so the committed fixtures are hand-written. Running `build --record`
    from a machine that *can* reach them replaces those with real payloads,
    which is the only honest way to test against what the outlets actually emit.
    """
    fixtures.mkdir(parents=True, exist_ok=True)
    saved = 0
    for result in results:
        if not result.ok or not result.body:
            continue
        (fixtures / f"{result.feed_id}.xml").write_bytes(result.body)
        saved += 1
    return saved


def _now(pinned: str | None) -> datetime:
    if pinned:
        return datetime.fromisoformat(pinned).replace(tzinfo=timezone.utc)
    return utcnow()


def _local_date(now: datetime, cfg: Config) -> date:
    try:
        from zoneinfo import ZoneInfo
        return now.astimezone(ZoneInfo(cfg.timezone)).date()
    except Exception:  # noqa: BLE001
        return now.date()


def _archive_dates(out: Path, today: date, cfg: Config) -> list[date]:
    dates = {today}
    archive = out / "archive"
    if archive.is_dir():
        for child in archive.iterdir():
            if child.is_dir():
                try:
                    dates.add(date.fromisoformat(child.name))
                except ValueError:
                    continue
    cutoff = today - timedelta(days=cfg.archive_retention_days)
    return sorted(d for d in dates if d >= cutoff)


def _print_cost(stats: EditionStats) -> None:
    if not stats.usage:
        print("cost: no model calls (deterministic ranker)")
        return
    print(f"{'tier':<8}{'provider':<11}{'model':<20}"
          f"{'calls':>6}{'in':>10}{'out':>9}{'cached':>9}{'usd':>10}")
    for u in stats.usage:
        print(f"{u.tier:<8}{u.provider:<11}{u.model:<20}{u.calls:>6}"
              f"{u.input_tokens:>10}{u.output_tokens:>9}"
              f"{u.cached_read_tokens:>9}{u.cost_usd:>10.4f}")
    total_in = sum(u.input_tokens for u in stats.usage)
    worker_in = sum(u.input_tokens for u in stats.usage if u.tier == "worker")
    editor_in = sum(u.input_tokens for u in stats.usage if u.tier != "worker")
    print(f"{'total':<8}{'':<11}{'':<20}"
          f"{sum(u.calls for u in stats.usage):>6}{total_in:>10}"
          f"{sum(u.output_tokens for u in stats.usage):>9}{'':>9}"
          f"{stats.total_cost_usd:>10.4f}")
    if worker_in and editor_in:
        print(f"\neditor saw {editor_in / max(1, worker_in):.1%} of the worker "
              f"tier's input tokens — that ratio is the design working.")


def _emit_github_output(day: date, mode: str, count: int) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"edition_date={day.isoformat()}\n")
            fh.write(f"mode={mode}\n")
            fh.write(f"story_count={count}\n")
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
