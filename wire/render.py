"""Edition -> a batch of files.

`render()` is pure: it returns a list of (path, bytes) and touches nothing. The
caller writes the whole batch at once. That means a template error cannot leave
docs/ half-overwritten — either the new site lands completely or yesterday's
stays exactly as it was.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from .models import Edition, Story
from .settings import Config

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class WriteOp:
    path: str  # relative to the output root
    data: bytes


def _env(templates: Path, cfg: Config) -> Any:
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    env = Environment(
        loader=FileSystemLoader(str(templates)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals["url_for"] = cfg.url_for
    env.filters["pkt"] = lambda dt: _local(dt, cfg)
    return env


def _local(dt: datetime, cfg: Config) -> str:
    try:
        from zoneinfo import ZoneInfo
        return dt.astimezone(ZoneInfo(cfg.timezone)).strftime("%d %b %Y, %H:%M")
    except Exception:  # noqa: BLE001
        return dt.strftime("%d %b %Y, %H:%M")


def _nav(cfg: Config) -> list[dict[str, str]]:
    return [
        {"slug": slug, "label": cat.label, "url": cfg.url_for(f"/category/{slug}/")}
        for slug, cat in cfg.categories.items()
        if cat.weight > 0
    ]


def render(edition: Edition, cfg: Config, templates: Path,
           archive_dates: list[date], assets: Path | None = None) -> list[WriteOp]:
    env = _env(templates, cfg)
    nav = _nav(cfg)
    ops: list[WriteOp] = []

    site = _SiteView(cfg)
    common = {
        "site": site,
        "nav": nav,
        "generated_at": edition.generated_at,
        "archive_dates": archive_dates,
    }

    day = edition.date.isoformat()
    prev_date, next_date = _neighbours(edition.date, archive_dates)

    # Home page and the dated archive copy are the same render.
    edition_html = env.get_template("index.html").render(
        **common,
        edition=edition,
        is_today=True,
        prev_date=prev_date,
        next_date=next_date,
        page_title=f"{cfg.title} — {day}",
    )
    ops.append(WriteOp("index.html", edition_html.encode("utf-8")))

    archive_html = env.get_template("index.html").render(
        **common,
        edition=edition,
        is_today=False,
        prev_date=prev_date,
        next_date=next_date,
        page_title=f"{cfg.title} — {day}",
    )
    ops.append(WriteOp(f"archive/{day}/index.html", archive_html.encode("utf-8")))

    # Archive index.
    editions = [{
        "date": d,
        "url": cfg.url_for(f"/archive/{d.isoformat()}/"),
        "story_count": len(edition.stories) if d == edition.date else None,
        "mode": edition.mode if d == edition.date else "",
    } for d in sorted(archive_dates, reverse=True)]
    ops.append(WriteOp("archive/index.html", env.get_template("archive_index.html").render(
        **common, editions=editions, page_title=f"Archive — {cfg.title}",
    ).encode("utf-8")))

    # One page per live category.
    for slug, cat in cfg.categories.items():
        if cat.weight <= 0:
            continue
        stories = [s for s in edition.stories if s.category == slug]
        ops.append(WriteOp(
            f"category/{slug}/index.html",
            env.get_template("category.html").render(
                **common, category=cat, stories=stories, date=edition.date,
                edition=edition,
                page_title=f"{cat.label} — {cfg.title}",
            ).encode("utf-8"),
        ))

    # Transparency: which feeds worked, which did not.
    ops.append(WriteOp("sources.html", env.get_template("sources.html").render(
        **common, rows=list(edition.feed_health),
        page_title=f"Sources — {cfg.title}",
    ).encode("utf-8")))

    ops.append(WriteOp("about.html", env.get_template("about.html").render(
        **common, page_title=f"About — {cfg.title}", edition=edition,
    ).encode("utf-8")))

    ops.append(WriteOp("feed.xml", build_feed(edition, cfg).encode("utf-8")))
    ops.append(WriteOp("sitemap.xml", build_sitemap(edition, cfg, archive_dates).encode("utf-8")))
    ops.append(WriteOp("robots.txt", _robots(cfg).encode("utf-8")))

    # Without .nojekyll, GitHub Pages runs Jekyll over docs/ and silently 404s
    # anything beginning with an underscore.
    ops.append(WriteOp(".nojekyll", b""))

    if assets is not None and assets.is_dir():
        for path in sorted(assets.iterdir()):
            if path.is_file():
                ops.append(WriteOp(path.name, path.read_bytes()))

    return ops


def build_feed(edition: Edition, cfg: Config) -> str:
    """Our own RSS.

    Items link to OUR page, not to the source. If they linked straight out, the
    feed would itself become a republication channel that bypasses the
    attribution the site is careful to show.
    """
    base = f"{cfg.base_url}{cfg.base_path}"
    day = edition.date.isoformat()
    items: list[str] = []

    for story in edition.stories:
        link = f"{base}/archive/{day}/#{_anchor(story)}"
        desc = f"{story.summary} (Reported by {story.lead.source_name}.)"
        items.append(
            "    <item>\n"
            f"      <title>{xml_escape(story.headline)}</title>\n"
            f"      <link>{xml_escape(link)}</link>\n"
            f"      <guid isPermaLink=\"false\">{xml_escape(story.cluster_key)}</guid>\n"
            f"      <pubDate>{format_datetime(story.lead.published_at)}</pubDate>\n"
            f"      <category>{xml_escape(cfg.categories[story.category].label)}</category>\n"
            f"      <description>{xml_escape(desc)}</description>\n"
            "    </item>"
        )

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">\n'
        "  <channel>\n"
        f"    <title>{xml_escape(cfg.title)}</title>\n"
        f"    <link>{xml_escape(base + '/')}</link>\n"
        f"    <description>{xml_escape(cfg.description)}</description>\n"
        f"    <language>{xml_escape(cfg.locale)}</language>\n"
        f"    <lastBuildDate>{format_datetime(edition.generated_at)}</lastBuildDate>\n"
        f'    <atom:link href="{xml_escape(base + "/feed.xml")}" rel="self" '
        'type="application/rss+xml"/>\n'
        + "\n".join(items) + "\n"
        "  </channel>\n"
        "</rss>\n"
    )


def build_sitemap(edition: Edition, cfg: Config, archive_dates: list[date]) -> str:
    base = f"{cfg.base_url}{cfg.base_path}"
    urls = [f"{base}/", f"{base}/archive/", f"{base}/sources.html", f"{base}/about.html"]
    urls += [f"{base}/category/{slug}/" for slug, c in cfg.categories.items() if c.weight > 0]
    urls += [f"{base}/archive/{d.isoformat()}/" for d in sorted(archive_dates, reverse=True)]

    body = "\n".join(f"  <url><loc>{xml_escape(u)}</loc></url>" for u in urls)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{body}\n</urlset>\n"
    )


def _robots(cfg: Config) -> str:
    return (
        "User-agent: *\n"
        "Allow: /\n"
        f"Sitemap: {cfg.base_url}{cfg.base_path}/sitemap.xml\n"
    )


def _anchor(story: Story) -> str:
    return f"s-{story.cluster_key}"


def _neighbours(day: date, dates: list[date]) -> tuple[date | None, date | None]:
    ordered = sorted(set(dates) | {day})
    i = ordered.index(day)
    return (ordered[i - 1] if i > 0 else None,
            ordered[i + 1] if i + 1 < len(ordered) else None)


def commit(ops: list[WriteOp], out: Path) -> int:
    """Write the whole batch. Called only after every render succeeded."""
    out.mkdir(parents=True, exist_ok=True)
    for op in ops:
        target = out / op.path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(op.data)
    return len(ops)


class _SiteView:
    """Read-only view of config for templates."""

    def __init__(self, cfg: Config) -> None:
        self.title = cfg.title
        self.tagline = cfg.tagline
        self.description = cfg.description
        self.base_url = cfg.base_url
        self.contact = cfg.contact
        self.categories = cfg.categories
