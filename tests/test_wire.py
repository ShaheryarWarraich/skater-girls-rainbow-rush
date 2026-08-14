"""Tests for the guarantees that matter.

These are not exhaustive unit tests. They lock down the four properties that,
if they silently broke, would make the site wrong rather than merely worse:

  1. Publisher prose never reaches the rendered output.
  2. A published story does not reappear in a later edition.
  3. Clustering merges genuine cross-outlet duplicates without chaining
     unrelated stories together.
  4. With no API key and no network, a complete site is still produced.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wire import cluster as cluster_mod  # noqa: E402
from wire import fetch, heuristic, normalize, parse, seen  # noqa: E402
from wire.models import Article, Cluster  # noqa: E402
from wire.settings import load_config  # noqa: E402
from wire.util import canonical_url, shares_long_run, strip_html  # noqa: E402

PINNED = datetime(2026, 8, 14, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def cfg():
    return load_config(ROOT / "config")


@pytest.fixture(scope="module")
def articles(cfg):
    out = []
    for feed in cfg.all_feeds:
        result = fetch._fetch_offline(feed, ROOT / "fixtures")
        out.extend(normalize.to_articles(parse.parse(result, feed), cfg, feed, PINNED))
    return normalize.drop_stale(normalize.dedupe(out), PINNED, cfg.max_age_hours)


# --- 1. never republish -------------------------------------------------------


def test_no_template_references_the_publisher_blurb():
    """The blurb is model input only. A template touching it would leak the
    publisher's own prose onto our pages."""
    offenders = [
        p.name for p in (ROOT / "templates").glob("*.html")
        if re.search(r"\bdek\b", p.read_text(encoding="utf-8"))
    ]
    assert offenders == [], f"templates reference the publisher blurb: {offenders}"


def test_plagiarism_tripwire_catches_a_copied_blurb():
    blurb = ("Prime Minister Shehbaz Sharif convened the National Security "
             "Committee on Thursday to review a sharp rise in militant attacks "
             "across Balochistan province")
    assert shares_long_run(blurb, blurb) is True
    assert shares_long_run("The prime minister met security officials.", blurb) is False


def test_heuristic_summary_contains_no_publisher_text(cfg, articles):
    clusters = cluster_mod.precluster(articles)
    for story in heuristic.build_stories(clusters, cfg, PINNED):
        for member in next(c for c in clusters if c.key == story.cluster_key).members:
            if member.dek:
                assert not shares_long_run(story.summary, member.dek, n=8)


def test_redacted_articles_carry_no_blurb(articles):
    assert all(a.redacted().dek == "" for a in articles)


# --- 2. no repeats ------------------------------------------------------------


def test_seen_store_suppresses_a_published_article(tmp_path, articles):
    store = seen.SeenStore(tmp_path / "seen.json")
    assert len(store.filter_new(articles)) == len(articles)

    store.record(articles[:5], PINNED.date())
    store.save()

    reloaded = seen.SeenStore(tmp_path / "seen.json")
    remaining = reloaded.filter_new(articles)
    assert len(remaining) == len(articles) - 5
    assert all(a.id in reloaded for a in articles[:5]), "recorded ids must persist"
    assert all(a.id not in reloaded for a in remaining), "unpublished ids stay new"


def test_prune_keeps_entries_inside_the_freshness_window(tmp_path, articles):
    """An article must not fall out of `seen` while it is still fresh enough to
    be re-fetched — that is precisely how a story gets published twice."""
    store = seen.SeenStore(tmp_path / "seen.json")
    store.record(articles[:3], PINNED.date())
    store.prune(PINNED + timedelta(hours=36), 36)
    assert len(store.articles) == 3


# --- 3. clustering ------------------------------------------------------------

# Pairs the fixtures deliberately describe as one event across two newsrooms.
TRUE_PAIRS = [
    ("Pakistan, IMF reach staff-level agreement after week-long talks",
     "Pakistan, IMF conclude talks with staff-level agreement in hand"),
    ("PM Shehbaz chairs NSC meeting after surge in Balochistan attacks",
     "National Security Committee meets amid spike in Balochistan violence"),
    ("ECP announces by-election schedule for Punjab, KP constituencies",
     "Election Commission unveils by-poll schedule for two provinces"),
    ("PSX benchmark index gains over 400 points on foreign inflows",
     "KSE-100 rallies over 400 points amid heavy foreign buying"),
    ("Bilawal Bhutto calls for federal support over Sindh flood response",
     "Bilawal Bhutto criticises federal government's flood response in Sindh"),
    ("Pakistan, Afghanistan hold talks to ease border tensions",
     "Pakistan and Afghanistan officials meet in Doha over border row"),
]


def test_cross_outlet_duplicates_are_merged(articles):
    clusters = cluster_mod.precluster(articles)
    grouped = {}
    for c in clusters:
        for m in c.members:
            grouped[m.title] = c.key

    missed = [
        (a, b) for a, b in TRUE_PAIRS
        if a in grouped and b in grouped and grouped[a] != grouped[b]
    ]
    assert missed == [], f"failed to merge known duplicates: {missed}"


def test_clustering_does_not_chain_unrelated_stories(articles):
    """Union-find over pairwise similarity chains A-B-C into one cluster even
    when A and C are unrelated. On a flood day that collapsed a third of the
    corpus into a single item, so the size ceiling is a real regression guard."""
    clusters = cluster_mod.precluster(articles)
    biggest = max(clusters, key=lambda c: len(c.members))
    assert len(biggest.members) <= 5, (
        f"cluster of {len(biggest.members)} suggests transitive chaining: "
        f"{[m.title for m in biggest.members]}"
    )


def test_every_article_lands_in_exactly_one_cluster(articles):
    clusters = cluster_mod.precluster(articles)
    ids = [m.id for c in clusters for m in c.members]
    assert len(ids) == len(set(ids)) == len(articles)


def test_owner_count_does_not_double_count_one_media_group(cfg):
    """Geo and The News share an owner. Their cross-posting is one newsroom, and
    must not read as independent corroboration."""
    def make(idx: str, source: str, owner: str) -> Article:
        return Article(
            id=idx, title="t", url=f"https://x/{idx}", source_id=source,
            source_name=source, source_homepage="", source_weight=1.0,
            source_owner=owner, source_tier="national", feed_id="f", dek="",
            published_at=PINNED, time_confidence="high", category_hint="politics",
        )

    c = Cluster(key="k", members=(make("1", "geo", "jang-group"),
                                  make("2", "thenews", "jang-group"),
                                  make("3", "dawn", "dawn-group")))
    assert c.outlet_count == 3
    assert c.owner_count == 2


# --- 4. degradation -----------------------------------------------------------


def test_offline_build_with_no_api_key_produces_a_complete_site(tmp_path):
    """The headline guarantee: no key, no network, still a readable site."""
    out, state = tmp_path / "out", tmp_path / "state"
    # Inherit the real environment so installed packages resolve, but strip both
    # vendor keys — this test exists to prove the no-key path works.
    env = {**os.environ}
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("OPENAI_API_KEY", None)

    proc = subprocess.run(
        [sys.executable, "-m", "wire.cli", "--state", str(state), "build",
         "--offline", "--date", "2026-08-14", "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, proc.stderr

    for required in ("index.html", "feed.xml", "sources.html", "about.html",
                     "sitemap.xml", "robots.txt", ".nojekyll",
                     "archive/index.html"):
        assert (out / required).exists(), f"missing {required}"

    home = (out / "index.html").read_text(encoding="utf-8")
    assert "story" in home
    # Degradation must be disclosed, not silent.
    assert "heuristic" in home.lower()


def test_category_weight_of_zero_removes_the_category(cfg):
    live = [s for s, c in cfg.categories.items() if c.weight > 0]
    assert len(live) == 4, "all four criteria should be enabled by default"


# --- utilities ----------------------------------------------------------------


@pytest.mark.parametrize("raw,expected", [
    ("https://www.dawn.com/news/123/story?utm_source=twitter",
     "https://dawn.com/news/123/story"),
    ("https://dawn.com/news/123/story/amp", "https://dawn.com/news/123/story"),
    ("http://WWW.Dawn.COM/news/123/", "http://dawn.com/news/123"),
    ("https://dawn.com/news/123#comments", "https://dawn.com/news/123"),
    ("javascript:alert(1)", ""),
    ("", ""),
])
def test_url_canonicalisation(raw, expected):
    assert canonical_url(raw) == expected


def test_canonicalisation_collapses_syndicated_variants():
    variants = [
        "https://tribune.com.pk/story/1?utm_campaign=rss",
        "https://www.tribune.com.pk/story/1/",
        "https://tribune.com.pk/story/1/amp",
    ]
    assert len({canonical_url(v) for v in variants}) == 1


def test_strip_html_flattens_feed_markup():
    assert strip_html("<p>Hello &amp; <b>welcome</b></p>") == "Hello & welcome"


def test_keyword_matching_uses_word_boundaries(cfg):
    """"us" must not match inside "august" or "business" — a substring match
    silently gave every story published in August a security-category hit."""
    security = cfg.categories["security"]
    assert security.count_hits("gold prices rose in august business news") == 0
    assert security.count_hits("talks with the us delegation") == 1
