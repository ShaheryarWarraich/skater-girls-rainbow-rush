"""The deterministic ranker: the floor the site never falls below.

Built before the AI path on purpose. A fallback written after the primary path
is a fallback that has never actually run, and this one has to work on the worst
day — no key, API down, everything refused.

It is also the shared assembly stage: both AI and heuristic curation end up in
`assemble()`, so quotas and the never-publish-nothing rule are enforced in
exactly one place regardless of which produced the scores.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .models import Article, Cluster, Story
from .settings import Config

# The theoretical maximum of the multiplied signals, used to map raw scores onto
# a 0-100 scale that means the same thing across runs.
RAW_CEILING = 6.0

# Hard lower bound the never-publish-nothing rule may not cross.
ABSOLUTE_FLOOR = 12.0


def score_cluster(cluster: Cluster, cfg: Config, now: datetime) -> tuple[float, str, dict]:
    """Return (score, category, breakdown). Breakdown feeds `--explain`."""
    sig = cfg.signals
    category = assign_category(cluster, cfg)
    cat_weight = cfg.categories[category].weight

    source = max(m.source_weight for m in cluster.members) * sig.get("source_weight", 1.0)

    # Corroboration counts distinct OWNERS, not outlets. Geo and The News share
    # an owner; their cross-posting is one newsroom, not two confirmations.
    corrob = 1.0 + sig.get("corroboration", 0.9) * math.log2(1 + cluster.owner_count)

    half_life = max(1.0, sig.get("recency_half_life_h", 10.0))
    age_h = max(0.0, (now - cluster.earliest).total_seconds() / 3600.0)
    recency = 0.5 ** (age_h / half_life)

    hits = _keyword_hits(cluster, cfg, category)
    keyword = 1.0 + sig.get("keyword_hit", 0.6) * min(1.0, hits / 3.0)

    best_pos = min(m.position for m in cluster.members)
    position = 1.0 + sig.get("lead_position", 0.25) * (1.0 - min(best_pos, 9) / 10.0)

    time_pen = 0.85 if cluster.lead.time_confidence == "low" else 1.0

    raw = source * corrob * recency * keyword * position * cat_weight * time_pen
    score = max(0.0, min(100.0, 100.0 * raw / RAW_CEILING))

    return score, category, {
        "source": round(source, 3),
        "corroboration": round(corrob, 3),
        "recency": round(recency, 3),
        "keyword": round(keyword, 3),
        "position": round(position, 3),
        "category_weight": round(cat_weight, 3),
        "time_penalty": time_pen,
        "owners": cluster.owner_count,
        "outlets": cluster.outlet_count,
        "age_hours": round(age_h, 1),
        "raw": round(raw, 4),
    }


def assign_category(cluster: Cluster, cfg: Config) -> str:
    """Pick the best-fitting category by weighted keyword hits.

    Ties fall back to the feed's own hint, then to `frontier` as the residual
    bucket — a story we cannot classify is more likely soft news than a missed
    front-page political story.
    """
    haystack = " ".join(f"{m.title} {m.dek}" for m in cluster.members).lower()

    best_slug, best_score = "", -1.0
    for slug, cat in cfg.categories.items():
        weighted = cat.count_hits(haystack) * max(cat.weight, 0.01)
        if weighted > best_score:
            best_slug, best_score = slug, weighted

    if best_score <= 0:
        hint = cluster.lead.category_hint
        if hint in cfg.categories and cfg.categories[hint].weight > 0:
            return hint
        return "frontier" if cfg.categories["frontier"].weight > 0 else best_slug
    return best_slug


def _keyword_hits(cluster: Cluster, cfg: Config, category: str) -> int:
    haystack = " ".join(f"{m.title} {m.dek}" for m in cluster.members).lower()
    return cfg.categories[category].count_hits(haystack)


def compose_summary(cluster: Cluster, cfg: Config, category: str) -> str:
    """Build a factual, non-infringing summary from metadata alone.

    Deliberately NOT the publisher's RSS blurb. That blurb is their copyrighted
    prose, and reproducing it is exactly the republication this project promises
    not to do — the fact that RSS makes it easy does not make it ours.
    """
    label = cfg.categories[category].label
    lead = cluster.lead
    others = cluster.outlet_count - 1

    if others == 1:
        coverage = f"Reported by {lead.source_name} and one other outlet"
    elif others > 1:
        coverage = f"Reported by {lead.source_name} and {others} other outlets"
    else:
        coverage = f"Reported by {lead.source_name}"

    try:
        local = cluster.earliest.astimezone(ZoneInfo(cfg.timezone))
        when = f" First filed {local:%H:%M} PKT."
    except Exception:  # noqa: BLE001 - a bad tz name must not break the build
        when = ""

    return f"{label} · {coverage}.{when}"


def build_stories(clusters: list[Cluster], cfg: Config, now: datetime) -> list[Story]:
    stories: list[Story] = []
    for cluster in clusters:
        score, category, _ = score_cluster(cluster, cfg, now)
        stories.append(Story(
            cluster_key=cluster.key,
            headline=cluster.lead.title,
            summary=compose_summary(cluster, cfg, category),
            why_it_matters="",
            category=category,
            score=score,
            lead=cluster.lead.redacted(),
            also_reported_by=tuple(m.redacted() for m in cluster.members[1:]),
            curated_by="heuristic",
            confidence="medium",
        ))
    return stories


def assemble(stories: list[Story], cfg: Config) -> list[Story]:
    """Apply the editorial quotas and the never-publish-nothing rule.

    Quotas are enforced here rather than asked of the model, because a model
    scoring one batch cannot see the global distribution it is supposed to
    respect.
    """
    ranked = sorted(stories, key=lambda s: (-s.score, s.cluster_key))

    picked = _greedy_pick(ranked, cfg, cfg.min_score)

    # A quiet news day, a threshold set too high, or a bad feed morning should
    # still produce a readable page. But "never publish nothing" is not "publish
    # anything": below the absolute floor we would rather show yesterday's
    # edition than pad today's with material we have already judged worthless.
    if len(picked) < 3 and ranked:
        relaxed = _greedy_pick(ranked, cfg, ABSOLUTE_FLOOR)
        if len(relaxed) > len(picked):
            picked = relaxed[:max(5, len(picked))]

    return picked


def _greedy_pick(ranked: list[Story], cfg: Config, min_score: float) -> list[Story]:
    picked: list[Story] = []
    per_category: dict[str, int] = {}
    per_lead: dict[str, int] = {}

    for story in ranked:
        if len(picked) >= cfg.max_stories:
            break
        if story.score < min_score:
            continue
        if per_category.get(story.category, 0) >= cfg.max_per_category:
            continue
        if per_lead.get(story.lead.source_id, 0) >= cfg.max_per_source_as_lead:
            continue
        picked.append(story)
        per_category[story.category] = per_category.get(story.category, 0) + 1
        per_lead[story.lead.source_id] = per_lead.get(story.lead.source_id, 0) + 1

    return picked


def explain(clusters: list[Cluster], cfg: Config, now: datetime) -> list[dict]:
    """Per-cluster score breakdown — the tuning tool for curation.yml."""
    rows = []
    for cluster in clusters:
        score, category, breakdown = score_cluster(cluster, cfg, now)
        rows.append({
            "score": round(score, 2),
            "category": category,
            "headline": cluster.lead.title,
            "outlets": cluster.outlet_count,
            "owners": cluster.owner_count,
            **breakdown,
        })
    rows.sort(key=lambda r: -r["score"])
    return rows
