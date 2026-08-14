"""Two-tier curation with a deterministic floor.

    clusters ──► WORKER TIER (cheap, parallel batches)
                 categorise · draft summary · score · drop filler
                 sees headlines AND publisher blurbs
                      │  compact digest; blurbs discarded here
                      ▼
                 EDITOR TIER (frontier model, one call)
                 merge across batches · re-rank · cut · rewrite weak drafts
                 sees ONLY the digest
                      │
                      ▼
                 assemble() ──► quotas, floor, never-publish-nothing

The editor never receives a publisher blurb. That is the whole reason a
frontier-priced model is affordable here: it processes a fraction of the corpus,
on the one job where its judgement changes the product. `max_input_tokens` is
asserted before the call so that invariant cannot silently rot.

Every failure degrades one step rather than aborting: editor down means
worker-ranked, workers down means editor-direct, both down means heuristic.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any

from .. import cluster as cluster_mod
from .. import heuristic
from ..models import Cluster, CurationUnavailable, Story, TierUsage
from ..settings import Config
from ..util import shares_long_run, truncate
from . import prompt as prompts
from . import schema as schemas
from .providers import CallResult, Curator, chain_for, usage_from

log = logging.getLogger(__name__)

# Rough characters-per-token, used only for the pre-flight guard.
CHARS_PER_TOKEN = 3.7


class CurationOutcome:
    def __init__(self, stories: list[Story], brief: str | None, mode: str,
                 reason: str | None, usage: list[TierUsage],
                 plagiarism_rejections: int = 0) -> None:
        self.stories = stories
        self.brief = brief
        self.mode = mode
        self.reason = reason
        self.usage = usage
        self.plagiarism_rejections = plagiarism_rejections


def curate(clusters: list[Cluster], cfg: Config, now: datetime,
           *, force_heuristic: bool = False) -> CurationOutcome:
    if not clusters:
        return CurationOutcome([], None, "empty", "no new articles", [])

    if force_heuristic:
        return _heuristic_only(clusters, cfg, now, "forced")

    worker_chain = chain_for(cfg.worker)
    editor_chain = chain_for(cfg.editor)

    if not worker_chain and not editor_chain:
        return _heuristic_only(clusters, cfg, now, "no api key")

    usage: list[TierUsage] = []
    rejections = 0
    today = now.date().isoformat()
    cfg_summary = prompts.config_summary(cfg)

    # --- worker tier -------------------------------------------------------
    digest: list[dict[str, Any]] = []
    worker_reason: str | None = None
    if worker_chain:
        try:
            digest, worker_usage, rejections = _run_workers(
                clusters, cfg, worker_chain, cfg_summary, today)
            usage.append(worker_usage)
        except CurationUnavailable as exc:
            worker_reason = f"worker:{exc.reason}"
            log.warning("worker tier unavailable (%s)", exc.reason)

    if not digest:
        # No workers. The editor can read the clusters directly — correct, just
        # more expensive, so it is logged loudly rather than done quietly.
        if editor_chain:
            log.warning("worker tier produced nothing; editor reads clusters directly")
            digest = _digest_from_clusters(clusters, cfg, now)
        else:
            return _heuristic_only(clusters, cfg, now, worker_reason or "worker tier down")

    # --- editor tier -------------------------------------------------------
    editor_stories: list[Story] | None = None
    editor_reason: str | None = None
    if editor_chain:
        try:
            editor_stories, editor_usage = _run_editor(
                clusters, digest, cfg, editor_chain, cfg_summary, today, now)
            usage.append(editor_usage)
        except CurationUnavailable as exc:
            editor_reason = f"editor:{exc.reason}"
            log.warning("editor tier unavailable (%s); ranking worker output", exc.reason)
    else:
        editor_reason = "editor:not configured"

    if editor_stories is not None:
        stories = editor_stories
        mode = "ai"
        reason = worker_reason
    else:
        stories = _stories_from_digest(clusters, digest, cfg, now)
        mode = "mixed"
        reason = editor_reason

    stories = heuristic.assemble(stories, cfg)

    # --- brief -------------------------------------------------------------
    brief = None
    if stories and editor_chain and mode == "ai":
        try:
            brief, brief_usage = _run_brief(stories, cfg, editor_chain, today)
            usage.append(brief_usage)
        except CurationUnavailable as exc:
            log.info("brief unavailable (%s); publishing without it", exc.reason)

    return CurationOutcome(stories, brief, mode, reason, usage, rejections)


# ---------------------------------------------------------------------------
# Tiers
# ---------------------------------------------------------------------------


def _run_workers(clusters: list[Cluster], cfg: Config, chain: list[Curator],
                 cfg_summary: str, today: str) -> tuple[list[dict], TierUsage, int]:
    batches = [
        clusters[i:i + cfg.worker.batch_size]
        for i in range(0, len(clusters), cfg.worker.batch_size)
    ]
    payloads = [_worker_payload(b) for b in batches]

    def run(payload: list[dict]) -> CallResult | None:
        user = prompts.worker_user_turn(payload, cfg_summary, today)
        try:
            return _try_chain(chain, prompts.WORKER_SYSTEM, user,
                              schemas.WORKER_SCHEMA, "worker_pass", cfg, cfg.worker)
        except CurationUnavailable as exc:
            # One bad batch should not lose the other three.
            log.warning("worker batch failed (%s); its clusters fall back", exc.reason)
            return None

    workers = max(1, min(cfg.worker.parallel, len(payloads)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run, payloads))

    ok = [r for r in results if r is not None]
    if not ok:
        raise CurationUnavailable("all worker batches failed")

    by_key = {c.key: c for c in clusters}
    digest: list[dict[str, Any]] = []
    rejections = 0

    for result in ok:
        for row in result.data.get("stories", []) or []:
            key = str(row.get("cluster_key", ""))
            source = by_key.get(key)
            if source is None:
                continue  # the model invented a key; drop it silently
            if not row.get("keep", True):
                continue
            summary, rejected = _safe_summary(str(row.get("summary", "")), source, cfg)
            rejections += rejected
            digest.append({
                "cluster_key": key,
                "headline": source.lead.title,
                "summary": summary,
                "category": _safe_category(row.get("category"), source, cfg),
                "score": _safe_score(row.get("score")),
                "outlets": source.outlet_count,
                "owners": source.owner_count,
            })

    return digest, usage_from("worker", ok[0].model and chain[0].name or "", ok), rejections


def _run_editor(clusters: list[Cluster], digest: list[dict], cfg: Config,
                chain: list[Curator], cfg_summary: str, today: str,
                now: datetime) -> tuple[list[Story], TierUsage]:
    user = prompts.editor_user_turn(digest, cfg_summary, today, cfg.max_stories)

    # The guard that protects the entire economic premise of this design. If the
    # editor's input has grown to corpus size, refuse rather than pay frontier
    # rates on everything.
    limit = cfg.editor.max_input_tokens
    if limit:
        estimate = int((len(user) + len(prompts.EDITOR_SYSTEM)) / CHARS_PER_TOKEN)
        if estimate > limit:
            raise CurationUnavailable(
                f"input_too_large:{estimate}>{limit}")

    result = _try_chain(chain, prompts.EDITOR_SYSTEM, user,
                        schemas.EDITOR_SCHEMA, "editor_pass", cfg, cfg.editor)

    rows = result.data.get("stories", []) or []
    merge_groups = [
        [str(row.get("cluster_key", ""))] + [str(k) for k in (row.get("merged_from") or [])]
        for row in rows
        if row.get("merged_from")
    ]
    merged = cluster_mod.merge(clusters, merge_groups) if merge_groups else clusters
    by_key = {c.key: c for c in merged}
    # After a merge the surviving cluster has a new key, so map every original
    # member id back to whatever cluster now contains it.
    by_member = {m.id: c for c in merged for m in c.members}
    original = {c.key: c for c in clusters}

    stories: list[Story] = []
    used: set[str] = set()

    for row in rows:
        key = str(row.get("cluster_key", ""))
        target = by_key.get(key)
        if target is None:
            seed = original.get(key)
            if seed is None:
                continue
            target = by_member.get(seed.lead.id)
            if target is None:
                continue
        if target.key in used:
            continue
        used.add(target.key)

        summary, _ = _safe_summary(str(row.get("summary", "")), target, cfg)
        stories.append(Story(
            cluster_key=target.key,
            headline=target.lead.title,
            summary=summary,
            why_it_matters=truncate(str(row.get("why_it_matters", "")).strip(), 140),
            category=_safe_category(row.get("category"), target, cfg),
            score=float(_safe_score(row.get("score"))),
            lead=target.lead.redacted(),
            also_reported_by=tuple(m.redacted() for m in target.members[1:]),
            curated_by=f"editor:{result.model}",
            confidence=_safe_confidence(row.get("confidence")),
        ))

    if not stories:
        raise CurationUnavailable("editor returned nothing usable")

    return stories, usage_from("editor", chain[0].name, [result])


def _run_brief(stories: list[Story], cfg: Config, chain: list[Curator],
               today: str) -> tuple[str, TierUsage]:
    payload = [
        {"headline": s.headline, "summary": s.summary,
         "category": cfg.categories[s.category].label}
        for s in stories[:12]
    ]
    user = prompts.brief_user_turn(payload, today)
    result = _try_chain(chain, prompts.BRIEF_SYSTEM, user, schemas.BRIEF_SCHEMA,
                        "editors_brief", cfg, cfg.editor)
    brief = str(result.data.get("brief", "")).strip()
    if not brief:
        raise CurationUnavailable("empty brief")
    return brief, usage_from("brief", chain[0].name, [result])


def _try_chain(chain: list[Curator], system: str, user: str, schema: dict,
               schema_name: str, cfg: Config, tier: Any) -> CallResult:
    """Try each provider in turn. Only the last failure propagates."""
    last: CurationUnavailable | None = None
    for provider in chain:
        try:
            return provider.call(system, user, schema, schema_name, cfg, tier)
        except CurationUnavailable as exc:
            log.warning("provider %s failed (%s)", provider.name, exc.reason)
            last = exc
    raise last or CurationUnavailable("no provider available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _worker_payload(batch: list[Cluster]) -> list[dict[str, Any]]:
    return [{
        "cluster_key": c.key,
        "headline": c.lead.title,
        "outlets": c.outlet_count,
        "owners": c.owner_count,
        "filed": c.earliest.isoformat(),
        # The publisher blurb goes to the worker tier only, and never travels
        # onward into the editor's context.
        "blurb": " / ".join(m.dek for m in c.members[:3] if m.dek)[:600],
    } for c in batch]


def _digest_from_clusters(clusters: list[Cluster], cfg: Config,
                          now: datetime) -> list[dict[str, Any]]:
    out = []
    for c in clusters:
        score, category, _ = heuristic.score_cluster(c, cfg, now)
        out.append({
            "cluster_key": c.key,
            "headline": c.lead.title,
            "summary": heuristic.compose_summary(c, cfg, category),
            "category": category,
            "score": int(score),
            "outlets": c.outlet_count,
            "owners": c.owner_count,
        })
    return out


def _stories_from_digest(clusters: list[Cluster], digest: list[dict],
                         cfg: Config, now: datetime) -> list[Story]:
    """Worker output ranked without an editor. Still a real edition."""
    by_key = {c.key: c for c in clusters}
    stories: list[Story] = []
    for row in digest:
        source = by_key.get(row["cluster_key"])
        if source is None:
            continue
        stories.append(Story(
            cluster_key=source.key,
            headline=source.lead.title,
            summary=row["summary"],
            why_it_matters="",
            category=row["category"],
            score=float(row["score"]),
            lead=source.lead.redacted(),
            also_reported_by=tuple(m.redacted() for m in source.members[1:]),
            curated_by="worker",
            confidence="medium",
        ))
    return stories


def _heuristic_only(clusters: list[Cluster], cfg: Config, now: datetime,
                    reason: str) -> CurationOutcome:
    stories = heuristic.assemble(heuristic.build_stories(clusters, cfg, now), cfg)
    return CurationOutcome(stories, None, "heuristic", reason, [])


def _safe_summary(text: str, source: Cluster, cfg: Config) -> tuple[str, int]:
    """Validate a generated summary, replacing it if it echoes the publisher.

    The prompt forbids copying, but an instruction is not enforcement. Any
    summary sharing a long contiguous run with a publisher blurb is discarded in
    favour of the mechanical one.
    """
    text = " ".join((text or "").split())
    category = heuristic.assign_category(source, cfg)

    if len(text) < cfg.summary_min_chars:
        return heuristic.compose_summary(source, cfg, category), 0

    for member in source.members:
        if member.dek and shares_long_run(text, member.dek):
            log.warning("summary echoed publisher text for %r; using metadata instead",
                        source.lead.title[:60])
            return heuristic.compose_summary(source, cfg, category), 1

    return truncate(text, cfg.summary_max_chars), 0


def _safe_category(value: Any, source: Cluster, cfg: Config) -> str:
    slug = str(value or "").strip().lower()
    if slug in cfg.categories:
        return slug
    return heuristic.assign_category(source, cfg)


def _safe_score(value: Any) -> int:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return 50


def _safe_confidence(value: Any) -> str:
    v = str(value or "").strip().lower()
    return v if v in ("high", "medium", "low") else "medium"


__all__ = ["curate", "CurationOutcome"]
