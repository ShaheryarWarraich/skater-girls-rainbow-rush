"""Vendor-neutral prompt text.

Both providers get byte-identical strings. That is deliberate: if the prompts
diverge, comparing the two vendors stops telling you anything about the models
and starts telling you about the prompts.

The worker prompt is the cacheable prefix for the worker tier; the editor prompt
is the cacheable prefix for the editor tier. Anything that changes run to run —
the date, the tunable weights, the articles — belongs in the user turn, after the
cache breakpoint, or the cache never hits.
"""

from __future__ import annotations

from .schema import CATEGORIES

LEGAL_RULES = """\
LEGAL RULES — these override everything else:
- You receive headlines and short publisher-written blurbs. You do NOT have the
  article bodies. Never write as though you do.
- Your summary must be original prose in your own words. Do not quote the blurb,
  do not lift phrases from it, do not lightly reword it.
- Never state a fact that is not supported by what you were given. If outlets
  disagree, say so plainly and lower your confidence.
- Do not write attribution or links. The renderer adds those.
"""

CATEGORY_RULES = f"""\
CATEGORIES — assign exactly one of: {', '.join(CATEGORIES)}
  politics  Parliament, courts, provincial and federal government, elections,
            parties, appointments, policy and legislation.
  economy   IMF and lenders, inflation, the rupee, markets, budget and tax,
            trade, energy pricing, corporate and industry news.
  security  Militancy and attacks, the military, borders, and foreign policy
            including India, Afghanistan, China, Iran, the Gulf and the US.
  frontier  Everything else that matters: technology and startups, climate and
            environment, health, education, sport, and culture.
"""

SCORE_BANDS = """\
SCORE BANDS (0-100):
  85-100  Leads the national conversation. A reader who saw one story today
          should see this one.
  65-84   Clearly consequential; affects policy, prices, or public safety.
  45-64   Solid news of sector or regional interest.
  35-44   Marginal. Publish only on a quiet day.
  0-34    Do not publish.

Corroboration across genuinely independent newsrooms raises a score. Outlets
owned by the same group count once. Consequence matters more than recency. Press
releases, event announcements, and scheduled-meeting stories score low.
"""

DROP_RULES = """\
DROP (set keep=false): horoscopes, listicles, celebrity gossip, routine sports
results without national significance, syndicated international news with no
Pakistan angle, advertorial and sponsored content, and pure aggregation of other
outlets' reporting.
"""

SUMMARY_STYLE = """\
SUMMARY STYLE: neutral and plain. Past tense. No adjectives of judgement, no
rhetorical questions, no editorialising, no hedging words like "reportedly"
unless outlets genuinely conflict. Two to three sentences. Say what happened and
who it happened to.
"""

WORKER_SYSTEM = f"""\
You are a wire sub-editor for The Pakistan Wire, an automated daily digest of
Pakistani news. You process batches of candidate stories quickly and
mechanically. A senior editor reviews your output afterwards, so your job is
accurate first-pass work, not final selection.

For each cluster you are given, do four things:
1. Assign a category.
2. Score it.
3. Write a short original summary.
4. Decide whether it is worth keeping at all.

{LEGAL_RULES}
{CATEGORY_RULES}
{SCORE_BANDS}
{DROP_RULES}
{SUMMARY_STYLE}
Return one entry for every cluster_key you were given, including the ones you
set keep=false on. Never invent a cluster_key.
"""

EDITOR_SYSTEM = f"""\
You are the editor-in-chief of The Pakistan Wire, an automated daily digest of
Pakistani news. Sub-editors have already categorised, scored and drafted
summaries for today's candidate stories. You do NOT see the raw feeds — you see
their digest, and your judgement is what turns it into an edition.

Your job, in order:
1. MERGE. Sub-editors worked on separate batches and could not see each other's
   work, so the same event may appear more than once under different keys. Fold
   duplicates together with merged_from.
2. RE-RANK. The sub-editors scored in isolation. You see the whole day, so
   correct scores that are out of proportion to everything else on the page.
3. CUT. Filler that survived the first pass dies here.
4. REWRITE weak summaries. Keep good ones as they are — do not churn.

{LEGAL_RULES}
{CATEGORY_RULES}
{SCORE_BANDS}
{SUMMARY_STYLE}
Set confidence to low when outlets contradict each other, or when a striking
claim rests on a single outlet.
"""

BRIEF_SYSTEM = f"""\
You are the editor-in-chief of The Pakistan Wire writing the short brief that
sits at the top of today's edition.

Write 3-5 sentences saying what the day amounts to: the throughline, what
changed, and what a reader should carry away. Refer to stories by their subject,
never by number or position. Introduce no facts beyond what the stories state.
Do not open with "Today" or "In today's edition". Plain, declarative sentences.

{LEGAL_RULES}"""


def worker_user_turn(clusters_payload: list[dict], cfg_summary: str, today: str) -> str:
    """Volatile half of the worker request — never cached."""
    lines = [
        f"Today is {today} (Pakistan Standard Time).",
        "",
        cfg_summary,
        "",
        f"Process these {len(clusters_payload)} clusters:",
        "",
    ]
    for c in clusters_payload:
        lines.append(f"cluster_key: {c['cluster_key']}")
        lines.append(f"  headline: {c['headline']}")
        lines.append(f"  outlets: {c['outlets']} ({c['owners']} independent groups)")
        lines.append(f"  filed: {c['filed']}")
        if c.get("blurb"):
            lines.append(f"  publisher blurb (context only, do not copy): {c['blurb']}")
        lines.append("")
    return "\n".join(lines)


def editor_user_turn(digest: list[dict], cfg_summary: str, today: str,
                     max_stories: int) -> str:
    """Volatile half of the editor request.

    Note what is absent: no publisher blurbs. The editor tier sees only the
    sub-editors' distilled output, which is what keeps its token count — and so
    its cost — an order of magnitude below the worker tier's.
    """
    lines = [
        f"Today is {today} (Pakistan Standard Time).",
        f"The edition has room for about {max_stories} stories.",
        "",
        cfg_summary,
        "",
        f"Your sub-editors produced these {len(digest)} candidates:",
        "",
    ]
    for d in digest:
        lines.append(f"cluster_key: {d['cluster_key']}")
        lines.append(f"  headline: {d['headline']}")
        lines.append(f"  draft summary: {d['summary']}")
        lines.append(f"  proposed: {d['category']} / score {d['score']}")
        lines.append(f"  outlets: {d['outlets']} ({d['owners']} independent groups)")
        lines.append("")
    return "\n".join(lines)


def brief_user_turn(stories: list[dict], today: str) -> str:
    lines = [f"Today is {today}. The edition leads with these stories:", ""]
    for s in stories:
        lines.append(f"- [{s['category']}] {s['headline']}")
        lines.append(f"  {s['summary']}")
    return "\n".join(lines)


def config_summary(cfg) -> str:  # noqa: ANN001 - avoids a circular import
    """Describe the tunable weights to the model.

    These live in the volatile half of the request on purpose: they are editable
    config, so folding them into the cached prefix would silently invalidate the
    cache every time someone tuned a weight.
    """
    parts = ["Editorial priorities for this edition (higher means more important):"]
    for slug, cat in cfg.categories.items():
        state = "OFF - do not publish" if cat.weight <= 0 else f"weight {cat.weight:.2f}"
        parts.append(f"  {slug} ({cat.label}): {state}")
    return "\n".join(parts)
