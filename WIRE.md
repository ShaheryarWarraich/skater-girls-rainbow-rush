# The Pakistan Wire

An automated daily digest of Pakistani news. It reads the RSS feeds of Pakistani
newsrooms, groups the coverage so one event appears once, summarises each story
in its own words, ranks everything against criteria you control, and publishes a
static site every morning at 7am Pakistan time.

**It never reproduces article text.** Every item is a headline, an original
short summary, and prominent links back to the outlets that did the reporting.

---

## Try it in thirty seconds

No API key and no network required — this is the normal development loop:

```bash
pip install -r requirements.txt
python -m wire.cli doctor
python -m wire.cli --state /tmp/state build --offline --date 2026-08-14 --out /tmp/out
python -m http.server -d /tmp/out 8000
```

`--offline` reads the committed sample feeds in `fixtures/` instead of the
network. A missing fixture is treated as a 404, so the offline path exercises
the dead-feed handling rather than routing around it.

## The file you actually edit

**`config/curation.yml`.** It holds the four criteria and their weights:

```yaml
categories:
  politics:  {label: Politics & Governance, weight: 1.00, ...}
  economy:   {label: Economy & Business,    weight: 1.00, ...}
  security:  {label: Security & Foreign Policy, weight: 0.95, ...}
  frontier:  {label: Tech, Climate, Health, Education, Sport & Culture, weight: 0.70, ...}
```

Raise a weight to see more of that subject. Set one to `0.0` to switch it off
entirely. Weights are relative, so only the ratios matter. To see how a change
moves the rankings before you commit it:

```bash
python -m wire.cli --state /tmp/state build --offline --explain --dry-run
```

`config/feeds.yml` is the source list, and `config/site.yml` holds the title and
URLs.

## How curation works

Two tiers, so the expensive model never reads the whole corpus:

```
~120 clusters ──► WORKER TIER (cheap model, parallel batches)
                  categorise · draft a summary · score · drop filler
                  reads headlines AND publisher blurbs
                        │  compact digest — blurbs stop here
                        ▼
                  EDITOR TIER (frontier model, one call)
                  merge duplicates across batches · re-rank the whole day ·
                  cut filler · rewrite weak drafts · write the brief
                  reads ONLY the digest
                        │
                        ▼
                  quotas, floor, never-publish-nothing
```

The editor tier sees roughly an eighth of the tokens the workers do, which is
what makes a frontier-priced model affordable for the one job where its
judgement changes the product. `tiers.editor.max_input_tokens` is asserted
before the call, so if that ratio ever collapses the run refuses rather than
silently paying frontier rates on everything. Run with `--explain-cost` to see
the actual split.

### It is not tied to one vendor

`provider: auto` uses whichever key is present — `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`. Both tiers are independently configurable, and if the
chosen provider fails the other is tried automatically. One Pydantic schema
generates the JSON Schema both vendors receive, so neither is a second-class
path.

Below both sits a deterministic ranker with no network calls at all. The
degradation chain is:

| Situation | What gets published |
|---|---|
| Everything working | Full AI edition with an editor's brief |
| Editor tier fails | Worker summaries, deterministically ranked |
| Worker tier fails | Editor reads clusters directly (pricier, logged loudly) |
| No key, or both vendors down | Deterministic ranking, mechanical summaries |
| No new stories at all | Yesterday's edition, untouched, with a dated banner |

Every degraded edition says so on the page. Silent degradation is how you ship a
broken site for three weeks without noticing.

## Commands

```bash
python -m wire.cli doctor                     # preflight: deps, keys, config, feed health
python -m wire.cli build --out docs           # the real thing
python -m wire.cli report-health              # per-feed status; exits 1 if any are quarantined
python -m wire.cli clean --older-than 0       # wipe seen-state and force a full republish
```

Useful `build` flags: `--offline`, `--force-heuristic`, `--dry-run`, `--explain`,
`--explain-cost`, `--date YYYY-MM-DD`, `--record` (refresh fixtures from live
feeds).

Note that global options come before the subcommand: `--state X build`, not
`build --state X`.

## Deployment

1. Settings → Pages → Source: **Deploy from a branch**, `main` / `/docs`.
2. Add `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` as repository secrets. With
   neither, the site still publishes — just deterministically ranked.
3. `.github/workflows/publish.yml` runs daily at 02:00 UTC and commits `docs/`
   and `state/` back to `main`.

The publish workflow has no `push` trigger. That is deliberate and load-bearing:
it commits to the repository, so a push trigger would make it retrigger itself
forever.

## A caveat about the feed list

The feed URLs in `config/feeds.yml` could not be reached from the environment
this was built in, so treat them as a starting point rather than verified truth.
The first live run validates every one of them and writes the result to
`docs/sources.html` and `state/feeds_health.json`. Feeds that fail five runs in
a row are quarantined, re-probed weekly, and reported as a GitHub issue.

News sites move their feeds without notice, so this is ongoing maintenance
rather than a one-time fix — which is why the health table is a public page
rather than a log line.

## Tests

```bash
python -m pytest tests/ -q
```

They cover the four things that would make the site *wrong* rather than merely
worse: publisher prose never reaching the output, published stories never
reappearing, clustering merging real duplicates without chaining unrelated
stories, and the no-key path still producing a complete site.

## Licence and attribution

The code here is yours. The reporting is not: every story links to the outlet
that produced it, and `docs/sources.html` lists every source with its homepage.
If you are an outlet listed here and want to be removed, delete the entry from
`config/feeds.yml` — or email the address in `config/site.yml` and it will be
done.
