"""Group articles that describe the same event.

Deterministic and always run — it is both the cost reducer for the AI pass (the
model refines groups instead of building them from nothing) and the only
clustering available when there is no API key.

The approach is IDF-weighted Jaccard over title tokens. Plain Jaccard fails here
because Pakistani news headlines share a large common vocabulary ("PM", "govt",
"Pakistan", "says"), so two unrelated stories can look similar. Weighting by
inverse document frequency means the words that actually decide a match are the
rare ones — place names, surnames, institutions.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict

from .models import Article, Cluster
from .util import stable_key

# High-frequency words that carry almost no identifying signal in this corpus.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by
with without into onto over under after before during about against between
is are was were be been being has have had do does did will would shall should
can could may might must not no nor so such as it its his her their our your my
he she they we you i who whom whose which what when where why how
said says say told tells according reported reports report new latest update
pakistan pakistani govt government minister ministry official officials pm cm
president chief secretary spokesperson today yesterday day week month year
amid over after ahead following ban call calls calling get gets got make makes
made take takes taken set sets put puts one two three first second last next
more most less least high low big small top down up out off back
news story video photo watch live breaking exclusive
""".split())

# Acronyms and short institution names that are highly identifying despite being
# short. Without this they would be filtered out by the length rule below.
KEEP_SHORT = frozenset("""
imf psx sbp fbr nab ecp pti pml ppp mqm jui ttp isi loc cpec adb wto gdp cpi
kse ogra nepra pia psl pcb pmd nadra fia ihc lhc shc phc bhc sc un us uk eu
""".split())

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'&-]*")

# Thresholds calibrated against a labelled set of cross-outlet pairs rather than
# guessed: at these values the fixture corpus yields 6/6 true merges and zero
# false merges. Re-run tests/test_cluster.py after changing any of them.
#
# Headline similarity sufficient on its own to call two stories one event.
TITLE_MERGE_THRESHOLD = 0.42
# Independently written headlines about one event often share only one or two
# words ("Balochistan"), so a weaker headline match can still merge — but only
# when the publisher blurbs independently corroborate it.
TITLE_ASSIST_THRESHOLD = 0.07
DEK_ASSIST_THRESHOLD = 0.13
DISTINCTIVE_MAX_DF = 3
DISTINCTIVE_MATCHES_REQUIRED = 1
DEK_TOKEN_CHARS = 240


def tokenize(text: str) -> set[str]:
    out: set[str] = set()
    for raw in TOKEN_RE.findall((text or "").lower()):
        token = raw.strip("'&-")
        if not token:
            continue
        if token in KEEP_SHORT:
            out.add(token)
            continue
        if token in STOPWORDS or len(token) < 4 or token.isdigit():
            continue
        out.add(token)
    return out


def _idf(docs: list[set[str]]) -> dict[str, float]:
    n = max(1, len(docs))
    df: dict[str, int] = defaultdict(int)
    for doc in docs:
        for token in doc:
            df[token] += 1
    return {t: math.log((n + 1) / (c + 0.5)) for t, c in df.items()}


def _weighted_jaccard(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    union = a | b
    num = sum(idf.get(t, 1.0) for t in inter)
    den = sum(idf.get(t, 1.0) for t in union)
    return num / den if den else 0.0


def precluster(articles: list[Article]) -> list[Cluster]:
    if not articles:
        return []
    if len(articles) == 1:
        a = articles[0]
        return [Cluster(key=stable_key([a.id]), members=(a,))]

    # Titles carry the identity of a story; the publisher blurb carries
    # corroborating detail. Two outlets covering one event often share only a
    # single title word ("Balochistan") but several blurb words, so both are
    # used for similarity — while a shared *title* token is still required
    # before a pair is even considered, which keeps blurb noise from merging
    # unrelated stories.
    titles = [tokenize(a.title) for a in articles]
    deks = [tokenize(a.dek[:DEK_TOKEN_CHARS]) for a in articles]
    idf = _idf([t | d for t, d in zip(titles, deks)])

    df: dict[str, int] = defaultdict(int)
    for doc in titles:
        for token in doc:
            df[token] += 1

    # Lead-anchored agglomeration rather than union-find.
    #
    # Transitive closure is the wrong model for news: on a flood day, "PM
    # announces relief package" resembles "PM chairs security meeting" (both
    # mention the PM), which resembles "security forces operation" — and
    # union-find happily chains all three into one cluster of unrelated
    # stories. Requiring every member to match the cluster's LEAD instead of
    # any member makes that impossible, because the lead is a fixed reference
    # point rather than a moving frontier.
    order = sorted(range(len(articles)),
                   key=lambda i: (-articles[i].source_weight,
                                  articles[i].published_at, articles[i].id))

    leads: list[int] = []
    members_of: dict[int, list[int]] = {}

    for i in order:
        best_lead, best_sim = -1, 0.0
        for lead in leads:
            if not (titles[i] & titles[lead]):
                # No shared headline word: two stories that merely mention the
                # same place or person.
                continue
            sim = _similarity(titles[i], titles[lead], deks[i], deks[lead], idf, df)
            if sim > best_sim:
                best_lead, best_sim = lead, sim
        if best_lead >= 0:
            members_of[best_lead].append(i)
        else:
            leads.append(i)
            members_of[i] = [i]

    clusters: list[Cluster] = []
    for lead, idxs in members_of.items():
        members = [articles[i] for i in idxs]
        # Lead = most trusted outlet, then earliest. The lead's headline is what
        # gets published, so it should come from the most reliable source that
        # carried the story.
        ordered = sorted(members, key=lambda a: (-a.source_weight, a.published_at, a.id))
        clusters.append(Cluster(
            key=stable_key([m.id for m in ordered]),
            members=tuple(ordered),
        ))

    clusters.sort(key=lambda c: (c.earliest, c.key))
    return clusters


def _similarity(title_a: set[str], title_b: set[str], dek_a: set[str],
                dek_b: set[str], idf: dict[str, float],
                df: dict[str, int]) -> float:
    """Similarity in [0, 1], or 0.0 when the pair fails to qualify at all.

    The headline is the primary evidence — it is what the newsroom chose to say
    the story *is*. The blurb only corroborates a headline match that is already
    plausible; it never carries a merge on its own, because two stories on the
    same topic share plenty of blurb vocabulary without being the same event.
    """
    title_sim = _weighted_jaccard(title_a, title_b, idf)

    if title_sim >= TITLE_MERGE_THRESHOLD:
        return title_sim

    shared = title_a & title_b
    distinctive = sum(1 for t in shared if df.get(t, 99) <= DISTINCTIVE_MAX_DF)
    if distinctive < DISTINCTIVE_MATCHES_REQUIRED or title_sim < TITLE_ASSIST_THRESHOLD:
        return 0.0

    # The blurb is corroboration, never the sole basis for a merge: two stories
    # on one topic share plenty of blurb vocabulary without being one event.
    dek_sim = _weighted_jaccard(dek_a, dek_b, idf)
    if dek_sim < DEK_ASSIST_THRESHOLD:
        return 0.0
    return title_sim + 0.25 * dek_sim


def merge(clusters: list[Cluster], groups: list[list[str]]) -> list[Cluster]:
    """Apply merge instructions (cluster_key lists) from the AI pass."""
    by_key = {c.key: c for c in clusters}
    merged: list[Cluster] = []
    consumed: set[str] = set()

    for group in groups:
        keys = [k for k in group if k in by_key and k not in consumed]
        if len(keys) < 2:
            continue
        members: list[Article] = []
        for k in keys:
            members.extend(by_key[k].members)
            consumed.add(k)
        seen: set[str] = set()
        unique = [m for m in members if not (m.id in seen or seen.add(m.id))]
        ordered = sorted(unique, key=lambda a: (-a.source_weight, a.published_at, a.id))
        merged.append(Cluster(key=stable_key([m.id for m in ordered]),
                              members=tuple(ordered)))

    merged.extend(c for c in clusters if c.key not in consumed)
    merged.sort(key=lambda c: (c.earliest, c.key))
    return merged
