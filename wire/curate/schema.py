"""The single source of truth for both vendors' structured output.

One Pydantic definition generates the JSON Schema that Anthropic and OpenAI each
receive. Keeping one definition is what stops one vendor quietly becoming "the
real one" while the other drifts into a second-class code path.

Pydantic is optional at import time: if it is missing we fall back to
hand-written JSON Schema dicts so the heuristic path still runs on a minimal
install.
"""

from __future__ import annotations

from typing import Any

CATEGORIES = ("politics", "economy", "security", "frontier")

try:
    from pydantic import BaseModel, Field

    HAVE_PYDANTIC = True

    class WorkerStory(BaseModel):
        """Worker tier output: mechanical, per-cluster, high volume."""

        cluster_key: str = Field(
            description="Echo the cluster_key exactly as given. Never invent one.")
        category: str = Field(
            description=f"Exactly one of: {', '.join(CATEGORIES)}.")
        score: int = Field(
            ge=0, le=100,
            description="Newsworthiness for a Pakistani general reader today.")
        summary: str = Field(
            description=(
                "2-3 sentences in your own words describing what happened. "
                "Never copy phrasing from the supplied blurb."))
        keep: bool = Field(
            description=(
                "False for filler: horoscopes, listicles, celebrity gossip, "
                "routine sports results, syndicated news with no Pakistan angle."))

    class WorkerPass(BaseModel):
        stories: list[WorkerStory]

    class EditorStory(BaseModel):
        """Editor tier output: judgment on the workers' digest."""

        cluster_key: str = Field(description="Echo exactly. Never invent.")
        merged_from: list[str] = Field(
            default_factory=list,
            description=(
                "Other cluster_keys in this batch that are the SAME event and "
                "should fold into this one. Empty if none."))
        category: str = Field(description=f"One of: {', '.join(CATEGORIES)}.")
        score: int = Field(ge=0, le=100, description="Final ranking score.")
        summary: str = Field(
            description=(
                "The worker's draft, kept as-is if it is good, or rewritten in "
                "your own words if it is weak. Never copy the publisher."))
        why_it_matters: str = Field(
            description="One short clause on the consequence. Empty if routine.")
        confidence: str = Field(
            description="high, medium, or low. Low when outlets contradict.")

    class EditorPass(BaseModel):
        stories: list[EditorStory]
        dropped_cluster_keys: list[str] = Field(default_factory=list)

    class BriefPass(BaseModel):
        brief: str = Field(
            description=(
                "3-5 sentences on what today's edition amounts to. Reference "
                "stories by subject, never by number. Introduce no new facts."))
        theme: str = Field(description="A short phrase naming the day's throughline.")

except ImportError:  # pragma: no cover - minimal install
    HAVE_PYDANTIC = False
    WorkerStory = WorkerPass = EditorStory = EditorPass = BriefPass = None  # type: ignore


# --- Hand-written schemas -----------------------------------------------------
# Used when pydantic is unavailable, and as the canonical wire shape for the
# OpenAI strict-mode path, which requires additionalProperties:false and every
# property listed in `required`.

WORKER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stories"],
    "properties": {
        "stories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cluster_key", "category", "score", "summary", "keep"],
                "properties": {
                    "cluster_key": {"type": "string"},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "summary": {"type": "string"},
                    "keep": {"type": "boolean"},
                },
            },
        }
    },
}

EDITOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["stories", "dropped_cluster_keys"],
    "properties": {
        "stories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cluster_key", "merged_from", "category", "score",
                             "summary", "why_it_matters", "confidence"],
                "properties": {
                    "cluster_key": {"type": "string"},
                    "merged_from": {"type": "array", "items": {"type": "string"}},
                    "category": {"type": "string", "enum": list(CATEGORIES)},
                    "score": {"type": "integer", "minimum": 0, "maximum": 100},
                    "summary": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "confidence": {"type": "string",
                                   "enum": ["high", "medium", "low"]},
                },
            },
        },
        "dropped_cluster_keys": {"type": "array", "items": {"type": "string"}},
    },
}

BRIEF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["brief", "theme"],
    "properties": {
        "brief": {"type": "string"},
        "theme": {"type": "string"},
    },
}
