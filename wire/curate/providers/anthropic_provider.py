"""Anthropic provider.

The current Claude models reject several parameters that older code habitually
sends, and each rejection is a hard 400 rather than a degradation, so the rules
are encoded here once:

  * No `temperature` / `top_p` / `top_k`.
  * No `budget_tokens` — thinking depth is controlled by `effort`.
  * No trailing assistant message (prefill).
  * `stop_reason == "refusal"` must be checked BEFORE reading `content`, which
    is empty on a pre-output refusal. Pakistani security coverage is exactly the
    shape that trips a safety classifier, and this runs unattended at 2am.
  * Claude Fable 5 has thinking permanently on: the `thinking` parameter must be
    omitted entirely rather than set to disabled.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ...models import CurationUnavailable
from ...settings import Config, TierCfg
from ...util import has_key
from .base import CallResult

log = logging.getLogger(__name__)

ENV_KEY = "ANTHROPIC_API_KEY"

# Models where thinking is always on and the parameter must be omitted.
ALWAYS_THINKING = {"claude-fable-5", "claude-mythos-5"}


class AnthropicCurator:
    name = "anthropic"

    def __init__(self, model: str, effort: str = "medium") -> None:
        self.model = model
        self.effort = effort
        self._client: Any = None

    def is_available(self) -> bool:
        if not has_key(ENV_KEY):
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            log.warning("ANTHROPIC_API_KEY is set but the anthropic SDK is not installed")
            return False
        return True

    def _client_or_raise(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise CurationUnavailable("anthropic sdk missing") from exc
            self._client = anthropic.Anthropic()
        return self._client

    def call(self, system: str, user: str, schema: dict[str, Any],
             schema_name: str, cfg: Config, tier: TierCfg) -> CallResult:
        client = self._client_or_raise()

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": cfg.max_output_tokens,
            "system": [{
                "type": "text",
                "text": system,
                # Cache the stable prefix. The volatile article list sits in the
                # user turn, after this breakpoint, so it never invalidates it.
                "cache_control": {"type": "ephemeral"},
            }],
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        }

        # Omitted entirely on always-thinking models; an explicit disabled value
        # is a 400 there.
        if self.model not in ALWAYS_THINKING:
            request["thinking"] = {"type": "adaptive"}

        try:
            resp = client.messages.create(timeout=cfg.timeout_s, **request)
        except Exception as exc:  # noqa: BLE001
            raise CurationUnavailable(_classify(exc)) from exc

        # Refusal check first. content is empty on a pre-output refusal, so
        # reading content[0] here would crash rather than degrade.
        stop = getattr(resp, "stop_reason", None)
        if stop == "refusal":
            details = getattr(resp, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise CurationUnavailable(f"refusal:{category or 'unspecified'}")
        if stop == "max_tokens":
            raise CurationUnavailable("truncated")

        text = _first_text(resp)
        if not text:
            raise CurationUnavailable("empty response")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CurationUnavailable("invalid json") from exc

        usage = getattr(resp, "usage", None)
        return CallResult(
            data=data,
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cached_read_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
            model=self.model,
        )


def _first_text(resp: Any) -> str:
    for block in getattr(resp, "content", None) or []:
        if getattr(block, "type", None) == "text":
            return getattr(block, "text", "") or ""
    return ""


def _classify(exc: Exception) -> str:
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if status == 401 or "Authentication" in name:
        return "auth"
    if status == 403 or "PermissionDenied" in name:
        return "permission"
    if status == 429 or "RateLimit" in name:
        return "rate_limit"
    if status and int(status) >= 500:
        return f"server_{status}"
    message = str(exc).lower()
    # Fable 5 is unavailable under zero data retention and returns 400 on every
    # request. It looks like a broken integration but is a policy setting, so it
    # is worth naming explicitly rather than reporting as a generic 400.
    if "retention" in message or "zdr" in message:
        return "data_retention_policy"
    if "timeout" in name.lower() or "timeout" in message:
        return "timeout"
    return f"error:{name}"
