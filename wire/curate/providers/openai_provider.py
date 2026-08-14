"""OpenAI provider.

Same prompts, same schema, same return shape as the Anthropic provider. The two
differences that matter:

  * Strict structured outputs require `additionalProperties: false` and every
    property present in `required` — our schemas are written that way already.
  * The Responses API and Chat Completions differ in shape, so we try Responses
    first and fall back, because which one an installed SDK version exposes
    varies more than the model does.
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

ENV_KEY = "OPENAI_API_KEY"


class OpenAICurator:
    name = "openai"

    def __init__(self, model: str) -> None:
        self.model = model
        self._client: Any = None

    def is_available(self) -> bool:
        if not has_key(ENV_KEY):
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            log.warning("OPENAI_API_KEY is set but the openai SDK is not installed")
            return False
        return True

    def _client_or_raise(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise CurationUnavailable("openai sdk missing") from exc
            self._client = openai.OpenAI()
        return self._client

    def call(self, system: str, user: str, schema: dict[str, Any],
             schema_name: str, cfg: Config, tier: TierCfg) -> CallResult:
        client = self._client_or_raise()
        fmt = {
            "type": "json_schema",
            "name": schema_name,
            "schema": schema,
            "strict": True,
        }

        try:
            resp = client.responses.create(
                model=self.model,
                instructions=system,
                input=user,
                max_output_tokens=cfg.max_output_tokens,
                text={"format": fmt},
                timeout=cfg.timeout_s,
            )
            text, usage = _read_responses(resp)
        except CurationUnavailable:
            raise
        except (AttributeError, TypeError) as exc:
            log.debug("responses API unavailable (%s); trying chat completions", exc)
            text, usage = self._chat_fallback(client, system, user, schema,
                                              schema_name, cfg)
        except Exception as exc:  # noqa: BLE001
            raise CurationUnavailable(_classify(exc)) from exc

        if not text:
            raise CurationUnavailable("empty response")

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise CurationUnavailable("invalid json") from exc

        return CallResult(
            data=data,
            input_tokens=usage.get("input", 0),
            output_tokens=usage.get("output", 0),
            cached_read_tokens=usage.get("cached", 0),
            model=self.model,
        )

    def _chat_fallback(self, client: Any, system: str, user: str,
                       schema: dict[str, Any], schema_name: str,
                       cfg: Config) -> tuple[str, dict[str, int]]:
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema,
                                    "strict": True},
                },
                timeout=cfg.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001
            raise CurationUnavailable(_classify(exc)) from exc

        choice = (getattr(resp, "choices", None) or [None])[0]
        if choice is None:
            raise CurationUnavailable("empty response")
        message = getattr(choice, "message", None)
        if getattr(message, "refusal", None):
            raise CurationUnavailable("refusal:content_policy")
        if getattr(choice, "finish_reason", None) == "length":
            raise CurationUnavailable("truncated")

        usage = getattr(resp, "usage", None)
        return (getattr(message, "content", "") or ""), {
            "input": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output": int(getattr(usage, "completion_tokens", 0) or 0),
            "cached": 0,
        }


def _read_responses(resp: Any) -> tuple[str, dict[str, int]]:
    status = getattr(resp, "status", None)
    if status == "incomplete":
        raise CurationUnavailable("truncated")

    text = getattr(resp, "output_text", None)
    if not text:
        chunks: list[str] = []
        for item in getattr(resp, "output", None) or []:
            for part in getattr(item, "content", None) or []:
                if getattr(part, "type", None) == "refusal":
                    raise CurationUnavailable("refusal:content_policy")
                if getattr(part, "type", None) in ("output_text", "text"):
                    chunks.append(getattr(part, "text", "") or "")
        text = "".join(chunks)

    usage = getattr(resp, "usage", None)
    cached = 0
    details = getattr(usage, "input_tokens_details", None)
    if details is not None:
        cached = int(getattr(details, "cached_tokens", 0) or 0)

    return text or "", {
        "input": int(getattr(usage, "input_tokens", 0) or 0),
        "output": int(getattr(usage, "output_tokens", 0) or 0),
        "cached": cached,
    }


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
    if "timeout" in name.lower() or "timeout" in str(exc).lower():
        return "timeout"
    return f"error:{name}"
