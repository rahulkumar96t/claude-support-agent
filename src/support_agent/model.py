"""The model boundary: a small protocol, the Claude implementation and history helpers.

The agent only needs ``create(system=, tools=, messages=, timeout=)`` returning a Messages API
message (``content`` blocks, ``stop_reason``, ``model``, ``usage``). Tests plug in a scripted
fake that returns real SDK message objects, so the agent code path is identical.
"""

from __future__ import annotations

from typing import Any, Protocol

import anthropic

from .config import Settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ModelTimeout(Exception):
    """The model call did not finish within its time budget."""


class ModelError(Exception):
    """The model API returned an error after the SDK's own retries (cause is attached)."""


class ModelClient(Protocol):
    def create(
        self,
        *,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        timeout: float,
    ) -> Any: ...


class ClaudeModel:
    """Calls the Claude Messages API with the settings from ``Settings``.

    * Credentials come from the environment only (``anthropic.Anthropic()`` resolves them).
    * No ``thinking`` parameter: current models run adaptive thinking by default, and effort
      (``output_config.effort``) is the control.
    * No sampling parameters (``temperature``/``top_p``/``top_k``): the API version the SDK
      sends rejects them.
    * Refusal fallback is on by default (``fallbacks="default"``): if a safety classifier
      declines a turn, the API retries it on the model Anthropic recommends for that category.
    """

    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None) -> None:
        self.settings = settings
        self._client = client if client is not None else anthropic.Anthropic(max_retries=2)

    def request_params(
        self, system: str, tools: list[dict[str, Any]], messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "tools": tools,
            "messages": messages,
            # Caches the stable prefix (tools + system + earlier turns) across loop steps.
            "cache_control": {"type": "ephemeral"},
        }
        if self.settings.effort:
            params["output_config"] = {"effort": self.settings.effort}
        return params

    def create(
        self,
        *,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        timeout: float,
    ) -> Any:
        client = self._client.with_options(timeout=timeout)
        params = self.request_params(system, tools, messages)
        try:
            if self.settings.fallbacks:
                return client.beta.messages.create(
                    **params, betas=[FALLBACK_BETA], fallbacks="default"
                )
            return client.messages.create(**params)
        except anthropic.APITimeoutError as exc:
            raise ModelTimeout(str(exc)) from exc
        except anthropic.APIError as exc:
            raise ModelError(str(exc)) from exc


# ---- history helpers --------------------------------------------------------------------------


def block_to_dict(block: Any) -> dict[str, Any]:
    if isinstance(block, dict):
        return dict(block)
    return block.model_dump(mode="json", by_alias=True, exclude_none=True)


def content_for_history(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepare an assistant turn for the append-only history.

    Normally the content is kept exactly as returned (thinking blocks included). After a
    server-side fallback, the blocks the declining model produced before the last
    ``fallback`` marker are dropped except for text, as the fallback docs require, and the
    marker itself is removed. Tool calls before the marker were never meant to run.
    """
    last = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=None)
    if last is None:
        return blocks
    return [
        b
        for i, b in enumerate(blocks)
        if b.get("type") != "fallback" and (i > last or b.get("type") == "text")
    ]


def text_of(blocks: list[dict[str, Any]]) -> str:
    return "\n\n".join(b["text"] for b in blocks if b.get("type") == "text" and b.get("text"))
