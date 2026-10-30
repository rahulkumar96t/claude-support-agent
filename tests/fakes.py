"""Scripted stand-ins for the model, built from real Anthropic SDK types."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable
from typing import Any

from anthropic.types.beta import (
    BetaContentBlock,
    BetaMessage,
    BetaTextBlock,
    BetaThinkingBlock,
    BetaToolUseBlock,
)

_ids = itertools.count(1)

Step = BetaMessage | Exception | Callable[[list[dict[str, Any]]], BetaMessage]


def text(value: str) -> BetaTextBlock:
    return BetaTextBlock(type="text", text=value)


def thinking(value: str = "") -> BetaThinkingBlock:
    return BetaThinkingBlock(type="thinking", thinking=value, signature="sig")


def tool(name: str, tool_input: dict[str, Any], tool_id: str | None = None) -> BetaToolUseBlock:
    return BetaToolUseBlock(
        type="tool_use", id=tool_id or f"toolu_{next(_ids):04d}", name=name, input=tool_input
    )


def message(
    *blocks: BetaContentBlock | dict[str, Any],
    stop_reason: str | None = None,
    stop_details: dict[str, Any] | None = None,
) -> BetaMessage:
    content = [b if isinstance(b, dict) else b.model_dump(by_alias=True) for b in blocks]
    if stop_reason is None:
        has_tool = any(b["type"] == "tool_use" for b in content)
        stop_reason = "tool_use" if has_tool else "end_turn"
    return BetaMessage.model_validate(
        {
            "id": f"msg_{next(_ids):04d}",
            "type": "message",
            "role": "assistant",
            "model": "scripted-model",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": {"input_tokens": 100, "output_tokens": 20},
        }
    )


class ScriptedModel:
    """Implements ``ModelClient``: returns scripted steps in order and records every request.

    A step is a message, an exception to raise, or a callable that receives the request's
    messages and returns a message (for steps that depend on what the agent sent).
    """

    def __init__(self, *steps: Step) -> None:
        self.steps = list(steps)
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        *,
        system: str,
        tools: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        timeout: float,
    ) -> BetaMessage:
        self.calls.append(
            {
                "system": system,
                "tools": tools,
                "messages": copy.deepcopy(messages),
                "timeout": timeout,
            }
        )
        if not self.steps:
            raise AssertionError("the model was called more often than scripted")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(messages)
        return step


def tool_results(request: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool_result blocks in the last user message of a recorded request."""
    last = request["messages"][-1]
    assert last["role"] == "user" and isinstance(last["content"], list)
    return [b for b in last["content"] if b["type"] == "tool_result"]


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds
