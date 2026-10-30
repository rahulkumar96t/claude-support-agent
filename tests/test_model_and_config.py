from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest

from support_agent.config import DEFAULT_MODEL, Settings
from support_agent.model import (
    FALLBACK_BETA,
    ClaudeModel,
    ModelError,
    ModelTimeout,
    content_for_history,
    text_of,
)


class RecordingMessages:
    def __init__(self, error: Exception | None = None) -> None:
        self.kwargs: dict[str, Any] | None = None
        self.error = error

    def create(self, **kwargs: Any) -> str:
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return "response"


class FakeAnthropic:
    """Just enough of ``anthropic.Anthropic`` to capture what ClaudeModel sends."""

    def __init__(self, error: Exception | None = None) -> None:
        self.messages = RecordingMessages(error)
        self.beta = SimpleNamespace(messages=RecordingMessages(error))
        self.options: dict[str, Any] = {}

    def with_options(self, **options: Any) -> FakeAnthropic:
        self.options = options
        return self


def call(settings: Settings, client: FakeAnthropic) -> Any:
    return ClaudeModel(settings, client).create(  # type: ignore[arg-type]
        system="sys",
        tools=[{"name": "t"}],
        messages=[{"role": "user", "content": "hi"}],
        timeout=12.5,
    )


def test_default_request_uses_beta_fallbacks_and_effort():
    client = FakeAnthropic()
    assert call(Settings(), client) == "response"
    sent = client.beta.messages.kwargs
    assert sent is not None and client.messages.kwargs is None
    assert sent["model"] == DEFAULT_MODEL == "claude-sonnet-5-5"
    assert sent["betas"] == [FALLBACK_BETA] and sent["fallbacks"] == "default"
    assert sent["output_config"] == {"effort": "medium"}
    assert sent["max_tokens"] == 16_000
    assert client.options == {"timeout": 12.5}
    # Parameters the current API rejects (or that this project leaves to defaults).
    for absent in ("temperature", "top_p", "top_k", "thinking", "tool_choice"):
        assert absent not in sent


def test_fallbacks_off_uses_the_plain_endpoint_and_effort_can_be_omitted():
    client = FakeAnthropic()
    call(Settings(fallbacks=False, effort=None), client)
    sent = client.messages.kwargs
    assert sent is not None and client.beta.messages.kwargs is None
    assert "betas" not in sent and "fallbacks" not in sent and "output_config" not in sent


def test_sdk_timeout_becomes_model_timeout():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    client = FakeAnthropic(error=anthropic.APITimeoutError(request=request))
    with pytest.raises(ModelTimeout):
        call(Settings(), client)


def test_sdk_api_errors_become_model_errors_with_the_cause_attached():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.AuthenticationError(
        "invalid x-api-key", response=httpx2.Response(401, request=request), body=None
    )
    with pytest.raises(ModelError) as info:
        call(Settings(), FakeAnthropic(error=error))
    assert isinstance(info.value.__cause__, anthropic.AuthenticationError)


def test_content_without_fallback_is_kept_verbatim():
    blocks = [{"type": "thinking", "thinking": "", "signature": "s"}, {"type": "text", "text": "a"}]
    assert content_for_history(blocks) is blocks


def test_content_after_fallback_keeps_text_and_drops_internal_blocks():
    blocks = [
        {"type": "text", "text": "partial"},
        {"type": "thinking", "thinking": "", "signature": "s"},
        {"type": "tool_use", "id": "t1", "name": "x", "input": {}},
        {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}},
        {"type": "thinking", "thinking": "", "signature": "s2"},
        {"type": "text", "text": "final"},
    ]
    assert content_for_history(blocks) == [
        {"type": "text", "text": "partial"},
        {"type": "thinking", "thinking": "", "signature": "s2"},
        {"type": "text", "text": "final"},
    ]
    assert text_of(content_for_history(blocks)) == "partial\n\nfinal"


def test_settings_from_env():
    settings = Settings.from_env(
        {
            "SUPPORT_AGENT_MODEL": "claude-opus-5-5",
            "SUPPORT_AGENT_EFFORT": "none",
            "SUPPORT_AGENT_FALLBACKS": "off",
            "SUPPORT_AGENT_APPROVAL_THRESHOLD": "$250.50",
            "SUPPORT_AGENT_MAX_TURNS": "4",
        }
    )
    assert settings.model == "claude-opus-5-5"
    assert settings.effort is None and settings.fallbacks is False
    assert settings.approval_threshold_cents == 25_050 and settings.max_turns == 4


def test_settings_defaults_and_validation():
    settings = Settings.from_env({})
    assert (settings.model, settings.effort, settings.fallbacks) == (DEFAULT_MODEL, "medium", True)
    with pytest.raises(ValueError):
        Settings.from_env({"SUPPORT_AGENT_EFFORT": "extreme"})
    with pytest.raises(ValueError):
        Settings.from_env({"SUPPORT_AGENT_APPROVAL_THRESHOLD": "lots"})
