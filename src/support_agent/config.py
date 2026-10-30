"""Runtime settings, read from environment variables.

The Anthropic API key is never part of these settings: the SDK client reads its credentials
from the environment (``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile) by itself.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

DEFAULT_MODEL = "claude-sonnet-5-5"
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class Settings:
    model: str = DEFAULT_MODEL
    effort: str | None = "medium"
    fallbacks: bool = True
    max_tokens: int = 16_000
    max_turns: int = 8
    request_timeout_s: float = 60.0
    deadline_s: float = 180.0
    approval_threshold_cents: int = 10_000
    db_path: str = "support-demo.db"
    trace_dir: str = "traces"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        effort = env.get("SUPPORT_AGENT_EFFORT", "medium").strip().lower()
        if effort not in (*EFFORT_LEVELS, "none"):
            raise ValueError(f"SUPPORT_AGENT_EFFORT must be one of {EFFORT_LEVELS} or 'none'")
        fallbacks = env.get("SUPPORT_AGENT_FALLBACKS", "default").strip().lower()
        if fallbacks not in ("default", "off"):
            raise ValueError("SUPPORT_AGENT_FALLBACKS must be 'default' or 'off'")
        return cls(
            model=env.get("SUPPORT_AGENT_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            effort=None if effort == "none" else effort,
            fallbacks=fallbacks == "default",
            max_turns=int(env.get("SUPPORT_AGENT_MAX_TURNS", cls.max_turns)),
            request_timeout_s=float(env.get("SUPPORT_AGENT_TIMEOUT", cls.request_timeout_s)),
            deadline_s=float(env.get("SUPPORT_AGENT_DEADLINE", cls.deadline_s)),
            approval_threshold_cents=_dollars_to_cents(
                env.get("SUPPORT_AGENT_APPROVAL_THRESHOLD", "100.00")
            ),
            db_path=env.get("SUPPORT_AGENT_DB", cls.db_path),
            trace_dir=env.get("SUPPORT_AGENT_TRACE_DIR", cls.trace_dir),
        )


def _dollars_to_cents(value: str) -> int:
    try:
        amount = Decimal(value.strip().lstrip("$"))
    except InvalidOperation as exc:
        raise ValueError(f"not a dollar amount: {value!r}") from exc
    return int((amount * 100).to_integral_value())
