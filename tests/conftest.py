from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from fakes import ScriptedModel, Step

from support_agent.agent import SupportAgent
from support_agent.config import Settings
from support_agent.db import Database, open_demo_db

TODAY = date(2026, 10, 15)


@pytest.fixture
def db() -> Iterator[Database]:
    database = open_demo_db(":memory:", today=TODAY)
    yield database
    database.close()


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    return tmp_path / "traces"


AgentFactory = Callable[..., tuple[SupportAgent, ScriptedModel]]


@pytest.fixture
def make_agent(db: Database, trace_dir: Path) -> AgentFactory:
    def factory(*steps: Step, **agent_kwargs: Any) -> tuple[SupportAgent, ScriptedModel]:
        model = ScriptedModel(*steps)
        settings = agent_kwargs.pop("settings", Settings())
        agent = SupportAgent(
            model, db, settings=settings, trace_dir=trace_dir, today=lambda: TODAY, **agent_kwargs
        )
        return agent, model

    return factory
