"""Wiring for the real (Claude-backed) agent used by the CLI and the HTTP service."""

from __future__ import annotations

from .agent import SupportAgent
from .config import Settings
from .db import open_demo_db
from .model import ClaudeModel


def build_agent(settings: Settings | None = None) -> SupportAgent:
    settings = settings or Settings.from_env()
    db = open_demo_db(settings.db_path)
    return SupportAgent(ClaudeModel(settings), db, settings=settings, trace_dir=settings.trace_dir)
