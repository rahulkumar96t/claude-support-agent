from __future__ import annotations

import json
import os

import pytest

from support_agent.cli import main


def test_seed_command_creates_the_database(tmp_path, capsys):
    db_path = tmp_path / "demo.db"
    assert main(["--db", str(db_path), "seed"]) == 0
    assert db_path.exists()
    out = capsys.readouterr().out
    assert "C-1001 Dana Whitfield" in out and "AC-10007" in out
    assert main(["--db", str(db_path), "seed", "--reset"]) == 0


def test_tools_command_prints_definitions(capsys):
    assert main(["tools"]) == 0
    names = [d["name"] for d in json.loads(capsys.readouterr().out)]
    assert "issue_refund" in names


@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("RUN_LIVE_TESTS") == "1"),
    reason="live test: set ANTHROPIC_API_KEY and RUN_LIVE_TESTS=1",
)
def test_live_shipment_question(db, tmp_path):  # pragma: no cover - needs network and a key
    from support_agent.agent import StopReason, SupportAgent
    from support_agent.config import Settings
    from support_agent.model import ClaudeModel

    settings = Settings.from_env()
    agent = SupportAgent(ClaudeModel(settings), db, settings=settings, trace_dir=tmp_path)
    conv = agent.start("C-1001")
    result = agent.send(conv, "Where is my order AC-10004? What's the tracking number?")
    assert result.stop_reason is StopReason.COMPLETED
    assert "get_shipment_status" in result.tool_calls or "lookup_order" in result.tool_calls
    assert "PW7730018842" in result.reply
