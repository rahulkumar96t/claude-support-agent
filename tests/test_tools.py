from __future__ import annotations

from typing import Any

import pytest
from conftest import TODAY

from support_agent.kb import KnowledgeBase
from support_agent.tools import Toolbox, ToolContext, strict_schema

UNSUPPORTED = {"minLength", "maxLength", "pattern", "minimum", "maximum", "title"}


def walk(node: Any):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


@pytest.fixture
def ctx(db) -> ToolContext:
    return ToolContext(db=db, kb=KnowledgeBase.bundled(), customer_id="C-1001", today=lambda: TODAY)


def test_definitions_are_strict_and_use_supported_schema_keywords():
    definitions = Toolbox().definitions()
    assert [d["name"] for d in definitions] == [
        "lookup_order",
        "get_shipment_status",
        "check_refund_eligibility",
        "issue_refund",
        "create_ticket",
        "search_knowledge_base",
    ]
    for definition in definitions:
        assert definition["strict"] is True
        assert definition["description"]
        schema = definition["input_schema"]
        for node in walk(schema):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
            assert not UNSUPPORTED & {k for k in node if k != "properties"}


def test_issue_refund_has_no_amount_parameter():
    (refund,) = [d for d in Toolbox().definitions() if d["name"] == "issue_refund"]
    assert set(refund["input_schema"]["properties"]) == {"order_id", "reason"}


def test_strict_schema_keeps_a_property_named_title():
    from pydantic import BaseModel, ConfigDict

    class Note(BaseModel):
        model_config = ConfigDict(extra="forbid")
        title: str

    assert "title" in strict_schema(Note)["properties"]


def test_validation_errors_become_tool_errors(ctx):
    outcome = Toolbox().execute("search_knowledge_base", {"query": "x", "top_k": 9}, ctx)
    assert outcome.is_error
    assert {p.split(":")[0] for p in outcome.payload["problems"]} == {"query", "top_k"}


def test_order_ids_are_normalised(ctx):
    outcome = Toolbox().execute("lookup_order", {"order_id": "  ac-10001 "}, ctx)
    assert not outcome.is_error and outcome.payload["order_id"] == "AC-10001"


def test_shipment_status_for_order_in_transit(ctx):
    outcome = Toolbox().execute("get_shipment_status", {"order_id": "AC-10004"}, ctx)
    assert outcome.payload["tracking_number"] == "PW7730018842"
    assert outcome.payload["shipment_status"] == "in_transit"
    assert len(outcome.payload["events"]) == 2


def test_shipment_status_before_shipping(db):
    ctx = ToolContext(db=db, kb=KnowledgeBase.bundled(), customer_id="C-1002", today=lambda: TODAY)
    outcome = Toolbox().execute("get_shipment_status", {"order_id": "AC-10009"}, ctx)
    assert outcome.payload["shipment"] is None and outcome.payload["order_status"] == "processing"


def test_create_ticket(ctx, db):
    outcome = Toolbox().execute(
        "create_ticket",
        {
            "subject": "Headlamp flickers",
            "summary": "Customer reports the headlamp from AC-10003 flickers after 6 weeks.",
            "priority": "normal",
            "order_id": "AC-10003",
        },
        ctx,
    )
    assert outcome.payload["status"] == "open"
    (ticket,) = db.tickets_for_customer("C-1001")
    assert ticket["id"] == outcome.payload["ticket_id"] and ticket["order_id"] == "AC-10003"


def test_create_ticket_rejects_another_customers_order(ctx, db):
    outcome = Toolbox().execute(
        "create_ticket",
        {"subject": "Paddle", "summary": "About someone else's order", "priority": "low",
         "order_id": "AC-10006"},
        ctx,
    )  # fmt: skip
    assert outcome.is_error and db.tickets_for_customer("C-1001") == []


def test_knowledge_base_search_finds_the_refund_window(ctx):
    outcome = Toolbox().execute(
        "search_knowledge_base", {"query": "refund window for a damaged item"}, ctx
    )
    sections = [(r["article"], r["section"]) for r in outcome.payload["results"]]
    assert ("refund-policy", "Refund windows") in sections


def test_knowledge_base_search_with_no_match_is_empty():
    assert KnowledgeBase.bundled().search("zebra quantum") == []
