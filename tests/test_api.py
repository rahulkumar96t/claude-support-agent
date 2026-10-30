from __future__ import annotations

import pytest
from fakes import message, text, tool
from fastapi.testclient import TestClient

from support_agent.api import create_app
from support_agent.model import ModelError


@pytest.fixture
def client_for(make_agent):
    def build(*steps):
        agent, model = make_agent(*steps)
        return TestClient(create_app(lambda: agent)), model

    return build


def start(client: TestClient, customer_id: str) -> str:
    response = client.post("/conversations", json={"customer_id": customer_id})
    assert response.status_code == 201
    return response.json()["conversation_id"]


def test_health(client_for):
    client, _ = client_for()
    assert client.get("/health").json()["status"] == "ok"


def test_conversation_flow_with_approval(client_for):
    client, _ = client_for(
        message(tool("issue_refund", {"order_id": "AC-10002", "reason": "damaged"})),
        message(text("Your refund of $249.00 was approved.")),
    )
    conv_id = start(client, "C-1001")

    turn = client.post(
        f"/conversations/{conv_id}/messages", json={"text": "Refund my torn tent AC-10002, yes."}
    )
    assert turn.status_code == 200
    body = turn.json()
    assert body["stop_reason"] == "awaiting_approval" and body["status"] == "awaiting_approval"
    approval_id = body["pending_approvals"][0]["approval_id"]

    busy = client.post(f"/conversations/{conv_id}/messages", json={"text": "hello?"})
    assert busy.status_code == 409

    queue = client.get("/approvals").json()
    assert [a["approval_id"] for a in queue] == [approval_id]

    decided = client.post(f"/approvals/{approval_id}", json={"approve": True, "reviewer": "sam"})
    assert decided.status_code == 200
    assert decided.json()["stop_reason"] == "completed"
    assert decided.json()["reply"] == "Your refund of $249.00 was approved."
    assert client.get("/approvals").json() == []

    again = client.post(f"/approvals/{approval_id}", json={"approve": False, "reviewer": "sam"})
    assert again.status_code == 409

    state = client.get(f"/conversations/{conv_id}").json()
    assert state["status"] == "active"
    assert [t["role"] for t in state["transcript"]] == ["user", "assistant", "assistant"]


def test_not_found_and_validation_errors(client_for):
    client, _ = client_for()
    assert client.post("/conversations", json={"customer_id": "C-9999"}).status_code == 404
    assert client.get("/conversations/conv_missing").status_code == 404
    decision = {"approve": True, "reviewer": "x"}
    assert client.post("/approvals/apr_missing", json=decision).status_code == 404
    conv_id = start(client, "C-1001")
    assert client.post(f"/conversations/{conv_id}/messages", json={"text": ""}).status_code == 422


def test_out_of_scope_message_over_http(client_for):
    client, model = client_for()
    conv_id = start(client, "C-1002")
    body = client.post(
        f"/conversations/{conv_id}/messages", json={"text": "Write me a haiku about rain"}
    ).json()
    assert body["stop_reason"] == "out_of_scope" and model.calls == []


def test_model_api_errors_map_to_502(client_for):
    client, _ = client_for(ModelError("overloaded"))
    conv_id = start(client, "C-1001")
    response = client.post(
        f"/conversations/{conv_id}/messages", json={"text": "Where is AC-10004?"}
    )
    assert response.status_code == 502
    assert "overloaded" in response.json()["detail"]
