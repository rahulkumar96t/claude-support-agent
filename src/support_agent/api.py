"""HTTP interface (FastAPI).

This is a demo service: there is no authentication. In a real deployment ``customer_id``
would come from the customer's session, and the approval endpoints would sit behind staff
authentication.
"""

# No ``from __future__ import annotations`` here: FastAPI resolves the Annotated dependency
# aliases defined inside ``create_app`` at decoration time.

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .agent import (
    Conversation,
    ConversationBusy,
    SupportAgent,
    TurnResult,
    UnknownApproval,
    UnknownCustomer,
)
from .model import ModelError


class StartConversationIn(BaseModel):
    customer_id: str = Field(min_length=1, max_length=40, examples=["C-1001"])


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


class DecisionIn(BaseModel):
    approve: bool
    reviewer: str = Field(min_length=1, max_length=80)
    note: str | None = Field(default=None, max_length=1000)


class ConversationOut(BaseModel):
    conversation_id: str
    customer_id: str
    status: str
    transcript: list[dict[str, str]]
    pending_approvals: list[dict[str, Any]]


class TurnOut(BaseModel):
    conversation_id: str
    stop_reason: str
    reply: str
    status: str
    tool_calls: list[str]
    pending_approvals: list[dict[str, Any]]
    model_calls: int


def _conversation_out(conv: Conversation) -> ConversationOut:
    return ConversationOut(
        conversation_id=conv.id,
        customer_id=conv.customer_id,
        status=str(conv.status),
        transcript=conv.transcript,
        pending_approvals=[a.public() for a in conv.pending_approvals()],
    )


def _turn_out(result: TurnResult) -> TurnOut:
    return TurnOut(
        conversation_id=result.conversation_id,
        stop_reason=str(result.stop_reason),
        reply=result.reply,
        status=str(result.status),
        tool_calls=result.tool_calls,
        pending_approvals=result.pending_approvals,
        model_calls=result.model_calls,
    )


def create_app(agent_factory: Callable[[], SupportAgent] | None = None) -> FastAPI:
    """Create the app. The agent is built lazily on first use (so importing needs no key)."""
    app = FastAPI(title="claude-support-agent", version=__version__)
    holder: dict[str, SupportAgent] = {}

    def get_agent() -> SupportAgent:
        if "agent" not in holder:
            if agent_factory is None:
                from .factory import build_agent

                holder["agent"] = build_agent()
            else:
                holder["agent"] = agent_factory()
        return holder["agent"]

    AgentDep = Annotated[SupportAgent, Depends(get_agent)]

    def get_conversation(conversation_id: str, agent: AgentDep) -> Conversation:
        conv = agent.get(conversation_id)
        if conv is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "conversation not found")
        return conv

    ConversationDep = Annotated[Conversation, Depends(get_conversation)]

    @app.exception_handler(ModelError)
    def model_error(_request: Request, exc: ModelError) -> JSONResponse:
        return JSONResponse({"detail": f"model API error: {exc}"}, status_code=502)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.post("/conversations", status_code=status.HTTP_201_CREATED)
    def start_conversation(body: StartConversationIn, agent: AgentDep) -> ConversationOut:
        try:
            conv = agent.start(body.customer_id)
        except UnknownCustomer as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown customer") from exc
        return _conversation_out(conv)

    @app.get("/conversations/{conversation_id}")
    def read_conversation(conv: ConversationDep) -> ConversationOut:
        return _conversation_out(conv)

    @app.post("/conversations/{conversation_id}/messages")
    def post_message(body: MessageIn, conv: ConversationDep, agent: AgentDep) -> TurnOut:
        try:
            return _turn_out(agent.send(conv, body.text))
        except ConversationBusy as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    @app.get("/approvals")
    def list_approvals(agent: AgentDep) -> list[dict[str, Any]]:
        return [a.public() for a in agent.pending_approvals()]

    @app.post("/approvals/{approval_id}")
    def decide_approval(approval_id: str, body: DecisionIn, agent: AgentDep) -> TurnOut:
        try:
            result = agent.decide(
                approval_id, approve=body.approve, reviewer=body.reviewer, note=body.note
            )
        except UnknownApproval as exc:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "approval not found") from exc
        except ConversationBusy as exc:
            raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
        return _turn_out(result)

    return app
