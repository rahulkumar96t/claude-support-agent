"""The agent loop: Messages API tool use with budgets, an approval gate and traces.

One call to ``send`` (or ``decide`` for an approval) runs the loop until the model answers
without tool calls, a refund needs a human decision, or a budget runs out:

    user message -> model -> tool_use blocks -> execute (validated, customer-scoped)
                 -> tool_result blocks -> model -> ... -> final text

The model-facing history (``Conversation.messages``) is append-only: assistant turns are
stored exactly as returned, and all tool results of a turn go back in one user message.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from .config import Settings
from .db import Database, format_cents
from .guardrails import injection_findings, out_of_scope, wrap_tool_output
from .kb import KnowledgeBase
from .model import (
    ModelClient,
    ModelError,
    ModelTimeout,
    block_to_dict,
    content_for_history,
    text_of,
)
from .policy import PolicyEngine, RefundPolicy
from .prompts import (
    MAX_TURNS_REPLY,
    OUT_OF_SCOPE_REPLY,
    REFUSAL_REPLY,
    SYSTEM_PROMPT,
    TIMEOUT_REPLY,
    approval_pending_reply,
)
from .redact import Redactor
from .tools import RefundRequest, Toolbox, ToolContext, ToolOutcome
from .trace import JsonlTracer, NullTracer, Tracer


class Status(StrEnum):
    ACTIVE = "active"
    AWAITING_APPROVAL = "awaiting_approval"


class StopReason(StrEnum):
    COMPLETED = "completed"
    AWAITING_APPROVAL = "awaiting_approval"
    MAX_TURNS = "max_turns"
    TIMEOUT = "timeout"
    REFUSAL = "refusal"
    OUT_OF_SCOPE = "out_of_scope"


class AgentError(Exception):
    """Base class for errors callers should map to client errors."""


class UnknownCustomer(AgentError):
    pass


class UnknownApproval(AgentError):
    pass


class ConversationBusy(AgentError):
    """The conversation is waiting for an approval or already processing a request."""


@dataclass
class PendingApproval:
    approval_id: str
    conversation_id: str
    tool_use_id: str
    request: RefundRequest
    requested_at: str
    decision: str | None = None  # "approved" | "denied"
    reviewer: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "conversation_id": self.conversation_id,
            "order_id": self.request.order_id,
            "amount": format_cents(self.request.amount_cents),
            "reason": self.request.reason,
            "requested_at": self.requested_at,
            "decision": self.decision,
            "reviewer": self.reviewer,
        }


@dataclass
class Conversation:
    id: str
    customer_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, str]] = field(default_factory=list)
    status: Status = Status.ACTIVE
    paused_tool_use_ids: list[str] = field(default_factory=list)
    held_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    approvals: dict[str, PendingApproval] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def pending_approvals(self) -> list[PendingApproval]:
        return [a for a in self.approvals.values() if a.decision is None]


@dataclass
class TurnResult:
    conversation_id: str
    stop_reason: StopReason
    reply: str
    status: Status
    tool_calls: list[str] = field(default_factory=list)
    pending_approvals: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0


class SupportAgent:
    def __init__(
        self,
        model: ModelClient,
        db: Database,
        *,
        settings: Settings | None = None,
        kb: KnowledgeBase | None = None,
        toolbox: Toolbox | None = None,
        trace_dir: str | Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        today: Callable[[], date] = date.today,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> None:
        self.settings = settings or Settings()
        self.model = model
        self.db = db
        self.kb = kb or KnowledgeBase.bundled()
        self.toolbox = toolbox or Toolbox()
        self.policy = PolicyEngine(
            RefundPolicy(approval_threshold_cents=self.settings.approval_threshold_cents)
        )
        self.trace_dir = trace_dir
        self.system_prompt = system_prompt
        self._clock = clock
        self._today = today
        self._tool_definitions = self.toolbox.definitions()  # fixed for every conversation
        self._conversations: dict[str, Conversation] = {}
        self._approval_index: dict[str, str] = {}
        self._tracers: dict[str, Tracer] = {}
        self._registry_lock = threading.Lock()

    # ---- conversation registry ---------------------------------------------------------------

    def start(self, customer_id: str) -> Conversation:
        customer = self.db.customer(customer_id)
        if customer is None:
            raise UnknownCustomer(customer_id)
        conv = Conversation(id=f"conv_{uuid.uuid4().hex[:12]}", customer_id=customer_id)
        # Known values of the signed-in customer, including each part of their name, are
        # redacted from traces in addition to the generic detectors.
        redactor = Redactor(
            [
                customer["name"],
                *customer["name"].split(),
                customer["email"],
                customer["phone"],
                customer["address"],
            ]
        )
        tracer: Tracer = (
            JsonlTracer(self.trace_dir, conv.id, redactor) if self.trace_dir else NullTracer()
        )
        with self._registry_lock:
            self._conversations[conv.id] = conv
            self._tracers[conv.id] = tracer
        tracer.emit("conversation_started", customer_id=customer_id, model=self.settings.model)
        return conv

    def get(self, conversation_id: str) -> Conversation | None:
        return self._conversations.get(conversation_id)

    def find_approval(self, approval_id: str) -> tuple[Conversation, PendingApproval] | None:
        conv_id = self._approval_index.get(approval_id)
        conv = self._conversations.get(conv_id) if conv_id else None
        if conv is None or approval_id not in conv.approvals:
            return None
        return conv, conv.approvals[approval_id]

    def pending_approvals(self) -> list[PendingApproval]:
        return [a for c in self._conversations.values() for a in c.pending_approvals()]

    # ---- public entry points -----------------------------------------------------------------

    def send(self, conv: Conversation, text: str) -> TurnResult:
        if not conv.lock.acquire(blocking=False):
            raise ConversationBusy("the conversation is already processing a message")
        try:
            if conv.status is Status.AWAITING_APPROVAL:
                raise ConversationBusy("the conversation is waiting for a refund decision")
            tracer = self._tracers[conv.id]
            tracer.emit("user_message", text=text)
            conv.transcript.append({"role": "user", "text": text})
            if out_of_scope(text):
                tracer.emit("guardrail", name="out_of_scope_prefilter")
                return self._finish(conv, StopReason.OUT_OF_SCOPE, OUT_OF_SCOPE_REPLY, [], 0)
            conv.messages.append({"role": "user", "content": text})
            return self._run(conv)
        finally:
            conv.lock.release()

    def decide(
        self, approval_id: str, *, approve: bool, reviewer: str, note: str | None = None
    ) -> TurnResult:
        found = self.find_approval(approval_id)
        if found is None:
            raise UnknownApproval(approval_id)
        conv, approval = found
        if not conv.lock.acquire(blocking=False):
            raise ConversationBusy("the conversation is already processing a request")
        try:
            if approval.decision is not None:
                raise ConversationBusy(f"approval {approval_id} was already {approval.decision}")
            approval.decision = "approved" if approve else "denied"
            approval.reviewer = reviewer
            ctx = self._context(conv)
            if approve:
                outcome = self.toolbox.execute_approved_refund(approval.request, ctx, reviewer)
            else:
                outcome = ToolOutcome(
                    {
                        "status": "denied_by_reviewer",
                        "refund_issued": False,
                        "order_id": approval.request.order_id,
                        "reviewer_note": note or "",
                    }
                )
            tracer = self._tracers[conv.id]
            tracer.emit(
                "approval_decided",
                approval_id=approval_id,
                decision=approval.decision,
                reviewer=reviewer,
                note=note,
            )
            conv.held_results[approval.tool_use_id] = self._tool_result(
                conv, approval.tool_use_id, "issue_refund", outcome
            )
            if conv.pending_approvals():
                return self._finish(
                    conv, StopReason.AWAITING_APPROVAL, self._pending_reply(conv), [], 0
                )
            # Every gated call is decided: hand all results of that turn back in one message.
            conv.messages.append(
                {
                    "role": "user",
                    "content": [conv.held_results[i] for i in conv.paused_tool_use_ids],
                }
            )
            conv.paused_tool_use_ids, conv.held_results = [], {}
            conv.status = Status.ACTIVE
            return self._run(conv)
        finally:
            conv.lock.release()

    # ---- the loop ------------------------------------------------------------------------

    def _run(self, conv: Conversation) -> TurnResult:
        settings, tracer = self.settings, self._tracers[conv.id]
        ctx = self._context(conv)
        deadline = self._clock() + settings.deadline_s
        calls = 0
        tool_calls: list[str] = []

        while True:
            if calls >= settings.max_turns:
                return self._finish(conv, StopReason.MAX_TURNS, MAX_TURNS_REPLY, tool_calls, calls)
            remaining = deadline - self._clock()
            if remaining <= 0:
                return self._finish(conv, StopReason.TIMEOUT, TIMEOUT_REPLY, tool_calls, calls)

            started = self._clock()
            try:
                response = self.model.create(
                    system=self.system_prompt,
                    tools=self._tool_definitions,
                    messages=conv.messages,
                    timeout=min(settings.request_timeout_s, remaining),
                )
            except ModelTimeout as exc:
                tracer.emit("model_timeout", error=str(exc))
                return self._finish(conv, StopReason.TIMEOUT, TIMEOUT_REPLY, tool_calls, calls)
            except ModelError as exc:
                tracer.emit("model_error", error=str(exc))
                raise
            calls += 1

            blocks = content_for_history([block_to_dict(b) for b in response.content])
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            usage = getattr(response, "usage", None)
            tracer.emit(
                "model_response",
                model=getattr(response, "model", None),
                stop_reason=response.stop_reason,
                latency_ms=round((self._clock() - started) * 1000),
                input_tokens=getattr(usage, "input_tokens", None),
                output_tokens=getattr(usage, "output_tokens", None),
                text=text_of(blocks),
                tool_uses=[{"id": b["id"], "name": b["name"]} for b in tool_uses],
            )

            if response.stop_reason == "refusal":
                # Content is empty or a partial to discard; it is not added to the history.
                details = getattr(response, "stop_details", None)
                tracer.emit("model_refusal", category=getattr(details, "category", None))
                return self._finish(conv, StopReason.REFUSAL, REFUSAL_REPLY, tool_calls, calls)

            conv.messages.append({"role": "assistant", "content": blocks})
            if response.stop_reason == "pause_turn":
                continue
            if not tool_uses:
                reply = text_of(blocks) or "Sorry, I don't have an answer for that."
                return self._finish(conv, StopReason.COMPLETED, reply, tool_calls, calls)

            truncated = response.stop_reason == "max_tokens"
            results: dict[str, dict[str, Any]] = {}
            gated: list[PendingApproval] = []
            for use in tool_uses:
                tool_calls.append(use["name"])
                tracer.emit(
                    "tool_call", name=use["name"], tool_use_id=use["id"], input=use["input"]
                )
                if truncated:  # a cut-off tool input can still parse; never run it
                    outcome = ToolOutcome(
                        {"error": "The tool input was cut off. Call the tool again."},
                        is_error=True,
                    )
                else:
                    outcome = self.toolbox.execute(use["name"], use["input"], ctx)
                if outcome.approval is not None:
                    gated.append(self._request_approval(conv, use["id"], outcome.approval))
                    continue
                results[use["id"]] = self._tool_result(conv, use["id"], use["name"], outcome)

            if gated:
                conv.paused_tool_use_ids = [u["id"] for u in tool_uses]
                conv.held_results = results
                conv.status = Status.AWAITING_APPROVAL
                return self._finish(
                    conv,
                    StopReason.AWAITING_APPROVAL,
                    self._pending_reply(conv),
                    tool_calls,
                    calls,
                )
            conv.messages.append({"role": "user", "content": [results[u["id"]] for u in tool_uses]})

    # ---- helpers -----------------------------------------------------------------------------

    def _context(self, conv: Conversation) -> ToolContext:
        return ToolContext(
            db=self.db,
            kb=self.kb,
            customer_id=conv.customer_id,
            policy=self.policy,
            today=self._today,
        )

    def _tool_result(
        self, conv: Conversation, tool_use_id: str, name: str, outcome: ToolOutcome
    ) -> dict[str, Any]:
        flags = injection_findings(outcome.payload)
        self._tracers[conv.id].emit(
            "tool_result",
            name=name,
            tool_use_id=tool_use_id,
            is_error=outcome.is_error,
            payload=outcome.payload,
            injection_flags=flags,
        )
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": wrap_tool_output(name, outcome.payload),
        }
        if outcome.is_error:
            block["is_error"] = True
        return block

    def _request_approval(
        self, conv: Conversation, tool_use_id: str, request: RefundRequest
    ) -> PendingApproval:
        approval = PendingApproval(
            approval_id=f"apr_{uuid.uuid4().hex[:10]}",
            conversation_id=conv.id,
            tool_use_id=tool_use_id,
            request=request,
            requested_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )
        conv.approvals[approval.approval_id] = approval
        with self._registry_lock:
            self._approval_index[approval.approval_id] = conv.id
        self._tracers[conv.id].emit("approval_requested", **approval.public())
        return approval

    @staticmethod
    def _pending_reply(conv: Conversation) -> str:
        return " ".join(
            approval_pending_reply(a.request.order_id, format_cents(a.request.amount_cents))
            for a in conv.pending_approvals()
        )

    def _finish(
        self,
        conv: Conversation,
        reason: StopReason,
        reply: str,
        tool_calls: list[str],
        calls: int,
    ) -> TurnResult:
        conv.transcript.append({"role": "assistant", "text": reply})
        self._tracers[conv.id].emit(
            "turn_finished", stop_reason=str(reason), reply=reply, model_calls=calls
        )
        return TurnResult(
            conversation_id=conv.id,
            stop_reason=reason,
            reply=reply,
            status=conv.status,
            tool_calls=tool_calls,
            pending_approvals=[a.public() for a in conv.pending_approvals()],
            model_calls=calls,
        )
