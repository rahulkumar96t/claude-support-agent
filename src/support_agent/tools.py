"""Tools exposed to the model, defined from Pydantic input models.

Each tool has a Pydantic input model. Its JSON schema becomes the tool's ``input_schema``
(with ``strict: true``), and the same model validates the model's arguments before anything
runs. Every handler is scoped to the conversation's signed-in customer, so the model cannot
read or refund another customer's orders no matter what it is asked or told.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .db import Database, Order, format_cents
from .kb import KnowledgeBase
from .policy import PolicyEngine, RefundReason

log = logging.getLogger(__name__)

ORDER_ID_PATTERN = r"^[A-Z]{2}-\d{5}$"


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _OrderInput(_Input):
    order_id: str = Field(
        description="Order number exactly as the customer gave it, e.g. AC-10001.",
        pattern=ORDER_ID_PATTERN,
    )

    @field_validator("order_id", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value


class LookupOrderInput(_OrderInput):
    """Look up one of the signed-in customer's orders."""


class ShipmentStatusInput(_OrderInput):
    """Get carrier, tracking number and tracking events for an order."""


class RefundEligibilityInput(_OrderInput):
    reason: RefundReason = Field(description="Why the customer wants a refund.")


class IssueRefundInput(_OrderInput):
    reason: RefundReason = Field(description="Why the customer wants a refund.")


class CreateTicketInput(_Input):
    subject: str = Field(description="One-line subject.", min_length=3, max_length=120)
    summary: str = Field(
        description="What the customer needs and what has been tried, in 1-5 sentences.",
        min_length=10,
        max_length=2000,
    )
    priority: Literal["low", "normal", "high"] = Field(description="Ticket priority.")
    order_id: str | None = Field(
        default=None, description="Related order number, if any.", pattern=ORDER_ID_PATTERN
    )

    @field_validator("order_id", mode="before")
    @classmethod
    def _normalise(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value


class SearchKnowledgeBaseInput(_Input):
    query: str = Field(description="Search terms, e.g. 'refund window damaged item'.", min_length=2)
    top_k: int = Field(default=3, ge=1, le=5, description="Number of sections to return (1-5).")


# ---- strict schema generation ----------------------------------------------------------------

# Strict tool use accepts a subset of JSON Schema. Pydantic still validates these constraints
# locally, so they are removed from the copy sent to the API.
_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "title",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "minItems",
        "maxItems",
        "uniqueItems",
    }
)
_NAMED_SCHEMA_MAPS = frozenset({"properties", "$defs"})


def strict_schema(model: type[BaseModel]) -> dict[str, Any]:
    def clean(node: Any, *, mapping: bool = False) -> Any:
        if isinstance(node, list):
            return [clean(item) for item in node]
        if not isinstance(node, dict):
            return node
        if mapping:  # keys are property names, values are schemas
            return {name: clean(schema) for name, schema in node.items()}
        out = {
            key: clean(value, mapping=key in _NAMED_SCHEMA_MAPS)
            for key, value in node.items()
            if key not in _UNSUPPORTED_KEYWORDS
        }
        if out.get("type") == "object":
            out["additionalProperties"] = False
        return out

    return clean(model.model_json_schema())


# ---- runtime -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class RefundRequest:
    """A refund the policy allows but that needs a human decision before it is issued."""

    order_id: str
    reason: RefundReason
    amount_cents: int


@dataclass
class ToolOutcome:
    payload: dict[str, Any]
    is_error: bool = False
    approval: RefundRequest | None = None


@dataclass
class ToolContext:
    db: Database
    kb: KnowledgeBase
    customer_id: str
    policy: PolicyEngine = field(default_factory=PolicyEngine)
    today: Callable[[], date] = date.today


Handler = Callable[[ToolContext, Any], ToolOutcome]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    handler: Handler

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": strict_schema(self.input_model),
            "strict": True,
        }


def _not_found(order_id: str) -> ToolOutcome:
    return ToolOutcome(
        {"error": f"No order {order_id} was found for this customer."}, is_error=True
    )


def _order_payload(ctx: ToolContext, order: Order) -> dict[str, Any]:
    refund = ctx.db.refund_for_order(order.id)
    return {
        "order_id": order.id,
        "status": order.status,
        "placed_on": order.placed_on.isoformat(),
        "delivered_on": order.delivered_on.isoformat() if order.delivered_on else None,
        "items": [
            {
                "sku": i.sku,
                "name": i.name,
                "category": i.category,
                "quantity": i.quantity,
                "unit_price": format_cents(i.unit_price_cents),
                "line_total": format_cents(i.line_total_cents),
            }
            for i in order.items
        ],
        "order_total": format_cents(order.total_cents),
        "customer_note": order.note,
        "refund": (
            {"refund_id": refund.id, "amount": format_cents(refund.amount_cents)}
            if refund
            else None
        ),
    }


def lookup_order(ctx: ToolContext, args: LookupOrderInput) -> ToolOutcome:
    order = ctx.db.order_for_customer(args.order_id, ctx.customer_id)
    return ToolOutcome(_order_payload(ctx, order)) if order else _not_found(args.order_id)


def get_shipment_status(ctx: ToolContext, args: ShipmentStatusInput) -> ToolOutcome:
    order = ctx.db.order_for_customer(args.order_id, ctx.customer_id)
    if order is None:
        return _not_found(args.order_id)
    shipment = ctx.db.shipment(order.id)
    if shipment is None:
        return ToolOutcome(
            {
                "order_id": order.id,
                "order_status": order.status,
                "shipment": None,
                "message": "This order has not shipped yet, so there is no tracking number.",
            }
        )
    return ToolOutcome(
        {
            "order_id": order.id,
            "order_status": order.status,
            "carrier": shipment["carrier"],
            "tracking_number": shipment["tracking_number"],
            "shipment_status": shipment["status"],
            "estimated_delivery": shipment["estimated_delivery"],
            "events": shipment["events"],
        }
    )


def check_refund_eligibility(ctx: ToolContext, args: RefundEligibilityInput) -> ToolOutcome:
    order = ctx.db.order_for_customer(args.order_id, ctx.customer_id)
    if order is None:
        return _not_found(args.order_id)
    result = ctx.policy.evaluate(order, args.reason, ctx.db.refund_for_order(order.id), ctx.today())
    return ToolOutcome(result.to_payload())


def issue_refund(ctx: ToolContext, args: IssueRefundInput) -> ToolOutcome:
    order = ctx.db.order_for_customer(args.order_id, ctx.customer_id)
    if order is None:
        return _not_found(args.order_id)
    result = ctx.policy.evaluate(order, args.reason, ctx.db.refund_for_order(order.id), ctx.today())
    if not result.eligible:
        return ToolOutcome({"status": "not_eligible", **result.to_payload()})
    if result.requires_approval:
        request = RefundRequest(order.id, args.reason, result.refundable_cents)
        return ToolOutcome(
            {
                "status": "pending_approval",
                "order_id": order.id,
                "amount": format_cents(result.refundable_cents),
            },
            approval=request,
        )
    return _execute_refund(ctx, order.id, args.reason, result.refundable_cents, "automatic")


def _execute_refund(
    ctx: ToolContext, order_id: str, reason: RefundReason, amount_cents: int, approved_by: str
) -> ToolOutcome:
    refund = ctx.db.create_refund(order_id, amount_cents, reason, approved_by)
    return ToolOutcome(
        {
            "status": "refunded",
            "refund_id": refund.id,
            "order_id": order_id,
            "amount": format_cents(amount_cents),
            "approved_by": approved_by,
            "expected_arrival": "5-10 business days, back to the original payment method",
        }
    )


def create_ticket(ctx: ToolContext, args: CreateTicketInput) -> ToolOutcome:
    if args.order_id and ctx.db.order_for_customer(args.order_id, ctx.customer_id) is None:
        return _not_found(args.order_id)
    ticket_id = ctx.db.create_ticket(
        ctx.customer_id, args.subject, args.summary, args.priority, args.order_id
    )
    return ToolOutcome({"ticket_id": ticket_id, "status": "open", "priority": args.priority})


def search_knowledge_base(ctx: ToolContext, args: SearchKnowledgeBaseInput) -> ToolOutcome:
    hits = ctx.kb.search(args.query, args.top_k)
    return ToolOutcome(
        {
            "query": args.query,
            "results": [
                {"article": h.article, "title": h.title, "section": h.heading, "text": h.snippet}
                for h in hits
            ],
        }
    )


DEFAULT_TOOLS: tuple[ToolSpec, ...] = (
    ToolSpec(
        "lookup_order",
        "Look up one of the signed-in customer's orders: status, dates, items, prices, any "
        "existing refund and the free-text note the customer left at checkout.",
        LookupOrderInput,
        lookup_order,
    ),
    ToolSpec(
        "get_shipment_status",
        "Get the carrier, tracking number, delivery estimate and tracking events for an order.",
        ShipmentStatusInput,
        get_shipment_status,
    ),
    ToolSpec(
        "check_refund_eligibility",
        "Check whether an order can be refunded under the written refund policy and how much "
        "would be refunded. Read-only: it never issues a refund. Always call this before "
        "offering a refund.",
        RefundEligibilityInput,
        check_refund_eligibility,
    ),
    ToolSpec(
        "issue_refund",
        "Issue the refund for an order after check_refund_eligibility said it is eligible and "
        "the customer confirmed they want it. The amount is computed by the system from the "
        "order; you cannot choose it. Refunds above the review threshold are held until a "
        "human reviewer approves them.",
        IssueRefundInput,
        issue_refund,
    ),
    ToolSpec(
        "create_ticket",
        "Open a support ticket for a human specialist, e.g. for warranty claims, lost "
        "packages, refunds above the self-service cap, or anything the tools cannot resolve.",
        CreateTicketInput,
        create_ticket,
    ),
    ToolSpec(
        "search_knowledge_base",
        "Search the store's help articles (refund policy, shipping, warranty, payments). Use "
        "it to check what is allowed, required or charged instead of answering from memory.",
        SearchKnowledgeBaseInput,
        search_knowledge_base,
    ),
)


class Toolbox:
    def __init__(self, specs: tuple[ToolSpec, ...] = DEFAULT_TOOLS) -> None:
        self._specs = {spec.name: spec for spec in specs}

    @property
    def names(self) -> list[str]:
        return list(self._specs)

    def definitions(self) -> list[dict[str, Any]]:
        return [spec.definition() for spec in self._specs.values()]

    def resolve(self, name: str) -> ToolSpec | None:
        """Exact match, or an unambiguous case-insensitive match (``Lookup_Order``)."""
        if name in self._specs:
            return self._specs[name]
        matches = [s for n, s in self._specs.items() if n.lower() == name.lower()]
        return matches[0] if len(matches) == 1 else None

    def execute(self, name: str, raw_input: Any, ctx: ToolContext) -> ToolOutcome:
        spec = self.resolve(name)
        if spec is None:
            return ToolOutcome(
                {"error": f"Unknown tool '{name}'. Valid tools: {', '.join(self.names)}."},
                is_error=True,
            )
        try:
            args = spec.input_model.model_validate(raw_input)
        except ValidationError as exc:
            problems = [
                f"{'.'.join(str(p) for p in err['loc']) or 'input'}: {err['msg']}"
                for err in exc.errors()
            ]
            return ToolOutcome(
                {"error": f"Invalid input for {spec.name}.", "problems": problems}, is_error=True
            )
        try:
            return spec.handler(ctx, args)
        except Exception:
            log.exception("tool %s failed", spec.name)
            return ToolOutcome({"error": f"{spec.name} failed unexpectedly."}, is_error=True)

    def execute_approved_refund(
        self, request: RefundRequest, ctx: ToolContext, reviewer: str
    ) -> ToolOutcome:
        """Run a reviewer-approved refund, re-checking the policy at decision time."""
        try:
            return self._approved_refund(request, ctx, reviewer)
        except Exception:
            log.exception("approved refund for %s failed", request.order_id)
            return ToolOutcome({"error": "The refund failed unexpectedly."}, is_error=True)

    def _approved_refund(
        self, request: RefundRequest, ctx: ToolContext, reviewer: str
    ) -> ToolOutcome:
        order = ctx.db.order_for_customer(request.order_id, ctx.customer_id)
        if order is None:
            return _not_found(request.order_id)
        existing = ctx.db.refund_for_order(order.id)
        result = ctx.policy.evaluate(order, request.reason, existing, ctx.today())
        if not result.eligible:
            return ToolOutcome({"status": "not_eligible", **result.to_payload()})
        return _execute_refund(
            ctx, order.id, request.reason, result.refundable_cents, f"reviewer:{reviewer}"
        )
