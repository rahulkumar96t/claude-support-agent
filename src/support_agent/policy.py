"""The refund policy from ``kb/refund-policy.md``, implemented as deterministic code.

The model never decides eligibility or amounts. It can only ask; this module answers, using
the order record from the database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

from .db import Order, Refund, format_cents

RefundReason = Literal["damaged", "wrong_item", "not_as_described", "changed_mind", "other"]


@dataclass(frozen=True)
class RefundPolicy:
    standard_window_days: int = 30
    extended_window_days: int = 60
    self_service_cap_cents: int = 50_000
    approval_threshold_cents: int = 10_000
    never_refundable: frozenset[str] = frozenset({"gift_card"})
    final_sale: frozenset[str] = frozenset({"clearance"})
    extended_reasons: frozenset[str] = frozenset({"damaged", "wrong_item"})


@dataclass(frozen=True)
class Eligibility:
    order_id: str
    eligible: bool
    refundable_cents: int = 0
    requires_approval: bool = False
    escalate: bool = False
    reasons: tuple[str, ...] = ()
    excluded_items: tuple[str, ...] = ()
    window_ends_on: date | None = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "order_id": self.order_id,
            "eligible": self.eligible,
            "refundable_amount": format_cents(self.refundable_cents),
            "requires_human_approval": self.requires_approval,
            "escalate_to_ticket": self.escalate,
            "policy_notes": list(self.reasons),
        }
        if self.excluded_items:
            payload["excluded_items"] = list(self.excluded_items)
        if self.window_ends_on:
            payload["refund_window_ends_on"] = self.window_ends_on.isoformat()
        return payload


@dataclass
class PolicyEngine:
    policy: RefundPolicy = field(default_factory=RefundPolicy)

    def evaluate(
        self,
        order: Order,
        reason: RefundReason,
        existing_refund: Refund | None,
        today: date,
    ) -> Eligibility:
        p = self.policy

        def no(
            *why: str,
            escalate: bool = False,
            excluded: tuple[str, ...] = (),
            window_ends_on: date | None = None,
        ) -> Eligibility:
            return Eligibility(
                order.id,
                eligible=False,
                escalate=escalate,
                reasons=why,
                excluded_items=excluded,
                window_ends_on=window_ends_on,
            )

        if order.status != "delivered" or order.delivered_on is None:
            return no(
                f"The order is {order.status}; refunds are only available after delivery. "
                "Offer to open a support ticket instead."
            )
        if existing_refund is not None:
            return no(
                f"The order was already refunded ({existing_refund.id}, "
                f"{format_cents(existing_refund.amount_cents)}). Each order is refunded once."
            )

        window = p.extended_window_days if reason in p.extended_reasons else p.standard_window_days
        window_ends = order.delivered_on + timedelta(days=window)
        if today > window_ends:
            days = (today - order.delivered_on).days
            return no(
                f"Delivered {days} days ago; the {window}-day window for reason '{reason}' "
                f"ended on {window_ends.isoformat()}.",
                window_ends_on=window_ends,
            )

        refundable = 0
        excluded: list[str] = []
        for item in order.items:
            if item.category in p.never_refundable:
                excluded.append(f"{item.name} ({item.category}: never refundable)")
            elif item.category in p.final_sale and reason not in p.extended_reasons:
                excluded.append(f"{item.name} ({item.category}: final sale)")
            else:
                refundable += item.line_total_cents

        if refundable == 0:
            return no(
                "None of the items on this order are refundable under the policy.",
                excluded=tuple(excluded),
                window_ends_on=window_ends,
            )
        if refundable > p.self_service_cap_cents:
            return no(
                f"The refundable amount {format_cents(refundable)} is above the "
                f"{format_cents(p.self_service_cap_cents)} self-service cap. "
                "Open a high-priority support ticket instead.",
                escalate=True,
                excluded=tuple(excluded),
                window_ends_on=window_ends,
            )

        needs_review = refundable > p.approval_threshold_cents
        notes = [f"Eligible within the {window}-day window."]
        if needs_review:
            notes.append(
                f"Amounts above {format_cents(p.approval_threshold_cents)} are held for a "
                "human reviewer before the refund is issued."
            )
        return Eligibility(
            order_id=order.id,
            eligible=True,
            refundable_cents=refundable,
            requires_approval=needs_review,
            reasons=tuple(notes),
            excluded_items=tuple(excluded),
            window_ends_on=window_ends,
        )
