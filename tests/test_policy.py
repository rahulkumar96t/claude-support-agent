from __future__ import annotations

import re
from datetime import date, timedelta
from importlib import resources

import pytest

from support_agent.db import Order, OrderItem, Refund
from support_agent.policy import PolicyEngine, RefundPolicy

TODAY = date(2026, 10, 15)


def order(
    *items: OrderItem, status: str = "delivered", delivered_days_ago: int | None = 5
) -> Order:
    delivered = (
        TODAY - timedelta(days=delivered_days_ago) if delivered_days_ago is not None else None
    )
    return Order("AC-20001", "C-1", TODAY - timedelta(days=40), status, delivered, "", items)


def item(category: str = "tents", cents: int = 5000, qty: int = 1) -> OrderItem:
    return OrderItem("SKU", f"{category} item", category, qty, cents)


engine = PolicyEngine()


def test_small_refund_is_eligible_without_review():
    result = engine.evaluate(order(item(cents=4000, qty=2)), "changed_mind", None, TODAY)
    assert result.eligible and result.refundable_cents == 8000 and not result.requires_approval


def test_amount_above_threshold_needs_review():
    result = engine.evaluate(order(item(cents=10_001)), "changed_mind", None, TODAY)
    assert result.eligible and result.requires_approval


def test_threshold_itself_does_not_need_review():
    result = engine.evaluate(order(item(cents=10_000)), "changed_mind", None, TODAY)
    assert result.eligible and not result.requires_approval


@pytest.mark.parametrize(
    ("days", "reason", "eligible"),
    [(30, "changed_mind", True), (31, "changed_mind", False), (60, "damaged", True),
     (61, "wrong_item", False), (45, "not_as_described", False)],
)  # fmt: skip
def test_refund_windows(days, reason, eligible):
    result = engine.evaluate(order(item(), delivered_days_ago=days), reason, None, TODAY)
    assert result.eligible is eligible


def test_gift_cards_are_never_refundable_and_clearance_is_final_sale():
    mixed = order(item("gift_card", 5000), item("clearance", 900), item("tents", 2000))
    result = engine.evaluate(mixed, "changed_mind", None, TODAY)
    assert result.refundable_cents == 2000
    assert len(result.excluded_items) == 2


def test_clearance_is_refundable_when_damaged():
    result = engine.evaluate(order(item("clearance", 900)), "damaged", None, TODAY)
    assert result.eligible and result.refundable_cents == 900


def test_only_gift_cards_means_nothing_to_refund():
    result = engine.evaluate(order(item("gift_card", 5000)), "damaged", None, TODAY)
    assert not result.eligible and result.refundable_cents == 0


def test_already_refunded_order_is_refused():
    previous = Refund("R-1", "AC-20001", 100, "damaged", "automatic", "2026-10-01T00:00:00")
    assert not engine.evaluate(order(item()), "damaged", previous, TODAY).eligible


@pytest.mark.parametrize("status", ["processing", "shipped", "cancelled"])
def test_undelivered_orders_are_refused(status):
    result = engine.evaluate(
        order(item(), status=status, delivered_days_ago=None), "damaged", None, TODAY
    )
    assert not result.eligible


def test_above_self_service_cap_escalates():
    result = engine.evaluate(order(item(cents=50_001)), "damaged", None, TODAY)
    assert not result.eligible and result.escalate


def test_written_policy_matches_the_code():
    """The numbers in kb/refund-policy.md are the numbers the code enforces."""
    doc = (resources.files("support_agent") / "kb" / "refund-policy.md").read_text("utf-8")
    table = dict(re.findall(r"^\| ([^|]+?) \| ([^|]+?) \|$", doc, flags=re.MULTILINE))
    policy = RefundPolicy()

    def days(key: str) -> int:
        return int(re.match(r"(\d+) days", table[key]).group(1))

    def cents(key: str) -> int:
        return int(table[key].split()[0].lstrip("$").replace(",", "").replace(".", ""))

    assert days("Standard refund window") == policy.standard_window_days
    assert days("Extended window (damaged or wrong item)") == policy.extended_window_days
    assert cents("Self-service refund cap") == policy.self_service_cap_cents
    assert cents("Human approval required above") == policy.approval_threshold_cents
    assert set(table["Never refundable categories"].split(", ")) == policy.never_refundable
    assert set(table["Final-sale categories"].split(", ")) == policy.final_sale
