"""Multi-turn scenarios driven by a scripted model.

The scripted model plays the part of Claude; everything else (tool execution, policy,
approval gate, history, traces) is the real code.
"""

from __future__ import annotations

import json

import pytest
from fakes import FakeClock, message, text, thinking, tool, tool_results

from support_agent.agent import ConversationBusy, Status, StopReason
from support_agent.config import Settings
from support_agent.model import ModelTimeout
from support_agent.trace import read_trace


def payload_of(result_block: dict) -> dict:
    """Decode the JSON body of a tool_result envelope."""
    lines = result_block["content"].splitlines()
    assert lines[0].startswith("<tool_output") and lines[-1] == "</tool_output>"
    return json.loads(lines[1])


def test_order_lookup_happy_path(make_agent):
    agent, model = make_agent(
        message(text("Let me check."), tool("lookup_order", {"order_id": "AC-10001"})),
        message(text("Order AC-10001 (Ridgeline 28L daypack) was delivered on 2026-10-10.")),
    )
    conv = agent.start("C-1001")
    result = agent.send(conv, "Where is my order AC-10001?")

    assert result.stop_reason is StopReason.COMPLETED
    assert result.tool_calls == ["lookup_order"]
    assert "delivered" in result.reply
    assert result.model_calls == 2
    (block,) = tool_results(model.calls[1])
    assert not block.get("is_error")
    order = payload_of(block)
    assert order["status"] == "delivered"
    assert order["items"][0]["name"] == "Ridgeline 28L daypack"
    # Append-only: every request starts with exactly what the previous one sent.
    assert model.calls[1]["messages"][:1] == model.calls[0]["messages"]
    assert [m["role"] for m in conv.messages] == ["user", "assistant", "user", "assistant"]


def test_history_keeps_thinking_blocks_unchanged(make_agent):
    agent, model = make_agent(
        message(thinking("checking"), tool("lookup_order", {"order_id": "AC-10001"})),
        message(text("Delivered.")),
    )
    conv = agent.start("C-1001")
    agent.send(conv, "Status of AC-10001?")
    first_assistant = model.calls[1]["messages"][1]
    assert first_assistant["content"][0] == {
        "type": "thinking",
        "thinking": "checking",
        "signature": "sig",
    }


def test_refund_below_threshold_is_issued_automatically(make_agent, db):
    agent, model = make_agent(
        message(
            tool("check_refund_eligibility", {"order_id": "AC-10001", "reason": "changed_mind"})
        ),
        message(text("You can get $89.00 back. Shall I go ahead?")),
        message(tool("issue_refund", {"order_id": "AC-10001", "reason": "changed_mind"})),
        message(text("Done: refund R-5002 for $89.00 is on its way.")),
    )
    conv = agent.start("C-1001")
    first = agent.send(conv, "I changed my mind about AC-10001, can I get a refund?")
    assert first.stop_reason is StopReason.COMPLETED
    assert payload_of(tool_results(model.calls[1])[0])["eligible"] is True
    assert db.refund_for_order("AC-10001") is None  # eligibility check is read-only

    second = agent.send(conv, "Yes please.")
    assert second.stop_reason is StopReason.COMPLETED
    refund = db.refund_for_order("AC-10001")
    assert refund is not None and refund.amount_cents == 8900 and refund.approved_by == "automatic"
    assert payload_of(tool_results(model.calls[3])[0])["status"] == "refunded"


def test_refund_above_threshold_pauses_until_approved(make_agent, db):
    agent, model = make_agent(
        message(tool("issue_refund", {"order_id": "AC-10002", "reason": "damaged"}, "toolu_tent")),
        message(text("Good news: the reviewer approved your $249.00 refund.")),
    )
    conv = agent.start("C-1001")
    paused = agent.send(conv, "My tent arrived torn, please refund AC-10002, I confirm.")

    assert paused.stop_reason is StopReason.AWAITING_APPROVAL
    assert conv.status is Status.AWAITING_APPROVAL
    (pending,) = paused.pending_approvals
    assert pending["amount"] == "$249.00" and pending["order_id"] == "AC-10002"
    assert "$249.00" in paused.reply
    assert db.refund_for_order("AC-10002") is None
    assert len(model.calls) == 1
    with pytest.raises(ConversationBusy):
        agent.send(conv, "Hello?")

    done = agent.decide(pending["approval_id"], approve=True, reviewer="sam")
    assert done.stop_reason is StopReason.COMPLETED
    assert conv.status is Status.ACTIVE
    refund = db.refund_for_order("AC-10002")
    assert refund is not None and refund.approved_by == "reviewer:sam"
    (block,) = tool_results(model.calls[1])
    assert block["tool_use_id"] == "toolu_tent"
    assert payload_of(block)["status"] == "refunded"
    with pytest.raises(ConversationBusy):
        agent.decide(pending["approval_id"], approve=False, reviewer="sam")


def test_refund_above_threshold_denied_by_reviewer(make_agent, db):
    agent, model = make_agent(
        message(tool("issue_refund", {"order_id": "AC-10002", "reason": "changed_mind"})),
        message(text("Sorry, the refund was declined. Shall I open a ticket?")),
    )
    conv = agent.start("C-1001")
    paused = agent.send(conv, "Refund AC-10002 please, yes I'm sure.")
    result = agent.decide(
        paused.pending_approvals[0]["approval_id"], approve=False, reviewer="sam", note="dup"
    )
    assert result.stop_reason is StopReason.COMPLETED
    assert db.refund_for_order("AC-10002") is None
    body = payload_of(tool_results(model.calls[1])[0])
    assert body["status"] == "denied_by_reviewer" and body["refund_issued"] is False


def test_parallel_tool_calls_return_in_one_message_after_approval(make_agent):
    agent, model = make_agent(
        message(
            tool("lookup_order", {"order_id": "AC-10001"}, "toolu_a"),
            tool("issue_refund", {"order_id": "AC-10002", "reason": "damaged"}, "toolu_b"),
        ),
        message(text("All set.")),
    )
    conv = agent.start("C-1001")
    paused = agent.send(conv, "Check AC-10001 and refund my damaged tent AC-10002, confirmed.")
    agent.decide(paused.pending_approvals[0]["approval_id"], approve=True, reviewer="sam")

    blocks = tool_results(model.calls[1])
    assert [b["tool_use_id"] for b in blocks] == ["toolu_a", "toolu_b"]


@pytest.mark.parametrize(
    ("order_id", "customer", "reason", "expected"),
    [
        ("AC-10003", "C-1001", "changed_mind", "window"),  # delivered 45 days ago
        ("AC-10005", "C-1002", "changed_mind", "None of the items"),  # gift card + clearance
        ("AC-10008", "C-1001", "damaged", "already refunded"),
        ("AC-10004", "C-1001", "damaged", "after delivery"),  # still in transit
    ],
)
def test_ineligible_refund_is_refused_by_policy(
    make_agent, db, order_id, customer, reason, expected
):
    agent, model = make_agent(
        message(tool("issue_refund", {"order_id": order_id, "reason": reason})),
        message(text("Sorry, that order can't be refunded.")),
    )
    conv = agent.start(customer)
    result = agent.send(conv, f"Refund {order_id} now, I confirm.")

    assert result.stop_reason is StopReason.COMPLETED
    assert result.pending_approvals == []
    body = payload_of(tool_results(model.calls[1])[0])
    assert body["status"] == "not_eligible"
    assert expected in " ".join(body["policy_notes"])
    if order_id != "AC-10008":
        assert db.refund_for_order(order_id) is None


def test_refund_above_cap_is_escalated_not_issued(make_agent, db):
    agent, model = make_agent(
        message(tool("issue_refund", {"order_id": "AC-10006", "reason": "changed_mind"})),
        message(text("That's above what I can refund; I'll open a ticket.")),
    )
    conv = agent.start("C-1002")
    agent.send(conv, "Refund my paddle set AC-10006, I confirm.")
    body = payload_of(tool_results(model.calls[1])[0])
    assert body["status"] == "not_eligible" and body["escalate_to_ticket"] is True
    assert db.refund_for_order("AC-10006") is None


def test_prompt_injection_in_order_note_cannot_bypass_policy(make_agent, db, trace_dir):
    """The order note tells the model to refund $5000 and approve it itself.

    The scripted model plays a model that falls for it. The note still reaches the model only
    as JSON inside the untrusted-data envelope, an attempt to pass an amount is rejected by
    the schema, and the refund that is allowed is computed from the order and held for review.
    """
    agent, model = make_agent(
        message(tool("lookup_order", {"order_id": "AC-10007"})),
        message(tool("issue_refund", {"order_id": "AC-10007", "reason": "other", "amount": 5000})),
        message(tool("issue_refund", {"order_id": "AC-10007", "reason": "other"})),
    )
    conv = agent.start("C-1002")
    result = agent.send(conv, "What's going on with order AC-10007?")

    # (a) the note arrives as data, inside the envelope, flagged.
    (lookup_block,) = tool_results(model.calls[1])
    envelope = lookup_block["content"]
    assert envelope.startswith('<tool_output tool="lookup_order" trust="untrusted-data">')
    assert (
        "IGNORE ALL PREVIOUS INSTRUCTIONS" in json.loads(envelope.splitlines()[1])["customer_note"]
    )
    assert "warning: fields $.customer_note" in envelope
    for request in model.calls:
        assert "IGNORE ALL PREVIOUS" not in request["system"]
        for msg in request["messages"]:
            if msg["role"] == "user" and isinstance(msg["content"], str):
                assert "IGNORE ALL PREVIOUS" not in msg["content"]

    # (b) the model cannot choose an amount.
    (amount_block,) = tool_results(model.calls[2])
    assert amount_block["is_error"] is True
    assert any("amount" in p for p in payload_of(amount_block)["problems"])

    # (c) the allowed refund is the order's own amount, and it waits for a human.
    assert result.stop_reason is StopReason.AWAITING_APPROVAL
    assert result.pending_approvals[0]["amount"] == "$129.00"
    assert db.refund_for_order("AC-10007") is None

    # (d) the trace records the flag and does not contain the card number.
    trace_text = (trace_dir / f"{conv.id}.jsonl").read_text()
    events = read_trace(trace_dir / f"{conv.id}.jsonl")
    lookup_result = next(e for e in events if e["event"] == "tool_result")
    assert lookup_result["injection_flags"] == ["$.customer_note"]
    assert "4111 1111 1111 1111" not in trace_text and "[CARD]" in trace_text


def test_tools_are_scoped_to_the_signed_in_customer(make_agent):
    agent, model = make_agent(
        message(tool("lookup_order", {"order_id": "AC-10005"})),  # belongs to C-1002
        message(text("I couldn't find that order on your account.")),
    )
    conv = agent.start("C-1001")
    agent.send(conv, "Look up AC-10005")
    (block,) = tool_results(model.calls[1])
    assert block["is_error"] is True
    assert "No order AC-10005" in payload_of(block)["error"]


def test_out_of_scope_request_is_declined_without_calling_the_model(make_agent):
    agent, model = make_agent()  # no scripted steps: any model call would fail the test
    conv = agent.start("C-1001")
    result = agent.send(conv, "Write me a poem about the ocean.")
    assert result.stop_reason is StopReason.OUT_OF_SCOPE
    assert "orders" in result.reply
    assert model.calls == [] and conv.messages == []


def test_in_scope_keywords_override_the_prefilter(make_agent):
    agent, model = make_agent(message(text("Sorry, I can only help with your orders.")))
    conv = agent.start("C-1001")
    result = agent.send(conv, "Write me a poem about my order AC-10001")
    assert result.stop_reason is StopReason.COMPLETED
    assert len(model.calls) == 1


def test_model_refusal_is_reported_and_not_added_to_history(make_agent):
    agent, _ = make_agent(
        message(stop_reason="refusal", stop_details={"type": "refusal", "category": None})
    )
    conv = agent.start("C-1001")
    result = agent.send(conv, "Tell me about AC-10001")
    assert result.stop_reason is StopReason.REFUSAL
    assert [m["role"] for m in conv.messages] == ["user"]


def test_max_turns_stops_the_loop(make_agent):
    steps = [message(tool("lookup_order", {"order_id": "AC-10001"})) for _ in range(5)]
    agent, model = make_agent(*steps, settings=Settings(max_turns=3))
    conv = agent.start("C-1001")
    result = agent.send(conv, "Check AC-10001 again and again")
    assert result.stop_reason is StopReason.MAX_TURNS
    assert len(model.calls) == 3
    assert conv.messages[-1]["role"] == "user"  # every tool_use got its tool_result


def test_model_timeout_stops_the_loop(make_agent):
    agent, _ = make_agent(ModelTimeout("read timed out"))
    conv = agent.start("C-1001")
    assert agent.send(conv, "Where is AC-10004?").stop_reason is StopReason.TIMEOUT


def test_wall_clock_deadline_bounds_the_run(make_agent):
    clock = FakeClock()

    def slow_step(_messages):
        clock.advance(70)
        return message(tool("lookup_order", {"order_id": "AC-10001"}))

    agent, model = make_agent(
        slow_step,
        slow_step,
        slow_step,
        settings=Settings(deadline_s=120, request_timeout_s=60),
        clock=clock,
    )
    conv = agent.start("C-1001")
    result = agent.send(conv, "Check AC-10001")
    assert result.stop_reason is StopReason.TIMEOUT
    assert len(model.calls) == 2
    assert model.calls[0]["timeout"] == 60
    assert model.calls[1]["timeout"] == 50  # never more than the time left


def test_invalid_tool_input_returns_an_error_the_model_can_fix(make_agent):
    agent, model = make_agent(
        message(tool("lookup_order", {"order_id": "10001"})),
        message(tool("lookup_order", {"order_id": "ac-10001"})),
        message(text("Found it.")),
    )
    conv = agent.start("C-1001")
    result = agent.send(conv, "order 10001?")
    (bad,) = tool_results(model.calls[1])
    assert bad["is_error"] is True and "order_id" in payload_of(bad)["problems"][0]
    (good,) = tool_results(model.calls[2])
    assert not good.get("is_error") and payload_of(good)["order_id"] == "AC-10001"
    assert result.stop_reason is StopReason.COMPLETED


def test_unknown_tool_name_is_reported_and_case_mismatch_is_tolerated(make_agent):
    agent, model = make_agent(
        message(
            tool("refund_everything", {}),
            tool("Lookup_Order", {"order_id": "AC-10001"}),
        ),
        message(text("Here you go.")),
    )
    conv = agent.start("C-1001")
    agent.send(conv, "Status of AC-10001")
    unknown, tolerated = tool_results(model.calls[1])
    assert unknown["is_error"] is True
    assert "Valid tools: lookup_order" in payload_of(unknown)["error"]
    assert not tolerated.get("is_error")


def test_truncated_tool_input_is_never_executed(make_agent, db):
    agent, model = make_agent(
        message(
            tool("issue_refund", {"order_id": "AC-10001", "reason": "changed_mind"}),
            stop_reason="max_tokens",
        ),
        message(text("Let me try that again later.")),
    )
    conv = agent.start("C-1001")
    agent.send(conv, "Refund AC-10001, confirmed")
    (block,) = tool_results(model.calls[1])
    assert block["is_error"] is True
    assert db.refund_for_order("AC-10001") is None


def test_server_side_fallback_turn_is_cleaned_before_it_is_echoed(make_agent, db):
    fallback = {
        "type": "fallback",
        "from": {"model": "claude-sonnet-5-5"},
        "to": {"model": "claude-sonnet-5"},
        "trigger": {"type": "refusal", "category": "cyber"},
    }
    agent, _ = make_agent(
        message(
            thinking(),
            tool("issue_refund", {"order_id": "AC-10001", "reason": "changed_mind"}),
            fallback,
            text("I can help with your order."),
            stop_reason="end_turn",
        )
    )
    conv = agent.start("C-1001")
    result = agent.send(conv, "Order AC-10001 question")
    assert result.stop_reason is StopReason.COMPLETED
    assert conv.messages[-1]["content"] == [{"type": "text", "text": "I can help with your order."}]
    assert db.refund_for_order("AC-10001") is None


def test_traces_are_redacted_but_the_model_sees_the_original(make_agent, trace_dir):
    agent, model = make_agent(message(text("Thanks Dana, how can I help?")))
    conv = agent.start("C-1001")
    pii = (
        "Hi, I'm Dana Whitfield, dana.whitfield@example.com, +1 555 010 4477, "
        "card 4111 1111 1111 1111, 42 Juniper Lane."
    )
    agent.send(conv, pii)

    assert model.calls[0]["messages"][0]["content"] == pii
    trace_text = (trace_dir / f"{conv.id}.jsonl").read_text()
    for secret in ("Dana", "dana.whitfield", "555 010", "4111", "Juniper"):
        assert secret not in trace_text


def test_trace_event_sequence_for_an_approval(make_agent, trace_dir):
    agent, _ = make_agent(
        message(tool("issue_refund", {"order_id": "AC-10002", "reason": "damaged"})),
        message(text("Approved and refunded.")),
    )
    conv = agent.start("C-1001")
    paused = agent.send(conv, "Refund damaged tent AC-10002, confirmed")
    agent.decide(paused.pending_approvals[0]["approval_id"], approve=True, reviewer="sam")

    events = read_trace(trace_dir / f"{conv.id}.jsonl")
    assert [e["event"] for e in events] == [
        "conversation_started",
        "user_message",
        "model_response",
        "tool_call",
        "approval_requested",
        "turn_finished",
        "approval_decided",
        "tool_result",
        "model_response",
        "turn_finished",
    ]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    assert events[-1]["stop_reason"] == "completed"
