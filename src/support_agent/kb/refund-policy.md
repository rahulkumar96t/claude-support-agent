# Refund policy

Acme Trail Supply is a fictional outdoor-gear shop used for this demo. This article is the
written refund policy. The agent's refund tools implement exactly these rules in code
(`support_agent/policy.py`); a test checks that the parameters below match the code.

## Policy parameters

| Parameter | Value |
|---|---|
| Standard refund window | 30 days after delivery |
| Extended window (damaged or wrong item) | 60 days after delivery |
| Self-service refund cap | $500.00 per order |
| Human approval required above | $100.00 |
| Never refundable categories | gift_card |
| Final-sale categories | clearance |

## Who can get a refund

- Refunds are available for orders that have been **delivered**. Orders that are still
  processing or in transit cannot be refunded yet; a support ticket can be opened instead.
- Each order can be refunded **once**. A second refund request for the same order is declined.
- The refund amount is the sum of the refundable items on the order, computed by the system
  from the order record. Customers and agents cannot choose a different amount.

## Refund windows

- The standard window is 30 days after the delivery date for any reason, including a change of mind.
- If the item arrived damaged or the wrong item was delivered, the window is 60 days.

## Excluded items

- Gift cards are never refundable.
- Clearance items are final sale. They are refundable only if they arrived damaged or the
  wrong item was delivered.

## Amount limits and review

- Refunds of $100.00 or less are issued immediately after the eligibility check.
- Refunds above $100.00 are held until a human reviewer approves them. The customer is told
  that the refund is pending review.
- Orders whose refundable amount is above $500.00 cannot be refunded through self-service.
  The agent opens a high-priority support ticket instead.
