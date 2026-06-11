"""Customer support LangChain tools."""

from langchain.tools import tool

from tools.customer_support.schemas import (
    CustomerLookupInput,
    EscalationInput,
    KnowledgeSearchInput,
    OrderLookupInput,
    RefundEligibilityInput,
    ReturnLabelInput,
    SentimentInput,
    ShippingStatusInput,
    SubscriptionUpdateInput,
    TicketSummaryInput,
)


@tool(args_schema=CustomerLookupInput)
def lookup_customer_profile(customer_id: str) -> str:
    """Look up customer profile, tier, and account standing."""
    return f"Customer {customer_id}: active account, gold tier, verified email, no payment holds."


@tool(args_schema=OrderLookupInput)
def lookup_order(order_id: str) -> str:
    """Look up order status, items, and fulfillment details."""
    return f"Order {order_id}: delivered, two items, paid by card, support window open."


@tool(args_schema=RefundEligibilityInput)
def check_refund_eligibility(order_id: str, reason: str) -> str:
    """Check whether an order is eligible for refund."""
    return f"Order {order_id} is eligible for manual review refund because the reason is: {reason}."


@tool(args_schema=ReturnLabelInput)
def create_return_label(order_id: str, email: str) -> str:
    """Create a return label for an eligible order."""
    return f"Return label for order {order_id} created and queued for delivery to {email}."


@tool(args_schema=ShippingStatusInput)
def track_shipping_status(tracking_number: str) -> str:
    """Track shipping status for a carrier tracking number."""
    return (
        f"Tracking {tracking_number}: delivered yesterday at 14:20 local time."
    )


@tool(args_schema=SubscriptionUpdateInput)
def update_subscription_plan(customer_id: str, plan: str) -> str:
    """Update a customer's subscription plan."""
    return f"Customer {customer_id} subscription change to {plan} is staged pending confirmation."


@tool(args_schema=KnowledgeSearchInput)
def search_support_knowledge_base(query: str) -> str:
    """Search the support knowledge base for policy or troubleshooting guidance."""
    return f"Knowledge base result for '{query}': verify identity, confirm entitlement, then apply the relevant policy."


@tool(args_schema=SentimentInput)
def classify_customer_sentiment(message: str) -> str:
    """Classify customer sentiment and support risk."""
    sentiment = (
        "urgent"
        if "angry" in message.lower() or "cancel" in message.lower()
        else "neutral"
    )
    return f"Customer sentiment: {sentiment}."


@tool(args_schema=EscalationInput)
def create_support_escalation(ticket_id: str, severity: str) -> str:
    """Create an escalation for a customer support ticket."""
    return f"Ticket {ticket_id} escalated with severity {severity}; owner group: tier-2 support."


@tool(args_schema=TicketSummaryInput)
def summarize_support_ticket(ticket_id: str) -> str:
    """Summarize a support ticket for handoff."""
    return f"Ticket {ticket_id}: customer reported delivery issue, order verified, next step is resolution offer."


CUSTOMER_SUPPORT_TOOLS = [
    lookup_customer_profile,
    lookup_order,
    check_refund_eligibility,
    create_return_label,
    track_shipping_status,
    update_subscription_plan,
    search_support_knowledge_base,
    classify_customer_sentiment,
    create_support_escalation,
    summarize_support_ticket,
]
