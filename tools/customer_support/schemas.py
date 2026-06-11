"""Pydantic schemas for customer support tools."""

from pydantic import BaseModel, Field


class CustomerLookupInput(BaseModel):
    """Input for finding a customer profile."""

    customer_id: str = Field(description="Internal customer identifier.")


class OrderLookupInput(BaseModel):
    """Input for finding an order."""

    order_id: str = Field(description="Order identifier.")


class RefundEligibilityInput(BaseModel):
    """Input for refund eligibility checks."""

    order_id: str = Field(description="Order identifier.")
    reason: str = Field(description="Customer's refund reason.")


class ReturnLabelInput(BaseModel):
    """Input for return label creation."""

    order_id: str = Field(description="Order identifier.")
    email: str = Field(description="Customer email address.")


class ShippingStatusInput(BaseModel):
    """Input for shipment tracking."""

    tracking_number: str = Field(description="Carrier tracking number.")


class SubscriptionUpdateInput(BaseModel):
    """Input for subscription plan updates."""

    customer_id: str = Field(description="Internal customer identifier.")
    plan: str = Field(description="Requested subscription plan.")


class KnowledgeSearchInput(BaseModel):
    """Input for knowledge base search."""

    query: str = Field(description="Support question or issue summary.")


class SentimentInput(BaseModel):
    """Input for customer sentiment analysis."""

    message: str = Field(description="Customer message to analyze.")


class EscalationInput(BaseModel):
    """Input for creating an escalation."""

    ticket_id: str = Field(description="Support ticket identifier.")
    severity: str = Field(
        description="Severity such as low, medium, high, or urgent."
    )


class TicketSummaryInput(BaseModel):
    """Input for summarizing a ticket."""

    ticket_id: str = Field(description="Support ticket identifier.")
