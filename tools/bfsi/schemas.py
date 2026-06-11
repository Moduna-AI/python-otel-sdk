"""Pydantic schemas for BFSI tools."""

from pydantic import BaseModel, Field


class AccountLookupInput(BaseModel):
    """Input for account lookups."""

    customer_id: str = Field(description="Banking customer identifier.")


class TransactionSearchInput(BaseModel):
    """Input for transaction searches."""

    account_id: str = Field(description="Account identifier.")
    amount: float | None = Field(
        default=None, description="Optional transaction amount filter."
    )


class FraudReviewInput(BaseModel):
    """Input for fraud risk reviews."""

    transaction_id: str = Field(description="Transaction identifier.")
    customer_confirmed: bool = Field(
        description="Whether the customer recognizes the transaction."
    )


class CardControlInput(BaseModel):
    """Input for card controls."""

    card_id: str = Field(description="Payment card identifier.")
    action: str = Field(
        description="Requested action such as freeze, unfreeze, or replace."
    )


class LoanEligibilityInput(BaseModel):
    """Input for loan eligibility checks."""

    customer_id: str = Field(description="Banking customer identifier.")
    product_type: str = Field(description="Loan product type.")


class KycStatusInput(BaseModel):
    """Input for KYC status checks."""

    customer_id: str = Field(description="Banking customer identifier.")


class PolicyLookupInput(BaseModel):
    """Input for insurance policy lookups."""

    policy_id: str = Field(description="Insurance policy identifier.")


class ClaimStatusInput(BaseModel):
    """Input for insurance claim status."""

    claim_id: str = Field(description="Insurance claim identifier.")


class DisputeInput(BaseModel):
    """Input for transaction disputes."""

    transaction_id: str = Field(description="Transaction identifier.")
    dispute_reason: str = Field(description="Customer's dispute reason.")


class ComplianceInput(BaseModel):
    """Input for compliance guidance."""

    activity: str = Field(description="Financial activity or customer request.")
