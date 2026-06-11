"""BFSI LangChain tools."""

from langchain.tools import tool

from tools.bfsi.schemas import (
    AccountLookupInput,
    CardControlInput,
    ClaimStatusInput,
    ComplianceInput,
    DisputeInput,
    FraudReviewInput,
    KycStatusInput,
    LoanEligibilityInput,
    PolicyLookupInput,
    TransactionSearchInput,
)


@tool(args_schema=AccountLookupInput)
def lookup_account_summary(customer_id: str) -> str:
    """Look up non-sensitive account summary information."""
    return f"Customer {customer_id}: two active accounts, KYC current, no collections flags."


@tool(args_schema=TransactionSearchInput)
def search_transactions(account_id: str, amount: float | None = None) -> str:
    """Search transactions using safe filters."""
    amount_note = f" matching amount {amount}" if amount is not None else ""
    return f"Account {account_id}: found three recent card transactions{amount_note}; details require verification."


@tool(args_schema=FraudReviewInput)
def review_fraud_risk(transaction_id: str, customer_confirmed: bool) -> str:
    """Review fraud risk for a transaction."""
    if customer_confirmed:
        return f"Transaction {transaction_id}: customer confirmed activity; fraud case not opened."
    return f"Transaction {transaction_id}: unrecognized activity; freeze card and open fraud review."


@tool(args_schema=CardControlInput)
def manage_card_control(card_id: str, action: str) -> str:
    """Apply a card control action after verification."""
    return f"Card {card_id}: {action} action queued pending step-up authentication."


@tool(args_schema=LoanEligibilityInput)
def check_loan_eligibility(customer_id: str, product_type: str) -> str:
    """Check preliminary loan eligibility without making a credit decision."""
    return f"Customer {customer_id}: preliminarily eligible for {product_type}; formal underwriting still required."


@tool(args_schema=KycStatusInput)
def check_kyc_status(customer_id: str) -> str:
    """Check KYC and document status."""
    return f"Customer {customer_id}: KYC complete, address proof valid, next periodic review due in 90 days."


@tool(args_schema=PolicyLookupInput)
def lookup_insurance_policy(policy_id: str) -> str:
    """Look up insurance policy status and coverage summary."""
    return f"Policy {policy_id}: active, premium current, comprehensive coverage with standard exclusions."


@tool(args_schema=ClaimStatusInput)
def get_claim_status(claim_id: str) -> str:
    """Get insurance claim status."""
    return f"Claim {claim_id}: assessor review complete, settlement calculation pending."


@tool(args_schema=DisputeInput)
def open_transaction_dispute(transaction_id: str, dispute_reason: str) -> str:
    """Open a regulated transaction dispute."""
    return f"Dispute opened for {transaction_id}: {dispute_reason}; provisional credit review started."


@tool(args_schema=ComplianceInput)
def get_compliance_guidance(activity: str) -> str:
    """Get compliance guidance for a BFSI activity."""
    return f"Compliance guidance for {activity}: verify identity, minimize data exposure, log rationale, and escalate exceptions."


BFSI_TOOLS = [
    lookup_account_summary,
    search_transactions,
    review_fraud_risk,
    manage_card_control,
    check_loan_eligibility,
    check_kyc_status,
    lookup_insurance_policy,
    get_claim_status,
    open_transaction_dispute,
    get_compliance_guidance,
]
