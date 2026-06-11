"""Customer support LangChain agent."""

from typing import Any

from services.base_agent_service import LangChainAgentService
from tools.customer_support.registry import CUSTOMER_SUPPORT_TOOLS


class CustomerSupportAgentService(LangChainAgentService):
    """Service for customer support agent workflows."""

    agent_name = "customer_support_agent"
    system_prompt = (
        "You are a customer support operations bot. Resolve customer issues with empathy, "
        "verify account and order details before actioning changes, and escalate risky requests."
    )

    def get_tools(self) -> list[Any]:
        """Return customer support tools."""
        return CUSTOMER_SUPPORT_TOOLS
