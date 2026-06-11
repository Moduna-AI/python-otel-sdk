"""BFSI LangChain agent."""

from typing import Any

from services.base_agent_service import LangChainAgentService
from tools.bfsi.registry import BFSI_TOOLS


class BFSIAgentService(LangChainAgentService):
    """Service for banking, financial services, and insurance workflows."""

    agent_name = "bfsi_agent"
    system_prompt = (
        "You are a BFSI operations bot. Prioritize compliance, customer privacy, fraud controls, "
        "and explain regulated decisions without exposing sensitive internals."
    )

    def get_tools(self) -> list[Any]:
        """Return BFSI tools."""
        return BFSI_TOOLS
