"""IT service LangChain agent."""

from typing import Any

from services.base_agent_service import LangChainAgentService
from tools.it_service.registry import IT_SERVICE_TOOLS


class ITServiceAgentService(LangChainAgentService):
    """Service for internal IT support workflows."""

    agent_name = "it_service_agent"
    system_prompt = (
        "You are an IT service desk bot. Triage incidents by business impact, use approved runbooks, "
        "protect credentials, and clearly separate remediation from escalation."
    )

    def get_tools(self) -> list[Any]:
        """Return IT service tools."""
        return IT_SERVICE_TOOLS
