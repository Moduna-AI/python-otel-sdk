"""Shared LangChain agent service base."""

from typing import Any

from langchain.agents import create_agent

DEFAULT_AGENT_MODEL = "google_genai:gemini-2.5-flash"


class LangChainAgentService:
    """Base service for domain-specific LangChain agents."""

    agent_name: str
    system_prompt: str

    def __init__(self, model: str = DEFAULT_AGENT_MODEL) -> None:
        """Initialize a LangChain agent with the configured tools."""
        self.model = model
        self.agent = create_agent(
            model=self.model,
            tools=self.get_tools(),
            system_prompt=self.system_prompt,
            name=self.agent_name,
        )

    def get_tools(self) -> list[Any]:
        """Return the tools available to this agent."""
        raise NotImplementedError

    def invoke(self, message: str) -> dict[str, Any]:
        """Invoke the agent with a single user message."""
        return self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]}
        )
