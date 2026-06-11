"""Command line entrypoint for domain agents."""

import argparse
import os
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from traceloop.sdk.tracing import set_conversation_id

from services.base_agent_service import (
    DEFAULT_AGENT_MODEL,
    LangChainAgentService,
)
from services.bfsi import BFSIAgentService
from services.customer_support import CustomerSupportAgentService
from services.it_service import ITServiceAgentService

AgentFactory = Callable[[str], LangChainAgentService]

AGENT_FACTORIES: dict[str, AgentFactory] = {
    "customer_support": CustomerSupportAgentService,
    "it_service": ITServiceAgentService,
    "bfsi": BFSIAgentService,
}


class DomainAgentCli:
    """CLI service for invoking a selected domain agent."""

    def run(self, argv: list[str] | None = None) -> int:
        """Run the CLI with optional argv overrides."""
        args = self.parse_args(argv)
        conversation_id = args.conversation_id or f"cli-{uuid4()}"

        if args.trace:
            self.setup_tracing(conversation_id)

        agent = AGENT_FACTORIES[args.agent](args.model)
        result = agent.invoke(args.prompt)
        print(self.extract_output(result))
        return 0

    def parse_args(self, argv: list[str] | None) -> argparse.Namespace:
        """Parse CLI arguments."""
        parser = argparse.ArgumentParser(
            description="Invoke a Moduna domain agent."
        )
        parser.add_argument(
            "--agent", choices=sorted(AGENT_FACTORIES), required=True
        )
        parser.add_argument("--prompt", required=True)
        parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
        parser.add_argument("--conversation-id")

        trace_group = parser.add_mutually_exclusive_group()
        trace_group.add_argument(
            "--trace", dest="trace", action="store_true", default=True
        )
        trace_group.add_argument(
            "--no-trace", dest="trace", action="store_false"
        )

        return parser.parse_args(argv)

    def setup_tracing(self, conversation_id: str) -> None:
        """Initialize Moduna tracing for CLI agent runs."""
        from moduna import Instruments, Moduna

        Moduna().init(
            {
                "app_name": f"moduna-{conversation_id}",
                "framework": Instruments.LANGCHAIN,
                "api_key": os.environ.get("MODUNA_API_KEY"),
            }
        )
        set_conversation_id(conversation_id)

    def extract_output(self, result: dict[str, Any]) -> str:
        """Extract the final assistant content from a LangChain agent result."""
        messages = result.get("messages", [])
        if messages:
            content = getattr(messages[-1], "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                text_blocks = [
                    block["text"]
                    for block in content
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                ]
                if text_blocks:
                    return "\n".join(text_blocks)
            if content is not None:
                return str(content)
        return str(result)


def main() -> int:
    """Run the domain agent CLI."""
    return DomainAgentCli().run()
