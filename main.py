"""Moduna SDK LangChain example."""

import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from moduna import Moduna
from moduna.sdk.tracing import Instruments, set_conversation_id
import uuid


def run_example() -> None:
    """Run a small LangChain request with Moduna tracing enabled."""
    Moduna().init(
        {
            "app_name": "langchain_trace",
            "framework": Instruments.LANGCHAIN,
        }
    )
    unique_id = uuid.uuid4()
    set_conversation_id(f"conv-{unique_id}")

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        temperature=0.3,
        max_tokens=40,
        timeout=None,
        max_retries=2,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a witty, sarcastic AI assistant."),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    print("\n--- Sending LLM Request ---")
    response = chain.invoke({"question": "Why is the water blue?"})

    print("\n--- LLM Response ---")
    print(response)
    print(
        "\n--- (Look above for the OpenTelemetry Spans printed by Traceloop) ---\n"
    )


if __name__ == "__main__":
    if "GOOGLE_API_KEY" not in os.environ:
        raise ValueError(
            "Set GOOGLE_API_KEY before running the LangChain example."
        )

    run_example()
