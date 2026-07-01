"""
claude_agent.py — Claude agent with You.com web search tool.

The agent decides when to search. You.com handles the retrieval.

    from agents.claude_agent import ClaudeAgent

    agent = ClaudeAgent()
    result = agent.ask("What happened in tech news this week?")

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY="sk-ant-..."
    export YDC_API_KEY="..."

Responses API note:
    Claude uses Anthropic's Messages API, not the OpenAI Responses API.
    force_chat_completions is accepted as a parameter for API consistency but
    has no effect — this agent always uses AnthropicAgent's loop.
"""

import os

from anthropic import Anthropic

from base_agent import AnthropicAgent

# Pinned model versions:
#   "claude-opus-4-8"              — most capable, highest cost
#   "claude-sonnet-4-6"            — balanced (recommended)
#   "claude-haiku-4-5-20251001"    — fastest, lowest cost
DEFAULT_MODEL = "claude-sonnet-4-6"


class ClaudeAgent(AnthropicAgent):
    """Claude-powered agent with You.com web search.

    Claude uses Anthropic's tool_use format (not OpenAI function calling),
    which is why this inherits from AnthropicAgent instead of
    OpenAICompatibleAgent.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        force_chat_completions: bool = False,  # accepted for API consistency, no effect
    ):
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not resolved_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. "
                "Pass api_key= or export ANTHROPIC_API_KEY."
            )
        client = Anthropic(api_key=resolved_key)
        super().__init__(client=client, model=model)


if __name__ == "__main__":
    import sys

    question = " ".join(sys.argv[1:]) or input("Ask something: ")
    agent = ClaudeAgent()
    result = agent.ask(question)
    print(result["answer"])
    if result["sources"]:
        print(f"\nSources: {result['sources']}")
