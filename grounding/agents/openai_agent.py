"""
openai_agent.py — OpenAI agent with You.com web search tool.

Default path: OpenAI Responses API (previous_response_id chaining, parallel
tool calls). Use force_chat_completions=True for A/B comparison against the
older chat.completions path.

    from agents.openai_agent import OpenAIAgent

    agent = OpenAIAgent()
    result = agent.ask("Compare Tesla and BYD delivery numbers this quarter")

    # Force chat.completions for comparison:
    agent = OpenAIAgent(force_chat_completions=True)
    result = agent.ask("Same question, older API path")

    # To use a specific model version:
    agent = OpenAIAgent(model="gpt-5.4-mini")
    result = agent.ask("Quick fact lookup")

Requirements:
    pip install openai
    export OPENAI_API_KEY="sk-..."
    export YDC_API_KEY="..."

API path selection:
    Responses API (default):   parallel tool calls, previous_response_id chaining,
                               no token accumulation across rounds. Tool is named
                               'you_search' via TOOL_SCHEMA_RESPONSES (not 'web_search')
                               so DashScope/Qwen can also reuse this class safely.
    chat.completions fallback: full message history re-sent every round,
                               sequential tool execution, tool named 'web_search'.
    See decisions/006_gpt_ydc_chat_vs_responses_api.md for full comparison.

Design note — factory function, not a class:
    OpenAIAgent() is a function that returns either OpenAIResponsesAgent or
    OpenAICompatibleAgent. This is intentional: the two paths have genuinely
    different loop contracts (stateful vs stateless), and inheriting from both
    would require runtime branching inside methods. The factory keeps each class
    pure and makes the selection point explicit at construction time.
    Consequence: isinstance(agent, OpenAIAgent) always returns False — use
    isinstance(agent, OpenAIResponsesAgent) or isinstance(agent, BaseAgent).

Model versions pinned here vs. run.py/compare.py:
    This file's DEFAULT_MODEL is used when the agent is instantiated directly
    (programmatic use, agent CLI). run.py and compare.py maintain their own
    model configs in GROUNDING_MODELS / pricing.json so that display name,
    cost, and API quirks are co-located with the model key. The two configs
    are intentionally separate and may differ. Do NOT auto-sync them.
"""

import os
import sys

from openai import OpenAI

from base_agent import OpenAICompatibleAgent, OpenAIResponsesAgent

# Pinned model versions:
#   "gpt-5.4"        — latest flagship (recommended)
#   "gpt-5.4-mini"   — balanced cost/performance
#   "gpt-5.4-nano"   — fastest, lowest cost
#   "gpt-4.1"        — previous gen, still in API
DEFAULT_MODEL = "gpt-5.4"


def OpenAIAgent(
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    force_chat_completions: bool = False,
):
    """Factory that returns the right agent class based on force_chat_completions.

    Returns OpenAIResponsesAgent by default (Responses API).
    Returns OpenAICompatibleAgent when force_chat_completions=True.

    The force_chat_completions flag is set once at construction — it determines
    which loop implementation the agent uses for its lifetime, not per-call.
    Pass --chat-completions on the CLI (run.py, benchmark_runner.py) to flip it.
    """
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not resolved_key:
        raise ValueError(
            "OPENAI_API_KEY not set. "
            "Pass api_key= or export OPENAI_API_KEY."
        )
    client = OpenAI(api_key=resolved_key)
    cls = OpenAICompatibleAgent if force_chat_completions else OpenAIResponsesAgent
    return cls(client=client, model=model)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    force_cc = "--chat-completions" in args
    args = [a for a in args if a != "--chat-completions"]
    question = " ".join(args) or input("Ask something: ")
    agent = OpenAIAgent(force_chat_completions=force_cc)
    result = agent.ask(question)
    if result is None:
        print("Error: no result returned")
        sys.exit(1)
    answer = result.get("answer") or "(no answer)"
    print(answer)
    if result.get("sources"):
        print(f"\nSources: {result['sources']}")
