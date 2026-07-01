"""
demo.py — Interactive demo showing how to use the grounded agent.

This is the simplest possible integration. It creates an agent,
registers the You.com search tool (automatically via the agent class),
and runs a conversation loop. The agent decides when to search.

Run it:
    python demo.py                    # Claude (default)
    python demo.py openai             # GPT-4.1
    python demo.py qwen               # Qwen via DashScope
    python demo.py qwen huggingface   # Qwen via Hugging Face
    python demo.py kimi               # Kimi K2.5
    python demo.py llama              # Llama via Together AI
    python demo.py llama ollama       # Llama via Ollama (local)

This is a demo. In your real application, you'd import the agent class
directly and call agent.ask() wherever you need grounded responses:

    from agents.claude_agent import ClaudeAgent

    agent = ClaudeAgent()

    # In your API handler, pipeline, CLI, wherever:
    result = agent.ask(user_question)
    print(result["answer"])
    print(result["sources"])
"""

import os
import sys
from dotenv import load_dotenv
# Supports both env.txt (visible in Finder) and .env (standard convention)
load_dotenv("env.txt") or load_dotenv(".env")

from search_tool import is_verbose, format_tool_log
from agents import ClaudeAgent, OpenAIAgent, QwenAgent, KimiAgent, LlamaAgent

AGENTS = {
    "claude":  lambda backend: ClaudeAgent(),
    "openai":  lambda backend: OpenAIAgent(),
    "qwen":    lambda backend: QwenAgent(backend=backend or "dashscope"),
    "kimi":    lambda backend: KimiAgent(),
    "llama":   lambda backend: LlamaAgent(backend=backend or "together"),
}


def main():
    # Parse args: python demo.py [provider] [backend] [--verbose]
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    provider = args[0] if len(args) > 0 else "claude"
    backend = args[1] if len(args) > 1 else None

    # --verbose flag sets GROUNDING_VERBOSE=1 for this session
    if "--verbose" in flags:
        os.environ["GROUNDING_VERBOSE"] = "1"

    if provider not in AGENTS:
        print(f"Unknown provider: {provider}")
        print(f"Available: {', '.join(AGENTS.keys())}")
        sys.exit(1)

    # Create the agent. This registers the web_search tool automatically.
    # If an API key is missing, the agent raises ValueError with a clear message.
    try:
        agent = AGENTS[provider](backend)
    except ValueError as e:
        print(f"Setup error: {e}")
        sys.exit(1)

    verbose = is_verbose()
    label = f"{provider}" + (f" ({backend})" if backend else "")
    print(f"Agent ready: {agent.model} via {label}")
    print(f"Interface:   direct_api (You.com Search API → ydc-index.io/v1/search)")
    print(f"Verbose:     {'ON' if verbose else 'OFF'} (toggle: --verbose flag or GROUNDING_VERBOSE=1)")
    print(f"Type a question. The agent decides whether to search.\n")

    # Conversation loop
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break

        result = agent.ask(question)

        # Show what the agent did
        if result["tool_calls"]:
            print(f"\n  [Agent searched {len(result['tool_calls'])} time(s) via {result.get('interface', 'direct_api')}]")
            for tc in result["tool_calls"]:
                mode = "livecrawl" if tc.get("livecrawl") else "snippets"
                print(f"  → \"{tc['query']}\" ({mode})")
                # Verbose mode: full tool call log is already printed inline
                # by base_agent.py during execution. Here we print it again
                # post-hoc only if verbose was toggled mid-session.

        # Verbose: print full structured log after response
        if is_verbose() and result["tool_calls"]:
            print(f"\n  ─── Tool Call Log ───")
            for tc in result["tool_calls"]:
                print(format_tool_log(tc))

        print(f"\nAgent: {result['answer']}")

        if result["sources"]:
            print(f"\nSources:")
            for i, url in enumerate(result["sources"], 1):
                print(f"  [{i}] {url}")

        # Token summary
        breakdown = result.get("token_breakdown", {})
        if breakdown:
            inp = breakdown.get('input', 0)
            out = breakdown.get('output', 0)
            search_ctx = breakdown.get('search_context', 0)
            llm_input = inp - search_ctx
            print(f"\n  Tokens: {result['tokens_used']:,} total = {inp:,} input + {out:,} output")
            print(f"          Input breakdown: ~{search_ctx:,} search context + ~{llm_input:,} prompt/schema")
        else:
            print(f"\n  ({result['tokens_used']:,} tokens)")
        print(f"  Model: {result['model']} | Interface: {result.get('interface', 'direct_api')}\n")


if __name__ == "__main__":
    main()
