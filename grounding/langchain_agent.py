"""
langchain_agent.py — LangChain/LangGraph agent with You.com search tools.

LangChain's agent pattern is already Architecture B: the LLM decides when
to search. YouSearchTool and YouContentsTool become tools the agent can
invoke autonomously.

Three patterns:
  1. ReAct agent   — LLM decides when and what to search (most common)
  2. Multi-tool    — search + content extraction for deep research
  3. RAG retriever — deterministic search-then-generate (Architecture A)

Requirements:
    pip install langchain-youdotcom langchain-openai langgraph
    export YDC_API_KEY="..."
    export OPENAI_API_KEY="..."

Run:
    python langchain_agent.py                 # ReAct agent (default)
    python langchain_agent.py research        # Multi-tool research agent
    python langchain_agent.py rag             # RAG retriever
    python langchain_agent.py react --verbose # With full tool call logging
"""

import os
import sys
import time
from dotenv import load_dotenv
# Supports both env.txt (visible in Finder) and .env (standard convention)
load_dotenv("env.txt") or load_dotenv(".env")

from search_tool import is_verbose, INTEGRATION_INTERFACE, set_interface

# Set interface label so any shared logging knows this is the LangChain path
set_interface("langchain")


# ─── Verbose callback handler ──────────────────────────────────────────────
# LangChain uses callbacks, not our base_agent loop. This handler prints
# tool call details in the same format as the direct API path.

def _make_token_tracker():
    """Create a LangChain callback handler that tracks token usage per LLM call.

    Returns (handler, get_stats_fn). Call get_stats_fn() after the agent run
    to get the accumulated token counts.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class TokenTracker(BaseCallbackHandler):
        def __init__(self):
            self.total_input = 0
            self.total_output = 0
            self.baseline_input = 0
            self.llm_call_count = 0

        def on_llm_end(self, response, **kwargs):
            """Called after each LLM call. Accumulates token usage."""
            usage = {}
            if hasattr(response, "llm_output") and response.llm_output:
                usage = response.llm_output.get("token_usage", {})
            if usage:
                inp = usage.get("prompt_tokens", 0)
                out = usage.get("completion_tokens", 0)
                self.total_input += inp
                self.total_output += out
                if self.llm_call_count == 0:
                    self.baseline_input = inp
                self.llm_call_count += 1

        def get_stats(self) -> dict:
            search_context = max(0, self.total_input - self.baseline_input) if self.llm_call_count > 1 else 0
            return {
                "input": self.total_input,
                "output": self.total_output,
                "search_context": search_context,
            }

    tracker = TokenTracker()
    return tracker, tracker.get_stats


def _make_verbose_handler():
    """Create a LangChain callback handler for verbose tool call logging.

    Only imported/used when verbose mode is on, so langchain isn't a hard
    dependency just for logging.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class VerboseToolHandler(BaseCallbackHandler):
        def __init__(self):
            self._tool_start_time = {}

        def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
            self._tool_start_time[run_id] = time.perf_counter()
            tool_name = serialized.get("name", "unknown")
            print(f"  ┌─ Tool Call: {tool_name}")
            print(f"  │ Interface:   langchain")
            print(f"  │ Input:       {input_str[:200]}")

        def on_tool_end(self, output, *, run_id, **kwargs):
            elapsed = 0.0
            if run_id in self._tool_start_time:
                elapsed = (time.perf_counter() - self._tool_start_time.pop(run_id)) * 1000
            preview = str(output)[:200] if output else ""
            print(f"  │ Latency:     {elapsed:.1f}ms")
            print(f"  │ Preview:     {preview}")
            print(f"  └─")

    return VerboseToolHandler()


# ─── Pattern 1: ReAct Agent (Architecture B) ────────────────────────────────
# The LLM decides when to search. This is the recommended pattern.

def react_agent(question: str) -> dict:
    """ReAct agent that autonomously calls You.com search.

    The LLM sees the user's question, decides whether it needs web data,
    and calls YouSearchTool as needed. It might search multiple times
    for multi-hop questions.

    Returns:
        dict with keys: answer, sources, model, interface, tokens_used, token_breakdown
    """
    from langchain_youdotcom import YouSearchTool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    tools = [YouSearchTool()]
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    tracker, get_stats = _make_token_tracker()
    callbacks = [tracker]
    if is_verbose():
        callbacks.append(_make_verbose_handler())

    agent = create_react_agent(llm, tools)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"callbacks": callbacks},
    )

    answer = result["messages"][-1].content

    # Extract source URLs from intermediate tool call messages
    sources = []
    for msg in result["messages"]:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and "URL:" in content:
            for line in content.split("\n"):
                if line.strip().startswith("URL:"):
                    sources.append(line.strip()[4:].strip())

    breakdown = get_stats()
    return {
        "answer": answer,
        "sources": sources,
        "model": "gpt-4.1",
        "interface": "langchain",
        "tokens_used": breakdown["input"] + breakdown["output"],
        "token_breakdown": breakdown,
    }


# ─── Pattern 2: Multi-Tool Agent ────────────────────────────────────────────
# Search for discovery, then extract full content from promising URLs.

def research_agent(question: str) -> dict:
    """Multi-tool agent: search to find sources, then deep-read them.

    Combines YouSearchTool (find relevant pages) with YouContentsTool
    (extract full content from specific URLs). Good for questions that
    need more than snippet-level context.

    Returns:
        dict with keys: answer, sources, model, interface, tokens_used, token_breakdown
    """
    from langchain_youdotcom import YouSearchTool, YouContentsTool
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent

    tools = [YouSearchTool(), YouContentsTool()]
    llm = ChatOpenAI(model="gpt-4.1", temperature=0)

    tracker, get_stats = _make_token_tracker()
    callbacks = [tracker]
    if is_verbose():
        callbacks.append(_make_verbose_handler())

    agent = create_react_agent(llm, tools)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"callbacks": callbacks},
    )

    answer = result["messages"][-1].content
    sources = []
    for msg in result["messages"]:
        content = getattr(msg, "content", "")
        if isinstance(content, str) and "URL:" in content:
            for line in content.split("\n"):
                if line.strip().startswith("URL:"):
                    sources.append(line.strip()[4:].strip())

    breakdown = get_stats()
    return {
        "answer": answer,
        "sources": sources,
        "model": "gpt-4.1",
        "interface": "langchain",
        "tokens_used": breakdown["input"] + breakdown["output"],
        "token_breakdown": breakdown,
    }


# ─── Pattern 3: RAG Retriever (Architecture A) ──────────────────────────────
# Deterministic: always search, then generate. No agent decision-making.
# Included for completeness; Patterns 1-2 are the recommended approach.

def rag_retriever(question: str, count: int = 5) -> dict:
    """Classic RAG: always retrieve, then generate. No agent autonomy.

    This is Architecture A — the application controls when search happens.
    Use this when you want to force-ground every single response.

    Returns:
        dict with keys: answer, sources, model, interface, tokens_used, token_breakdown
    """
    from langchain_youdotcom import YouRetriever
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    retriever = YouRetriever(count=count, livecrawl="web")

    prompt = ChatPromptTemplate.from_template(
        "Answer using only the provided context. Cite sources.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )

    tracker, get_stats = _make_token_tracker()
    callbacks = [tracker]
    llm = ChatOpenAI(model="gpt-4.1", temperature=0, callbacks=callbacks)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke(question)
    breakdown = get_stats()
    return {
        "answer": answer,
        "sources": [],  # RAG retriever doesn't expose URLs through the chain
        "model": "gpt-4.1",
        "interface": "langchain",
        "tokens_used": breakdown["input"] + breakdown["output"],
        "token_breakdown": breakdown,
    }


# ─── Interactive demo ──────────────────────────────────────────────────────

PATTERNS = {
    "react": ("ReAct Agent (recommended)", react_agent),
    "research": ("Multi-Tool Research Agent", research_agent),
    "rag": ("RAG Retriever", rag_retriever),
}

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    pattern = args[0] if args else "react"

    if "--verbose" in flags:
        os.environ["GROUNDING_VERBOSE"] = "1"

    if pattern not in PATTERNS:
        print(f"Unknown pattern: {pattern}")
        print(f"Available: {', '.join(PATTERNS.keys())}")
        sys.exit(1)

    label, fn = PATTERNS[pattern]
    verbose = is_verbose()
    print(f"LangChain pattern: {label}")
    print(f"Interface:         langchain (langchain-youdotcom → You.com Search API)")
    print(f"Verbose:           {'ON' if verbose else 'OFF'}")
    print(f"Type a question. The agent decides whether to search.\n")

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

        result = fn(question)

        print(f"\nAgent: {result['answer']}")

        if result.get("sources"):
            print(f"\nSources:")
            for i, url in enumerate(result["sources"], 1):
                print(f"  [{i}] {url}")

        breakdown = result.get("token_breakdown", {})
        if breakdown:
            inp = breakdown.get('input', 0)
            out = breakdown.get('output', 0)
            search_ctx = breakdown.get('search_context', 0)
            llm_input = inp - search_ctx
            print(f"\n  Tokens: {result.get('tokens_used', 0):,} total = {inp:,} input + {out:,} output")
            print(f"          Input breakdown: ~{search_ctx:,} search context + ~{llm_input:,} prompt/schema")
        print(f"  Model: {result['model']} | Interface: {result['interface']}\n")


if __name__ == "__main__":
    main()
