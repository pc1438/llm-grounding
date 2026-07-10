"""
qwen_agent.py — Qwen agent with You.com web search tool.

Runs against DashScope (Alibaba's first-party API). Supports both the shared
DashScope endpoint and MaaS workspace deployments (set DASHSCOPE_BASE_URL).

Default path: OpenAI Responses API (same as GPT). DashScope confirmed to
support /v1/responses as of 2026-06-30. See decisions/006 for test results.

    from agents.qwen_agent import QwenAgent

    agent = QwenAgent()                        # DashScope, Responses API (default)
    agent = QwenAgent(force_chat_completions=True)  # DashScope, chat.completions

    # Native search (DashScope only):
    result = agent.run_native_search("What happened at the 2026 AI Summit?")

Requirements:
    pip install openai
    export YDC_API_KEY="..."
    export DASHSCOPE_API_KEY="sk-..."    # DashScope

Tool naming (DashScope-specific quirk):
    DashScope intercepts any function named 'web_search' and routes it to
    Qwen's native search instead of calling our custom function. We use
    'you_search' via TOOL_SCHEMA_RESPONSES to avoid this. This only applies
    to the Responses API path; chat.completions uses TOOL_SCHEMA ('web_search')
    which is fine because DashScope's chat.completions function calling does
    not have this interception behavior.
    Confirmed empirically 2026-06-30. See decisions/006.

Endpoint configuration (verified 2026-07-06):
    Both YDC and native search paths use the international endpoint by default.
      - International (dashscope-intl.aliyuncs.com): supports both YDC
        (chat.completions tool-use loop, confirmed 5 rounds) and native search
        (Responses API + web_search). Default for all paths.
      - Domestic (dashscope.aliyuncs.com): fallback for YDC chat.completions only.
        Does NOT support /responses — returns hard 401 (endpoint-level block, not
        a key issue; same key works fine on domestic for chat.completions).
      - Both endpoints share the same DASHSCOPE_API_KEY.

Native search reliability:
    run_native_search() uses Qwen's built-in web_search tool, which the model calls
    at its own discretion. This works reliably for questions the model knows it
    cannot answer (prices, weather, ephemeral data). It is unreliable when the
    model has strong training-data beliefs about an event's timing — for example,
    if training data says "the 2026 World Cup runs June–July 2026" the model may
    answer "the tournament hasn't started yet" without searching, even after the
    event has concluded. Phrasing with "most recent / latest / recently" does not
    help and can reinforce the stale reasoning. Phrasing with "as of today" or
    "current result" reliably triggers search for these edge cases.
    For sports results and event outcomes, the YDC path (ask()) is more reliable
    because it always searches regardless of model confidence.
    Verified 2026-07-06. See test_qwen_intl_endpoint.py for the test harness.

MaaS workspace note:
    When DASHSCOPE_BASE_URL is set (MaaS workspace deployment), force_chat_completions
    is automatically set to True. MaaS workspace endpoints do not support stateful
    Responses API chaining (previous_response_id fails on round 2 with 400). Chat.completions
    tool-use loop with role="tool" messages works correctly on MaaS workspace.

Subclassing:
    QwenAgent is a proper class — subclass it to build your own Qwen-based agent:

        class MyQwenAgent(QwenAgent):
            def __init__(self):
                super().__init__(model="qwen3-max")

            def run_native_search(self, question, **kwargs):
                return super().run_native_search(question, **kwargs)
"""

import os
import re
import time

from openai import OpenAI

from base_agent import BaseAgent, OpenAICompatibleAgent, OpenAIResponsesAgent, _empty_stats, _get_agent_default_model
from search_tool import get_system_prompt, MAX_TOKENS, NATIVE_SEARCH_BASELINE_TOKENS, make_progress_reporter, make_elapsed_timer

# DashScope endpoints — verified 2026-07-06:
#
#   International: dashscope-intl.aliyuncs.com    — DEFAULT for both paths.
#                  Supports YDC (chat.completions tool-use loop, confirmed 5 rounds)
#                  and native search (Responses API + web_search). Use this endpoint
#                  for all new code.
#
#   Domestic:      dashscope.aliyuncs.com         — FALLBACK for YDC chat.completions
#                  only. Does NOT support the Responses API (/responses returns hard
#                  401 — endpoint-level block, not a key issue). Retained here in case
#                  the international endpoint is unavailable for the YDC path.
#
#   Both endpoints use the same DASHSCOPE_API_KEY.
#
#   MaaS workspace: *.maas.aliyuncs.com           — Regional MaaS workspace endpoint.
#                  Set via DASHSCOPE_BASE_URL env var. Supports chat.completions tool-use
#                  loop (role="tool"). Does NOT support stateful Responses API chaining
#                  (previous_response_id fails on round 2 with 400). force_chat_completions
#                  is auto-set to True when DASHSCOPE_BASE_URL is configured.
#
# BACKENDS["dashscope"]["base_url"] defaults to international. Overridden by DASHSCOPE_BASE_URL.
# run_native_search() also uses _DASHSCOPE_INTL_BASE_URL.
_DASHSCOPE_DOMESTIC_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DASHSCOPE_INTL_BASE_URL     = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

_DASHSCOPE_CFG = {
    "api_key_env": "DASHSCOPE_API_KEY",
    "base_url": _DASHSCOPE_INTL_BASE_URL,  # overridden by DASHSCOPE_BASE_URL env var for MaaS workspace
    # Default model comes from pricing.json (agent_default: true for provider "qwen").
    # To change the default, update pricing.json — do not edit this line directly.
    "default_model": _get_agent_default_model("qwen", fallback="qwen3.7-max"),
    # Also: "qwen3-max", "qwen3.5-plus", "qwen3.5-flash"
    "supports_responses_api": True,
    "supports_native_search": True,
}


class QwenAgent(BaseAgent):
    """Qwen-powered agent with You.com web search.

    Selects Responses API or chat.completions at construction time based on
    backend and force_chat_completions. Both YDC and native search paths are
    available as methods on this single class.

    Return value of ask() and run_native_search(): see base_agent._empty_stats()
    for the full field-by-field reference.

    Subclass this to build your own Qwen-based grounded agent:

        class MyQwenAgent(QwenAgent):
            def __init__(self):
                super().__init__(model="qwen3-max")
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        force_chat_completions: bool = False,
    ):
        cfg = _DASHSCOPE_CFG
        resolved_key = api_key or os.environ.get(cfg["api_key_env"], "")
        if not resolved_key:
            raise ValueError(
                f"{cfg['api_key_env']} not set. "
                f"Pass api_key= or export {cfg['api_key_env']}."
            )

        self._api_key = resolved_key
        self._backend_cfg = cfg
        # DASHSCOPE_BASE_URL env var overrides the default base_url — used for MaaS
        # workspace deployments (e.g. ap-southeast-1) which have their own endpoint.
        maas_url = os.environ.get("DASHSCOPE_BASE_URL")
        base_url = maas_url or cfg["base_url"]
        client = OpenAI(api_key=resolved_key, base_url=base_url, timeout=120.0)
        resolved_model = model or cfg["default_model"]

        # MaaS workspace endpoints don't support stateful Responses API
        # (previous_response_id chaining fails with 400 on round 2). Fall back to
        # chat.completions, which the MaaS endpoint does support for tool-use loops.
        if maas_url:
            force_chat_completions = True

        use_responses = cfg["supports_responses_api"] and not force_chat_completions
        cls = OpenAIResponsesAgent if use_responses else OpenAICompatibleAgent
        self._impl = cls(client=client, model=resolved_model)
        super().__init__(model=resolved_model)

    def stream(self, question: str, max_rounds: int = None):
        yield from self._impl.stream(question, max_rounds=max_rounds)

    def run_native_search(
        self,
        question: str,
        system_prompt: str = None,
        on_progress=None,
    ) -> dict:
        """Run a query using Qwen's built-in web_search tool (no You.com).

        Always uses the international endpoint (dashscope-intl.aliyuncs.com) —
        the only DashScope endpoint that supports the Responses API with web_search.
        Not available on MaaS workspace endpoints (supports_native_search=False).

        Source URLs are returned in item.action.sources on the web_search_call
        output item (not as url_citation annotations like OpenAI).
        """
        if not self._backend_cfg.get("supports_native_search"):
            return {
                **_empty_stats(self.model),
                "answer": "Native search not supported for this backend.",
                "interface": "not_supported",
            }

        prompt = system_prompt or get_system_prompt()
        stats = _empty_stats(self.model)
        stats["interface"] = "native_responses"
        _notify = make_progress_reporter("Native", on_progress)
        # Native search always uses the international endpoint — it's the only one that
        # supports the Responses API with web_search. See _DASHSCOPE_INTL_BASE_URL above.
        native_client = OpenAI(api_key=self._api_key, base_url=_DASHSCOPE_INTL_BASE_URL, timeout=120.0)
        t0, _elapsed = make_elapsed_timer()

        _notify(f"POST dashscope-intl · model={self.model} · tools=[web_search] ({_elapsed()})")

        try:
            t_connect = time.perf_counter()
            response = native_client.responses.create(
                model=self.model,
                instructions=prompt,
                input=question,
                tools=[{"type": "web_search"}],
                extra_body={"enable_thinking": False},
            )
            stats["connect_ms"] = round((time.perf_counter() - t_connect) * 1000)

            stats["model_confirmed"] = getattr(response, "model", None)
            stats["latency_ms"] = (time.perf_counter() - t0) * 1000
            stats["api_calls"] = 1

            if hasattr(response, "usage") and response.usage:
                stats["token_breakdown"]["input"] = getattr(response.usage, "input_tokens", 0)
                stats["token_breakdown"]["output"] = getattr(response.usage, "output_tokens", 0)
            stats["tokens_used"] = stats["token_breakdown"]["input"] + stats["token_breakdown"]["output"]

            for item in response.output:
                item_type = getattr(item, "type", "")

                if item_type == "web_search_call":
                    stats["search_calls"] += 1
                    action = getattr(item, "action", None)
                    display_query = "(internal)"
                    if action:
                        raw_query = getattr(action, "query", "") or ""
                        display_query = raw_query if raw_query and raw_query.lower() != "web search" else "(internal)"
                        for src in getattr(action, "sources", []):
                            url = getattr(src, "url", "")
                            if url and url not in stats["sources"]:
                                stats["sources"].append(url)
                    _notify(f"Search {stats['search_calls']}: query={display_query} · {len(stats['sources'])} sources ({_elapsed()})")

                elif item_type == "message":
                    _notify(f"Generating answer... ({_elapsed()})")
                    for part in getattr(item, "content", []):
                        if getattr(part, "type", "") == "output_text":
                            stats["answer"] += getattr(part, "text", "")

            if stats["search_calls"] > 0 and stats["token_breakdown"]["input"] > NATIVE_SEARCH_BASELINE_TOKENS:
                stats["token_breakdown"]["search_context"] = stats["token_breakdown"]["input"] - NATIVE_SEARCH_BASELINE_TOKENS

            # DashScope international endpoint sometimes returns empty action.sources.
            # Fall back to extracting [N] url patterns the model embedded in the answer text.
            if not stats["sources"] and stats["answer"]:
                for url in re.findall(r'\[[\d,\s]+\]\s*(https?://\S+)', stats["answer"]):
                    url = url.rstrip(".,)")
                    if url not in stats["sources"]:
                        stats["sources"].append(url)

            _notify(f"Complete — {stats['latency_ms']:.0f}ms total ({_elapsed()})")

        except Exception as e:
            stats["answer"] = f"Error: {e}"
            stats["latency_ms"] = (time.perf_counter() - t0) * 1000

        return stats


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    force_cc = "--chat-completions" in args
    args = [a for a in args if a != "--chat-completions"]
    question = " ".join(args) or input("Ask something: ")
    agent = QwenAgent(force_chat_completions=force_cc)
    result = agent.ask(question)
    if result is None:
        print("Error: no result returned")
        sys.exit(1)
    answer = result.get("answer") or "(no answer)"
    print(answer)
    if result.get("sources"):
        print(f"\nSources: {result['sources']}")
