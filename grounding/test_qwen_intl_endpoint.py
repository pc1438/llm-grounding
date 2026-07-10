"""
test_qwen_intl_endpoint.py — Throwaway test script.

Tests whether the international DashScope endpoint (dashscope-intl.aliyuncs.com)
supports the YDC path (chat.completions tool-use loop with role="tool") past round 1.

Goal: if this passes, both YDC and native search can consolidate to the
international endpoint. See the TODO in qwen_agent.py docstring.

Usage:
    cd grounding
    python3 test_qwen_intl_endpoint.py

Does NOT modify qwen_agent.py — swaps the endpoint on the live instance only.
"""

from dotenv import load_dotenv
load_dotenv("env.txt")

from openai import OpenAI
from agents.qwen_agent import QwenAgent, _DASHSCOPE_INTL_BASE_URL
from base_agent import OpenAICompatibleAgent, OpenAIResponsesAgent

QUESTION = "What was the last soccer game the United States played during qualifying?"

print("=" * 60)
print("Qwen international endpoint — YDC path test")
print(f"Endpoint: {_DASHSCOPE_INTL_BASE_URL}")
print(f"Question: {QUESTION}")
print("=" * 60)

# Instantiate normally — this sets up the domestic endpoint client internally.
agent = QwenAgent(model="qwen3.7-max")

# Swap the underlying client to point to the international endpoint.
# No changes to qwen_agent.py — we're patching the live instance only.
intl_client = OpenAI(api_key=agent._api_key, base_url=_DASHSCOPE_INTL_BASE_URL)
cls = type(agent._impl)
agent._impl = cls(client=intl_client, model=agent.model)

print(f"Client swapped to international endpoint (was domestic).\n")

def on_progress(msg):
    print(f"  > {msg}")

result = agent.ask(QUESTION, on_progress=on_progress)

print()
print("-" * 60)
print(f"search_calls : {result['search_calls']}")
print(f"api_calls    : {result['api_calls']}")
print(f"sources      : {result['sources'][:3]}")
print(f"answer       : {result['answer'][:400]}")
print("-" * 60)

if result['search_calls'] > 0 and result['answer']:
    print("\nPASS — international endpoint supports YDC path past round 1.")
    print("TODO: consolidate both paths to _DASHSCOPE_INTL_BASE_URL in qwen_agent.py.")
else:
    print("\nFAIL — search_calls=0 or empty answer.")
    print("International endpoint does not support YDC tool-use loop.")
    print("Keep the domestic/international split as-is.")
