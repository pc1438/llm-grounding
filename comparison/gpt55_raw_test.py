#!/usr/bin/env python3
"""
Raw GPT-5.5 API test — no benchmark machinery.
Tests a single chat.completions call with a 30s timeout to see if the model
responds at all, and how long it takes.
"""
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent.parent / "grounding/env.txt")
load_dotenv(Path(__file__).parent.parent / "grounding/.env")

api_key = os.environ.get("OPENAI_API_KEY", "")
if not api_key:
    raise SystemExit("OPENAI_API_KEY not set")

client = OpenAI(api_key=api_key, timeout=30)
model = "gpt-5.5"

question = "What is the current price of NVIDIA stock?"

print(f"Model:    {model}")
print(f"Query:    {question}")
print(f"Timeout:  30s")
print(f"Started:  {datetime.now(timezone.utc).isoformat()}")
print()

t0 = time.perf_counter()
try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        max_completion_tokens=512,
    )
    elapsed = time.perf_counter() - t0
    print(f"✓ Response in {elapsed:.2f}s")
    print(f"  finish_reason: {response.choices[0].finish_reason}")
    print(f"  model:         {response.model}")
    usage = response.usage
    if usage:
        print(f"  tokens:        {getattr(usage, 'prompt_tokens', 0)} in / {getattr(usage, 'completion_tokens', 0)} out")
    print(f"  answer:        {response.choices[0].message.content[:200]}")
except Exception as e:
    elapsed = time.perf_counter() - t0
    print(f"✗ Failed after {elapsed:.2f}s: {type(e).__name__}: {e}")
