"""
mcp_demo.py — Runnable demo of the MCP integration path.

Shows how to connect to You.com's MCP server programmatically using the
Model Context Protocol. This is the same server that Claude Desktop,
Cursor, VS Code, and Claude Code connect to — but called from Python
so you can see the request/response flow.

In production, MCP is used via IDE integrations (see mcp_config.json).
This demo exists so developers can understand and test the MCP path
without needing an IDE set up.

Requirements:
    pip install mcp httpx python-dotenv
    export YDC_API_KEY="..."

Run:
    python mcp_demo.py                    # List available tools
    python mcp_demo.py search "query"     # Run a search via MCP
    python mcp_demo.py search "query" --verbose  # With full logging
"""

import asyncio
import json
import os
import sys
import time
from dotenv import load_dotenv
# Supports both env.txt (visible in Finder) and .env (standard convention)
load_dotenv("env.txt") or load_dotenv(".env")

from search_tool import is_verbose, set_interface

set_interface("mcp")

MCP_SERVER_URL = "https://api.you.com/mcp"


async def list_tools(api_key: str) -> list[dict]:
    """Connect to You.com's MCP server and list available tools.

    This is the equivalent of what your IDE does when it first connects
    to the MCP server — it discovers which tools are available.
    """
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        print("MCP client not installed. Run: pip install mcp httpx")
        sys.exit(1)

    headers = {"X-API-Key": api_key}
    async with streamablehttp_client(MCP_SERVER_URL, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.list_tools()
            tools = []
            for tool in result.tools:
                tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema if hasattr(tool, "inputSchema") else {},
                })
            return tools


async def call_tool(api_key: str, tool_name: str, args: dict) -> str:
    """Call a specific tool on You.com's MCP server.

    This is the equivalent of what your IDE does when the LLM decides
    to invoke a tool — it sends the tool call to the MCP server and
    gets results back.
    """
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:
        print("MCP client not installed. Run: pip install mcp httpx")
        sys.exit(1)

    headers = {"X-API-Key": api_key}

    t0 = time.perf_counter()
    async with streamablehttp_client(MCP_SERVER_URL, headers=headers) as (r, w, _):
        async with ClientSession(r, w) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            elapsed_ms = (time.perf_counter() - t0) * 1000

    # MCP returns content as a list of content blocks
    output_parts = []
    for block in result.content:
        if hasattr(block, "text"):
            output_parts.append(block.text)

    output = "\n".join(output_parts)

    if is_verbose():
        print(f"  ┌─ MCP Tool Call: {tool_name}")
        print(f"  │ Interface:   mcp")
        print(f"  │ Server:      {MCP_SERVER_URL}")
        print(f"  │ Args:        {json.dumps(args)}")
        print(f"  │ Latency:     {elapsed_ms:.1f}ms")
        print(f"  │ Response:    {len(output)} chars")
        print(f"  └─")

    return output


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = [a for a in sys.argv[1:] if a.startswith("--")]

    if "--verbose" in flags:
        os.environ["GROUNDING_VERBOSE"] = "1"

    api_key = os.environ.get("YDC_API_KEY", "")
    if not api_key:
        print("Error: YDC_API_KEY not set. Get one at https://you.com/platform")
        sys.exit(1)

    command = args[0] if args else "list"

    if command == "list":
        # List available MCP tools
        print(f"Connecting to You.com MCP server: {MCP_SERVER_URL}")
        print(f"Interface: mcp\n")
        tools = asyncio.run(list_tools(api_key))
        print(f"Available tools ({len(tools)}):\n")
        for tool in tools:
            print(f"  {tool['name']}")
            print(f"    {tool['description'][:120]}")
            print()
        print("These are the tools your IDE sees when it connects to the MCP server.")
        print("The IDE's built-in agent decides when to call them, just like")
        print("the direct API agents decide when to call web_search.\n")
        print(f"To test a search: python mcp_demo.py search \"your query here\"")

    elif command == "search":
        query = " ".join(args[1:]) if len(args) > 1 else input("Search query: ")
        if not query:
            print("Error: query is required")
            sys.exit(1)

        print(f"MCP search: \"{query}\"")
        print(f"Server:     {MCP_SERVER_URL}")
        print(f"Interface:  mcp")
        print(f"Verbose:    {'ON' if is_verbose() else 'OFF'}\n")

        # Call the web_search tool via MCP
        result = asyncio.run(call_tool(api_key, "web_search", {"query": query}))
        print(result[:1500])

        print(f"\n  (interface: mcp | server: {MCP_SERVER_URL})")

    else:
        print(f"Unknown command: {command}")
        print(f"Usage:")
        print(f"  python mcp_demo.py              # List MCP tools")
        print(f"  python mcp_demo.py search \"...\" # Search via MCP")
        sys.exit(1)


if __name__ == "__main__":
    main()
