"""Standalone MCP server entry point for Web Eyes.

Usage:
    python run_mcp.py              # stdio (default, for Claude Desktop/Code)
    python run_mcp.py stdio        # same as above
    python run_mcp.py http         # streamable-http on port 3001
    python run_mcp.py sse          # SSE on port 3001
"""
from __future__ import annotations

import sys

import config
from mcp_server import server


def main():
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"

    if transport == "stdio":
        server.run(transport="stdio")
    elif transport == "http":
        server.run(transport="http", host=config.MCP_HOST, port=config.MCP_PORT)
    elif transport == "sse":
        server.run(transport="sse", host=config.MCP_HOST, port=config.MCP_PORT)
    else:
        print(f"Unknown transport: {transport}. Use: stdio, http, sse")
        sys.exit(1)


if __name__ == "__main__":
    main()
