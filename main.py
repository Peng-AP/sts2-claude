"""
Entry point: wire up the mod client + Claude agent and start a run.

Usage:
    1. Copy .env.example to .env and set ANTHROPIC_API_KEY.
    2. Launch Slay the Spire 2 with the STS2MCP mod installed.
    3. python main.py
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

from agent import SpireAgent
from sts2mcp_client import STS2MCPClient, STS2MCPError


def main() -> int:
    load_dotenv()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it in.")
        return 1

    base_url = os.environ.get("STS2MCP_BASE_URL", "http://127.0.0.1:15526")
    client = STS2MCPClient(base_url=base_url)

    print(f"Connecting to STS2MCP at {base_url} ...")
    agent = SpireAgent(client)
    try:
        agent.run()
    except STS2MCPError as e:
        print(f"\nMod connection problem: {e}")
        return 2
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
