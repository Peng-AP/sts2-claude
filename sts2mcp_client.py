"""
Thin HTTP client for the STS2MCP mod's localhost REST API.

STS2MCP (Gennadiyev/STS2MCP) starts a server on http://127.0.0.1:15526 when
Slay the Spire 2 launches. It exposes the current game state and accepts
actions.

⚠️ VERSION-DEPENDENT: The two endpoint paths below (STATE_PATH, ACTION_PATH)
and the exact action payload shape depend on the STS2MCP build you have
installed. They're isolated here so we can confirm and adjust them in one
place once the mod is actually running. Everything else in the project is
written against this client, not against raw HTTP.
"""

from __future__ import annotations

import time
from typing import Any

import requests


# --- Confirmed against STS2MCP v0.4.0 (raw-full.md) -----------------------
# State and actions share ONE path; GET reads state, POST performs an action.
# (Multiplayer uses /api/v1/multiplayer; mixing the two returns HTTP 409.)
STATE_PATH = "/api/v1/singleplayer"   # GET  -> current game state as JSON
ACTION_PATH = "/api/v1/singleplayer"  # POST -> {"action": <verb>, ...params}
WIKI_PATH = "/api/v1/wiki"            # GET  -> fuzzy card/relic lookup (read-only)
HEALTH_PATH = "/"                     # GET  -> {"message": "...", "status": "ok"}
# --------------------------------------------------------------------------


class STS2MCPError(RuntimeError):
    """Raised when the mod server is unreachable or returns an error."""


class STS2MCPClient:
    def __init__(self, base_url: str = "http://127.0.0.1:15526", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_state(self) -> dict[str, Any]:
        """Fetch the current game state. Returns parsed JSON."""
        try:
            resp = self._session.get(self._url(STATE_PATH), timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            raise STS2MCPError(f"Failed to read state from {STATE_PATH}: {e}") from e

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """POST an action to the game. Returns the server response (often the
        resulting state, depending on mod version)."""
        try:
            resp = self._session.post(
                self._url(ACTION_PATH), json=action, timeout=self.timeout
            )
            resp.raise_for_status()
            # Some mod versions return empty body on success.
            return resp.json() if resp.content else {}
        except requests.RequestException as e:
            raise STS2MCPError(f"Failed to send action to {ACTION_PATH}: {e}") from e

    def wiki(self, query: str, item_type: str = "all", limit: int = 5) -> dict[str, Any]:
        """Fuzzy-search the mod's card/relic wiki. Read-only — does NOT change
        game state. For cards it returns both `base` and `upgraded` variants.
        Scope is limited to items the active profile has discovered."""
        try:
            resp = self._session.get(
                self._url(WIKI_PATH),
                params={"query": query, "item_type": item_type, "limit": limit},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        except requests.RequestException as e:
            raise STS2MCPError(f"Wiki lookup failed for {query!r}: {e}") from e

    def wait_until_ready(self, attempts: int = 30, delay: float = 1.0) -> dict[str, Any]:
        """Poll get_state() until the mod server responds. Call this at startup
        so the agent waits for the game/mod to come up instead of crashing."""
        last_err: Exception | None = None
        for _ in range(attempts):
            try:
                return self.get_state()
            except STS2MCPError as e:
                last_err = e
                time.sleep(delay)
        raise STS2MCPError(
            f"STS2MCP server at {self.base_url} never responded. "
            f"Is StS2 running with the mod loaded? Last error: {last_err}"
        )
