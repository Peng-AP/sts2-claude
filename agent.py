"""
The agent loop: read state -> ask Claude -> apply action -> repeat.

Uses the Anthropic SDK with claude-opus-4-8 and adaptive thinking. Claude is
given the current game state plus the tool set from tools.py, and may only act
through those tools.
"""

from __future__ import annotations

import json
from typing import Any

from anthropic import Anthropic

from state_compaction import compact_state, estimate_tokens
from sts2mcp_client import STS2MCPClient
from tools import TOOLS, action_from_tool_call

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """\
You are an expert Slay the Spire 2 player controlling a full run. You will be \
given the current game state as JSON on each step. Decide the single best next \
action and take it by calling exactly one tool.

Principles:
- Win the run: survive, build a coherent deck, and beat each act's boss.
- Think about the whole run, not just the current turn. Deck quality and HP \
  preservation usually matter more than greedy short-term value.
- In combat: account for enemy intents, your block, energy, and incoming \
  damage. Don't take avoidable damage; set up for upcoming tougher fights.
- Only choose from options the game is actually offering right now. The state \
  lists what's currently legal — never invent an option that isn't there.
- When a screen needs no real decision, use `proceed`. When an optional reward \
  isn't worth it, use `skip`.

Call exactly one tool per step. Keep reasoning focused and decisive."""


class SpireAgent:
    def __init__(self, client: STS2MCPClient, anthropic: Anthropic | None = None,
                 max_steps: int = 2000, verbose: bool = True):
        self.client = client
        self.anthropic = anthropic or Anthropic()
        self.max_steps = max_steps
        self.verbose = verbose

    def _log(self, *a: Any) -> None:
        if self.verbose:
            print(*a, flush=True)

    def _decide(self, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Ask Claude for one action given the current state."""
        compact = compact_state(state)
        if self.verbose:
            self._log(
                f"    state tokens ~{estimate_tokens(state)} -> ~{estimate_tokens(compact)} "
                "(compacted)"
            )
        user_content = (
            "Current Slay the Spire 2 game state:\n\n"
            f"```json\n{json.dumps(compact, indent=2)}\n```\n\n"
            "Take the single best next action by calling one tool."
        )
        resp = self.anthropic.messages.create(
            model=MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            # Cache the static prefix (system prompt + tools) so it isn't
            # re-billed at full price every step. Marking the last item of each
            # caches that whole block. The per-turn state stays uncached.
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            tools=_tools_with_cache(),
            tool_choice={"type": "any"},  # force a tool call every step
            messages=[{"role": "user", "content": user_content}],
        )
        for block in resp.content:
            if block.type == "tool_use":
                return block.name, dict(block.input)
        raise RuntimeError("Claude returned no tool_use block")

    def run(self) -> None:
        """Drive the run from current state until the game ends or we hit max_steps."""
        state = self.client.wait_until_ready()
        self._log("Mod connected. Starting run.")

        for step in range(self.max_steps):
            if _run_is_over(state):
                self._log(f"Run ended after {step} steps.")
                return

            name, args = self._decide(state)
            self._log(f"[{step}] -> {name}({args})")

            action = action_from_tool_call(name, args)
            result = self.client.send_action(action)

            # Some mod versions return the new state from the action POST; if not,
            # re-read it. Either way we end the loop body holding fresh state.
            state = result if _looks_like_state(result) else self.client.get_state()

        self._log(f"Hit max_steps ({self.max_steps}); stopping.")


def _tools_with_cache() -> list[dict[str, Any]]:
    """Tools list with a cache breakpoint on the last entry, so the whole tool
    block is cached and reused across steps instead of re-billed each turn."""
    tools = [dict(t) for t in TOOLS]
    tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    return tools


def _looks_like_state(obj: Any) -> bool:
    return isinstance(obj, dict) and len(obj) > 0 and "action" not in obj


def _run_is_over(state: dict[str, Any]) -> bool:
    """Heuristic end-of-run detection. ⚠️ Confirm the real field once the mod
    is running — adjust the keys/values to match STS2MCP's state schema."""
    screen = str(state.get("screen") or state.get("screen_type") or "").lower()
    return screen in {"game_over", "victory", "death", "main_menu"}
