"""
The agent loop: read state -> ask Claude -> apply action -> repeat.

Uses the Anthropic SDK with claude-opus-4-8 and adaptive thinking. Claude is
given the current game state plus the tool set from tools.py, and may only act
through those tools.
"""

from __future__ import annotations

import json
import time
from typing import Any

from anthropic import Anthropic

from state_compaction import compact_state, estimate_tokens
from sts2mcp_client import STS2MCPClient
from tools import TOOLS, IllegalActionError, action_from_tool_call

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """\
You are an expert Slay the Spire 2 player controlling a full run. You will be \
given the current game state as JSON on each step. Decide the single best next \
action and take it by calling exactly one tool.

The state's `state_type` tells you which screen you're on (monster/elite/boss \
combat, rewards, card_reward, map, event, rest_site, shop, treasure, various \
selection overlays, or menu). Act accordingly:
- In combat, play cards with `play_card` using a card's `index`; if it targets \
  an enemy, pass `target` as that enemy's `entity_id` (from battle.enemies). \
  Use `end_turn` when done.
- For any "pick option N" screen (rewards, card_reward, map, event, rest_site, \
  shop, treasure, selection overlays) use `choose` with the option's `index`.
- `confirm`/`cancel` resolve selection overlays; `skip` declines an optional \
  reward; `proceed` advances a screen needing no decision; `advance_dialogue` \
  steps through event text; `menu_select` picks a menu option string. To start
  a run from menus: singleplayer -> standard, then on character_select pick a
  character and choose `embark` to begin. Don't re-pick an option that's already
  selected — if a screen didn't change, the action had no effect.

Principles:
- Win the run: survive, build a coherent deck, and beat each act's boss.
- Think about the whole run, not just the current turn. Deck quality and HP \
  preservation usually matter more than greedy short-term value.
- In combat: account for enemy intents, your block, energy, and incoming \
  damage. Don't take avoidable damage; set up for upcoming tougher fights.
- Only choose from options the game is actually offering right now (respect \
  `can_play`, `can_skip`, `can_confirm`, available indices). Never invent one.

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

    def _decide(self, state: dict[str, Any], hint: str | None = None) -> tuple[str, dict[str, Any]]:
        """Ask Claude for one action given the current state. `hint` carries a
        note about the previous step (e.g. it had no effect) to break loops."""
        compact = compact_state(state)
        if self.verbose:
            self._log(
                f"    state tokens ~{estimate_tokens(state)} -> ~{estimate_tokens(compact)} "
                "(compacted)"
            )
        nudge = f"\n\nNOTE: {hint}\n" if hint else ""
        user_content = (
            "Current Slay the Spire 2 game state:\n\n"
            f"```json\n{json.dumps(compact, indent=2)}\n```\n"
            f"{nudge}\n"
            "Take the single best next action by calling one tool."
        )
        messages = [{"role": "user", "content": user_content}]

        # Primary call: extended thinking ON for strategic reasoning. The API
        # forbids forcing tool use while thinking is enabled, so tool_choice is
        # "auto" — with the system prompt's "call exactly one tool" instruction,
        # Opus calls a tool essentially every time.
        resp = self._create(messages, thinking=True, force_tool=False)
        tool = _first_tool_use(resp)
        if tool is not None:
            return tool

        # Fallback for the rare turn where it replied without a tool: force a
        # tool call (which requires thinking OFF) so the loop never stalls.
        self._log("    no tool in thinking response; forcing a tool call")
        resp = self._create(messages, thinking=False, force_tool=True)
        tool = _first_tool_use(resp)
        if tool is not None:
            return tool
        raise RuntimeError("Claude returned no tool_use block")

    def _create(self, messages: list[dict[str, Any]], *, thinking: bool, force_tool: bool):
        kwargs: dict[str, Any] = dict(
            model=MODEL,
            max_tokens=4096,
            # Cache the static prefix (system prompt + tools) so it isn't
            # re-billed at full price every step.
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            tools=_tools_with_cache(),
            messages=messages,
        )
        if thinking:
            kwargs["thinking"] = {"type": "adaptive"}
        if force_tool:
            kwargs["tool_choice"] = {"type": "any"}
        return self.anthropic.messages.create(**kwargs)

    def run(self) -> None:
        """Drive the run from current state until the game ends or we hit max_steps."""
        state = self.client.wait_until_ready()
        self._log("Mod connected. Starting run.")

        hint: str | None = None
        for step in range(self.max_steps):
            if _run_is_over(state):
                self._log(f"Run ended after {step} steps.")
                return

            # Enemy turn: the player can't act, so don't spend a model call —
            # just poll until control returns (or the screen changes, e.g. the
            # fight ends). Saves an API call on every enemy turn.
            if _is_enemy_turn(state):
                self._log(f"[{step}] enemy turn — waiting for player control")
                state = self._await_player_turn(state)
                hint = None
                continue

            name, args = self._decide(state, hint)
            self._log(f"[{step}] -> {name}({args})")

            try:
                action = action_from_tool_call(name, args, state)
            except IllegalActionError as e:
                # Claude picked a tool that doesn't apply to this screen.
                self._log(f"    illegal action skipped: {e}")
                hint = f"`{name}` is not valid on the '{state.get('state_type')}' screen. Choose a different action."
                state = self.client.get_state()
                continue

            before = _signature(state)
            self.client.send_action(action)
            # POST may or may not echo state; always re-read so we hold the
            # authoritative current state for the next decision.
            state = self.client.get_state()

            # No-progress guard: if the action left the state byte-for-byte
            # identical, it was a no-op (e.g. re-selecting an already-selected
            # menu item). Tell the model so it tries something else instead of
            # looping forever.
            if _signature(state) == before:
                hint = (f"Your previous action {name}({args}) did NOT change the game "
                        "state. It had no effect — choose a DIFFERENT option this time.")
                self._log("    (no state change — nudging model)")
            else:
                hint = None

        self._log(f"Hit max_steps ({self.max_steps}); stopping.")

    def _await_player_turn(self, state: dict[str, Any], max_polls: int = 90,
                           delay: float = 0.4) -> dict[str, Any]:
        """Poll state (no model calls) until it's the player's turn again, the
        combat ends, or we give up. Returns the latest state."""
        for _ in range(max_polls):
            if not _is_enemy_turn(state):
                return state
            time.sleep(delay)
            state = self.client.get_state()
        return state


def _is_enemy_turn(state: dict[str, Any]) -> bool:
    """True when we're in combat but it's not the player's actionable window."""
    if str(state.get("state_type", "")) not in ("monster", "elite", "boss"):
        return False
    battle = state.get("battle") or {}
    if battle.get("is_play_phase") is False:
        return True
    return str(battle.get("turn", "")).lower() == "enemy"


def _signature(state: dict[str, Any]) -> str:
    """Stable string fingerprint of a state, for detecting no-op actions."""
    return json.dumps(state, sort_keys=True, default=str)


def _first_tool_use(resp: Any) -> tuple[str, dict[str, Any]] | None:
    """Return (tool_name, input) for the first tool_use block, or None."""
    for block in resp.content:
        if block.type == "tool_use":
            return block.name, dict(block.input)
    return None


def _tools_with_cache() -> list[dict[str, Any]]:
    """Tools list with a cache breakpoint on the last entry, so the whole tool
    block is cached and reused across steps instead of re-billed each turn."""
    tools = [dict(t) for t in TOOLS]
    tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
    return tools


def _run_is_over(state: dict[str, Any]) -> bool:
    """End-of-run detection against STS2MCP's schema: `game_over` is the
    terminal run screen. We stop there rather than at `menu` so the agent can
    still be pointed at the menu to start a run (see future menu-driving work)."""
    return str(state.get("state_type", "")) == "game_over"
