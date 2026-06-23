"""
The agent loop: read state -> ask Claude -> apply action -> repeat.

Uses the Anthropic SDK with claude-opus-4-8 and adaptive thinking. Claude is
given the current game state plus the tool set from tools.py, and may only act
through those tools.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from anthropic import Anthropic

from state_compaction import compact_state, estimate_tokens
from sts2mcp_client import STS2MCPClient
from tools import TOOLS, IllegalActionError, action_from_tool_call

MODEL = "claude-sonnet-4-6"

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

Combat strategy (this is where most runs are won or lost):
- SPEND YOUR ENERGY. Each turn, keep playing affordable cards (cost <= your \
  current `energy`, `can_play` true) until you have no useful play left. Ending \
  a turn with unspent energy and playable cards in hand is almost always a \
  mistake — only do it if you are deliberately holding a card for next turn.
- DEFEND, don't just attack. Read each enemy's `intents`: an Attack intent's \
  `label` is the damage incoming this turn. Sum the incoming damage and compare \
  it to your `block`. If enemies will hit you for meaningful damage, play your \
  block/Defend skills (cards whose description says "Gain N Block") to absorb \
  it. Trading a chunk of HP for a little extra damage is usually a bad trade — \
  blunt big hits, especially from elites and bosses.
- Sequence within a turn: apply debuffs (Weak/Vulnerable) before big attacks, \
  focus one enemy down to reduce incoming damage, then block the rest.
- Think about the whole run: HP and a coherent deck matter more than greedy \
  short-term value. Skip cards that dilute your deck.

General:
- Win the run: survive, build a focused deck, beat each act's boss.
- Only choose from options actually offered right now (respect `can_play`, \
  `can_skip`, `can_confirm`, available indices). Never invent one.

Call exactly one tool per step. Keep reasoning focused and decisive."""


class SpireAgent:
    def __init__(self, client: STS2MCPClient, anthropic: Anthropic | None = None,
                 max_steps: int = 2000, verbose: bool = True, log_dir: str | None = "runs",
                 show_thoughts: bool = True, action_delay: float = 0.4):
        self.client = client
        self.anthropic = anthropic or Anthropic()
        self.max_steps = max_steps
        self.verbose = verbose
        self.show_thoughts = show_thoughts
        self.log_dir = log_dir
        self.action_delay = action_delay   # buffer for the mod to apply an action
        self._log_file = None              # opened lazily in run()
        self._last_nudge_sig: str | None = None  # end-turn nudge de-dupe

    def _log(self, *a: Any) -> None:
        if not self.verbose:
            return
        msg = " ".join(str(x) for x in a)
        # Windows consoles are often cp1252; thinking text/emoji can contain
        # characters it can't encode. Degrade gracefully instead of crashing.
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            enc = sys.stdout.encoding or "ascii"
            print(msg.encode(enc, errors="replace").decode(enc), flush=True)

    def _open_log(self) -> None:
        if not self.log_dir:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
        self._log_file = open(path, "w", encoding="utf-8")
        self._log_file.write(
            f"Slay the Spire 2 — agent run {time.strftime('%Y-%m-%d %H:%M:%S')} ({MODEL})\n"
            + "=" * 60 + "\n\n"
        )
        self._log_file.flush()
        self._log(f"Logging run to {path}")

    def _record_step(self, step: int, state_type: str, thinking: str,
                     name: str, args: dict[str, Any]) -> None:
        """Append a human-readable step to the run log."""
        if not self._log_file:
            return
        self._log_file.write(f"[{step:>4}] {state_type}\n")
        if thinking:
            self._log_file.write("  thinking:\n")
            for line in thinking.strip().splitlines():
                self._log_file.write(f"    {line}\n" if line.strip() else "\n")
        self._log_file.write(f"  action: {_format_action(name, args)}\n\n")
        self._log_file.flush()

    def _note(self, text: str) -> None:
        """Append a one-line note (waits, nudges) to the run log."""
        if self._log_file:
            self._log_file.write(f"  ({text})\n\n")
            self._log_file.flush()

    def _decide(self, state: dict[str, Any], hint: str | None = None) -> tuple[str, dict[str, Any], str]:
        """Ask Claude for one action given the current state. `hint` carries a
        note about the previous step (e.g. it had no effect) to break loops.
        Returns (tool_name, args, thinking_text)."""
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
        thinking = _thinking_text(resp)
        tool = _first_tool_use(resp)
        if tool is not None:
            return tool[0], tool[1], thinking

        # Fallback for the rare turn where it replied without a tool: force a
        # tool call (which requires thinking OFF) so the loop never stalls.
        self._log("    no tool in thinking response; forcing a tool call")
        resp = self._create(messages, thinking=False, force_tool=True)
        tool = _first_tool_use(resp)
        if tool is not None:
            return tool[0], tool[1], thinking
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
            # display="summarized" opts into readable thinking text; on Opus 4.8
            # the default is "omitted" (empty thinking blocks). The raw chain of
            # thought is never returned — this is a summary of it.
            kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}
        if force_tool:
            kwargs["tool_choice"] = {"type": "any"}
        return self.anthropic.messages.create(**kwargs)

    def run(self) -> None:
        """Drive the run from current state until the game ends or we hit max_steps."""
        state = self.client.wait_until_ready()
        self._log("Mod connected. Starting run.")
        self._open_log()
        try:
            self._loop(state)
        finally:
            if self._log_file:
                self._log_file.close()
                self._log_file = None

    def _loop(self, state: dict[str, Any]) -> None:
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
                self._note("enemy turn — waited for player control")
                state = self._await_player_turn(state)
                hint = None
                continue

            name, args, thinking = self._decide(state, hint)
            if self.show_thoughts and thinking:
                self._log("    [think] " + thinking.replace("\n", "\n    "))
            self._log(f"[{step}] -> {_format_action(name, args)}")
            self._record_step(step, str(state.get("state_type")), thinking, name, args)

            # End-turn sanity nudge: if it tries to end the turn with energy and
            # playable cards still in hand, push back once (per state) so it
            # doesn't waste energy / skip blocking. If it insists, let it.
            if name == "end_turn" and _has_unspent_play(state):
                sig = _signature(state)
                if sig != self._last_nudge_sig:
                    self._last_nudge_sig = sig
                    hint = ("You're ending your turn with unspent energy and playable "
                            "cards. Don't waste energy — play affordable cards, and add "
                            "Block if any enemy intends to attack. Only end the turn if "
                            "you have a real reason to hold.")
                    self._log("    (end_turn with resources left — nudging to reconsider)")
                    self._note("end_turn with resources left — nudged to reconsider")
                    continue

            try:
                action = action_from_tool_call(name, args, state)
            except IllegalActionError as e:
                # Claude picked a tool that doesn't apply to this screen.
                self._log(f"    illegal action skipped: {e}")
                self._note(f"illegal action skipped: {e}")
                hint = f"`{name}` is not valid on the '{state.get('state_type')}' screen. Choose a different action."
                state = self.client.get_state()
                continue

            before = _signature(state)
            self.client.send_action(action)
            # The mod applies actions on an async queue (animations, queued
            # effects), so the state right after the POST can predate the card
            # landing. Buffer, then read — polling until the state actually
            # changes so we capture the post-action state, not a stale frame.
            state = self._read_after_action(before)

            # No-progress guard: if the action left the state byte-for-byte
            # identical, it was a no-op (e.g. re-selecting an already-selected
            # menu item). Tell the model so it tries something else instead of
            # looping forever.
            if _signature(state) == before:
                hint = (f"Your previous action {name}({args}) did NOT change the game "
                        "state. It had no effect — choose a DIFFERENT option this time.")
                self._log("    (no state change — nudging model)")
                self._note("no state change — nudged model")
            else:
                hint = None

        self._log(f"Hit max_steps ({self.max_steps}); stopping.")

    def _read_after_action(self, before_sig: str) -> dict[str, Any]:
        """Read state after an action, waiting for the mod to apply it. Sleeps a
        buffer, then re-reads until the state changes from `before_sig` (up to a
        few tries). A genuine no-op settles to the unchanged state after the
        tries, which the no-progress guard then detects."""
        state = self.client.get_state()
        for _ in range(4):
            time.sleep(self.action_delay)
            state = self.client.get_state()
            if _signature(state) != before_sig:
                break
        return state

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


def _format_action(name: str, args: dict[str, Any]) -> str:
    """Render an action like play_card(card_index=0, target=JAW_WORM_0)."""
    if not args:
        return f"{name}()"
    inner = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{name}({inner})"


def _signature(state: dict[str, Any]) -> str:
    """Stable string fingerprint of a state, for detecting no-op actions."""
    return json.dumps(state, sort_keys=True, default=str)


def _first_tool_use(resp: Any) -> tuple[str, dict[str, Any]] | None:
    """Return (tool_name, input) for the first tool_use block, or None."""
    for block in resp.content:
        if block.type == "tool_use":
            return block.name, dict(block.input)
    return None


def _thinking_text(resp: Any) -> str:
    """Concatenate the model's visible thinking blocks (empty if none)."""
    parts = []
    for block in resp.content:
        if block.type == "thinking":
            parts.append(getattr(block, "thinking", "") or "")
    return "\n".join(p for p in parts if p).strip()


def _has_unspent_play(state: dict[str, Any]) -> bool:
    """True if it's combat, the player has energy, and at least one hand card is
    playable and affordable — i.e. ending the turn now would waste energy."""
    if str(state.get("state_type", "")) not in ("monster", "elite", "boss"):
        return False
    player = state.get("player") or {}
    energy = player.get("energy") or 0
    if energy <= 0:
        return False
    for card in player.get("hand") or []:
        if not card.get("can_play", False):
            continue
        cost = str(card.get("cost", ""))
        if cost == "X" or (cost.isdigit() and int(cost) <= energy):
            return True
    return False


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
