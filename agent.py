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
from collections import namedtuple
from typing import Any

# One model decision, plus the message blocks needed to extend turn history.
Decision = namedtuple("Decision", "name args thinking tool_id assistant_content user_message")

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
- On non-combat screens a `your_deck` field (when present) lists your full
  current deck as card-name -> count. Use it for deck-shaping decisions: pick
  card rewards that fit a coherent plan, skip cards that dilute the deck, and
  decide at rest sites whether to upgrade (smith) vs heal based on the deck.
- Only choose from options actually offered right now (respect `can_play`, \
  `can_skip`, `can_confirm`, available indices). Never invent one.
- If you're unsure what a card or relic does, or want to compare a card to its \
  upgraded version (before a card reward, a smith/upgrade at a rest site, or a \
  shop buy), call `look_up` first. It's information only — it doesn't use a \
  turn — so look up what you need, then make your real move. A card that's \
  already enchanted/modified shows its true current effect in its own \
  `description`; the lookup gives base vs upgraded rules text.

Call exactly one tool per step. Keep reasoning focused and decisive."""


class SpireAgent:
    def __init__(self, client: STS2MCPClient, anthropic: Anthropic | None = None,
                 max_steps: int = 2000, verbose: bool = True, log_dir: str | None = "runs",
                 show_thoughts: bool = True, action_delay: float = 0.4,
                 max_lookups: int = 4, lookup_limit: int = 5):
        self.client = client
        self.anthropic = anthropic or Anthropic()
        self.max_steps = max_steps
        self.verbose = verbose
        self.show_thoughts = show_thoughts
        self.log_dir = log_dir
        self.action_delay = action_delay   # buffer for the mod to apply an action
        self.max_lookups = max_lookups     # max look_up calls per decision episode
        self.lookup_limit = lookup_limit   # results requested per wiki lookup
        self._deck_cards: list[str] = []   # latest known master deck (from combat)
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

    def _decide(self, state: dict[str, Any], hint: str | None = None,
                history: list | None = None,
                pending: tuple[str, str] | None = None) -> Decision:
        """Ask Claude for one action. `hint` carries a note about the previous
        step (loop-breaking). `history` + `pending` thread per-episode context:
        history is the prior messages of the current episode (a combat turn, or a
        stay on one non-combat screen), and `pending` is (tool_use_id, result_text)
        for the previous step — opened as a tool_result so the model sees its own
        move sequence and any look_up results within the episode.

        Returns a Decision(name, args, thinking, tool_id, assistant_content,
        user_message) — the last two let the caller extend the episode history."""
        compact = compact_state(state)
        deck = self._deck_for(state)
        if deck is not None:
            compact = {**compact, "your_deck": deck}
        if self.verbose:
            self._log(
                f"    state tokens ~{estimate_tokens(state)} -> ~{estimate_tokens(compact)} "
                "(compacted)" + ("" if not history else f", turn history {len(history)//2} step(s)")
            )
        nudge = f"\n\nNOTE: {hint}\n" if hint else ""
        text = (
            "Current Slay the Spire 2 game state:\n\n"
            f"```json\n{json.dumps(compact, indent=2)}\n```\n"
            f"{nudge}\n"
            "Take the single best next action by calling one tool."
        )
        # When continuing an episode, the new user message must open with a
        # tool_result for the previous step's tool_use before the new state. The
        # result text is "Action applied…" for a real move, or the wiki data for
        # a look_up (so the model actually sees what it looked up).
        if pending:
            pending_tool_id, result_text = pending
            content: Any = [
                {"type": "tool_result", "tool_use_id": pending_tool_id,
                 "content": result_text},
                {"type": "text", "text": text},
            ]
        else:
            content = text
        user_message = {"role": "user", "content": content}
        messages = (history or []) + [user_message]

        # Primary call: extended thinking ON. The API forbids forcing tool use
        # while thinking is on, so tool_choice is "auto"; the system prompt's
        # "call exactly one tool" makes the model call one essentially always.
        resp = self._create(messages, thinking=True, force_tool=False)
        thinking = _thinking_text(resp)
        tool = _first_tool_use(resp)
        if tool is None:
            # Rare: replied without a tool. Force one (requires thinking OFF).
            self._log("    no tool in thinking response; forcing a tool call")
            resp = self._create(messages, thinking=False, force_tool=True)
            tool = _first_tool_use(resp)
        if tool is None:
            raise RuntimeError("Claude returned no tool_use block")
        name, args, tool_id = tool
        return Decision(name, args, thinking, tool_id, list(resp.content), user_message)

    def _deck_for(self, state: dict[str, Any]) -> dict[str, int] | None:
        """A compact name->count view of the known deck, for non-combat decision
        screens (card reward, rest site, shop, …) where the deck isn't in state.
        None during combat (the piles are already present) or before we've seen
        any combat to snapshot from."""
        if str(state.get("state_type", "")) in ("monster", "elite", "boss"):
            return None
        if not self._deck_cards:
            return None
        counts: dict[str, int] = {}
        for name in self._deck_cards:
            counts[name] = counts.get(name, 0) + 1
        return dict(sorted(counts.items()))

    def _update_deck_snapshot(self, state: dict[str, Any]) -> None:
        """Snapshot the master deck from a round-1 combat state: the union of all
        four piles is the full deck, cleanest at the start of a fight."""
        if str(state.get("state_type", "")) not in ("monster", "elite", "boss"):
            return
        battle = state.get("battle") or {}
        if battle.get("round") != 1:
            return
        player = state.get("player") or {}
        cards: list[str] = []
        for c in player.get("hand") or []:
            cards.append(str(c.get("name", "?")))
        for pile in ("draw_pile", "discard_pile", "exhaust_pile"):
            for c in player.get(pile) or []:
                cards.append(str(c.get("name", "?")))
        if cards:
            self._deck_cards = cards

    def _create(self, messages: list[dict[str, Any]], *, thinking: bool, force_tool: bool):
        if not thinking:
            messages = _strip_thinking(messages)
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
        # Per-episode history: messages accumulate while we stay in one decision
        # context — a combat player turn, or a contiguous stay on one non-combat
        # screen (event, card_reward, rest_site, shop…) — so the model sees its
        # own move sequence and any look_up results it requested. It resets when
        # the episode changes (end_turn, enemy turn, or moving to a new screen).
        # Cross-turn / cross-room history is intentionally dropped: it's expensive
        # and low-value since each state is fully observable.
        history: list = []
        pending: tuple[str, str] | None = None
        episode: str | None = None
        lookups = 0  # look_up calls used in the current episode

        for step in range(self.max_steps):
            if _run_is_over(state):
                self._log(f"Run ended after {step} steps.")
                return

            self._update_deck_snapshot(state)

            # Enemy turn: the player can't act, so don't spend a model call —
            # just poll until control returns (or the screen changes, e.g. the
            # fight ends). Saves an API call on every enemy turn.
            if _is_enemy_turn(state):
                self._log(f"[{step}] enemy turn — waiting for player control")
                self._note("enemy turn — waited for player control")
                state = self._await_player_turn(state)
                hint = None
                history, pending, episode, lookups = [], None, None, 0
                continue

            # New episode? Reset the threaded context when the screen/context
            # changes (combat resets additionally at end_turn, handled below).
            key = _episode_key(state)
            if key != episode:
                history, pending, episode, lookups = [], None, key, 0

            d = self._decide(state, hint, history, pending)
            if self.show_thoughts and d.thinking:
                self._log("    [think] " + d.thinking.replace("\n", "\n    "))
            self._log(f"[{step}] -> {_format_action(d.name, d.args)}")
            self._record_step(step, str(state.get("state_type")), d.thinking, d.name, d.args)

            # look_up is information only: fetch the wiki entry and thread it back
            # as a tool_result on the same state — no game action is taken.
            if d.name == "look_up":
                lookups += 1
                result_text = self._lookup_result(d.args, lookups)
                history.append(d.user_message)
                history.append({"role": "assistant", "content": d.assistant_content})
                pending = (d.tool_id, result_text)
                hint = None
                continue

            # End-turn sanity nudge: if it tries to end the turn with energy and
            # playable cards still in hand, push back once (per state) so it
            # doesn't waste energy / skip blocking. If it insists, let it.
            if d.name == "end_turn" and _has_unspent_play(state):
                sig = _signature(state)
                if sig != self._last_nudge_sig:
                    self._last_nudge_sig = sig
                    hint = ("You're ending your turn with unspent energy and playable "
                            "cards. Don't waste energy — play affordable cards, and add "
                            "Block if any enemy intends to attack. Only end the turn if "
                            "you have a real reason to hold.")
                    self._log("    (end_turn with resources left — nudging to reconsider)")
                    self._note("end_turn with resources left — nudged to reconsider")
                    continue  # don't commit; re-decide on the same state

            try:
                action = action_from_tool_call(d.name, d.args, state)
            except IllegalActionError as e:
                # Claude picked a tool that doesn't apply to this screen.
                self._log(f"    illegal action skipped: {e}")
                self._note(f"illegal action skipped: {e}")
                hint = f"`{d.name}` is not valid on the '{state.get('state_type')}' screen. Choose a different action."
                state = self.client.get_state()
                continue  # don't commit; re-decide

            before = _signature(state)
            self.client.send_action(action)
            # The mod applies actions on an async queue (animations, queued
            # effects), so the state right after the POST can predate the card
            # landing. Buffer, then read — polling until the state actually
            # changes so we capture the post-action state, not a stale frame.
            state = self._read_after_action(before)

            # Commit to the episode history now that the action ran. If the move
            # changed the screen, the next iteration's episode-key check resets
            # this; while we stay on one screen it threads the context forward.
            history.append(d.user_message)
            history.append({"role": "assistant", "content": d.assistant_content})
            pending = (d.tool_id, "Action applied. Updated state:")
            if d.name == "end_turn":
                history, pending, episode, lookups = [], None, None, 0  # turn over

            # No-progress guard: if the action left the state byte-for-byte
            # identical, it was a no-op (e.g. re-selecting an already-selected
            # menu item). Tell the model so it tries something else instead of
            # looping forever.
            if _signature(state) == before:
                hint = (f"Your previous action {d.name}({d.args}) did NOT change the game "
                        "state. It had no effect — choose a DIFFERENT option this time.")
                self._log("    (no state change — nudging model)")
                self._note("no state change — nudged model")
            else:
                hint = None

        self._log(f"Hit max_steps ({self.max_steps}); stopping.")

    def _lookup_result(self, args: dict[str, Any], lookups: int) -> str:
        """Run a wiki look_up and return the tool_result text to thread back to
        the model. Caps lookups per episode so it can't loop on info-gathering."""
        query = str(args.get("query", "")).strip()
        item_type = str(args.get("item_type") or "all")
        self._log(f"    [look_up] {query!r} ({item_type})")
        self._note(f"look_up: {query!r} ({item_type})")
        if lookups > self.max_lookups:
            return ("Lookup limit reached for this decision. Decide now and take a "
                    "real game action.")
        if not query:
            return "Provide a non-empty `query` (a card or relic name) to look up."
        try:
            raw = self.client.wiki(query, item_type, limit=self.lookup_limit)
        except Exception as e:  # network/mod error — tell the model, don't crash
            return f"Lookup failed: {e}. Proceed using the info already in the state."
        return "Wiki lookup result:\n" + json.dumps(_format_wiki(raw), indent=2)

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


def _episode_key(state: dict[str, Any]) -> str:
    """Identify the current decision 'episode' for history scoping. All combat
    player turns share the key "combat" (per-turn reset is handled explicitly at
    end_turn / enemy turn); each non-combat screen type is its own episode, so
    reasoning and look_ups thread across the steps of one event / card reward /
    rest site but reset when we move to a different screen."""
    st = str(state.get("state_type", ""))
    if st in ("monster", "elite", "boss"):
        return "combat"
    return f"screen:{st}"


def _format_wiki(resp: dict[str, Any]) -> dict[str, Any]:
    """Trim a /api/v1/wiki response to the rules text the model needs (id, name,
    rarity, type, base + upgraded cost/description), dropping scores, keywords,
    and profile metadata."""
    if not isinstance(resp, dict):
        return {"note": "Unexpected lookup response."}
    if resp.get("status") and resp.get("status") != "ok":
        return {"status": resp.get("status"), "message": resp.get("message", "")}
    out: list[dict[str, Any]] = []
    for r in resp.get("results") or []:
        item: dict[str, Any] = {k: r.get(k) for k in ("item_type", "id", "name", "rarity", "type")
                                if r.get(k) is not None}
        if r.get("is_upgradable") is not None:
            item["is_upgradable"] = r["is_upgradable"]
        for variant in ("base", "upgraded"):
            v = r.get(variant)
            if isinstance(v, dict):
                item[variant] = {k: v.get(k) for k in ("cost", "description")
                                 if v.get(k) is not None}
        if "base" not in item and r.get("description"):   # relic (no variants)
            item["description"] = r["description"]
        out.append(item)
    if not out:
        return {"results": [],
                "note": "No discovered card/relic matched. You can only look up "
                        "items you've already encountered; otherwise rely on the state."}
    return {"query": resp.get("query"), "results": out}


def _is_enemy_turn(state: dict[str, Any]) -> bool:
    """True when we're in combat but it's not the player's actionable window."""
    if str(state.get("state_type", "")) not in ("monster", "elite", "boss"):
        return False
    battle = state.get("battle") or {}
    if battle.get("is_play_phase") is False:
        return True
    return str(battle.get("turn", "")).lower() == "enemy"


def _strip_thinking(messages: list) -> list:
    """Remove thinking blocks from assistant messages — needed when sending
    history on a request that has thinking disabled (the forced-tool fallback)."""
    out = []
    for m in messages:
        content = m.get("content")
        if m.get("role") == "assistant" and isinstance(content, list):
            kept = [b for b in content
                    if getattr(b, "type", None) not in ("thinking", "redacted_thinking")]
            out.append({**m, "content": kept})
        else:
            out.append(m)
    return out


def _format_action(name: str, args: dict[str, Any]) -> str:
    """Render an action like play_card(card_index=0, target=JAW_WORM_0)."""
    if not args:
        return f"{name}()"
    inner = ", ".join(f"{k}={v}" for k, v in args.items())
    return f"{name}({inner})"


def _signature(state: dict[str, Any]) -> str:
    """Stable string fingerprint of a state, for detecting no-op actions."""
    return json.dumps(state, sort_keys=True, default=str)


def _first_tool_use(resp: Any) -> tuple[str, dict[str, Any], str] | None:
    """Return (tool_name, input, tool_use_id) for the first tool_use block, or None."""
    for block in resp.content:
        if block.type == "tool_use":
            return block.name, dict(block.input), block.id
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
