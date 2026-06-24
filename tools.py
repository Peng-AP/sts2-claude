"""
Tool definitions exposed to Claude, plus translation to STS2MCP action verbs.

Confirmed against STS2MCP v0.4.0 (raw-full.md). The mod exposes ~25 specific
action verbs, most of which are an index choice on a particular screen
(claim_reward, choose_map_node, select_relic, ...). Rather than surface all 25
as near-identical tools, we give Claude a small, legible tool set and dispatch
the generic ones to the correct verb using the current `state_type`. The
screen->verb mapping below is the one coupling point to the mod's action schema;
it's kept as explicit tables so it's easy to audit against the docs.

Combat actions (play_card/use_potion/discard_potion/end_turn) stay explicit
because they carry distinct parameters (a card index plus a string enemy
`entity_id` target, or a potion `slot`).
"""

from __future__ import annotations

from typing import Any

# Screen (state_type) -> the action verb a single "choose by index" maps to,
# and the parameter name that verb expects for the index.
CHOOSE_DISPATCH: dict[str, tuple[str, str]] = {
    "rewards": ("claim_reward", "index"),
    "card_reward": ("select_card_reward", "card_index"),
    "event": ("choose_event_option", "index"),
    "rest_site": ("choose_rest_option", "index"),
    "shop": ("shop_purchase", "index"),
    "fake_merchant": ("shop_purchase", "index"),
    "map": ("choose_map_node", "index"),
    "card_select": ("select_card", "index"),
    "bundle_select": ("select_bundle", "index"),
    "relic_select": ("select_relic", "index"),
    "treasure": ("claim_treasure_relic", "index"),
    "hand_select": ("combat_select_card", "card_index"),
}

# Screen -> verb for a "confirm" with no index (overlays / hand selection).
CONFIRM_DISPATCH: dict[str, str] = {
    "hand_select": "combat_confirm_selection",
    "card_select": "confirm_selection",
    "bundle_select": "confirm_bundle_selection",
}

# Screen -> verb for a "cancel".
CANCEL_DISPATCH: dict[str, str] = {
    "card_select": "cancel_selection",
    "bundle_select": "cancel_bundle_selection",
}

# Screen -> verb for a "skip"/decline.
SKIP_DISPATCH: dict[str, str] = {
    "card_reward": "skip_card_reward",
    "relic_select": "skip_relic_selection",
}


TOOLS: list[dict[str, Any]] = [
    {
        "name": "play_card",
        "description": (
            "Play a card from your hand in combat. Use the card's `index`. If the "
            "card's target_type names an enemy (AnyEnemy etc.), pass `target` as "
            "that enemy's `entity_id` string (e.g. 'JAW_WORM_0') from battle.enemies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "card_index": {"type": "integer", "description": "Card's `index` in your hand."},
                "target": {"type": "string", "description": "Target enemy's `entity_id`. Omit for untargeted cards."},
            },
            "required": ["card_index"],
        },
    },
    {
        "name": "use_potion",
        "description": "Use a potion by its `slot`. Pass `target` (an enemy entity_id) if its target_type needs one.",
        "input_schema": {
            "type": "object",
            "properties": {
                "slot": {"type": "integer"},
                "target": {"type": "string"},
            },
            "required": ["slot"],
        },
    },
    {
        "name": "discard_potion",
        "description": "Discard a potion you don't want, by its `slot`, to free a belt slot.",
        "input_schema": {"type": "object", "properties": {"slot": {"type": "integer"}}, "required": ["slot"]},
    },
    {
        "name": "end_turn",
        "description": "End your combat turn once you've played what you want.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "choose",
        "description": (
            "Pick an option by its `index` on whatever screen is active: a reward, "
            "card reward, event option, rest option, shop item, map node, or a "
            "selection overlay. Use the `index` shown next to the option in the state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"index": {"type": "integer", "description": "0-based `index` of the option to take."}},
            "required": ["index"],
        },
    },
    {
        "name": "confirm",
        "description": "Confirm the current selection (in-combat card selection or a selection overlay).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "cancel",
        "description": "Cancel/back out of the current selection overlay.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "skip",
        "description": "Skip/decline the current optional choice (e.g. skip a card reward or relic choice) when nothing is worth taking.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "proceed",
        "description": "Advance/confirm a screen that needs no real decision (leave a rewards screen, continue past a result).",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "advance_dialogue",
        "description": "Advance event/story dialogue to the next line.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "menu_select",
        "description": "On a menu screen, select an advertised `option` string (e.g. 'singleplayer', 'standard', 'main_menu', a character id).",
        "input_schema": {"type": "object", "properties": {"option": {"type": "string"}}, "required": ["option"]},
    },
    {
        "name": "look_up",
        "description": (
            "Look up what a card or relic does — INCLUDING its upgraded version — "
            "by name. Use it when you're unsure of an exact effect or want to "
            "compare base vs upgraded before a card reward, a smith/upgrade, or a "
            "shop buy. This is INFORMATION ONLY: it does NOT take a game turn, so "
            "you can look up a couple of things and then make your real move. "
            "`query` is fuzzy (e.g. 'perfected strike', 'ironclad block', 'silver "
            "spoon'). Only items you've already discovered are searchable. For a "
            "card that's currently enchanted/modified, trust its live `description` "
            "in the state — the lookup shows the base/upgraded rules text."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Card or relic name / fuzzy search text."},
                "item_type": {"type": "string", "enum": ["card", "relic", "all"],
                              "description": "Restrict the search. Default 'all'."},
            },
            "required": ["query"],
        },
    },
]


class IllegalActionError(ValueError):
    """The chosen tool doesn't apply to the current screen."""


def action_from_tool_call(name: str, args: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Translate a Claude tool call into a POST body for /api/v1/singleplayer.

    `state` is the current game state; its `state_type` is used to dispatch the
    generic choose/confirm/cancel/skip tools to the correct screen-specific verb.
    """
    screen = str(state.get("state_type", ""))

    if name == "play_card":
        body: dict[str, Any] = {"action": "play_card", "card_index": args["card_index"]}
        if args.get("target"):
            body["target"] = args["target"]
        return body
    if name == "use_potion":
        body = {"action": "use_potion", "slot": args["slot"]}
        if args.get("target"):
            body["target"] = args["target"]
        return body
    if name == "discard_potion":
        return {"action": "discard_potion", "slot": args["slot"]}
    if name == "end_turn":
        return {"action": "end_turn"}
    if name == "advance_dialogue":
        return {"action": "advance_dialogue"}
    if name == "proceed":
        return {"action": "proceed"}
    if name == "menu_select":
        return {"action": "menu_select", "option": args["option"]}

    if name == "choose":
        verb_param = CHOOSE_DISPATCH.get(screen)
        if verb_param is None:
            raise IllegalActionError(f"`choose` has no mapping on screen '{screen}'")
        verb, param = verb_param
        return {"action": verb, param: args["index"]}
    if name == "confirm":
        verb = CONFIRM_DISPATCH.get(screen)
        if verb is None:
            raise IllegalActionError(f"`confirm` has no mapping on screen '{screen}'")
        return {"action": verb}
    if name == "cancel":
        verb = CANCEL_DISPATCH.get(screen)
        if verb is None:
            raise IllegalActionError(f"`cancel` has no mapping on screen '{screen}'")
        return {"action": verb}
    if name == "skip":
        verb = SKIP_DISPATCH.get(screen)
        if verb is None:
            raise IllegalActionError(f"`skip` has no mapping on screen '{screen}'")
        return {"action": verb}

    raise ValueError(f"Unknown tool: {name}")
