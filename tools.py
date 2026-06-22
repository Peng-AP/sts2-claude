"""
Tool definitions exposed to Claude.

Design choice (from the plan): we don't let Claude emit free text actions.
Instead it can only call these tools, and we validate the inputs against the
*currently legal* options before forwarding to the mod. This keeps the agent
from hallucinating illegal moves (playing a card it can't afford, choosing a
reward index that doesn't exist, etc.).

Each tool maps to a category of STS2MCP action. The exact action payload that
gets POSTed to the mod is assembled in `action_from_tool_call` — that's the
other place (besides sts2mcp_client.py) that's coupled to the mod's action
schema, so it's kept small and explicit.
"""

from __future__ import annotations

from typing import Any


# The tool schemas Claude sees. Keep descriptions tight and action-focused.
TOOLS: list[dict[str, Any]] = [
    {
        "name": "play_card",
        "description": (
            "Play a card from your hand during combat. Provide the card's hand "
            "index and, if the card requires a target, the enemy's index."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "card_index": {
                    "type": "integer",
                    "description": "0-based index of the card in your hand.",
                },
                "target_index": {
                    "type": "integer",
                    "description": "0-based index of the target enemy. Omit for untargeted cards.",
                },
            },
            "required": ["card_index"],
        },
    },
    {
        "name": "end_turn",
        "description": "End your turn in combat once you've played the cards you want.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "use_potion",
        "description": "Use a potion. Provide the potion slot index and a target if needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "potion_index": {"type": "integer"},
                "target_index": {"type": "integer"},
            },
            "required": ["potion_index"],
        },
    },
    {
        "name": "choose",
        "description": (
            "Make a non-combat choice: pick a card reward, relic, shop item, "
            "event option, map node, or any menu choice the game is currently "
            "presenting. Provide the 0-based index of the option you want."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "option_index": {
                    "type": "integer",
                    "description": "0-based index into the list of currently available choices.",
                },
            },
            "required": ["option_index"],
        },
    },
    {
        "name": "skip",
        "description": (
            "Skip / decline the current optional choice (e.g. skip a card reward, "
            "leave a shop, proceed without resting). Use when no option is worth taking."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "proceed",
        "description": (
            "Advance the game when no real decision is required — confirm a "
            "result screen, continue past a transition, etc."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


def action_from_tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Translate a Claude tool call into a mod action payload.

    ⚠️ VERSION-DEPENDENT shape: this assembles the dict POSTed to STS2MCP's
    action endpoint. Adjust the field names here if your mod build expects a
    different schema. Kept intentionally flat and explicit.
    """
    if name == "play_card":
        action: dict[str, Any] = {"action": "play_card", "card_index": args["card_index"]}
        if "target_index" in args and args["target_index"] is not None:
            action["target_index"] = args["target_index"]
        return action
    if name == "end_turn":
        return {"action": "end_turn"}
    if name == "use_potion":
        action = {"action": "use_potion", "potion_index": args["potion_index"]}
        if "target_index" in args and args["target_index"] is not None:
            action["target_index"] = args["target_index"]
        return action
    if name == "choose":
        return {"action": "choose", "option_index": args["option_index"]}
    if name == "skip":
        return {"action": "skip"}
    if name == "proceed":
        return {"action": "proceed"}
    raise ValueError(f"Unknown tool: {name}")
