"""
State compaction: shrink the raw STS2MCP game-state JSON before sending it to
Claude, without losing decision-relevant information.

Why: the agent sends state to the model every step, and a run is hundreds of
steps. Raw mod state is full of rendering/engine noise (sprite paths, colors,
screen coordinates, animation flags, internal ids) that costs input tokens but
tells Claude nothing about how to play. Stripping it is the biggest per-turn
cost lever.

The pruner is schema-agnostic on purpose, because STS2MCP's exact field names
are version-dependent. It works by key-name patterns plus a couple of structural
rules, so it degrades gracefully on a schema we haven't seen. Once we know the
real schema we can tighten KEEP_ALWAYS / NOISE_KEY_PATTERNS to match.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Keys whose VALUES are presentation/engine noise, not gameplay decisions.
# We act on cards/options by index, never by internal id, so ids are noise too.
NOISE_KEY_PATTERNS = [
    r"image", r"sprite", r"texture", r"icon", r"asset", r"portrait", r"art",
    r"url", r"uri", r"path", r"uuid", r"guid", r"hash",
    r"colou?r", r"tint", r"alpha", r"opacity",
    r"position", r"coord", r"offset", r"scale", r"rotation", r"^[xy]$", r"^z$",
    r"anim", r"sound", r"sfx", r"vfx", r"shader", r"font", r"tween",
    r"^id$", r"_id$", r"internal", r"^seed$", r"render", r"layout", r"pixel",
]
_NOISE_RE = re.compile("|".join(NOISE_KEY_PATTERNS), re.IGNORECASE)

# Keys we never prune even if a value looks empty/falsy — these carry meaning
# (0 block, 0 energy, empty hand are all decision-relevant facts).
KEEP_ALWAYS = {
    "block", "energy", "hp", "current_hp", "max_hp", "cost", "damage",
    "name", "intent", "screen", "screen_type", "hand", "monsters", "enemies",
    "powers", "relics", "potions", "choices", "options", "floor", "act",
}

MAX_STR_LEN = 400  # truncate verbose description strings


def _prune(value: Any, key: str | None = None) -> Any:
    """Recursively strip noise. Returns a cleaned copy."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if _NOISE_RE.search(k) and k.lower() not in KEEP_ALWAYS:
                continue
            cleaned = _prune(v, k)
            if _is_empty(cleaned) and (k.lower() not in KEEP_ALWAYS):
                continue
            out[k] = cleaned
        return out
    if isinstance(value, list):
        return [_prune(v, key) for v in value]
    if isinstance(value, str) and len(value) > MAX_STR_LEN:
        return value[:MAX_STR_LEN] + "…"
    return value


def _is_empty(v: Any) -> bool:
    return v is None or v == "" or v == [] or v == {}


def estimate_tokens(obj: Any) -> int:
    """Rough token estimate (~4 chars/token) for logging savings."""
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return len(text) // 4


def compact_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return a pruned copy of the state suitable for sending to Claude.

    Pure function — does not mutate the input. The agent keeps the raw state for
    its own logic (end-of-run detection) and only compacts what it sends.
    """
    if not isinstance(state, dict):
        return state
    return _prune(state)
