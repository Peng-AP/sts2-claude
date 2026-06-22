"""
State compaction: shrink the STS2MCP game-state JSON before sending it to Claude.

Tuned to the real STS2MCP v0.4.0 schema (raw-full.md). That schema is already
clean semantic JSON — there's no sprite/color/coordinate noise to strip. The
actual token bloat is:

  * `keywords` arrays — every card/relic/potion/power repeats keyword name+text
    (Vulnerable, Exhaust, ...). Claude knows these; the card's own `description`
    already states the effect. Dropping them is a big, low-risk saving.
  * `combat_id` — an internal numeric id we never use (we target by `entity_id`).
  * Verbose `description` strings — truncated past a cap.

We deliberately PRESERVE the handles actions need: `entity_id`, `index`, `slot`,
`id`, and gameplay fields. The optional `drop_piles` flag can also strip the
draw/discard/exhaust pile listings, which are large; off by default since pile
knowledge helps planning.
"""

from __future__ import annotations

import json
from typing import Any

# Keys removed wherever they appear — pure token bloat for decision-making.
DROP_KEYS = {"keywords", "combat_id"}

# Pile listings are large; dropping them is opt-in.
PILE_KEYS = {"draw_pile", "discard_pile", "exhaust_pile"}

MAX_STR_LEN = 300  # truncate verbose description strings


def _prune(value: Any, drop_piles: bool) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if k in DROP_KEYS:
                continue
            if drop_piles and k in PILE_KEYS:
                continue
            out[k] = _prune(v, drop_piles)
        return out
    if isinstance(value, list):
        return [_prune(v, drop_piles) for v in value]
    if isinstance(value, str) and len(value) > MAX_STR_LEN:
        return value[:MAX_STR_LEN] + "…"
    return value


def estimate_tokens(obj: Any) -> int:
    """Rough token estimate (~4 chars/token) for logging savings."""
    text = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return len(text) // 4


def compact_state(state: dict[str, Any], drop_piles: bool = False) -> dict[str, Any]:
    """Return a pruned copy of the state suitable for sending to Claude.

    Pure function — does not mutate the input. The agent keeps the raw state for
    its own logic (end-of-run detection) and only compacts what it sends.
    """
    if not isinstance(state, dict):
        return state
    return _prune(state, drop_piles)
