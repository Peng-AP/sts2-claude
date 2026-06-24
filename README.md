# sts2-claude

A Python agent that plays **Slay the Spire 2** using the Claude API. It reads
structured game state from the **STS2MCP** mod's localhost REST API, asks Claude
(`claude-sonnet-4-6`, adaptive thinking) for the next move, and sends the action back.

```
StS2 + STS2MCP  <--HTTP :15526-->  Python agent loop  --Claude API-->  claude-sonnet-4-6
```

## Setup

1. Install the **STS2MCP** mod (`Gennadiyev/STS2MCP`) for Slay the Spire 2. It
   serves a REST API on `http://127.0.0.1:15526` while the game runs.
2. `python -m venv .venv && .venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `copy .env.example .env` and set `ANTHROPIC_API_KEY`.
5. Launch StS2 (with the mod loaded), then: `python main.py`

## Files

| File | Role |
|------|------|
| `sts2mcp_client.py` | HTTP client for the mod's state/action API |
| `tools.py` | Claude tool definitions + tool→action translation |
| `state_compaction.py` | Strips engine/render noise from state before sending |
| `agent.py` | The read→decide→act loop |
| `main.py` | Entry point |

## Keeping run cost down

A run is hundreds of model calls, so two levers are built in:

- **State compaction** (`state_compaction.py`): prunes rendering/engine noise
  (sprite paths, colors, coordinates, animation flags, internal ids) and
  truncates verbose card text before each turn's state goes to Claude. The agent
  keeps the raw state for its own logic and only compacts what it sends. The
  per-step log prints the before/after token estimate.
- **Prompt caching** (`agent.py`): the static prefix — system prompt + tool
  definitions — is marked with `cache_control`, so it's cached and reused across
  steps instead of re-billed at full price every call.

## Mod API (confirmed against STS2MCP v0.4.0)

The code is written against the real STS2MCP API (see the mod's own
`docs/raw-full.md` in the [STS2MCP repo](https://github.com/Gennadiyev/STS2MCP)):

- **State + actions share one path**: `GET /api/v1/singleplayer` reads state,
  `POST /api/v1/singleplayer` performs an action `{"action": <verb>, ...}`.
  (Multiplayer uses `/api/v1/multiplayer`; mixing them returns HTTP 409.)
- **Screens** are identified by `state_type` (monster/elite/boss, rewards,
  card_reward, map, event, rest_site, shop, treasure, selection overlays, menu,
  game_over). `tools.py` maps the generic `choose`/`confirm`/`skip` tools to the
  correct screen-specific verb.
- **Targeting** uses an enemy's `entity_id` string (e.g. `"JAW_WORM_0"`).
- **Card/relic lookups**: `GET /api/v1/wiki?query=…&item_type=card|relic` fuzzy-
  searches discovered items and returns each card's `base` *and* `upgraded`
  variant. The agent's `look_up` tool uses this so the model can check a card's
  exact effect (and what upgrading does) before card rewards, smith, or shop
  buys. Lookups are information-only (no game turn) and capped per decision.

## History scoping

The agent threads message history only *within a decision episode* — a single
combat player turn, or a contiguous stay on one non-combat screen (event, card
reward, rest site, shop). Within an episode the model sees its own move sequence
and any `look_up` results; the history resets when the episode changes (end of
turn, enemy turn, or moving to a new screen). Cross-turn / cross-room context is
intentionally dropped: state is fully observable, so it adds cost without value.

Inspect live state anytime with the game running:

```bash
curl http://127.0.0.1:15526/api/v1/singleplayer     # or: python probe.py
```

Verified working on game **v0.107.1** with mod **v0.4.0**.
