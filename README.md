# sts2-claude

A Python agent that plays **Slay the Spire 2** using the Claude API. It reads
structured game state from the **STS2MCP** mod's localhost REST API, asks Claude
(`claude-opus-4-8`, adaptive thinking) for the next move, and sends the action back.

```
StS2 + STS2MCP  <--HTTP :15526-->  Python agent loop  --Claude API-->  claude-opus-4-8
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

## ⚠️ Confirm against your mod build

A few spots are coupled to STS2MCP's exact API and are marked in-code with
`VERSION-DEPENDENT`. Once the mod is running, hit it and adjust these to match:

- `sts2mcp_client.py`: `STATE_PATH`, `ACTION_PATH` (the GET/POST endpoint paths).
- `tools.py` → `action_from_tool_call`: the action payload field names.
- `agent.py` → `_run_is_over` / `_looks_like_state`: the state schema keys used
  to detect end-of-run and whether an action response already contains new state.

Quick way to inspect the real schema:

```bash
curl http://127.0.0.1:15526/state
```

then align the field names above with what you see.
