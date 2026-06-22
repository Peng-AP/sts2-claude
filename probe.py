"""
Capture a live game-state sample from STS2MCP for inspection.

Run this with Slay the Spire 2 open (mod loaded) and the game in whatever screen
you want to inspect — ideally an active combat, so the sample includes the
battle/hand/enemy shapes.

    python probe.py

Endpoints are now known (STS2MCP v0.4.0): health at `/`, singleplayer state at
`/api/v1/singleplayer`. This saves the state to sample_state.json and prints its
top-level keys plus the current state_type.
"""

from __future__ import annotations

import json
import os

import requests

BASE = os.environ.get("STS2MCP_BASE_URL", "http://127.0.0.1:15526")


def main() -> None:
    # Health check first so a missing mod gives a clear message.
    try:
        h = requests.get(f"{BASE}/", timeout=4)
        print(f"health: HTTP {h.status_code} {h.text.strip()[:80]}")
    except requests.RequestException as e:
        print(f"Can't reach {BASE} — is StS2 running with the mod loaded? ({e})")
        return

    try:
        r = requests.get(f"{BASE}/api/v1/singleplayer", timeout=6)
    except requests.RequestException as e:
        print(f"state request failed: {e}")
        return

    if r.status_code == 409:
        print("HTTP 409: a multiplayer run is active — use /api/v1/multiplayer instead.")
        return
    if not r.ok:
        print(f"state request returned HTTP {r.status_code}: {r.text[:200]}")
        return

    state = r.json()
    with open("sample_state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    print(f"\nSaved sample_state.json  (state_type = {state.get('state_type')!r})")
    print("Top-level keys:")
    for k, v in state.items():
        kind = type(v).__name__
        size = f"[{len(v)}]" if isinstance(v, (list, dict)) else ""
        print(f"  {k:<16} {kind}{size}")


if __name__ == "__main__":
    main()
