"""
One-off discovery tool. Run this with Slay the Spire 2 open (mod loaded) to find
out STS2MCP's actual endpoint paths and state schema, so we can lock down the
VERSION-DEPENDENT spots in the rest of the project.

    python probe.py

It does three things:
  1. Probes a list of likely GET paths and reports which ones respond.
  2. Saves the first JSON state it finds to sample_state.json for inspection.
  3. Prints the top-level keys of that state so we can tune the compactor.

It only does GETs — it never sends an action, so it's safe to run anytime.
"""

from __future__ import annotations

import json
import os

import requests

BASE = os.environ.get("STS2MCP_BASE_URL", "http://127.0.0.1:15526")

# Likely paths for reading state / discovering the API. We don't know which the
# mod uses, so we try the common conventions and see what answers.
CANDIDATE_PATHS = [
    "/", "/state", "/game_state", "/gamestate", "/status", "/api", "/api/state",
    "/v1/state", "/get_state", "/current_state", "/observe", "/snapshot",
    "/openapi.json", "/docs", "/health", "/ping",
]


def probe() -> None:
    print(f"Probing STS2MCP at {BASE}\n")
    found_state: dict | None = None

    for path in CANDIDATE_PATHS:
        url = f"{BASE}{path}"
        try:
            r = requests.get(url, timeout=4)
        except requests.RequestException as e:
            print(f"  {path:<16} -- no response ({type(e).__name__})")
            continue

        ctype = r.headers.get("content-type", "")
        is_json = "json" in ctype or r.text.strip()[:1] in "{["
        tag = "JSON" if is_json else ctype.split(";")[0] or "?"
        print(f"  {path:<16} HTTP {r.status_code}  [{tag}]  {len(r.content)} bytes")

        if r.ok and is_json and found_state is None:
            try:
                data = r.json()
                if isinstance(data, dict) and len(data) > 1:
                    found_state = data
                    print(f"       ^ looks like state — captured from {path}")
            except ValueError:
                pass

    print()
    if found_state is None:
        print("No state-like JSON found. Either the game isn't in a run yet, or the")
        print("path differs — check the mod's README/Nexus page for its API and add")
        print("the path to CANDIDATE_PATHS above, then re-run.")
        return

    with open("sample_state.json", "w", encoding="utf-8") as f:
        json.dump(found_state, f, indent=2)
    print("Saved sample_state.json")
    print("\nTop-level keys:")
    for k in found_state:
        v = found_state[k]
        kind = type(v).__name__
        size = f"[{len(v)}]" if isinstance(v, (list, dict)) else ""
        print(f"  {k:<20} {kind}{size}")


if __name__ == "__main__":
    probe()
