# Astar Island — Quickstart

Get up and running in four steps: authenticate, fetch the active round, query the simulator, and submit predictions.

---

## Authentication

Log in at [app.ainm.no](https://app.ainm.no), then grab your `access_token` JWT from your browser cookies. You can pass it as either a cookie or a Bearer header — both work identically.

```python
import requests

BASE = "https://api.ainm.no"

# Option 1: Cookie-based auth
session = requests.Session()
session.cookies.set("access_token", "YOUR_JWT_TOKEN")

# Option 2: Bearer token auth
session = requests.Session()
session.headers["Authorization"] = "Bearer YOUR_JWT_TOKEN"
```

---

## Step 1 — Get the Active Round

```python
rounds = session.get(f"{BASE}/astar-island/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)

if active:
    round_id = active["id"]
    print(f"Active round: {active['round_number']}")
```

---

## Step 2 — Get Round Details

Fetch the detail endpoint for full round info: map dimensions, seed count, and the initial terrain state for each seed.

```python
detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()

width  = detail["map_width"]   # 40
height = detail["map_height"]  # 40
seeds  = detail["seeds_count"] # 5
print(f"Round: {width}x{height}, {seeds} seeds")

for i, state in enumerate(detail["initial_states"]):
    grid        = state["grid"]         # height × width terrain codes
    settlements = state["settlements"]  # [{x, y, has_port, alive}, ...]
    print(f"Seed {i}: {len(settlements)} settlements")
```

> Initial states expose settlement position and port status only. Internal stats (population, food, wealth, defense) are hidden until you query the simulator.

---

## Step 3 — Query the Simulator

You have **50 queries per round**, shared across all 5 seeds. Each query runs one stochastic simulation and returns a viewport of the result — between 5 and 15 cells wide.

```python
result = session.post(f"{BASE}/astar-island/simulate", json={
    "round_id":   round_id,
    "seed_index": 0,
    "viewport_x": 10,
    "viewport_y": 5,
    "viewport_w": 15,
    "viewport_h": 15,
}).json()

grid        = result["grid"]         # 15×15 terrain after simulation
settlements = result["settlements"]  # settlements in viewport with full stats
viewport    = result["viewport"]     # {x, y, w, h}
```

Each call uses a different random seed, so querying the same area multiple times gives you different stochastic outcomes — useful for building a probability distribution.

---

## Step 4 — Build and Submit Predictions

For each seed, submit an `H × W × 6` probability tensor. Each cell holds 6 values — one per terrain class — that must sum to `1.0`.

```python
import numpy as np

for seed_idx in range(seeds):
    prediction = np.full((height, width, 6), 1/6)  # uniform baseline

    # Replace with your model's predictions:
    # prediction[y][x] = [p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]

    resp = session.post(f"{BASE}/astar-island/submit", json={
        "round_id":   round_id,
        "seed_index": seed_idx,
        "prediction": prediction.tolist(),
    })
    print(f"Seed {seed_idx}: {resp.status_code}")
```

A uniform prediction scores roughly **1–5**. Use your 50 queries to build better estimates.

### Class index reference

| Index | Class |
|-------|-------|
| `0` | Empty (Ocean, Plains, Empty) |
| `1` | Settlement |
| `2` | Port |
| `3` | Ruin |
| `4` | Forest |
| `5` | Mountain |

### ⚠️ Never use `0.0` probability

If the ground truth assigns any non-zero probability to a class you've marked as `0.0`, KL divergence becomes infinite and your score for that cell is wiped out. Always apply a minimum floor and renormalise:

```python
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)
```

This costs almost nothing on confident cells but protects against catastrophic scoring blowups. See the [scoring docs](astar_island_scoring.md) for details.

---

## MCP Server (Claude Code)

Add the documentation server to Claude Code for AI-assisted development:

```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```