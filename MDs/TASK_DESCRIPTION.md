# 🏝️ Astar Island — Norse Civilisation Prediction

A machine learning challenge where you observe a black-box Norse civilisation simulator through a limited viewport and predict the final world state. The simulator runs a procedurally generated Norse world for 50 years — settlements grow, factions clash, trade routes form, and harsh winters reshape entire civilisations.

**Your goal:** observe the world through limited windows, learn its hidden rules, and submit probability distributions for the final terrain state across the entire map.

- **Platform:** [app.ainm.no](https://app.ainm.no)
- **API base URL:** `https://api.ainm.no/astar-island`

---

## 🧩 What You're Solving

Each round gives you a 40×40 map with 5 random seeds. You have **50 simulation queries** per round (shared across all 5 seeds). Each query lets you peek through a viewport (up to 15×15 cells) to observe one stochastic run of the simulation.

After observing, you submit a **W×H×6 probability tensor** per seed — your prediction of the probability of each terrain class at each cell after 50 simulated years.

Scoring uses entropy-weighted KL divergence. Only dynamic (uncertain) cells count — static ocean and mountain cells are excluded.

---

## 🔑 Authentication

All team endpoints require a JWT. Log in at [app.ainm.no](https://app.ainm.no), then grab your `access_token` from cookies.

Two auth options — pick whichever suits your setup:

```python
import requests

BASE = "https://api.ainm.no"

# Option A: Cookie
session = requests.Session()
session.cookies.set("access_token", "YOUR_JWT_TOKEN")

# Option B: Bearer header
session = requests.Session()
session.headers["Authorization"] = "Bearer YOUR_JWT_TOKEN"
```

---

## 🚀 Quick Start

### 1. Get the active round

```python
rounds = session.get(f"{BASE}/astar-island/rounds").json()
active = next((r for r in rounds if r["status"] == "active"), None)
round_id = active["id"]
```

### 2. Fetch round details & initial map state

```python
detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()

width = detail["map_width"]    # 40
height = detail["map_height"]  # 40
seeds = detail["seeds_count"]  # 5

for i, state in enumerate(detail["initial_states"]):
    grid = state["grid"]              # H×W terrain codes
    settlements = state["settlements"]  # [{x, y, has_port, alive}, ...]
```

> **Note:** Initial states expose settlement positions and port status only. Internal stats (population, food, wealth, defense) are hidden until you query the simulator.

### 3. Query the simulator (costs 1 budget query)

```python
result = session.post(f"{BASE}/astar-island/simulate", json={
    "round_id": round_id,
    "seed_index": 0,
    "viewport_x": 10,
    "viewport_y": 5,
    "viewport_w": 15,
    "viewport_h": 15,
}).json()

grid = result["grid"]                # viewport_h × viewport_w terrain after 50 years
settlements = result["settlements"]  # settlements in viewport, with full stats
# queries_used / queries_max also returned
```

Each query uses a different random sim seed → different stochastic outcome. Use this to build distributions.

### 4. Submit predictions for all 5 seeds

```python
import numpy as np

for seed_idx in range(seeds):
    prediction = np.full((height, width, 6), 1/6)  # uniform baseline

    # ⚠️ CRITICAL: Never leave any probability as 0.0 — see scoring section
    prediction = np.maximum(prediction, 0.01)
    prediction = prediction / prediction.sum(axis=-1, keepdims=True)

    resp = session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_idx,
        "prediction": prediction.tolist(),
    })
    print(f"Seed {seed_idx}: {resp.status_code}")
```

> Always submit something for every seed — even a uniform prediction beats a score of 0.

---

## 🗺️ Terrain Classes

| Class Index | Terrain Types | Notes |
|---|---|---|
| 0 | Ocean, Plains, Empty | Static borders / flat land |
| 1 | Settlement | Active Norse settlement |
| 2 | Port | Coastal settlement with harbour |
| 3 | Ruin | Collapsed settlement |
| 4 | Forest | Provides food to adjacent settlements |
| 5 | Mountain | Impassable, never changes |

Ocean, Plains, and Empty all map to class 0. Mountains are static. The interesting cells to predict are those that become Settlements (1), Ports (2), or Ruins (3).

---

## 📊 Scoring

### Ground truth
The organizers pre-compute ground truth by running the simulator hundreds of times per seed. Each cell gets a probability distribution — e.g. `[0.0, 0.60, 0.25, 0.15, 0.0, 0.0]` (60% Settlement, 25% Port, 15% Ruin).

### Formula
```
weighted_kl = Σ entropy(cell) × KL(ground_truth[cell], prediction[cell])
              ─────────────────────────────────────────────────────────
                            Σ entropy(cell)

score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```

- **100** = perfect match with ground truth
- **0** = catastrophically wrong

Only high-entropy (uncertain) cells count heavily — static cells are effectively excluded.

### ⚠️ The most important rule: never use 0.0

KL divergence is undefined when your prediction assigns 0 probability to a class the ground truth considers possible. A single zero can destroy your entire cell score.

**Always apply a probability floor:**

```python
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)
```

### Per-round score
```
round_score = average of 5 seed scores
```
Missing seeds score 0 — always submit all 5.

### Leaderboard score
```
leaderboard_score = best round_score × round_weight (across all rounds)
```
Later rounds may have higher weights. A "hot streak score" (avg of last 3 rounds) is also tracked.

---

## 🔌 API Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| GET | `/rounds` | Public | List all rounds |
| GET | `/rounds/{round_id}` | Public | Round details + initial states |
| GET | `/budget` | Team | Your query budget for the active round |
| POST | `/simulate` | Team | Run one simulation through a viewport (costs 1 query) |
| POST | `/submit` | Team | Submit prediction tensor for a seed |
| GET | `/my-rounds` | Team | Your scores, rank, budget per round |
| GET | `/my-predictions/{round_id}` | Team | Your submitted predictions + argmax/confidence |
| GET | `/analysis/{round_id}/{seed_index}` | Team | Post-round comparison vs ground truth |
| GET | `/leaderboard` | Public | Global leaderboard |

### Rate limits

| Endpoint | Limit |
|---|---|
| POST `/simulate` | 5 req/sec per team |
| POST `/submit` | 2 req/sec per team |

Exceeding limits returns `429 Too Many Requests`. You also have a hard budget of **50 simulation queries per round**.

---

## 📐 Prediction Format

Submit a `height × width × 6` float tensor. Each cell holds 6 probabilities (one per class) that **must sum to 1.0** (±0.01 tolerance).

```
prediction[y][x] = [p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]
```

Resubmitting for the same seed overwrites your previous prediction — only the last submission counts.

### Common validation errors

| Error | Cause |
|---|---|
| `Expected H rows, got N` | Wrong number of rows |
| `Row Y: expected W cols, got N` | Wrong number of columns |
| `Cell (Y,X): expected 6 probs, got N` | Wrong probability vector length |
| `Cell (Y,X): probs sum to S, expected 1.0` | Doesn't sum to 1 |
| `Cell (Y,X): negative probability` | Negative value in vector |

---

## 📡 Simulation Mechanics (Summary)

The simulator runs 50 years of Norse civilisation. Each year cycles through:

1. **Growth** — settlements produce food, expand populations, found new settlements, build ports and longships
2. **Conflict** — settlements raid each other; desperate (low-food) settlements raid more aggressively; longships extend raiding range
3. **Trade** — ports trade with each other, generating wealth and diffusing technology
4. **Winter** — all settlements lose food; some collapse from starvation or sustained raids, becoming Ruins
5. **Environment** — ruins may be reclaimed by nearby settlements or slowly revert to forest/plains

The map seed (terrain layout) is visible. The hidden parameters controlling the world's behaviour are the same across all 5 seeds in a round — observations from one seed can inform predictions on others.

---

## 🛠️ Using the MCP Server (Claude Code)

Add the docs server to Claude Code for AI-assisted development:

```bash
claude mcp add --transport http nmiai https://mcp-docs.ainm.no/mcp
```

---

## 💡 Strategy Tips

- **Budget wisely** — 50 queries across 5 seeds = ~10 per seed. Prioritize uncertain or high-settlement-density regions.
- **Cross-seed learning** — hidden parameters are shared across seeds in a round. Patterns you observe in seed 0 apply to seed 4.
- **Never assign 0.0** — always enforce a minimum probability floor of `0.01` and renormalize.
- **Always submit all 5 seeds** — an unsubmitted seed scores 0, dragging down your round average.
- **Use the initial grid** — you can reconstruct the starting terrain locally from the map seed. Mountains never change; ocean never changes. Focus your queries on dynamic regions.
- **Post-round analysis** — after rounds complete, use `GET /analysis/{round_id}/{seed_index}` to compare your predictions against ground truth and improve your model.