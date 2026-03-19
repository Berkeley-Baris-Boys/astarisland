# Astar Island Solver

> **NM i AI 2026 — Norwegian AI Championship**  
> Observe a Norse civilisation simulator. Predict the world state. Outscore everyone.

## Workflow

```
GitHub (this repo)          Google Cloud Shell
─────────────────           ──────────────────
solver.py          ──→      upload solver.py
                            python solver.py --token $TOKEN
                            ↓
                            observations.json  (saved after every query)
                            predictions_seed_*.npy
                            round_N.log
```

Develop locally → upload to GCP Cloud Shell → run → analyse output → improve → repeat.

---

## Quick Start

```bash
# One-time setup in Cloud Shell
pip install requests numpy scipy google-cloud-aiplatform -q
gcloud config set project ai-nm26osl-1722
export TOKEN="your_jwt_token"   # app.ainm.no → F12 → Cookies → access_token

# Each round
rm -f observations.json predictions_seed_*.npy
python solver.py --token $TOKEN 2>&1 | tee round_N.log
```

| Flag | Purpose |
|------|---------|
| *(none)* | Full run: query → predict → submit |
| `--resume` | Load `observations.json`, rebuild predictions, submit |
| `--dry-run` | Everything except final submit |
| `--no-query` | Predict from initial state only, no queries used |
| `--check-only` | Build predictions + offline KL estimate, no submit |
| `--submit-only` | Load saved `.npy` files and submit directly |

---

## The Task

**Platform:** [app.ainm.no](https://app.ainm.no)  
**API base:** `https://api.ainm.no/astar-island`

Each round:
1. A 40×40 Norse world is generated with **hidden parameters** (faction aggression, forest growth, winter severity, trade activity)
2. A civilisation simulator runs for **50 years** — settlements grow, factions clash, forests spread, winters kill
3. You get **50 queries total** (shared across 5 seeds) to observe the final state through a 15×15 viewport
4. You submit a **40×40×6 probability tensor** per seed
5. Scored by **entropy-weighted KL divergence** against a ground truth computed from **hundreds of Monte Carlo runs**

The simulation is **stochastic** — same map + same parameters → different outcome every run. The ground truth is a true probability distribution, not a single outcome. You're matching that distribution.

---

## Terrain System — CRITICAL

### Internal codes vs prediction classes

The initial grid and simulate responses use **internal terrain codes**. The prediction tensor uses **6 class indices**. They are NOT the same numbers.

| Internal Code | Terrain | Prediction Class | Notes |
|--------------|---------|-----------------|-------|
| 10 | Ocean | **0 (Empty)** | Static. Borders map. Excluded from scoring. |
| 11 | Plains | **0 (Empty)** | Dynamic. Can become Settlement, Forest, Ruin. |
| 0 | Empty | **0 (Empty)** | Generic empty cell |
| 1 | Settlement | **1** | Active Norse village |
| 2 | Port | **2** | Coastal trading settlement |
| 3 | Ruin | **3** | Collapsed settlement |
| 4 | Forest | **4** | Provides food to adjacent settlements |
| 5 | Mountain | **5** | Static. Excluded from scoring. |

**Ocean (10), Plains (11), and Empty (0) all map to class index 0.**  
**Mountains and Ocean are static — they never change and are excluded from scoring.**

### What this means for predictions

For initial code 10 (Ocean): predict almost all class 0.  
For initial code 11 (Plains): predict a mix — Plains mostly stays Empty but can be settled.  
For initial code 5 (Mountain): predict almost all class 5 — but it doesn't even count toward score.

---

## Scoring — Full Details

### Ground truth
Pre-computed by running the simulation **hundreds of times** with the true hidden parameters. Each cell gets a probability distribution across 6 classes. Example:
```
cell (5, 12) ground truth: [0.0, 0.60, 0.25, 0.15, 0.0, 0.0]
# 60% Settlement, 25% Port, 15% Ruin after 50 years
```

### Entropy weighting
**Only dynamic cells contribute to score.** Static cells (Ocean, Mountain) have near-zero entropy and are excluded. Cells with more uncertain outcomes (higher entropy) count more.

### Score formula
```python
weighted_kl = sum(entropy(cell) * KL(ground_truth[cell], prediction[cell])) / sum(entropy(cell))
score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
```

- `100` = perfect prediction
- `0` = terrible prediction
- Uniform prediction (~1/6 each) scores ~1–5
- The exponential decay means improvements get harder as you approach 100

### Leaderboard
Your leaderboard score = **best single round score × round weight** across all rounds.  
Hot streak score = average of last 3 rounds.

### CRITICAL: Never assign 0.0 probability
If ground truth has `p > 0` for a class you predicted as `0.0`, KL divergence = infinity → that cell scores 0. Always apply:
```python
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)
```

---

## API Reference

### Auth
```python
session.headers["Authorization"] = f"Bearer {YOUR_JWT}"
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/rounds` | List all rounds |
| GET | `/rounds/{id}` | Round details + initial states |
| GET | `/budget` | Your remaining query budget |
| POST | `/simulate` | Query one viewport (costs 1 query) |
| POST | `/submit` | Submit prediction tensor for one seed |
| GET | `/my-rounds` | Your scores, ranks, budgets per round |
| GET | `/my-predictions/{round_id}` | Your submitted predictions with argmax grid |
| GET | `/analysis/{round_id}/{seed_index}` | **Post-round: your prediction vs ground truth** |
| GET | `/leaderboard` | Public leaderboard |

### Rate limits
- POST `/simulate`: 5 requests/second
- POST `/submit`: 2 requests/second

### Simulate request/response
```python
# Request
{
  "round_id": str, "seed_index": int,
  "viewport_x": int, "viewport_y": int,
  "viewport_w": int (5-15), "viewport_h": int (5-15)
}

# Response
{
  "grid": [[terrain_codes]],      # viewport_h × viewport_w, AFTER 50 years of simulation
  "settlements": [{               # only settlements within viewport
    "x": int, "y": int,
    "population": float,          # ONLY available here, not in initial_states
    "food": float,
    "wealth": float,
    "defense": float,
    "has_port": bool,
    "alive": bool,
    "owner_id": int               # faction — useful for detecting conflict
  }],
  "viewport": {"x", "y", "w", "h"},
  "queries_used": int,            # running total — use to track budget
  "queries_max": 50
}
```

### Submit request
```python
{
  "round_id": str,
  "seed_index": int,              # 0–4
  "prediction": [[[6 floats]]]   # height × width × 6, each cell sums to 1.0 ±0.01
}
# class order: [Empty/Ocean/Plains, Settlement, Port, Ruin, Forest, Mountain]
```

### Analysis endpoint (post-round gold)
```python
GET /analysis/{round_id}/{seed_index}
# Returns your prediction AND the ground truth distribution
# Use this after every round to measure exactly where you went wrong
```

---

## Simulation Mechanics

### World phases each year (50 total)
1. **Growth** — settlements produce food from adjacent terrain, grow population, found new settlements on nearby land, build ports on coasts
2. **Conflict** — settlements raid neighbours; longships extend range; desperate settlements (low food) raid more; conquered settlements change faction
3. **Trade** — ports in range trade if not at war; generates wealth and food; diffuses technology
4. **Winter** — all settlements lose food; settlements collapse from starvation/raids/harsh winters → become Ruins
5. **Environment** — ruins reclaimed by nearby settlements or overtaken by forest growth; plains slowly revert to forest if unclaimed

### Hidden parameters (same for all 5 seeds per round)
| Parameter | Effect |
|-----------|--------|
| `faction_aggression` | How often settlements raid each other |
| `forest_growth_rate` | How fast forests spread and reclaim ruins/plains |
| `winter_severity` | How often harsh winters kill settlements |
| `trade_activity` | How often ports form trade routes |

### Settlement properties (visible via simulate)
`population`, `food`, `wealth`, `defense`, `tech_level`, `has_port`, `alive`, `owner_id`  
Initial states only expose `x`, `y`, `has_port`, `alive` — internal stats require queries.

---

## Round History

| Round | Score | Rank | Queries | Key Issue |
|-------|-------|------|---------|-----------|
| 1 | 0.1 pts | #113/117 | 46/50 | Crashed on terrain code >5 bug. Lost all query data. Submitted uniform prediction. |
| 2 | 0.2 pts | TBD | 50/50 | 100% coverage but wrong terrain code mapping. Codes 10/11 predicted wrong. |

### Why Round 2 scored only 0.2

We got 100% map coverage but **misunderstood the terrain code mapping**:

| What we predicted | What we should have predicted |
|-------------------|-------------------------------|
| Code 10: `[0.35, 0.15, 0.12, 0.05, 0.18, 0.05]` | Code 10 (Ocean): `[0.96, 0.01, 0.01, 0.01, 0.01, 0.00]` |
| Code 11: `[0.88, 0.02, 0.04, 0.02, 0.02, 0.02]` | Code 11 (Plains): `[0.84, 0.13, 0.00, 0.01, 0.02, 0.00]` |

Codes 10 and 11 cover **62% of the map**. Getting those wrong dominates the score.

Also: Settlement → Plains (class 0) transition at 46% was not being captured at all.

### Round 2 empirical transitions (useful prior for similar rounds)
```
Code 10 (Ocean)      → 0(Empty): 100%
Code 11 (Plains)     → 0(Empty): 84%,  1(Settlement): 13%, 4(Forest): 2%
Code  1 (Settlement) → 0(Empty): 46%,  1(Settlement): 46%, 4(Forest): 8%
Code  4 (Forest)     → 4(Forest): 86%, 1(Settlement): 9%,  0(Empty): 3%, 2(Port): 2%
Code  5 (Mountain)   → 5(Mountain): 100%
Settlement survival: 100% alive — faction_aggression ≈ 0, winter_severity ≈ 0
```

---

## Key Fixes Needed in Solver

### Fix 1: Correct terrain code → class mapping in `build_predictions()`

```python
# WRONG (what we had):
if init_class == 11:
    context = np.array([0.88, 0.02, 0.04, 0.02, 0.02, 0.02])  # Wrong!
elif init_class == 10:
    context = np.array([0.35, 0.15, 0.12, 0.05, 0.18, 0.05])  # Wrong!

# CORRECT:
if init_class == 10:
    # Ocean — static, maps to class 0 (Empty), excluded from scoring
    context = np.array([0.96, 0.01, 0.01, 0.01, 0.01, 0.00])
elif init_class == 11:
    # Plains — maps to class 0 (Empty), but dynamic — can be settled
    # Prior from Round 2: 84% stays plains/empty, 13% becomes settlement
    context = np.array([0.84, 0.13, 0.00, 0.01, 0.02, 0.00])
```

### Fix 2: Use the analysis endpoint after every round

```python
# Get ground truth for free after round closes
resp = session.get(f"{BASE}/analysis/{round_id}/{seed_index}")
data = resp.json()
ground_truth = np.array(data["ground_truth"])  # H×W×6 true distribution
your_pred = np.array(data["prediction"])        # what you submitted
# Compare to find exactly where your model was wrong
```

### Fix 3: Use `queries_used` from simulate response

The simulate response includes `queries_used` — use this instead of a local counter to avoid budget surprises.

### Fix 4: Use `population` and `owner_id` from settlements

The simulate response exposes settlement stats not available in initial_states:
- `population` — low population → likely to collapse this winter
- `food` — near zero → high probability of becoming Ruin
- `owner_id` — faction tracking → detect conflicts between factions

---

## Post-Round Analysis

Run this immediately after a round closes:

```bash
python3 - <<'EOF'
import json, requests, os, numpy as np
from collections import defaultdict, Counter

TOKEN = os.environ["TOKEN"]
s = requests.Session()
s.headers["Authorization"] = f"Bearer {TOKEN}"
data = json.load(open("observations.json"))
round_id = data["round_id"]
detail = s.get(f"https://api.ainm.no/astar-island/rounds/{round_id}").json()

# 1. Transition rates
classes = {0:"Empty",1:"Settlement",2:"Port",3:"Ruin",4:"Forest",5:"Mountain",10:"Ocean",11:"Plains"}
transitions = defaultdict(lambda: defaultdict(int))
for i, state in enumerate(detail["initial_states"]):
    for y, row in enumerate(state["grid"]):
        for x, iv in enumerate(row):
            fv = data["latest"][str(i)][y][x]
            if fv is not None:
                transitions[iv][fv] += 1

print("=== TRANSITION RATES ===")
for ic in sorted(transitions):
    total = sum(transitions[ic].values())
    print(f"{ic} ({classes.get(ic,'?')}):")
    for fc, cnt in sorted(transitions[ic].items(), key=lambda x: -x[1]):
        pct = 100*cnt/total
        if pct >= 1.0:
            print(f"  → {fc} ({classes.get(fc,'?')}): {pct:.1f}%")

# 2. Settlement survival
print("\n=== SETTLEMENT SURVIVAL ===")
for seed_str, snaps in data["settlements"].items():
    alive = sum(1 for s in snaps for st in s.get("settlements",[]) if st.get("alive",True))
    dead  = sum(1 for s in snaps for st in s.get("settlements",[]) if not st.get("alive",True))
    print(f"  Seed {seed_str}: {alive} alive / {alive+dead} total")

# 3. Ground truth comparison (post-round only)
print("\n=== GROUND TRUTH vs YOUR PREDICTION (seed 0) ===")
resp = s.get(f"https://api.ainm.no/astar-island/analysis/{round_id}/0")
if resp.status_code == 200:
    analysis = resp.json()
    gt = np.array(analysis["ground_truth"])
    pred = np.array(analysis["prediction"])
    diff = np.abs(gt - pred).mean(axis=2)  # H×W mean absolute difference
    worst_cells = np.unravel_index(np.argsort(diff.ravel())[-10:], diff.shape)
    print(f"  Score: {analysis['score']}")
    print(f"  Mean abs diff: {diff.mean():.4f}")
    print("  Worst 10 cells:")
    for y, x in zip(*worst_cells):
        print(f"    ({x},{y}): gt={np.round(gt[y,x],2)} pred={np.round(pred[y,x],2)}")
else:
    print(f"  Not available yet: {resp.status_code}")
EOF
```

---

## Improvement Roadmap

### Round 3 priorities (high impact)
1. **Fix code 10/11 priors** — Ocean = class 0 static, Plains = class 0 dynamic with correct transition rates
2. **Use analysis endpoint** — fetch ground truth after round 2 closes, compute exact per-class error, retune priors
3. **Use settlement stats** — `population`, `food`, `defense` from simulate response predict collapse probability
4. **Use `owner_id`** — settlements of different factions nearby → higher Ruin probability
5. **Re-query high-entropy cells** — spend reserve queries re-sampling cells that showed mixed outcomes, not just unseen area

### Medium impact
- Learn code 10/11 transition rates fresh each round — don't hardcode Round 2 values
- Submit interim predictions after 25 queries, improve and resubmit after 50
- Build a proper Bayesian update from multiple stochastic observations of the same cell

### Low impact
- Visualise your argmax grid vs initial grid using `/my-predictions/{round_id}` 
- Auto-tune smoothing and alpha parameters using `--check-only` offline self-check

---

## Bugs Fixed

| Bug | Symptom | Fix |
|-----|---------|-----|
| Terrain code >5 crashes counts array | `IndexError` at query 1 | `MAX_CLASS=16`, `[:CLASS_COUNT]` slicing |
| Terrain code >5 crashes transition matrix | `ValueError: shapes (6,) (16,)` | Slice `[:, :CLASS_COUNT]` in all matrix ops |
| `observations.json` lost on crash | All data gone if crash | Save after every single query |
| Budget exhaustion crashes solver | Dies instead of submitting | Catch `BudgetExhaustedError`, proceed to predict |
| Reserve queries waste budget | Queries 46–50 got 0 new cells | `viewport_hits` revisit penalty |
| Alpha too high | Single observation got 27% weight | `alpha = max(0.30, 1.2/n_obs)` |
| Gemini model 404 | `gemini-2.0-flash-001` not found | Updated to `gemini-2.0-flash-002` |
| `has_port` variable shadowing | Wrong port boost | Renamed to `s_has_port` in loop |
| `hidden_params` not returned | Gemini result discarded | Added `nonlocal hidden_params` |
| **Code 10/11 wrong class mapping** | **0.2 score despite 100% coverage** | **Fix: Ocean→class 0 static, Plains→class 0 dynamic** |

---

## Environment

- **GCP Project:** `ai-nm26osl-1722`
- **Cloud Shell Python:** 3.12
- **Vertex AI region:** `us-central1`
- **Gemini model:** `gemini-2.0-flash-002`
- **Dependencies:** `requests numpy scipy google-cloud-aiplatform`