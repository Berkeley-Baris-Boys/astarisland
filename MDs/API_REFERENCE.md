# Astar Island — API Reference

A reference guide for the Astar Island prediction API. All endpoints live under:

```
https://api.ainm.no/astar-island
```

---

## Authentication

Every team endpoint requires a JWT token, passed as either:

- A cookie: `Cookie: access_token` (set automatically on login at `app.ainm.no`)
- A bearer header: `Authorization: Bearer <token>`

Both methods use the same token — use whichever suits your setup.

---

## Endpoints at a Glance

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/rounds` | Public | List all rounds |
| `GET` | `/rounds/{round_id}` | Public | Round details + initial states |
| `GET` | `/budget` | Team | Remaining query budget for active round |
| `POST` | `/simulate` | Team | Run one simulation through a viewport |
| `POST` | `/submit` | Team | Submit a prediction tensor |
| `GET` | `/my-rounds` | Team | Your scores, rank, and budget per round |
| `GET` | `/my-predictions/{round_id}` | Team | Your predictions with argmax & confidence |
| `GET` | `/analysis/{round_id}/{seed_index}` | Team | Post-round ground truth comparison |
| `GET` | `/leaderboard` | Public | All-time leaderboard |

---

## Round Lifecycle

| Status | Meaning |
|--------|---------|
| `pending` | Round created, not yet started |
| `active` | Queries and submissions open |
| `scoring` | Submissions closed, scoring in progress |
| `completed` | Scores finalised |

---

## GET `/rounds`

List all rounds with status and timing.

```json
[
  {
    "id": "uuid",
    "round_number": 1,
    "event_date": "2026-03-19",
    "status": "active",
    "map_width": 40,
    "map_height": 40,
    "prediction_window_minutes": 165,
    "started_at": "2026-03-19T10:00:00Z",
    "closes_at": "2026-03-19T12:45:00Z",
    "round_weight": 1,
    "created_at": "2026-03-19T09:00:00Z"
  }
]
```

---

## GET `/rounds/{round_id}`

Returns round details including **initial map states for all seeds**. Use this to reconstruct the starting terrain locally.

> Settlement data in initial states exposes only position and port status. Internal stats (population, food, wealth, defense) are not included.

```json
{
  "id": "uuid",
  "round_number": 1,
  "status": "active",
  "map_width": 40,
  "map_height": 40,
  "seeds_count": 5,
  "initial_states": [
    {
      "grid": [[10, 10, 10], ["..."]],
      "settlements": [
        { "x": 5, "y": 12, "has_port": true, "alive": true }
      ]
    }
  ]
}
```

### Grid Cell Values

| Value | Terrain |
|-------|---------|
| `0` | Empty |
| `1` | Settlement |
| `2` | Port |
| `3` | Ruin |
| `4` | Forest |
| `5` | Mountain |
| `10` | Ocean |
| `11` | Plains |

---

## GET `/budget`

Check your team's remaining query budget for the active round.

```json
{
  "round_id": "uuid",
  "queries_used": 23,
  "queries_max": 50,
  "active": true
}
```

---

## POST `/simulate`

The core observation endpoint. Each call runs **one stochastic simulation** and reveals a rectangular viewport of the result. Costs **one query** from your budget (50 per round). Each call uses a different random `sim_seed`, so repeated calls on the same seed give different stochastic outcomes.

### Rate limit: 5 requests/second per team

### Request

```json
{
  "round_id": "uuid-of-active-round",
  "seed_index": 3,
  "viewport_x": 10,
  "viewport_y": 5,
  "viewport_w": 15,
  "viewport_h": 15
}
```

| Field | Type | Description |
|-------|------|-------------|
| `round_id` | string | UUID of the active round |
| `seed_index` | int (0–4) | Which of the 5 seeds to simulate |
| `viewport_x` | int (≥0) | Left edge of viewport (default `0`) |
| `viewport_y` | int (≥0) | Top edge of viewport (default `0`) |
| `viewport_w` | int (5–15) | Viewport width (default `15`) |
| `viewport_h` | int (5–15) | Viewport height (default `15`) |

### Response

```json
{
  "grid": [[4, 11, 1], ["..."]],
  "settlements": [
    {
      "x": 12, "y": 7,
      "population": 2.8,
      "food": 0.4,
      "wealth": 0.7,
      "defense": 0.6,
      "has_port": true,
      "alive": true,
      "owner_id": 3
    }
  ],
  "viewport": { "x": 10, "y": 5, "w": 15, "h": 15 },
  "width": 40,
  "height": 40,
  "queries_used": 24,
  "queries_max": 50
}
```

- `grid` contains only the viewport region (`viewport_h × viewport_w`), not the full map.
- `settlements` includes only settlements within the viewport.
- `viewport` confirms the actual bounds (clamped to map edges if needed).

### Error Codes

| Status | Meaning |
|--------|---------|
| `400` | Round not active, or invalid `seed_index` |
| `403` | Not on a team |
| `404` | Round not found |
| `429` | Budget exhausted (50/50) or rate limit exceeded |

---

## POST `/submit`

Submit your prediction for one seed. **You must submit all 5 seeds for a complete score.** Resubmitting for the same seed overwrites the previous prediction — only the last submission counts.

### Rate limit: 2 requests/second per team

### Request

```json
{
  "round_id": "uuid-of-active-round",
  "seed_index": 3,
  "prediction": [
    [
      [0.85, 0.05, 0.02, 0.03, 0.03, 0.02],
      ["..."]
    ]
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `round_id` | string | UUID of the active round |
| `seed_index` | int (0–4) | Which seed this prediction is for |
| `prediction` | float[][][] | `H × W × 6` tensor — probability per cell per class |

### Prediction Format

The prediction tensor is indexed as `prediction[y][x][class]`:

- Outer dimension: `H` rows
- Middle dimension: `W` columns
- Inner dimension: 6 class probabilities

Each cell's 6 probabilities must **sum to 1.0** (±0.01 tolerance), and all values must be **non-negative**.

### Class Indices

| Index | Class |
|-------|-------|
| `0` | Empty (Ocean, Plains, Empty) |
| `1` | Settlement |
| `2` | Port |
| `3` | Ruin |
| `4` | Forest |
| `5` | Mountain |

### Response

```json
{ "status": "accepted", "round_id": "uuid", "seed_index": 3 }
```

### Validation Errors

| Error | Cause |
|-------|-------|
| `Expected H rows, got N` | Wrong number of rows |
| `Row Y: expected W cols, got N` | Wrong number of columns |
| `Cell (Y,X): expected 6 probs, got N` | Wrong probability vector length |
| `Cell (Y,X): probs sum to S, expected 1.0` | Probabilities don't sum to 1.0 |
| `Cell (Y,X): negative probability` | Negative value in probability vector |

---

## GET `/my-rounds`

All rounds enriched with your team's scores, rank, and query usage. Team-specific version of `/rounds`.

```json
[
  {
    "id": "uuid",
    "round_number": 1,
    "status": "completed",
    "round_score": 72.5,
    "seed_scores": [80.1, 65.3, 71.9],
    "seeds_submitted": 5,
    "rank": 3,
    "total_teams": 12,
    "queries_used": 48,
    "queries_max": 50,
    "initial_grid": [[10, 10, 10], ["..."]]
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `round_score` | float \| null | Average score across all seeds (null if unscored) |
| `seed_scores` | float[] \| null | Per-seed scores (null if unscored) |
| `seeds_submitted` | int | Number of seeds with submitted predictions |
| `rank` | int \| null | Your rank for this round (null if unscored) |
| `total_teams` | int \| null | Total teams scored in this round |
| `queries_used` | int | Simulation queries used by your team |
| `queries_max` | int | Maximum queries allowed (default 50) |
| `initial_grid` | int[][] | Initial terrain grid for the first seed |

---

## GET `/my-predictions/{round_id}`

Your submitted predictions for a given round, with derived `argmax` and `confidence` grids for easy visualisation.

```json
[
  {
    "seed_index": 0,
    "argmax_grid": [[0, 4, 5], ["..."]],
    "confidence_grid": [[0.85, 0.72, 0.93], ["..."]],
    "score": 78.2,
    "submitted_at": "2026-03-19T10:30:00+00:00"
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `argmax_grid` | int[][] | `H × W` grid of predicted class indices (argmax of probability vector) |
| `confidence_grid` | float[][] | `H × W` grid of max probability per cell (rounded to 3 decimal places) |
| `score` | float \| null | Score for this seed (null if unscored) |
| `submitted_at` | string \| null | ISO 8601 submission timestamp |

---

## GET `/analysis/{round_id}/{seed_index}`

Post-round analysis. Returns your prediction alongside the **ground truth** for detailed comparison. Only available once a round reaches `completed` or `scoring` status.

```json
{
  "prediction": [[["..."]]], 
  "ground_truth": [[["..."]]],
  "score": 78.2,
  "width": 40,
  "height": 40,
  "initial_grid": [[10, 10, 10], ["..."]]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | float[][][] | Your submitted `H × W × 6` probability tensor |
| `ground_truth` | float[][][] | Actual `H × W × 6` distribution computed from Monte Carlo simulations |
| `score` | float \| null | Your score for this seed |
| `initial_grid` | int[][] \| null | Initial terrain grid for this seed |

### Error Codes

| Status | Meaning |
|--------|---------|
| `400` | Round not yet completed/scoring, or invalid `seed_index` |
| `403` | Not on a team |
| `404` | Round not found |

---

## GET `/leaderboard`

Public all-time leaderboard. Each team's score is their **best round score**, weighted by `round_weight`.

```json
[
  {
    "team_id": "uuid",
    "team_name": "Vikings ML",
    "team_slug": "vikings-ml",
    "weighted_score": 72.5,
    "rounds_participated": 3,
    "hot_streak_score": 78.1,
    "rank": 1,
    "is_verified": true
  }
]
```

| Field | Type | Description |
|-------|------|-------------|
| `weighted_score` | float | Best `round_score × round_weight` across all rounds |
| `rounds_participated` | int | Total rounds with submitted predictions |
| `hot_streak_score` | float | Average score over the last 3 rounds |
| `is_verified` | bool | Whether all team members are Vipps-verified |
| `rank` | int | Current leaderboard rank |