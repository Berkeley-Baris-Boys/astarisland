# Astar Island Competition Baseline

## What This Does

This repository implements a practical baseline for the Astar Island Norse civilisation prediction challenge. It is built around the documented constraints:

- 5 seeds per round on `40x40` maps
- 50 total simulation queries per round
- shared hidden simulator tendencies across all 5 seeds
- entropy-weighted KL scoring on full per-cell probability distributions
- heavy penalty for zero probabilities

The system combines:

- deterministic priors for static physics and obvious terrain rules
- adaptive viewport selection over likely dynamic regions
- cross-seed evidence aggregation to infer round-level tendencies
- feature-based transfer from observed windows to unseen cells
- calibrated probability floors and final validation
- optional learned feature-conditioned priors from completed rounds via the analysis endpoint
- optional tree-based residual calibration learned from archived prediction and ground-truth pairs

## Markdown Facts Used As Primary Context

The implementation follows the attached docs in `MDs/`:

- `TASK_DESCRIPTION.md`
- `API_REFERENCE.md`
- `SIMULATION_MECHANICS.md`
- `SCORING.md`
- `QUICKSTART.md`
- `CONCRETE_TWEAKS.md`

Key facts encoded in the baseline:

- `GET /rounds/{round_id}` exposes the full initial terrain plus settlement positions and port flags.
- `POST /simulate` returns one stochastic viewport outcome and detailed settlement stats within that window.
- Mountains are immutable, ocean borders are effectively static, and only classes `1/2/3/4` are meaningfully dynamic.
- Ports require coast access.
- Hidden simulator parameters are shared across seeds, so evidence is aggregated round-wide.
- Final predictions must be `H x W x 6`, non-negative, sum to 1, and should never include zeros.

## Project Layout

```text
src/astar_island/
  api.py
  aggregator.py
  config.py
  features.py
  learned_prior.py
  predictor.py
  query_planner.py
  residual_calibrator.py
  scoring.py
  submit.py
  types.py
  utils.py
  visualize.py
scripts/
  archive_completed_rounds.py
  build_learned_prior.py
  build_residual_calibrator.py
  evaluate_residual_calibrator.py
  run_round.py
  analyze_round.py
  inspect_predictions.py
requirements.txt
```

## Baseline Design

### 1. Deterministic Priors

The predictor starts from visible map structure:

- mountain cells are clamped to near-certain mountain
- ocean and hard static cells heavily favor class `0`
- inland cells are discouraged from becoming ports
- initial settlements/ports/ruins receive transition priors centered on settlement, port, and ruin classes

### 2. Feature Engineering

Per seed, the code computes:

- buildable and dynamic masks
- coastal adjacency
- distance to nearest initial settlement
- distance to coast and ruin
- local settlement / forest / mountain densities
- expansion frontier and conflict corridor proxies
- reclaimable ruin zones
- settlement intensity map from a Gaussian prior around known initial settlements

### 3. Query Planning

Queries are allocated in three phases:

- `recon`: broad coverage of settlement-heavy, coastal, and frontier regions
- `targeted`: focus on conflict corridors, ambiguous frontiers, and under-observed high-value regions
- `calibration`: repeat high-entropy windows to estimate stochasticity and sharpen local distributions

The planner scores candidate windows using feature density, current coverage, observed entropy, overlap penalties, and per-seed fairness.

### 4. Observation Aggregation

Every viewport updates:

- per-cell class counts
- per-cell observation counts
- bucketed feature-conditioned class counts
- round-wide settlement stat summaries from query-visible internals

Those summaries produce latent round-level tendencies such as:

- settlement activity rate
- port share among active cells
- ruin share among active cells
- forest share among dynamic cells
- average visible food, wealth, defense, and population

### 5. Prediction

For each seed:

1. Build a terrain-aware prior distribution.
2. Add transfer mass from:
   - same-seed locally observed cells using a spatial kernel
   - round-wide feature buckets learned across all seeds
   - global class prevalence from all observations
3. Blend in direct empirical counts where cells were actually observed.
4. Apply the settlement shape prior from `CONCRETE_TWEAKS.md`.
5. Enforce physical constraints and apply a probability floor.

This is not a learned offline model; it is a round-adaptive empirical Bayes baseline designed for the challenge’s limited-query setting.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If you prefer not to install editable mode:

```bash
export PYTHONPATH=src
```

## Authentication

Set your JWT token from `app.ainm.no`:

```bash
export ASTAR_ISLAND_TOKEN="your_jwt_here"
export ASTAR_ISLAND_AUTH_MODE="bearer"   # or cookie
```

Optional historical prior artifact path:

```bash
export ASTAR_ISLAND_HISTORICAL_PRIOR_PATH="artifacts/historical_priors.json"
export ASTAR_ISLAND_HISTORY_DIR="artifacts/history"
```

## Learn Priors From Completed Rounds

Archive completed rounds in a structured format:

```bash
PYTHONPATH=src python scripts/archive_completed_rounds.py --max-rounds 9
```

This creates a directory structure like:

```text
artifacts/history/
  manifest.json
  round_09_<round_id>/
    round_detail.json
    ground_truth_priors.json
    seed_0/
      analysis.json
      ground_truth.npy
      initial_grid.npy
      summary.json
```

If completed-round analysis is available for your team, build a historical prior artifact first:

```bash
PYTHONPATH=src python scripts/build_historical_priors.py --max-rounds 9
```

To refresh the archive and rebuild in one command:

```bash
PYTHONPATH=src python scripts/build_historical_priors.py --max-rounds 9 --refresh-archive
```

This fetches completed rounds, stores the raw per-seed analysis plus normalized `.npy` tensors, buckets cells by structural features, and stores aggregated class counts in a local artifact. The live predictor automatically loads that artifact if it exists.

Build the learned static prior:

```bash
PYTHONPATH=src python scripts/build_learned_prior.py
```

Build the residual calibrator:

```bash
PYTHONPATH=src python scripts/build_residual_calibrator.py
```

Evaluate the residual calibrator with leave-one-round-out exact scoring:

```bash
PYTHONPATH=src python scripts/evaluate_residual_calibrator.py --rounds 8 9 10 11 12
```

## Run The Full Pipeline

Dry run without submission:

```bash
PYTHONPATH=src python scripts/run_round.py --no-submit
```

Live run with submission:

```bash
PYTHONPATH=src python scripts/run_round.py
```

Submit predictions from an earlier `--no-submit` run without using any more simulation budget:

```bash
PYTHONPATH=src python scripts/submit_saved_predictions.py artifacts/<run_dir>
```

Artifacts are written under `artifacts/<timestamp>/`:

- `run.log`
- `active_round.json`
- `round_detail.json`
- `metadata.json`
- `observations.json`
- `query_events.jsonl`
- `submission_events.jsonl`
- `class_counts.npy`
- `observation_counts.npy`
- `conditional_counts.json`
- `prediction_seed_*.npy`
- `diagnostics/index.json`
- `diagnostics/seed_<n>/summary.json`
- `diagnostics/seed_<n>/*.npy` for stage tensors like `prior`, `transfer`, `combined`, `learned_prior`, `post_structural`, `rare_port_support`, `rare_ruin_support`, and `final_prediction`
- `artifacts/learned_prior.json`
- `artifacts/residual_calibrator.joblib`
- visualization PNGs

## Inspection Utilities

Inspect a saved prediction tensor:

```bash
PYTHONPATH=src python scripts/inspect_predictions.py artifacts/<run>/prediction_seed_0.npy
```

Fetch post-round analysis:

```bash
PYTHONPATH=src python scripts/analyze_round.py <round_id> --seed 0
```

## Important Practical Notes

- The API client supports bearer or cookie auth.
- Retries and `429` handling are built in.
- Public endpoints can be disk-cached for development.
- Prediction tensors are validated before submission.
- The predictor uses a non-zero floor everywhere to avoid catastrophic KL failures.
- Query execution is logged in structured JSON as well as human-readable logs.

## Current Limitations

- No historical backtesting dataset is included here.
- The transfer model is heuristic rather than a trained probabilistic classifier.
- Query planning is adaptive but still hand-engineered.
- The post-round analysis script assumes the analysis payload can be serialized directly; adjust if the live schema differs.

## Highest-Leverage Next Improvements

1. Weight historical-prior fitting by entropy and per-cell difficulty instead of raw cell count.
2. Add a query planner that explicitly optimizes information gain under shared hidden parameters.
3. Fit a lightweight per-class logistic or isotonic calibration layer on post-round data.
4. Add a local simulator mock from archived rounds for regression testing.
