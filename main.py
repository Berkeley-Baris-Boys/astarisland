#!/usr/bin/env python3
"""
Astar Island — Norse Civilisation Prediction Pipeline
NM i AI 2026

Simulator-backed pipeline:
  1. Spend the query budget on broad coverage plus selective re-sampling
  2. Pool observations across seeds to infer round-level hidden dynamics
  3. Run local Monte Carlo rollouts for each seed using the inferred regime
  4. Blend rollout probabilities with direct empirical counts from observations
  5. Submit H×W×6 tensors for all 5 seeds

Usage:
  export ASTAR_TOKEN="<your JWT>"

  python main.py                   # Full pipeline: query → predict → submit
  python main.py --resume          # Continue from saved observations.json
  python main.py --submit-only     # Resubmit saved predictions_seed_*.npy
  python main.py --dry-run         # Everything except actual submission
  python main.py --check-only      # Build + self-check, no submission
  python main.py --no-query        # Skip querying (use saved observations or priors)
  python main.py --fetch-analysis  # Fetch ground truth after round completes
  python main.py --show-history    # Print cross-round metrics summary and exit
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from api import AstarAPI, APIError, BudgetExhausted
from config import TOTAL_BUDGET, N_CLASSES, OBSERVATIONS_FILE, PRED_TEMPLATE
from state import ObservationStore, plan_core_queries, all_viewports
from world_dynamics import estimate_world_dynamics
from predictor import build_predictions, validate_prediction
from metrics import MetricsLogger, holdout_self_check, fetch_and_log_analysis


# ── Load round ────────────────────────────────────────────────────────────────

def load_active_round(api: AstarAPI) -> tuple[str, int, int, int, float, list[dict]]:
    active = api.get_active_round()
    if active is None:
        raise SystemExit("No active round found. Check app.ainm.no.")
    round_id = str(active["id"])
    detail   = api.get_round_detail(round_id)
    W        = int(detail["map_width"])
    H        = int(detail["map_height"])
    seeds    = int(detail["seeds_count"])
    rn       = active.get("round_number", "?")
    wt       = float(active.get("round_weight", detail.get("round_weight", 1.0)))
    states   = detail["initial_states"]
    if not isinstance(states, list) or len(states) != seeds:
        raise SystemExit("initial_states missing or wrong length")
    print(f"Round {rn}  weight={wt}  map={W}×{H}  seeds={seeds}")
    return round_id, W, H, seeds, wt, states


# ── Query phase ───────────────────────────────────────────────────────────────

def run_query_phase(
    api: AstarAPI,
    store: ObservationStore,
    initial_states: list[dict],
    budget: int = TOTAL_BUDGET,
) -> None:
    """
    Execute core tiling + adaptive reserve queries.

    Core: non-overlapping 15×15 tiles cover every seed fully (~45 queries for 5 seeds).
    Reserve: re-sample highest-entropy viewports to improve empirical distributions.
    """
    core_plan = plan_core_queries(store.width, store.height, store.seeds_count,
                                   initial_states=initial_states, budget=budget)
    viewports = all_viewports(store.width, store.height)
    vp_hits: dict[int, dict] = {s: {} for s in range(store.seeds_count)}

    def execute(seed: int, vx: int, vy: int, vw: int, vh: int) -> None:
        result = api.simulate(store.round_id, seed, vx, vy, vw, vh)
        new_cells = store.ingest(seed, vx, vy, result)
        vp_hits[seed][(vx, vy, vw, vh)] = vp_hits[seed].get((vx, vy, vw, vh), 0) + 1
        budget_used = result.get("queries_used", store.queries_used)
        budget_max  = result.get("queries_max", budget)
        print(f"  Q{store.queries_used:2d}/{budget}: seed={seed} "
              f"({vx},{vy},{vw}×{vh}) +{new_cells} new  [{budget_used}/{budget_max}]")
        store.save()

    # Core coverage
    for seed, vx, vy, vw, vh in core_plan:
        if store.queries_used >= budget:
            break
        try:
            execute(seed, vx, vy, vw, vh)
        except BudgetExhausted:
            print("Budget exhausted in core queries.")
            break
        except APIError as exc:
            print(f"API error (continuing): {exc}")

    # Reserve: re-sample areas with most uncertainty
    while store.queries_used < budget:
        # Prioritise seeds with incomplete coverage first
        coverages = {s: store.coverage(s) for s in range(store.seeds_count)}
        if min(coverages.values()) < 0.99:
            seed = min(coverages, key=coverages.get)
        else:
            # All seeds fully covered — pick seed with highest average cell entropy
            seed = max(
                range(store.seeds_count),
                key=lambda s: _mean_entropy(store.counts[s]),
            )
        vp = store.best_reserve_viewport(seed, viewports, vp_hits[seed])
        vx, vy, vw, vh = vp
        try:
            execute(seed, vx, vy, vw, vh)
        except BudgetExhausted:
            print("Budget exhausted in reserve queries.")
            break
        except APIError as exc:
            print(f"API error (continuing): {exc}")

    store.print_summary()


def _mean_entropy(counts: np.ndarray) -> float:
    tot = counts.sum(axis=2, keepdims=True).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tot > 0, counts / tot, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0), axis=2)
    return float(ent.mean())


# ── Submission ────────────────────────────────────────────────────────────────

def submit_predictions(
    api: AstarAPI,
    round_id: str,
    predictions: dict[int, np.ndarray],
    dry_run: bool = False,
) -> dict[int, str]:
    print("\nSubmitting...")
    statuses: dict[int, str] = {}
    for seed in sorted(predictions.keys()):
        pred = predictions[seed]
        try:
            validate_prediction(pred, seed)
        except ValueError as exc:
            print(f"  Seed {seed}: VALIDATION FAILED — {exc}")
            statuses[seed] = f"validation_failed: {exc}"
            continue
        if dry_run:
            print(f"  Seed {seed}: DRY RUN")
            statuses[seed] = "dry_run"
            continue
        try:
            resp = api.submit(round_id, seed, pred.tolist())
            status = str(resp.get("status", resp))
            print(f"  Seed {seed}: {status}")
            statuses[seed] = status
        except APIError as exc:
            print(f"  Seed {seed}: ERROR — {exc}")
            statuses[seed] = f"error: {exc}"
    return statuses


def save_predictions(predictions: dict[int, np.ndarray]) -> None:
    for seed, pred in predictions.items():
        path = PRED_TEMPLATE.format(seed=seed)
        np.save(path, pred)


def load_predictions(seeds_count: int) -> dict[int, np.ndarray]:
    preds: dict[int, np.ndarray] = {}
    for seed in range(seeds_count):
        path = Path(PRED_TEMPLATE.format(seed=seed))
        if not path.exists():
            raise SystemExit(f"Missing: {path}")
        preds[seed] = np.load(path)
    return preds


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Astar Island pipeline")
    parser.add_argument("--token",          default=None)
    parser.add_argument("--resume",         action="store_true", help="Load saved observations.json")
    parser.add_argument("--submit-only",    action="store_true", help="Resubmit saved predictions")
    parser.add_argument("--no-query",       action="store_true", help="Skip queries")
    parser.add_argument("--dry-run",        action="store_true", help="No actual submission")
    parser.add_argument("--check-only",     action="store_true", help="Build + self-check only")
    parser.add_argument("--self-check",     action="store_true", help="Holdout self-check")
    parser.add_argument("--fetch-analysis", action="store_true", help="Fetch ground truth")
    parser.add_argument("--show-history",   action="store_true", help="Print cross-round summary")
    args = parser.parse_args()

    token = args.token or os.getenv("ASTAR_TOKEN") or os.getenv("ASTAR_ISLAND_TOKEN")
    if not token:
        parser.error("Set ASTAR_TOKEN env var or pass --token.")

    api    = AstarAPI(token)
    logger = MetricsLogger()

    if args.show_history:
        logger.print_cross_round_summary()
        return

    # Load round
    try:
        round_id, W, H, seeds_count, round_weight, initial_states = load_active_round(api)
    except APIError as exc:
        raise SystemExit(f"Failed to load round: {exc}") from exc

    logger.log_round_start(round_id, round_id, W, H, seeds_count, round_weight)

    # Post-round analysis mode
    if args.fetch_analysis:
        predictions = {}
        for s in range(seeds_count):
            p = Path(PRED_TEMPLATE.format(seed=s))
            if p.exists():
                predictions[s] = np.load(p)
        fetch_and_log_analysis(api, round_id, seeds_count, predictions, logger)
        return

    # Submit-only
    if args.submit_only:
        predictions = load_predictions(seeds_count)
        statuses = submit_predictions(api, round_id, predictions, dry_run=args.dry_run)
        logger.log_submission(round_id, statuses)
        return

    # Load or create observation store
    if args.resume and Path(OBSERVATIONS_FILE).exists():
        store = ObservationStore.load(OBSERVATIONS_FILE)
        print(f"Resumed: {store.queries_used} queries used")
    else:
        store = ObservationStore(round_id, W, H, seeds_count)

    # Query phase
    if not args.no_query:
        remaining = TOTAL_BUDGET - store.queries_used
        if remaining > 0:
            print(f"\nQuerying ({remaining} queries remaining)...")
            run_query_phase(api, store, initial_states, budget=TOTAL_BUDGET)
            logger.log_queries(round_id, store)
        else:
            print("Budget exhausted — skipping queries.")
    else:
        print("--no-query: using existing observations only.")

    # Estimate world dynamics (pooled across all seeds)
    print("\nEstimating world dynamics...")
    dynamics = estimate_world_dynamics(store, initial_states)
    logger.log_params(round_id, {
        "survival_rate":    dynamics.settlement_survival_rate,
        "ruin_rate":        dynamics.settlement_ruin_rate,
        "expansion_rate":   dynamics.expansion_rate,
        "forest_growth":    dynamics.forest_growth_signal,
        "avg_food":         dynamics.avg_food_survivors,
        "avg_port_wealth":  dynamics.avg_port_wealth,
    })

    # Build predictions
    print("\nBuilding predictions...")
    predictions = build_predictions(initial_states, store, dynamics, verbose=True)
    save_predictions(predictions)
    logger.log_predictions(round_id, predictions, initial_states, store)

    # Self-check
    if args.self_check or args.check_only:
        sc = holdout_self_check(initial_states, store, predictions)
        logger.log_self_check(round_id, sc)

    if args.check_only:
        print("\n--check-only: not submitting.")
        return

    # Submit
    statuses = submit_predictions(api, round_id, predictions, dry_run=args.dry_run)
    logger.log_submission(round_id, statuses)
    logger.log_transition_priors(round_id, store, initial_states)

    # Summary
    print(f"\n── Done ──────────────────────────────────────────")
    print(f"  Queries: {store.queries_used}/{TOTAL_BUDGET}")
    for s in range(seeds_count):
        print(f"  Seed {s}: {store.coverage(s):.1%} coverage  status={statuses.get(s, 'n/a')}")


if __name__ == "__main__":
    main()
