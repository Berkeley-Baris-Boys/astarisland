#!/usr/bin/env python3
"""
Test the modular pipeline (main.py) against local ground truth.

Mirrors the real task flow exactly:
- 40x40 map, 5 seeds, 50 viewport queries shared across seeds
- Viewport queries via local simulator (no API calls, no submission)
- Pipeline: plan_core_queries -> simulate -> ObservationStore.ingest
            -> estimate_world_dynamics -> predictor.build_predictions
- Compare predictions vs Monte Carlo ground truth using official scoring

Usage:
  python -m testing.test_main_local
  python -m testing.test_main_local --budget 50 --gt-runs 100 --quiet
  python -m testing.test_main_local --initial-state path/to/file.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from testing.simulator import (
    OCEAN,
    PLAINS,
    FOREST,
    MOUNTAIN,
    SETTLEMENT,
    TERRAIN_TO_CLASS,
    SimParams,
    monte_carlo,
    simulate_viewport_query,
)

from state import ObservationStore, plan_core_queries, all_viewports
from world_dynamics import estimate_world_dynamics
from predictor import build_predictions

EPS = 1e-12


def _builtin_initial_state():
    H, W = 40, 40
    grid = np.full((H, W), PLAINS, dtype=np.int64)
    grid[0, :] = OCEAN
    grid[-1, :] = OCEAN
    grid[:, 0] = OCEAN
    grid[:, -1] = OCEAN
    grid[8, 15] = MOUNTAIN
    grid[25, 20] = MOUNTAIN
    grid[30, 35] = MOUNTAIN
    for dy, dx in [(10, 10), (10, 11), (11, 10), (11, 11), (12, 10)]:
        grid[dy, dx] = FOREST
    for dy, dx in [(28, 28), (28, 29), (29, 28)]:
        grid[dy, dx] = FOREST
    grid[12, 12] = SETTLEMENT
    grid[20, 20] = SETTLEMENT
    grid[32, 8] = SETTLEMENT

    settlements_data = [
        {"x": 12, "y": 12, "has_port": False, "alive": True},
        {"x": 20, "y": 20, "has_port": False, "alive": True},
        {"x": 8, "y": 32, "has_port": False, "alive": True},
    ]
    init_state = {"grid": grid.tolist(), "settlements": settlements_data}
    return [init_state] * 5, H, W


def _load_initial_state(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    init_states = data["initial_states"]
    width = data.get("map_width", len(init_states[0]["grid"][0]))
    height = data.get("map_height", len(init_states[0]["grid"]))
    return init_states, height, width


def _run_queries(
    store: ObservationStore,
    init_states: list,
    params: SimParams,
    budget: int,
    base_seed: int,
    quiet: bool,
) -> None:
    width, height = store.width, store.height
    seeds_count = store.seeds_count

    core_plan = plan_core_queries(
        width, height, seeds_count,
        initial_states=init_states, budget=budget,
    )
    viewports = all_viewports(width, height)
    vp_hits: dict[int, dict] = {s: {} for s in range(seeds_count)}
    queries_used = 0

    def execute(seed: int, vx: int, vy: int, vw: int, vh: int) -> None:
        nonlocal queries_used
        init_grid = np.array(init_states[seed]["grid"], dtype=np.int64)
        settle_data = init_states[seed].get("settlements", [])
        rng_seed = base_seed + queries_used * 7919

        vp_grid, snap_list = simulate_viewport_query(
            init_grid, settle_data, params,
            seed=rng_seed,
            viewport_x=vx, viewport_y=vy,
            viewport_w=vw, viewport_h=vh,
        )

        api_result = {
            "grid": vp_grid.tolist(),
            "settlements": snap_list,
            "viewport": {"x": vx, "y": vy, "w": vw, "h": vh},
        }

        new_cells = store.ingest(seed, vx, vy, api_result)
        queries_used += 1
        vp_hits[seed][(vx, vy, vw, vh)] = vp_hits[seed].get((vx, vy, vw, vh), 0) + 1

        if not quiet:
            cov = store.coverage(seed)
            print(
                f"  Q{queries_used:02d}/{budget} seed={seed} "
                f"({vx},{vy},{vw}x{vh}) +{new_cells} new [{cov:.0%}]"
            )

    for seed, vx, vy, vw, vh in core_plan:
        if queries_used >= budget:
            break
        execute(seed, vx, vy, vw, vh)

    while queries_used < budget:
        coverages = {s: store.coverage(s) for s in range(seeds_count)}
        if min(coverages.values()) < 0.99:
            seed = min(coverages, key=coverages.get)
        else:
            seed = max(range(seeds_count), key=lambda s: float(store.counts[s].sum()))
        vp = store.best_reserve_viewport(seed, viewports, vp_hits[seed])
        vx, vy, vw, vh = vp
        execute(seed, vx, vy, vw, vh)


def _weighted_kl(gt: np.ndarray, pred: np.ndarray, static_mask: np.ndarray) -> float:
    gt = np.clip(gt, EPS, 1.0)
    pred = np.clip(pred, EPS, 1.0)
    gt = gt / gt.sum(axis=2, keepdims=True)
    pred = pred / pred.sum(axis=2, keepdims=True)

    entropy_gt = -np.sum(gt * np.log(gt), axis=2)
    kl_per_cell = np.sum(gt * (np.log(gt) - np.log(pred)), axis=2)

    dynamic = ~static_mask
    if not np.any(dynamic):
        return 0.0
    ent_sum = entropy_gt[dynamic].sum()
    if ent_sum <= 0:
        return 0.0
    return float((entropy_gt[dynamic] * kl_per_cell[dynamic]).sum() / ent_sum)


def _score(weighted_kl: float) -> float:
    return float(max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl))))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test modular pipeline (main.py) against local ground truth"
    )
    parser.add_argument("--initial-state", default=None,
                        help="Load initial states from JSON")
    parser.add_argument("--budget", type=int, default=50,
                        help="Viewport query budget (default: 50)")
    parser.add_argument("--gt-runs", type=int, default=100,
                        help="MC runs for ground truth (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-query output")
    args = parser.parse_args()

    if args.initial_state:
        init_states, height, width = _load_initial_state(args.initial_state)
        source = f"file {args.initial_state}"
    else:
        init_states, height, width = _builtin_initial_state()
        source = "built-in 40x40, 5 seeds"

    seeds_count = len(init_states)

    print("=== main.py Pipeline Local Test ===")
    print(f"Initial state: {source} ({seeds_count} seed(s))")
    print(f"Map size: {width}x{height}")
    print(f"Query budget: {args.budget} viewport queries")
    print(f"Ground truth: {args.gt_runs} MC runs")
    print()

    init_grid = np.array(init_states[0]["grid"], dtype=np.int64)
    static_mask = (init_grid == 10) | (init_grid == 5)

    print("Computing ground truth...")
    ground_truth: dict[int, np.ndarray] = {}
    for seed in range(seeds_count):
        state = init_states[seed]
        ig_arr = np.array(state["grid"], dtype=np.int64)
        settle_data = state.get("settlements", [])
        gt = monte_carlo(
            ig_arr, settle_data, SimParams(),
            n_runs=args.gt_runs, n_years=50, param_noise=0.0,
            seed=args.seed + seed * 1000,
        )
        ground_truth[seed] = gt
    print("  Done.\n")

    store = ObservationStore(
        round_id="local-test",
        width=width,
        height=height,
        seeds_count=seeds_count,
    )

    print("Running viewport queries (local simulator)...")
    _run_queries(
        store, init_states,
        params=SimParams(),
        budget=args.budget,
        base_seed=args.seed,
        quiet=args.quiet,
    )
    store.print_summary()

    print("\nEstimating world dynamics...")
    dynamics = estimate_world_dynamics(store, init_states)

    print("\nBuilding predictions...")
    predictions = build_predictions(
        init_states, store, dynamics, verbose=not args.quiet,
    )

    print("\n--- Results ---")
    scores: list[float] = []
    for seed in range(seeds_count):
        gt = ground_truth[seed]
        pred = predictions[seed]
        wkl = _weighted_kl(gt, pred, static_mask)
        sc = _score(wkl)
        scores.append(sc)
        cov = store.coverage(seed)
        print(f"Seed {seed}: {cov:.0%} coverage  weighted_kl={wkl:.4f}  score={sc:.1f}")

    overall = float(np.mean(scores)) if scores else 0.0
    print(f"\nOverall: score={overall:.1f}")
    print(
        "\n(Note: testing simulator may differ from API simulator — scores are approximate)"
    )


if __name__ == "__main__":
    main()
