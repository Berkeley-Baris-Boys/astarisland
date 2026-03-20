#!/usr/bin/env python3
"""
Canonical Astar Island solver entrypoint.

This module owns the full end-to-end flow:
  auth -> round load -> query -> infer dynamics -> predict -> submit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from api import APIError, AstarAPI, BudgetExhausted
from config import (
    INITIAL_STATES_FILE,
    OBSERVATIONS_FILE,
    PRED_TEMPLATE,
    TOTAL_BUDGET,
)
from metrics import MetricsLogger, fetch_and_log_analysis, holdout_self_check
from predictor import build_predictions, validate_prediction
from state import ObservationStore, all_viewports, plan_core_queries, candidate_viewports
from world_dynamics import estimate_world_dynamics

MAX_CONSECUTIVE_RESERVE_API_ERRORS = 8


@dataclass(frozen=True)
class RoundInfo:
    round_id: str
    round_number: int | None
    width: int
    height: int
    seeds_count: int
    round_weight: float
    initial_states: list[dict]


def load_active_round_info(api: AstarAPI) -> RoundInfo:
    active = api.get_active_round()
    if active is None:
        raise SystemExit("No active round found. Check app.ainm.no.")

    round_id = str(active["id"])
    detail = api.get_round_detail(round_id)
    width = int(detail["map_width"])
    height = int(detail["map_height"])
    seeds_count = int(detail["seeds_count"])

    round_number_raw = active.get("round_number", detail.get("round_number"))
    try:
        round_number = int(round_number_raw) if round_number_raw is not None else None
    except (TypeError, ValueError):
        round_number = None

    round_weight = float(active.get("round_weight", detail.get("round_weight", 1.0)))
    initial_states = detail["initial_states"]
    if not isinstance(initial_states, list) or len(initial_states) != seeds_count:
        raise SystemExit("initial_states missing or wrong length")

    display = round_number if round_number is not None else "?"
    print(
        f"Round {display}  weight={round_weight}  "
        f"map={width}x{height}  seeds={seeds_count}"
    )

    return RoundInfo(
        round_id=round_id,
        round_number=round_number,
        width=width,
        height=height,
        seeds_count=seeds_count,
        round_weight=round_weight,
        initial_states=initial_states,
    )


def load_active_round(api: AstarAPI) -> tuple[str, int | None, int, int, int, float, list[dict]]:
    """
    Compatibility wrapper retained for existing tests/imports.
    """
    info = load_active_round_info(api)
    return (
        info.round_id,
        info.round_number,
        info.width,
        info.height,
        info.seeds_count,
        info.round_weight,
        info.initial_states,
    )


def _mean_entropy(counts: np.ndarray) -> float:
    totals = counts.sum(axis=2, keepdims=True).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.where(totals > 0, counts / totals, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(probs > 0, probs * np.log(probs + 1e-12), 0.0), axis=2)
    return float(entropy.mean())


def run_query_phase(
    api: AstarAPI,
    store: ObservationStore,
    initial_states: list[dict],
    budget: int = TOTAL_BUDGET,
    *,
    autosave: bool = True,
    save_path: str = OBSERVATIONS_FILE,
) -> None:
    """
    Execute core tiling plus adaptive reserve queries.
    """
    core_plan = plan_core_queries(
        store.width,
        store.height,
        store.seeds_count,
        initial_states=initial_states,
        budget=budget,
    )
    # Use denser reserve candidates than coarse 15x15 tiling to spend
    # leftover budget on locally high-information regions.
    viewports = candidate_viewports(store.width, store.height, stride=5)
    viewport_hits: dict[int, dict[tuple[int, int, int, int], int]] = {
        seed: {} for seed in range(store.seeds_count)
    }

    def execute(seed: int, vx: int, vy: int, vw: int, vh: int) -> None:
        result = api.simulate(store.round_id, seed, vx, vy, vw, vh)
        new_cells = store.ingest(seed, vx, vy, result)
        vp_key = (vx, vy, vw, vh)
        viewport_hits[seed][vp_key] = viewport_hits[seed].get(vp_key, 0) + 1
        print(
            f"  Q{store.queries_used:2d}/{budget}: seed={seed} "
            f"({vx},{vy},{vw}x{vh}) +{new_cells} new"
        )
        if autosave:
            store.save(save_path)

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

    reserve_api_errors = 0
    while store.queries_used < budget:
        coverages = {seed: store.coverage(seed) for seed in range(store.seeds_count)}
        if min(coverages.values()) < 0.99:
            seed = min(coverages, key=coverages.get)
        else:
            seed = max(range(store.seeds_count), key=lambda s: _mean_entropy(store.counts[s]))

        vx, vy, vw, vh = store.best_reserve_viewport(seed, viewports, viewport_hits[seed])
        try:
            execute(seed, vx, vy, vw, vh)
            reserve_api_errors = 0
        except BudgetExhausted:
            print("Budget exhausted in reserve queries.")
            break
        except APIError as exc:
            reserve_api_errors += 1
            print(f"API error (continuing): {exc}")
            if reserve_api_errors >= MAX_CONSECUTIVE_RESERVE_API_ERRORS:
                print(
                    "Stopping reserve queries after repeated API errors "
                    f"({reserve_api_errors} consecutive failures)."
                )
                break

    store.print_summary()


def save_initial_states(initial_states: list[dict]) -> None:
    Path(INITIAL_STATES_FILE).write_text(
        json.dumps(initial_states, indent=2),
        encoding="utf-8",
    )


def save_predictions(predictions: dict[int, np.ndarray]) -> None:
    for seed, prediction in predictions.items():
        np.save(PRED_TEMPLATE.format(seed=seed), prediction)


def load_predictions(seeds_count: int) -> dict[int, np.ndarray]:
    predictions: dict[int, np.ndarray] = {}
    for seed in range(seeds_count):
        path = Path(PRED_TEMPLATE.format(seed=seed))
        if not path.exists():
            raise SystemExit(f"Missing: {path}")
        predictions[seed] = np.load(path)
    return predictions


def submit_predictions(
    api: AstarAPI,
    round_id: str,
    predictions: dict[int, np.ndarray],
    *,
    dry_run: bool = False,
) -> dict[int, str]:
    print("\nSubmitting...")
    statuses: dict[int, str] = {}
    for seed in sorted(predictions.keys()):
        prediction = predictions[seed]
        try:
            validate_prediction(prediction, seed)
        except ValueError as exc:
            statuses[seed] = f"validation_failed: {exc}"
            print(f"  Seed {seed}: VALIDATION FAILED - {exc}")
            continue

        if dry_run:
            statuses[seed] = "dry_run"
            print(f"  Seed {seed}: DRY RUN")
            continue

        try:
            response = api.submit(round_id, seed, prediction.tolist())
            status = str(response.get("status", response))
            statuses[seed] = status
            print(f"  Seed {seed}: {status}")
        except APIError as exc:
            statuses[seed] = f"error: {exc}"
            print(f"  Seed {seed}: ERROR - {exc}")
    return statuses


def archive_round(round_id: str, seeds_count: int) -> None:
    label = round_id[:8]
    destination = Path("data_prev_rounds") / label
    destination.mkdir(parents=True, exist_ok=True)

    files_to_copy = [OBSERVATIONS_FILE, INITIAL_STATES_FILE]
    for seed in range(seeds_count):
        files_to_copy.append(PRED_TEMPLATE.format(seed=seed))

    copied: list[str] = []
    for name in files_to_copy:
        source = Path(name)
        if source.exists():
            shutil.copy2(source, destination / source.name)
            copied.append(source.name)

    if copied:
        print(f"  Archived to data_prev_rounds/{label}/: {', '.join(copied)}")


def _load_or_create_store(args: argparse.Namespace, info: RoundInfo) -> ObservationStore:
    if args.resume and Path(OBSERVATIONS_FILE).exists():
        store = ObservationStore.load(OBSERVATIONS_FILE)
        if (
            store.round_id != info.round_id
            or store.width != info.width
            or store.height != info.height
            or store.seeds_count != info.seeds_count
        ):
            print("Saved observations mismatch active round; starting fresh.")
            return ObservationStore(
                round_id=info.round_id,
                width=info.width,
                height=info.height,
                seeds_count=info.seeds_count,
            )
        print(f"Resumed: {store.queries_used} queries used")
        return store

    return ObservationStore(
        round_id=info.round_id,
        width=info.width,
        height=info.height,
        seeds_count=info.seeds_count,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Astar Island solver")
    parser.add_argument("--token", default=None)
    parser.add_argument("--resume", action="store_true", help="Load saved observations")
    parser.add_argument("--submit-only", action="store_true", help="Submit saved predictions")
    parser.add_argument("--no-query", action="store_true", help="Skip simulator queries")
    parser.add_argument("--dry-run", action="store_true", help="Do everything except submit")
    parser.add_argument("--check-only", action="store_true", help="Build + self-check only")
    parser.add_argument("--self-check", action="store_true", help="Run holdout self-check")
    parser.add_argument("--fetch-analysis", action="store_true", help="Fetch post-round analysis")
    parser.add_argument("--show-history", action="store_true", help="Print metrics history")
    return parser


def run_solver(args: argparse.Namespace) -> None:
    token = args.token or os.getenv("ASTAR_TOKEN") or os.getenv("ASTAR_ISLAND_TOKEN")
    if not token:
        raise SystemExit("Set ASTAR_TOKEN or ASTAR_ISLAND_TOKEN, or pass --token.")

    api = AstarAPI(token)
    logger = MetricsLogger()

    if args.show_history:
        logger.print_cross_round_summary()
        return

    try:
        info = load_active_round_info(api)
    except APIError as exc:
        raise SystemExit(f"Failed to load round: {exc}") from exc

    logger.log_round_start(
        info.round_id,
        info.round_number,
        info.width,
        info.height,
        info.seeds_count,
        info.round_weight,
    )
    save_initial_states(info.initial_states)

    if args.fetch_analysis:
        predictions: dict[int, np.ndarray] = {}
        for seed in range(info.seeds_count):
            path = Path(PRED_TEMPLATE.format(seed=seed))
            if path.exists():
                predictions[seed] = np.load(path)
        fetch_and_log_analysis(api, info.round_id, info.seeds_count, predictions, logger)
        return

    if args.submit_only:
        predictions = load_predictions(info.seeds_count)
        statuses = submit_predictions(api, info.round_id, predictions, dry_run=args.dry_run)
        logger.log_submission(info.round_id, statuses)
        return

    store = _load_or_create_store(args, info)

    if not args.no_query:
        remaining = TOTAL_BUDGET - store.queries_used
        if remaining > 0:
            print(f"\nQuerying ({remaining} queries remaining)...")
            run_query_phase(api, store, info.initial_states, budget=TOTAL_BUDGET)
            logger.log_queries(info.round_id, store)
        else:
            print("Budget exhausted - skipping queries.")
    else:
        print("--no-query: using existing observations only.")

    print("\nEstimating world dynamics...")
    dynamics = estimate_world_dynamics(store, info.initial_states)
    logger.log_params(
        info.round_id,
        {
            "survival_rate": dynamics.settlement_survival_rate,
            "ruin_rate": dynamics.settlement_ruin_rate,
            "expansion_rate": dynamics.expansion_rate,
            "forest_growth": dynamics.forest_growth_signal,
            "avg_food": dynamics.avg_food_survivors,
            "avg_port_wealth": dynamics.avg_port_wealth,
        },
    )

    print("\nBuilding predictions...")
    predictions = build_predictions(info.initial_states, store, dynamics, verbose=True)
    save_predictions(predictions)
    logger.log_predictions(info.round_id, predictions, info.initial_states, store)

    if args.self_check or args.check_only:
        self_check_result = holdout_self_check(info.initial_states, store, predictions)
        logger.log_self_check(info.round_id, self_check_result)

    if args.check_only:
        print("\n--check-only: not submitting.")
        return

    statuses = submit_predictions(api, info.round_id, predictions, dry_run=args.dry_run)
    logger.log_submission(info.round_id, statuses)
    logger.log_transition_priors(info.round_id, store, info.initial_states)

    print("\nArchiving round data...")
    archive_round(info.round_id, info.seeds_count)

    print("\n-- Done --------------------------------------------")
    print(f"  Queries: {store.queries_used}/{TOTAL_BUDGET}")
    for seed in range(info.seeds_count):
        print(f"  Seed {seed}: {store.coverage(seed):.1%} coverage  status={statuses.get(seed, 'n/a')}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_solver(args)


if __name__ == "__main__":
    main()
