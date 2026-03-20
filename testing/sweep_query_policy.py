#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from predictor import build_predictions
from state import ObservationStore, plan_core_queries, candidate_viewports
from testing.simulator import (
    FOREST,
    MOUNTAIN,
    OCEAN,
    PLAINS,
    SETTLEMENT,
    SimParams,
    monte_carlo,
    simulate_viewport_query,
)
from world_dynamics import estimate_world_dynamics

N_CLASSES = 6
EPS = 1e-12


@dataclass(frozen=True)
class QueryPolicy:
    stride: int
    seed_mode: str
    unseen_w: float
    single_w: float
    double_w: float
    scarcity_w: float
    unresolved_w: float
    revisit_penalty: float

    def label(self) -> str:
        return (
            f"stride={self.stride} seed={self.seed_mode} "
            f"u={self.unseen_w:.2f} s1={self.single_w:.2f} s2={self.double_w:.2f} "
            f"sc={self.scarcity_w:.2f} ur={self.unresolved_w:.2f} rp={self.revisit_penalty:.2f}"
        )


def _builtin_initial_state() -> tuple[list[dict], int, int]:
    h, w = 40, 40
    grid = np.full((h, w), PLAINS, dtype=np.int64)
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
    return [init_state] * 5, h, w


def _choose_seed(store: ObservationStore, policy: QueryPolicy) -> int:
    coverages = {s: store.coverage(s) for s in range(store.seeds_count)}
    if min(coverages.values()) < 0.99:
        return min(coverages, key=coverages.get)

    if policy.seed_mode == "entropy":
        def _ent(seed: int) -> float:
            counts = store.counts[seed]
            totals = counts.sum(axis=2, keepdims=True).astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.where(totals > 0, counts / totals, 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ent = -np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0.0), axis=2)
            return float(ent.mean())

        return max(range(store.seeds_count), key=_ent)

    def _unresolved(seed: int) -> float:
        counts = store.counts[seed]
        n_obs = counts.sum(axis=2)
        unresolved = (n_obs <= 2).astype(np.float64)
        obs = store.latest[seed]
        dyn_mask = np.zeros((store.height, store.width), dtype=np.float64)
        for y, row in enumerate(obs):
            for x, raw in enumerate(row):
                if raw in (1, 2, 3):
                    dyn_mask[y, x] = 1.0
        return float((unresolved * (1.0 + dyn_mask)).mean())

    return max(range(store.seeds_count), key=_unresolved)


def _choose_viewport(
    store: ObservationStore,
    seed: int,
    viewports: list[tuple[int, int, int, int]],
    hits: dict[tuple[int, int, int, int], int],
    policy: QueryPolicy,
) -> tuple[int, int, int, int]:
    obs_mask = store.observed_mask(seed)
    cnt = store.counts[seed]
    totals = cnt.sum(axis=2, keepdims=True).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(totals > 0, cnt / totals, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0.0), axis=2)
    entropy[~obs_mask] = np.log(N_CLASSES)

    n_obs = cnt.sum(axis=2)
    single_mask = n_obs == 1
    double_mask = n_obs == 2
    scarcity = 1.0 / np.maximum(n_obs.astype(np.float64), 1.0)
    scarcity[~obs_mask] = 1.0

    unresolved_dynamic = np.zeros((store.height, store.width), dtype=np.float64)
    for y, row in enumerate(store.latest[seed]):
        for x, raw in enumerate(row):
            if raw in (1, 2, 3) and n_obs[y, x] <= 2:
                unresolved_dynamic[y, x] = 1.0

    best_vp = viewports[0]
    best_score = -1e18
    for vp in viewports:
        vx, vy, vw, vh = vp
        rs = slice(vy, vy + vh)
        cs = slice(vx, vx + vw)
        unseen_frac = float((~obs_mask[rs, cs]).mean())
        single_frac = float(single_mask[rs, cs].mean())
        double_frac = float(double_mask[rs, cs].mean())
        scarcity_mean = float(scarcity[rs, cs].mean())
        unresolved_frac = float(unresolved_dynamic[rs, cs].mean())
        revisit = hits.get(vp, 0)

        score = (
            float(entropy[rs, cs].mean())
            + policy.unseen_w * unseen_frac
            + policy.single_w * single_frac
            + policy.double_w * double_frac
            + policy.scarcity_w * scarcity_mean
            + policy.unresolved_w * unresolved_frac
            - policy.revisit_penalty * np.log1p(revisit)
        )
        if score > best_score:
            best_score = score
            best_vp = vp
    return best_vp


def _weighted_kl(gt: np.ndarray, pred: np.ndarray, static_mask: np.ndarray) -> float:
    gt = np.clip(gt, EPS, 1.0)
    pred = np.clip(pred, EPS, 1.0)
    gt = gt / gt.sum(axis=2, keepdims=True)
    pred = pred / pred.sum(axis=2, keepdims=True)

    ent = -np.sum(gt * np.log(gt), axis=2)
    kl = np.sum(gt * (np.log(gt) - np.log(pred)), axis=2)
    dynamic = ~static_mask
    if not np.any(dynamic):
        return 0.0
    ent_sum = float(ent[dynamic].sum())
    if ent_sum <= 0:
        return 0.0
    return float((ent[dynamic] * kl[dynamic]).sum() / ent_sum)


def _score(weighted_kl: float) -> float:
    return float(max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl))))


def _run_policy(
    init_states: list[dict],
    ground_truth: dict[int, np.ndarray],
    policy: QueryPolicy,
    *,
    budget: int,
    base_seed: int,
) -> float:
    height = len(init_states[0]["grid"])
    width = len(init_states[0]["grid"][0])
    seeds_count = len(init_states)
    store = ObservationStore("sweep-local", width, height, seeds_count)

    core_plan = plan_core_queries(
        width,
        height,
        seeds_count,
        initial_states=init_states,
        budget=budget,
    )
    reserve_viewports = candidate_viewports(width, height, stride=policy.stride)
    vp_hits: dict[int, dict[tuple[int, int, int, int], int]] = {s: {} for s in range(seeds_count)}

    queries_used = 0

    def execute(seed: int, vx: int, vy: int, vw: int, vh: int) -> None:
        nonlocal queries_used
        init_grid = np.array(init_states[seed]["grid"], dtype=np.int64)
        settle_data = init_states[seed].get("settlements", [])
        rng_seed = base_seed + queries_used * 7919
        vp_grid, snap_list = simulate_viewport_query(
            init_grid,
            settle_data,
            SimParams(),
            seed=rng_seed,
            viewport_x=vx,
            viewport_y=vy,
            viewport_w=vw,
            viewport_h=vh,
        )
        api_result = {
            "grid": vp_grid.tolist(),
            "settlements": snap_list,
            "viewport": {"x": vx, "y": vy, "w": vw, "h": vh},
        }
        store.ingest(seed, vx, vy, api_result)
        queries_used += 1
        vp = (vx, vy, vw, vh)
        vp_hits[seed][vp] = vp_hits[seed].get(vp, 0) + 1

    for seed, vx, vy, vw, vh in core_plan:
        if queries_used >= budget:
            break
        execute(seed, vx, vy, vw, vh)

    while queries_used < budget:
        seed = _choose_seed(store, policy)
        vx, vy, vw, vh = _choose_viewport(store, seed, reserve_viewports, vp_hits[seed], policy)
        execute(seed, vx, vy, vw, vh)

    dynamics = estimate_world_dynamics(store, init_states)
    preds = build_predictions(init_states, store, dynamics, verbose=False)

    scores: list[float] = []
    for seed in range(seeds_count):
        gt = ground_truth[seed]
        pred = preds[seed]
        init_grid = np.array(init_states[seed]["grid"], dtype=np.int64)
        static_mask = (init_grid == MOUNTAIN) | (init_grid == OCEAN)
        wkl = _weighted_kl(gt, pred, static_mask)
        scores.append(_score(wkl))
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep local query policy settings")
    parser.add_argument("--gt-runs", type=int, default=60)
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    init_states, _, _ = _builtin_initial_state()
    seeds_count = len(init_states)
    print(f"Building ground truth with gt_runs={args.gt_runs}...")
    gt: dict[int, np.ndarray] = {}
    for seed in range(seeds_count):
        state = init_states[seed]
        gt[seed] = monte_carlo(
            np.array(state["grid"], dtype=np.int64),
            state.get("settlements", []),
            SimParams(),
            n_runs=args.gt_runs,
            n_years=50,
            param_noise=0.0,
            seed=args.seed + seed * 1000,
        )

    policies = [
        QueryPolicy(3, "entropy", 2.1, 0.8, 0.35, 0.55, 0.50, 0.22),
        QueryPolicy(3, "unresolved", 2.1, 0.8, 0.35, 0.55, 0.65, 0.22),
        QueryPolicy(5, "entropy", 2.1, 0.8, 0.35, 0.55, 0.50, 0.22),
        QueryPolicy(5, "unresolved", 2.1, 0.8, 0.35, 0.55, 0.65, 0.22),
        QueryPolicy(7, "entropy", 2.1, 0.8, 0.35, 0.55, 0.50, 0.22),
        QueryPolicy(7, "unresolved", 2.1, 0.8, 0.35, 0.55, 0.65, 0.22),
        QueryPolicy(5, "unresolved", 2.3, 1.0, 0.45, 0.70, 0.90, 0.30),
        QueryPolicy(3, "unresolved", 2.3, 1.0, 0.45, 0.70, 0.90, 0.30),
    ]

    results: list[tuple[float, QueryPolicy]] = []
    for idx, policy in enumerate(policies, start=1):
        score = _run_policy(
            init_states,
            gt,
            policy,
            budget=args.budget,
            base_seed=args.seed,
        )
        print(f"[{idx:02d}/{len(policies)}] score={score:.3f}  {policy.label()}")
        results.append((score, policy))

    results.sort(key=lambda x: x[0], reverse=True)
    print("\nTop policies:")
    for rank, (score, policy) in enumerate(results[:5], start=1):
        print(f"  #{rank}: score={score:.3f}  {policy.label()}")


if __name__ == "__main__":
    main()
