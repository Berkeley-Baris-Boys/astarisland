#!/usr/bin/env python3
"""
Backtest the current prediction model against real ground truth from all rounds.

For each round with ig+gt data:
1. Reconstruct initial_states from ig numpy files
2. Run simulated viewport queries (local simulator)
3. Run the full prediction pipeline (world dynamics + predictor)
4. Score predictions against real API ground truth
5. Produce per-class and per-initial-terrain error breakdowns

Usage:
  python -m testing.backtest_all_rounds
  python -m testing.backtest_all_rounds --rounds 5 6 7
  python -m testing.backtest_all_rounds --budget 50 --quiet
  python -m testing.backtest_all_rounds --no-query   # pure prior (no observations)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from testing.simulator import SimParams, simulate_viewport_query
from state import ObservationStore, plan_core_queries, all_viewports
from world_dynamics import estimate_world_dynamics
from predictor import build_predictions
from config import CLASS_NAMES, OCEAN_CODE, MOUNTAIN_CODE

DATA_DIR = PROJECT_ROOT / "data_prev_rounds"
EPS = 1e-12

INIT_CODE_NAMES = {
    0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin",
    4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains",
}


def discover_rounds() -> list[int]:
    rounds = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("round"):
            continue
        r = int(d.name.replace("round", ""))
        ig_path = d / f"ig_r{r}_seed0.npy"
        gt_path = d / f"gt_r{r}_seed0.npy"
        if ig_path.exists() and gt_path.exists():
            rounds.append(r)
    return rounds


def load_round_data(r: int) -> tuple[list[dict], dict[int, np.ndarray], int, int]:
    """Load ig/gt for a round, reconstruct initial_states dicts."""
    round_dir = DATA_DIR / f"round{r}"
    seed = 0
    init_states = []
    ground_truths = {}

    while True:
        ig_path = round_dir / f"ig_r{r}_seed{seed}.npy"
        gt_path = round_dir / f"gt_r{r}_seed{seed}.npy"
        if not ig_path.exists() or not gt_path.exists():
            break

        ig = np.load(ig_path)
        gt = np.load(gt_path)
        ground_truths[seed] = gt

        settlements = []
        ys, xs = np.where((ig == 1) | (ig == 2))
        for y, x in zip(ys.tolist(), xs.tolist()):
            settlements.append({
                "x": x, "y": y,
                "has_port": bool(ig[y, x] == 2),
                "alive": True,
            })

        init_states.append({
            "grid": ig.tolist(),
            "settlements": settlements,
        })
        seed += 1

    H, W = ig.shape
    return init_states, ground_truths, H, W


def run_queries(
    store: ObservationStore,
    init_states: list,
    params: SimParams,
    budget: int,
    base_seed: int,
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
        store.ingest(seed, vx, vy, api_result)
        queries_used += 1
        vp_hits[seed][(vx, vy, vw, vh)] = vp_hits[seed].get((vx, vy, vw, vh), 0) + 1

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


def weighted_kl(gt: np.ndarray, pred: np.ndarray, static_mask: np.ndarray) -> float:
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


def kl_score(wkl: float) -> float:
    return float(max(0.0, min(100.0, 100.0 * np.exp(-3.0 * wkl))))


def per_class_kl(
    gt: np.ndarray, pred: np.ndarray, static_mask: np.ndarray,
) -> dict[int, float]:
    """Mean absolute error per class on dynamic cells."""
    gt_c = np.clip(gt, EPS, 1.0)
    pred_c = np.clip(pred, EPS, 1.0)
    gt_c = gt_c / gt_c.sum(axis=2, keepdims=True)
    pred_c = pred_c / pred_c.sum(axis=2, keepdims=True)
    dynamic = ~static_mask
    result = {}
    for c in range(6):
        diff = np.abs(gt_c[:, :, c] - pred_c[:, :, c])
        result[c] = float(diff[dynamic].mean())
    return result


def per_init_code_kl(
    gt: np.ndarray, pred: np.ndarray, ig: np.ndarray,
) -> dict[int, tuple[float, int]]:
    """Weighted KL per initial terrain code (excluding static)."""
    gt_c = np.clip(gt, EPS, 1.0)
    pred_c = np.clip(pred, EPS, 1.0)
    gt_c = gt_c / gt_c.sum(axis=2, keepdims=True)
    pred_c = pred_c / pred_c.sum(axis=2, keepdims=True)
    kl_per_cell = np.sum(gt_c * (np.log(gt_c) - np.log(pred_c)), axis=2)
    result = {}
    for code in sorted(set(ig.ravel().tolist())):
        if code in (OCEAN_CODE, MOUNTAIN_CODE):
            continue
        mask = ig == code
        count = int(mask.sum())
        if count == 0:
            continue
        result[code] = (float(kl_per_cell[mask].mean()), count)
    return result


def run_one_round(
    r: int, budget: int, base_seed: int, no_query: bool, quiet: bool,
) -> dict:
    init_states, ground_truths, H, W = load_round_data(r)
    seeds_count = len(init_states)
    ig0 = np.array(init_states[0]["grid"], dtype=np.int64)
    static_mask = (ig0 == OCEAN_CODE) | (ig0 == MOUNTAIN_CODE)

    store = ObservationStore(
        round_id=f"backtest-r{r}",
        width=W, height=H, seeds_count=seeds_count,
    )

    if not no_query:
        run_queries(store, init_states, SimParams(), budget, base_seed + r * 10000)

    import io, contextlib
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        dynamics = estimate_world_dynamics(store, init_states)
        predictions = build_predictions(
            init_states, store, dynamics, verbose=False,
        )

    seed_scores = []
    all_class_mae = {c: [] for c in range(6)}
    all_init_kl: dict[int, list[tuple[float, int]]] = {}

    for seed in range(seeds_count):
        gt = ground_truths[seed]
        pred = predictions[seed]
        wkl = weighted_kl(gt, pred, static_mask)
        sc = kl_score(wkl)
        seed_scores.append(sc)

        cls_mae = per_class_kl(gt, pred, static_mask)
        for c, v in cls_mae.items():
            all_class_mae[c].append(v)

        ig_s = np.array(init_states[seed]["grid"], dtype=np.int64)
        init_kl = per_init_code_kl(gt, pred, ig_s)
        for code, (kl, cnt) in init_kl.items():
            all_init_kl.setdefault(code, []).append((kl, cnt))

    overall = float(np.mean(seed_scores))

    avg_class_mae = {}
    for c in range(6):
        if all_class_mae[c]:
            avg_class_mae[c] = float(np.mean(all_class_mae[c]))

    avg_init_kl = {}
    for code, entries in all_init_kl.items():
        total_kl = sum(kl * cnt for kl, cnt in entries)
        total_cnt = sum(cnt for _, cnt in entries)
        avg_init_kl[code] = (total_kl / total_cnt if total_cnt > 0 else 0.0, total_cnt)

    return {
        "round": r,
        "seeds": seeds_count,
        "overall_score": overall,
        "seed_scores": seed_scores,
        "class_mae": avg_class_mae,
        "init_code_kl": avg_init_kl,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest predictor against real GT from all rounds"
    )
    parser.add_argument("--rounds", type=int, nargs="+", default=None,
                        help="Specific rounds to test (default: all)")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-query", action="store_true",
                        help="Pure prior prediction (no simulated observations)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    available = discover_rounds()
    rounds = args.rounds if args.rounds else available
    rounds = [r for r in rounds if r in available]

    mode = "PURE PRIOR (no queries)" if args.no_query else f"FULL PIPELINE ({args.budget} queries)"
    print(f"=== Backtest: {mode} ===")
    print(f"Rounds: {rounds}")
    print()

    results = []
    for r in rounds:
        print(f"--- Round {r} ---")
        res = run_one_round(r, args.budget, args.seed, args.no_query, args.quiet)
        results.append(res)

        print(f"  Score: {res['overall_score']:.1f}  "
              f"(seeds: {', '.join(f'{s:.1f}' for s in res['seed_scores'])})")

        print(f"  Per-class MAE:  ", end="")
        for c in range(6):
            if c in res["class_mae"]:
                print(f"{CLASS_NAMES[c][:5]}={res['class_mae'][c]:.4f}  ", end="")
        print()

        print(f"  Per-init-code KL:")
        for code in sorted(res["init_code_kl"]):
            kl, cnt = res["init_code_kl"][code]
            name = INIT_CODE_NAMES.get(code, f"code{code}")
            print(f"    {name:>10s} (n={cnt:5d}): mean_kl={kl:.4f}")
        print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_scores = [r["overall_score"] for r in results]
    print(f"\n  {'Round':>6s}  {'Score':>6s}  {'Seeds':>5s}")
    print(f"  {'-----':>6s}  {'-----':>6s}  {'-----':>5s}")
    for res in results:
        print(f"  R{res['round']:>4d}  {res['overall_score']:6.1f}  {res['seeds']:5d}")
    print(f"  {'AVG':>6s}  {np.mean(all_scores):6.1f}")

    # Aggregate class MAE across all rounds
    print(f"\n  Avg class MAE across all rounds:")
    agg_class = {c: [] for c in range(6)}
    for res in results:
        for c, v in res["class_mae"].items():
            agg_class[c].append(v)
    for c in range(6):
        if agg_class[c]:
            print(f"    {CLASS_NAMES[c]:>10s}: {np.mean(agg_class[c]):.4f}")

    # Aggregate init-code KL across all rounds
    print(f"\n  Avg init-code KL across all rounds:")
    agg_init: dict[int, list[tuple[float, int]]] = {}
    for res in results:
        for code, (kl, cnt) in res["init_code_kl"].items():
            agg_init.setdefault(code, []).append((kl, cnt))
    for code in sorted(agg_init):
        entries = agg_init[code]
        total_kl = sum(kl * cnt for kl, cnt in entries)
        total_cnt = sum(cnt for _, cnt in entries)
        mean_kl = total_kl / total_cnt if total_cnt > 0 else 0.0
        name = INIT_CODE_NAMES.get(code, f"code{code}")
        print(f"    {name:>10s} (n={total_cnt:6d}): mean_kl={mean_kl:.4f}")

    # Worst class per round
    print(f"\n  Worst class per round (highest MAE):")
    for res in results:
        worst_c = max(res["class_mae"], key=res["class_mae"].get)
        print(f"    R{res['round']}: {CLASS_NAMES[worst_c]} (MAE={res['class_mae'][worst_c]:.4f})")


if __name__ == "__main__":
    main()
