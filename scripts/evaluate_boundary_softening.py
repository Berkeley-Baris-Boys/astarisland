"""Offline sweep: apply boundary softening at various alpha values to stored predictions.

The mask used here is the forest↔plains boundary derived from initial_grid — a conservative
subset of the production mask (which also includes frontier cells and intermediate
settlement-intensity cells). If this conservative sweep shows improvement, the production
mask will do at least as well.

Usage:
    .venv/bin/python3 scripts/evaluate_boundary_softening.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

from astar_island.scoring import score_prediction
from astar_island.utils import normalize_probabilities

# Terrain constants
TERRAIN_FOREST = 4
TERRAIN_MOUNTAIN = 5
TERRAIN_OCEAN = 10
TERRAIN_PLAINS = 11

MIN_PROB = 0.0025
ALPHAS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]


def build_offline_mask(initial_grid: np.ndarray) -> np.ndarray:
    """Forest cells adjacent to non-forest, non-mountain, non-ocean cells."""
    forest = initial_grid == TERRAIN_FOREST
    mountain = initial_grid == TERRAIN_MOUNTAIN
    ocean = initial_grid == TERRAIN_OCEAN
    non_forest_buildable = ~forest & ~mountain & ~ocean
    return forest & binary_dilation(non_forest_buildable, iterations=1)


def apply_softening(pred: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return pred
    out = pred.copy().astype(np.float64)
    out[mask] += alpha
    return normalize_probabilities(out, MIN_PROB)


def main() -> None:
    history_dir = Path("artifacts/history")
    round_dirs = sorted(history_dir.glob("round_*"))

    results: dict[str, dict[float, list[float]]] = {}

    for rdir in round_dirs:
        round_name = rdir.name[:8]
        seed_dirs = sorted(rdir.glob("seed_*"))
        if not seed_dirs:
            continue

        per_alpha: dict[float, list[float]] = {a: [] for a in ALPHAS}

        for sdir in seed_dirs:
            data = json.loads((sdir / "analysis.json").read_text())
            pred = np.array(data["prediction"], dtype=np.float64)
            gt = np.array(data["ground_truth"], dtype=np.float64)
            initial_grid = np.array(data["initial_grid"], dtype=np.int32)
            mask = build_offline_mask(initial_grid)
            base_score = score_prediction(gt, pred)

            for alpha in ALPHAS:
                softened = apply_softening(pred, mask, alpha)
                delta = score_prediction(gt, softened) - base_score
                per_alpha[alpha].append(delta)

        results[round_name] = {a: float(np.mean(per_alpha[a])) for a in ALPHAS}

    # Print table
    alpha_strs = [f"α={a:.2f}" for a in ALPHAS]
    col_w = 8
    header = f"{'Round':<10}" + "".join(f"{s:>{col_w}}" for s in alpha_strs)
    print(header)
    print("-" * len(header))
    for rname in sorted(results.keys()):
        row = f"{rname:<10}" + "".join(f"{results[rname][a]:>+{col_w}.3f}" for a in ALPHAS)
        print(row)

    recent = [k for k in results if k in {"round_13", "round_14", "round_15", "round_16", "round_17"}]
    print()
    print("R13-R17 mean:")
    for a in ALPHAS:
        mean_delta = float(np.mean([results[r][a] for r in recent]))
        print(f"  α={a:.2f}: {mean_delta:+.3f}")

    print()
    print("All-rounds worst case:")
    for a in ALPHAS:
        worst = min(results[r][a] for r in results)
        print(f"  α={a:.2f}: worst={worst:+.3f}")

    # Acceptance check
    print()
    print("Acceptance gate (mean R13-R17 ≥ 0, worst ≥ -0.30, R14 ≥ -0.10):")
    for a in ALPHAS:
        if a == 0.0:
            continue
        mean_recent = float(np.mean([results[r][a] for r in recent]))
        worst = min(results[r][a] for r in results)
        r14 = results.get("round_14", {}).get(a, 0.0)
        ok = mean_recent >= 0.0 and worst >= -0.30 and r14 >= -0.10
        print(f"  α={a:.2f}: {'PASS' if ok else 'FAIL'}  (mean={mean_recent:+.3f}, worst={worst:+.3f}, R14={r14:+.3f})")


if __name__ == "__main__":
    main()
