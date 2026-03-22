"""Evaluate the villager recalibration fix across archived rounds.

Two modes per round×seed:
  oracle   – treat ground truth as perfect observation coverage (upper bound)
  sampled  – subsample ~80% of buildable cells randomly (realistic bound)

Reports score deltas at strength=[0.0, 0.3, 0.5, 0.7, 1.0] to help choose the
right default for villager_recalib_strength.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HISTORY_DIR = Path("artifacts/history")
STRENGTHS = [0.0, 0.3, 0.5, 0.7, 1.0]
CLIP_LO = 0.6
CLIP_HI = 1.8
SAMPLE_FRACTION = 0.80  # fraction of buildable cells to treat as "observed" in sampled mode
RANDOM_SEED = 42
MIN_OBS_CELLS = 150

CLASS_SETTLEMENT = 1
CLASS_MOUNTAIN = 5
TERRAIN_OCEAN = 10
TERRAIN_MOUNTAIN = 5

EPS = 1e-12


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def cell_entropy(probs: np.ndarray) -> np.ndarray:
    safe = np.maximum(probs, EPS)
    return -np.sum(np.where(probs > 0.0, probs * np.log(safe), 0.0), axis=-1)


def cell_kl(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    safe_gt = np.maximum(gt, EPS)
    safe_pred = np.maximum(pred, EPS)
    return np.sum(np.where(gt > 0.0, gt * (np.log(safe_gt) - np.log(safe_pred)), 0.0), axis=-1)


def score_prediction(gt: np.ndarray, pred: np.ndarray) -> float:
    entropy = cell_entropy(gt)
    w = float(np.sum(entropy))
    if w <= 0.0:
        return 0.0
    kl = cell_kl(gt, pred)
    wkl = float(np.sum(entropy * kl) / w)
    return float(max(0.0, min(100.0, 100.0 * math.exp(-3.0 * wkl))))


# ---------------------------------------------------------------------------
# Recalibration
# ---------------------------------------------------------------------------

def normalize(arr: np.ndarray, min_prob: float = 0.0025) -> np.ndarray:
    out = np.maximum(arr, min_prob)
    totals = out.sum(axis=-1, keepdims=True)
    return out / np.maximum(totals, EPS)


def apply_recalib(
    prediction: np.ndarray,
    obs_counts: np.ndarray,      # H×W×6  (raw cell-class counts on observed cells)
    obs_mask: np.ndarray,        # H×W bool  (which cells were "observed")
    buildable: np.ndarray,       # H×W bool
    strength: float,
) -> np.ndarray:
    observed_buildable = obs_mask & buildable
    n_obs = int(np.sum(observed_buildable))
    if n_obs < MIN_OBS_CELLS:
        return prediction

    obs_settlement = float(np.sum(obs_counts[observed_buildable, CLASS_SETTLEMENT]))
    obs_total = float(np.sum(obs_counts[observed_buildable]))
    if obs_total <= 0.0:
        return prediction
    obs_rate = obs_settlement / obs_total

    pred_rate = float(np.mean(prediction[observed_buildable, CLASS_SETTLEMENT]))
    if pred_rate <= 1e-6:
        return prediction

    raw_scale = obs_rate / max(pred_rate, EPS)
    clipped = float(np.clip(raw_scale, CLIP_LO, CLIP_HI))
    scale = 1.0 + strength * (clipped - 1.0)

    out = prediction.copy()
    out[..., CLASS_SETTLEMENT][buildable] *= scale
    return normalize(out)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def buildable_mask_from_grid(grid: np.ndarray) -> np.ndarray:
    """Cells that are not ocean and not mountain terrain."""
    return (grid != TERRAIN_OCEAN) & (grid != TERRAIN_MOUNTAIN)


def oracle_obs_counts(gt: np.ndarray) -> np.ndarray:
    """Perfect-information oracle: use the GT probability vector directly as expected
    observation counts. With all cells covered once, obs_rate converges to the true
    mean GT class probability — the theoretically correct upper bound."""
    return gt.copy()


def sampled_obs_counts(
    gt: np.ndarray, buildable: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate realistic observations: sample SAMPLE_FRACTION of buildable cells,
    draw ONE discrete outcome per cell from the GT distribution (as the real simulator
    would return a single state per observation)."""
    H, W, C = gt.shape
    buildable_indices = list(zip(*np.where(buildable)))
    k = max(MIN_OBS_CELLS, int(len(buildable_indices) * SAMPLE_FRACTION))
    k = min(k, len(buildable_indices))
    chosen_idx = rng.choice(len(buildable_indices), size=k, replace=False)

    obs = np.zeros((H, W, C), dtype=np.float64)
    mask = np.zeros((H, W), dtype=bool)
    for idx in chosen_idx:
        r, c = buildable_indices[idx]
        probs = gt[r, c]
        total = probs.sum()
        if total > 0:
            probs = probs / total
        observed_class = int(rng.choice(C, p=probs))
        obs[r, c, observed_class] += 1.0
        mask[r, c] = True
    return obs, mask


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_round(round_dir: Path, rng: np.random.Generator) -> dict | None:
    detail_path = round_dir / "round_detail.json"
    if not detail_path.exists():
        return None

    detail = json.loads(detail_path.read_text())
    round_number = detail.get("round_number", "?")

    per_seed: list[dict] = []

    for seed_index in range(detail.get("seeds_count", 5)):
        analysis_path = round_dir / f"seed_{seed_index}" / "analysis.json"
        gt_path = round_dir / f"seed_{seed_index}" / "ground_truth.npy"
        if not analysis_path.exists() or not gt_path.exists():
            continue

        analysis = json.loads(analysis_path.read_text())
        pred = np.asarray(analysis["prediction"], dtype=np.float64)
        gt = np.load(gt_path).astype(np.float64)
        grid = np.asarray(analysis["initial_grid"], dtype=np.int32)
        buildable = buildable_mask_from_grid(grid)

        baseline_score = score_prediction(gt, pred)

        # ---- oracle mode: expected GT probabilities as observation counts ----
        oracle_obs = oracle_obs_counts(gt)
        oracle_full_mask = buildable.copy()  # only buildable cells count

        oracle_scores: dict[float, float] = {}
        for s in STRENGTHS:
            adj = apply_recalib(pred, oracle_obs, oracle_full_mask, buildable, strength=s)
            oracle_scores[s] = score_prediction(gt, adj)

        # ---- sampled mode: 80% random coverage, discrete draws ----
        sampled_obs, sampled_mask = sampled_obs_counts(gt, buildable, rng)
        sampled_scores: dict[float, float] = {}
        for s in STRENGTHS:
            adj = apply_recalib(pred, sampled_obs, sampled_mask, buildable, strength=s)
            sampled_scores[s] = score_prediction(gt, adj)

        # Diagnostics: actual bias
        # oracle obs_rate = mean of GT class-1 probability on buildable cells
        obs_rate_oracle = float(np.mean(gt[buildable, CLASS_SETTLEMENT]))
        pred_rate = float(np.mean(pred[buildable, CLASS_SETTLEMENT]))

        per_seed.append({
            "seed": seed_index,
            "baseline": round(baseline_score, 3),
            "obs_rate_oracle": round(obs_rate_oracle, 4),
            "pred_rate": round(pred_rate, 4),
            "raw_scale": round(obs_rate_oracle / max(pred_rate, EPS), 4),
            "oracle": {str(s): round(oracle_scores[s], 3) for s in STRENGTHS},
            "sampled": {str(s): round(sampled_scores[s], 3) for s in STRENGTHS},
        })

    if not per_seed:
        return None

    return {"round_number": round_number, "per_seed": per_seed}


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    all_results: list[dict] = []

    round_dirs = sorted(HISTORY_DIR.glob("round_*"), key=lambda p: int(p.name.split("_")[1]))
    for round_dir in round_dirs:
        result = evaluate_round(round_dir, rng)
        if result is None:
            continue
        all_results.append(result)

    # Print summary table
    header = (
        f"{'Rnd':>4}  {'Seed':>4}  {'Baseline':>8}  "
        f"{'obs_rt':>6}  {'pred_rt':>7}  {'scale':>6}  "
        f"{'Oracle Δ @0.5':>13}  {'Oracle Δ @1.0':>13}  "
        f"{'Samp Δ @0.5':>11}  {'Samp Δ @1.0':>11}"
    )
    print(header)
    print("-" * len(header))

    for result in all_results:
        rn = result["round_number"]
        seed_rows = result["per_seed"]
        for row in seed_rows:
            b = row["baseline"]
            o05 = row["oracle"]["0.5"] - b
            o10 = row["oracle"]["1.0"] - b
            s05 = row["sampled"]["0.5"] - b
            s10 = row["sampled"]["1.0"] - b
            print(
                f"{rn:>4}  {row['seed']:>4}  {b:>8.3f}  "
                f"{row['obs_rate_oracle']:>6.3f}  {row['pred_rate']:>7.3f}  {row['raw_scale']:>6.3f}  "
                f"{o05:>+13.3f}  {o10:>+13.3f}  "
                f"{s05:>+11.3f}  {s10:>+11.3f}"
            )

    # Per-round average gain at each strength (sampled mode, which is realistic)
    print()
    print("Average score delta by strength (sampled 80% coverage, all rounds):")
    strength_deltas: dict[str, list[float]] = {str(s): [] for s in STRENGTHS}
    for result in all_results:
        for row in result["per_seed"]:
            b = row["baseline"]
            for s in STRENGTHS:
                strength_deltas[str(s)].append(row["sampled"][str(s)] - b)

    for s in STRENGTHS:
        deltas = strength_deltas[str(s)]
        if deltas:
            mean_d = sum(deltas) / len(deltas)
            negative = sum(1 for d in deltas if d < -0.1)
            print(f"  strength={s:.1f}: mean_delta={mean_d:+.3f}, n_hurt={negative}/{len(deltas)}")

    # Focus on recent rounds (15-18)
    recent = [r for r in all_results if r["round_number"] >= 15]
    if recent:
        print()
        print("Average delta on rounds 15-18 (sampled):")
        for s in STRENGTHS:
            deltas = [row["sampled"][str(s)] - row["baseline"]
                      for r in recent for row in r["per_seed"]]
            if deltas:
                print(f"  strength={s:.1f}: mean={sum(deltas)/len(deltas):+.3f}")


if __name__ == "__main__":
    main()
