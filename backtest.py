"""
Back-test with leave-one-round-out cross-validation.

Compares:
  - baseline: no historical prior (heuristic only)
  - new:      historical prior built from all OTHER rounds (leave-one-out)

Usage:
    python backtest.py [--rounds 1,2,3]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import N_CLASSES, PROB_FLOOR, OCEAN_CODE, MOUNTAIN_CODE, SETTLEMENT_CODE, PORT_CODE
from state import ObservationStore
from world_dynamics import WorldDynamics
from historical_prior import _build_from_pairs, _load_pairs_from_dir


# ── Scoring ────────────────────────────────────────────────────────────────────

def score(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.maximum(gt,   1e-9)
    q = np.maximum(pred, PROB_FLOOR)
    q = q / q.sum(axis=-1, keepdims=True)
    entropy = -(p * np.log(p)).sum(axis=-1)
    kl      = (p * (np.log(p) - np.log(q))).sum(axis=-1)
    w = entropy.sum()
    if w <= 0:
        return 100.0
    return float(100.0 * np.exp(-3.0 * (entropy * kl).sum() / w))


# ── Minimal stores / states ────────────────────────────────────────────────────

def _make_empty_store(ig: np.ndarray, n_seeds: int = 1) -> ObservationStore:
    H, W = ig.shape
    store = ObservationStore.__new__(ObservationStore)
    store.height = H; store.width = W; store.seeds_count = n_seeds
    store.counts = [np.zeros((H, W, N_CLASSES), dtype=np.int32) for _ in range(n_seeds)]
    store.latest = [[[None] * W for _ in range(H)] for _ in range(n_seeds)]
    store.settlement_snaps = {s: {} for s in range(n_seeds)}
    store.query_log = []
    return store


def _make_initial_state(ig: np.ndarray) -> dict:
    settlements = [
        {"x": int(x), "y": int(y),
         "population": 2.0, "food": 2.5, "wealth": 0.8,
         "defense": 1.2, "port": (int(ig[y, x]) == 2), "alive": True}
        for y in range(ig.shape[0]) for x in range(ig.shape[1])
        if ig[y, x] in (1, 2)
    ]
    return {"grid": ig.tolist(), "settlements": settlements}


# ── Prediction helpers ─────────────────────────────────────────────────────────

def predict_with_hist(ig, hist_prior, dynamics):
    """Run predictor with the given historical prior patched in."""
    import historical_prior as hp_module
    import predictor as pred_module

    original = hp_module.get_historical_prior
    hp_module.get_historical_prior = lambda *a, **kw: hist_prior
    # Also patch the reference already imported in predictor
    original_pred = pred_module.get_historical_prior
    pred_module.get_historical_prior = lambda *a, **kw: hist_prior

    try:
        from predictor import build_predictions
        store = _make_empty_store(ig)
        preds = build_predictions([_make_initial_state(ig)], store, dynamics, verbose=False)
        return preds[0].astype(np.float64)
    finally:
        hp_module.get_historical_prior = original
        pred_module.get_historical_prior = original_pred


def predict_baseline(ig, dynamics):
    """Run predictor with historical prior zeroed out (conf=0 → heuristic only)."""
    from historical_prior import HistoricalPrior
    uniform = np.full(N_CLASSES, 1.0 / N_CLASSES)
    dummy = HistoricalPrior({}, {}, uniform, 0)   # conf always 0 → heuristic dominates
    return predict_with_hist(ig, dummy, dynamics)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    data_dir = Path(__file__).parent / "data_prev_rounds"
    dynamics  = WorldDynamics()

    # Load all pairs upfront
    all_pairs = {}   # (r, s) -> (ig, gt)
    for r in range(1, 9):
        for s in range(5):
            ig_p = data_dir / f"ig_r{r}_seed{s}.npy"
            gt_p = data_dir / f"gt_r{r}_seed{s}.npy"
            if ig_p.exists() and gt_p.exists():
                all_pairs[(r, s)] = (np.load(ig_p), np.load(gt_p))

    baseline_scores = []
    new_scores      = []

    for r in range(1, 9):
        test_keys   = [(r, s) for s in range(5) if (r, s) in all_pairs]
        train_pairs = [v for (rr, ss), v in all_pairs.items() if rr != r]

        if not test_keys or not train_pairs:
            continue

        hist = _build_from_pairs(train_pairs)
        print(f"Round {r} — training on {len(train_pairs)} pairs from other rounds:")

        for (_, s) in test_keys:
            ig, gt = all_pairs[(r, s)]

            b = score(predict_baseline(ig, dynamics), gt)
            n = score(predict_with_hist(ig, hist, dynamics), gt)
            delta = n - b
            sign  = "+" if delta >= 0 else ""
            print(f"  seed {s}: baseline={b:.2f}  new={n:.2f}  ({sign}{delta:.2f})")
            baseline_scores.append(b)
            new_scores.append(n)

        print()

    print("=" * 50)
    print(f"Baseline avg : {np.mean(baseline_scores):.2f}")
    print(f"New avg      : {np.mean(new_scores):.2f}")
    print(f"Delta        : {np.mean(new_scores) - np.mean(baseline_scores):+.2f}")


if __name__ == "__main__":
    main()
