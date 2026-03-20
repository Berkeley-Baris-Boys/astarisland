"""
Prediction builder.

Approach (no local simulation):

1. Observed cells (n ≥ 1):
   Use empirical counts as the primary distribution.
   Blend with a dynamics-adjusted prior to smooth single-sample noise.
   Weight shifts toward data as n grows.

2. Unobserved cells:
   Use dynamics-adjusted terrain prior (conditioned on initial terrain,
   proximity to settlements, coastal status, forest adjacency).
   Smooth spatially from nearby observed cells.

3. Settlement-specific overrides:
   If we directly observed a settlement as alive/dead, apply that signal
   on top of everything else.

All 5 seeds share the same hidden parameters, so world_dynamics (estimated
by pooling across all seeds) applies uniformly to every seed's unobserved cells.

Never assign 0.0 to any class — always apply PROB_FLOOR and renormalize.
"""
from __future__ import annotations

import numpy as np

from config import (
    N_CLASSES, PROB_FLOOR, CLASS_NAMES,
    OCEAN_CODE, MOUNTAIN_CODE, SETTLEMENT_CODE, PORT_CODE,
)
from state import (
    ObservationStore, forest_adjacency, coastal_mask, settlement_influence,
    distance_to_coast, settlement_density,
)
from world_dynamics import WorldDynamics, dynamics_adjusted_prior


def safe_normalize(v: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    v = np.maximum(v.astype(np.float64), floor)
    s = v.sum()
    return v / s if s > 0 else np.full(N_CLASSES, 1.0 / N_CLASSES)


def build_predictions(
    initial_states: list[dict],
    store: ObservationStore,
    dynamics: WorldDynamics,
    *,
    verbose: bool = True,
) -> dict[int, np.ndarray]:
    """
    Build per-seed H×W×6 probability tensors.

    Parameters
    ----------
    initial_states : list of {grid, settlements} dicts from the API
    store          : ObservationStore with counts and settlement snapshots
    dynamics       : WorldDynamics estimated from pooled observations
    """
    H, W = store.height, store.width
    predictions: dict[int, np.ndarray] = {}

    for seed in range(store.seeds_count):
        ig      = np.asarray(initial_states[seed]["grid"], dtype=np.int32)
        counts  = store.counts[seed]              # H×W×6 int
        n_samp  = counts.sum(axis=2)              # H×W int
        obs_mask = n_samp > 0

        # Pre-compute spatial context
        has_settle, has_port, near_settle = settlement_influence(
            initial_states[seed], H, W, radius=4
        )
        fadj      = forest_adjacency(ig)    # H×W int: n adjacent forest cells
        coast     = coastal_mask(ig)         # H×W bool
        dist_coast= distance_to_coast(ig)    # H×W float: cells to nearest ocean
        settle_den= settlement_density(ig)   # H×W float: n settlements within r=5

        pred = np.zeros((H, W, N_CLASSES), dtype=np.float64)

        for y in range(H):
            for x in range(W):
                init_code = int(ig[y, x])
                n         = int(n_samp[y, x])
                cell_cnt  = counts[y, x].astype(np.float64)

                # ── Static cells (always the same class) ──────────────────────
                if init_code == OCEAN_CODE:
                    pred[y, x] = safe_normalize(
                        np.array([0.96, 0.01, 0.01, 0.01, 0.01, 0.00])
                    )
                    continue
                if init_code == MOUNTAIN_CODE:
                    pred[y, x] = safe_normalize(
                        np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95])
                    )
                    continue

                # ── Dynamics-adjusted prior for this cell ─────────────────────
                prior = dynamics_adjusted_prior(
                    init_code=init_code,
                    dyn=dynamics,
                    is_near_settlement=bool(near_settle[y, x]),
                    is_coastal=bool(coast[y, x]),
                    n_forest_adj=int(fadj[y, x]),
                    is_near_port=bool(has_port[y, x]) or (
                        bool(coast[y, x]) and bool(near_settle[y, x])
                    ),
                    dist_to_coast=float(dist_coast[y, x]),
                    local_settle_density=float(settle_den[y, x]),
                )

                # ── Observed cell: empirical counts blend with prior ───────────
                if n > 0:
                    empirical = cell_cnt / float(n)
                    # Trust data more as n grows:
                    # n=1 → 65% data, n=3 → 82%, n=5 → 90%
                    data_weight = 1.0 - 1.0 / (1.0 + 0.9 * n)
                    blended = data_weight * empirical + (1.0 - data_weight) * prior
                else:
                    # Unobserved: pure prior
                    blended = prior

                pred[y, x] = safe_normalize(blended)

        # ── Settlement alive/dead override ────────────────────────────────────
        _apply_settlement_signals(pred, store.settlement_snaps[seed], W, H, dynamics)

        # ── Spatial smoothing (unobserved non-static cells only) ──────────────
        pred = _spatial_smooth(pred, obs_mask, ig)

        # ── Final floor + renormalize ─────────────────────────────────────────
        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=2, keepdims=True)

        predictions[seed] = pred.astype(np.float32)

        if verbose:
            _print_seed_summary(seed, pred, obs_mask, n_samp)

    return predictions


def _apply_settlement_signals(
    pred: np.ndarray,
    snaps: list[dict],
    W: int,
    H: int,
    dynamics: WorldDynamics,
) -> None:
    """
    Apply last-observed settlement state as a strong signal.
    Alive settlements → high Settlement/Port probability.
    Dead settlements  → high Ruin probability.
    Low food          → increased risk of Ruin.
    """
    # Use last snapshot for each position
    last: dict[tuple[int, int], dict] = {}
    for snap in snaps:
        for s in snap["settlements"]:
            if not isinstance(s, dict):
                continue
            x, y = int(s.get("x", -1)), int(s.get("y", -1))
            if 0 <= x < W and 0 <= y < H:
                last[(x, y)] = s

    for (x, y), s in last.items():
        alive    = bool(s.get("alive", True))
        has_port = bool(s.get("has_port", False))
        food     = s.get("food")
        pop      = s.get("population")

        if not alive:
            # Dead → Ruin almost certainly
            pred[y, x] = safe_normalize(np.array([0.07, 0.03, 0.01, 0.82, 0.05, 0.02]))
        else:
            if has_port:
                base = np.array([0.02, 0.30, 0.54, 0.08, 0.04, 0.02])
            else:
                base = np.array([0.02, 0.86, 0.03, 0.05, 0.03, 0.01])

            # Low food → risk of future collapse (but we observed it alive, so it's
            # not yet a ruin — nudge probability, don't override)
            if food is not None and float(food) < 0.3:
                collapse_risk = min(0.35, (0.3 - float(food)) * 1.5)
                base[3] += collapse_risk
                base[1] -= collapse_risk * 0.7
                if has_port:
                    base[2] -= collapse_risk * 0.3

            pred[y, x] = safe_normalize(base * pred[y, x])


def _spatial_smooth(
    pred: np.ndarray,
    obs_mask: np.ndarray,
    init_grid: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Gaussian smooth unobserved cells using nearby observed values.
    Static terrain (ocean, mountain) is never smoothed.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return pred

    static = np.isin(init_grid, [OCEAN_CODE, MOUNTAIN_CODE])
    smooth_target = ~obs_mask & ~static

    for c in range(N_CLASSES):
        blurred = gaussian_filter(pred[:, :, c].astype(np.float32), sigma=sigma)
        pred[:, :, c] = np.where(
            smooth_target,
            0.55 * pred[:, :, c] + 0.45 * blurred,
            pred[:, :, c],
        )
    return pred


def validate_prediction(pred: np.ndarray, seed: int) -> None:
    if pred.ndim != 3 or pred.shape[2] != N_CLASSES:
        raise ValueError(f"Seed {seed}: invalid shape {pred.shape}")
    sums = pred.sum(axis=2)
    max_dev = float(np.abs(sums - 1.0).max())
    if max_dev > 1e-3:
        raise ValueError(f"Seed {seed}: probs don't sum to 1.0 (max dev={max_dev:.6f})")
    if (pred < 0).any():
        raise ValueError(f"Seed {seed}: negative probabilities")


def _print_seed_summary(
    seed: int, pred: np.ndarray, obs_mask: np.ndarray, n_samp: np.ndarray
) -> None:
    argmax     = pred.argmax(axis=2)
    cls_counts = " ".join(
        f"{CLASS_NAMES[c][0]}:{int((argmax==c).sum())}" for c in range(N_CLASSES)
    )
    cov      = 100.0 * obs_mask.sum() / obs_mask.size
    avg_samp = float(n_samp[obs_mask].mean()) if obs_mask.any() else 0.0
    avg_conf = float(pred.max(axis=2).mean())
    print(f"  Seed {seed}: {cov:.0f}% obs, {avg_samp:.1f} avg_samples, "
          f"avg_conf={avg_conf:.3f} | {cls_counts}")
