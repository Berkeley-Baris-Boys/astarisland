"""
Prediction builder.

Primary strategy:

1. Infer round-level simulator parameters from observed terrain outcomes and
   settlement snapshots.
2. Run local Monte Carlo rollouts for each seed with the inferred parameters.
3. Blend rollout probabilities with direct empirical counts at observed cells.
4. Apply a probability floor and keep a heuristic fallback path.

If rollout inference fails for any reason, fall back to the older heuristic
model so the solver still produces a valid submission.
"""
from __future__ import annotations

import numpy as np

from config import (
    N_CLASSES,
    PROB_FLOOR,
    CLASS_NAMES,
    OCEAN_CODE,
    MOUNTAIN_CODE,
    SETTLEMENT_CODE,
    PORT_CODE,
)
from state import (
    ObservationStore,
    forest_adjacency,
    coastal_mask,
    settlement_influence,
    distance_to_coast,
    settlement_density,
)
from world_dynamics import WorldDynamics, dynamics_adjusted_prior

ROLLOUT_MC_RUNS = 80
ROLLOUT_BASE_SEED = 20260320

try:
    from testing.simulator import estimate_params_from_observations, monte_carlo
except Exception:
    estimate_params_from_observations = None
    monte_carlo = None


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
    Build per-seed HxWx6 probability tensors.

    Preferred path is simulator-backed Monte Carlo. The heuristic path remains
    as a safety net.
    """
    if estimate_params_from_observations is not None and monte_carlo is not None:
        try:
            return _build_rollout_predictions(
                initial_states, store, dynamics, verbose=verbose
            )
        except Exception as exc:
            if verbose:
                print(f"Rollout predictor failed, falling back to heuristics: {exc}")

    return _build_heuristic_predictions(
        initial_states, store, dynamics, verbose=verbose
    )


def _build_rollout_predictions(
    initial_states: list[dict],
    store: ObservationStore,
    dynamics: WorldDynamics,
    *,
    verbose: bool = True,
) -> dict[int, np.ndarray]:
    H, W = store.height, store.width
    observations = _store_to_observations(store)
    inferred_params = estimate_params_from_observations(
        observations,
        initial_states,
        {s: store.settlement_snaps[s] for s in range(store.seeds_count)},
        verbose=verbose,
    )

    if verbose:
        print(f"Rollout predictor: {ROLLOUT_MC_RUNS} simulations per seed")

    predictions: dict[int, np.ndarray] = {}
    for seed in range(store.seeds_count):
        init_state = initial_states[seed]
        init_grid = np.asarray(init_state["grid"], dtype=np.int64)

        pred = monte_carlo(
            init_grid,
            init_state.get("settlements", []),
            inferred_params,
            n_runs=ROLLOUT_MC_RUNS,
            seed=ROLLOUT_BASE_SEED + seed * 1009,
        ).astype(np.float64, copy=False)

        counts = store.counts[seed].astype(np.float64)
        n_samp = counts.sum(axis=2, keepdims=True)
        empirical = np.divide(
            counts,
            np.maximum(n_samp, 1.0),
            out=np.zeros_like(counts),
            where=n_samp > 0,
        )
        # Observed cells should dominate quickly; the rollout model fills the map.
        alpha = np.clip(n_samp / 3.0, 0.0, 1.0)
        pred = (1.0 - alpha) * pred + alpha * empirical

        _force_static_cells(pred, init_grid)
        # Snapshot terrain outcomes are already represented in the empirical
        # counts above. A single settlement snapshot should shape parameter
        # inference, not overwrite the full rollout distribution.

        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=2, keepdims=True)
        predictions[seed] = pred.astype(np.float32)

        if verbose:
            obs_mask = n_samp[:, :, 0] > 0
            _print_seed_summary(seed, pred, obs_mask, n_samp[:, :, 0])

    return predictions


def _build_heuristic_predictions(
    initial_states: list[dict],
    store: ObservationStore,
    dynamics: WorldDynamics,
    *,
    verbose: bool = True,
) -> dict[int, np.ndarray]:
    """
    Legacy heuristic predictor.

    Kept as a fallback when simulator-backed rollouts are unavailable.
    """
    H, W = store.height, store.width
    predictions: dict[int, np.ndarray] = {}

    for seed in range(store.seeds_count):
        ig = np.asarray(initial_states[seed]["grid"], dtype=np.int32)
        counts = store.counts[seed]
        n_samp = counts.sum(axis=2)
        obs_mask = n_samp > 0

        has_settle, has_port, near_settle = settlement_influence(
            initial_states[seed], H, W, radius=4
        )
        fadj = forest_adjacency(ig)
        coast = coastal_mask(ig)
        dist_coast = distance_to_coast(ig)
        settle_den = settlement_density(ig)

        pred = np.zeros((H, W, N_CLASSES), dtype=np.float64)

        for y in range(H):
            for x in range(W):
                init_code = int(ig[y, x])
                n = int(n_samp[y, x])
                cell_cnt = counts[y, x].astype(np.float64)

                if init_code == OCEAN_CODE:
                    pred[y, x] = _ocean_distribution()
                    continue
                if init_code == MOUNTAIN_CODE:
                    pred[y, x] = _mountain_distribution()
                    continue

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

                if n > 0:
                    empirical = cell_cnt / float(n)
                    data_weight = 1.0 - 1.0 / (1.0 + 0.9 * n)
                    blended = data_weight * empirical + (1.0 - data_weight) * prior
                else:
                    blended = prior

                pred[y, x] = safe_normalize(blended)

        _apply_settlement_signals(pred, store.settlement_snaps[seed], W, H, dynamics)
        pred = _spatial_smooth(pred, obs_mask, ig)
        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=2, keepdims=True)

        predictions[seed] = pred.astype(np.float32)

        if verbose:
            _print_seed_summary(seed, pred, obs_mask, n_samp)

    return predictions


def _store_to_observations(store: ObservationStore) -> dict[int, np.ndarray]:
    observations: dict[int, np.ndarray] = {}
    for seed in range(store.seeds_count):
        obs = np.full((store.height, store.width), -1, dtype=np.int32)
        for y, row in enumerate(store.latest[seed]):
            for x, raw_val in enumerate(row):
                if raw_val is not None:
                    obs[y, x] = int(raw_val)
        observations[seed] = obs
    return observations


def _ocean_distribution() -> np.ndarray:
    return safe_normalize(np.array([0.96, 0.01, 0.01, 0.01, 0.01, 0.00]))


def _mountain_distribution() -> np.ndarray:
    return safe_normalize(np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95]))


def _force_static_cells(pred: np.ndarray, init_grid: np.ndarray) -> None:
    ocean_mask = init_grid == OCEAN_CODE
    mountain_mask = init_grid == MOUNTAIN_CODE
    if ocean_mask.any():
        pred[ocean_mask] = _ocean_distribution()
    if mountain_mask.any():
        pred[mountain_mask] = _mountain_distribution()


def _apply_settlement_signals(
    pred: np.ndarray,
    snaps: list[dict],
    W: int,
    H: int,
    dynamics: WorldDynamics,
) -> None:
    """
    Apply last-observed settlement state as a strong signal.
    Alive settlements -> high Settlement/Port probability.
    Dead settlements  -> high Ruin probability.
    Low food          -> increased risk of Ruin.
    """
    last: dict[tuple[int, int], dict] = {}
    for snap in snaps:
        for s in snap["settlements"]:
            if not isinstance(s, dict):
                continue
            x, y = int(s.get("x", -1)), int(s.get("y", -1))
            if 0 <= x < W and 0 <= y < H:
                last[(x, y)] = s

    for (x, y), s in last.items():
        alive = bool(s.get("alive", True))
        has_port = bool(s.get("has_port", False))
        food = s.get("food")

        if not alive:
            pred[y, x] = safe_normalize(np.array([0.07, 0.03, 0.01, 0.82, 0.05, 0.02]))
        else:
            if has_port:
                base = np.array([0.02, 0.30, 0.54, 0.08, 0.04, 0.02])
            else:
                base = np.array([0.02, 0.86, 0.03, 0.05, 0.03, 0.01])

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
    argmax = pred.argmax(axis=2)
    cls_counts = " ".join(
        f"{CLASS_NAMES[c][0]}:{int((argmax == c).sum())}" for c in range(N_CLASSES)
    )
    cov = 100.0 * obs_mask.sum() / obs_mask.size
    avg_samp = float(n_samp[obs_mask].mean()) if obs_mask.any() else 0.0
    avg_conf = float(pred.max(axis=2).mean())
    print(
        f"  Seed {seed}: {cov:.0f}% obs, {avg_samp:.1f} avg_samples, "
        f"avg_conf={avg_conf:.3f} | {cls_counts}"
    )
