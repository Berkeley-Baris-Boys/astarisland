"""
Prediction builder.

Primary strategy:

1. Build a current-round empirical transition model from observed cells.
2. Infer round-level simulator parameters from observed terrain outcomes and
   settlement snapshots.
3. Run local Monte Carlo rollouts for each seed with the inferred parameters.
4. Blend empirical transitions, calibrated rollouts, and heuristic priors.
5. Apply a probability floor and keep a heuristic fallback path.

If rollout inference fails for any reason, fall back to the older heuristic
model so the solver still produces a valid submission.
"""
from __future__ import annotations

import math

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
ROLLOUT_MAX_WEIGHT = 0.26

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

    Preferred path is a hybrid empirical + simulator predictor. The heuristic
    path remains as a safety net.
    """
    if estimate_params_from_observations is not None and monte_carlo is not None:
        try:
            return _build_hybrid_predictions(
                initial_states, store, dynamics, verbose=verbose
            )
        except Exception as exc:
            if verbose:
                print(f"Hybrid predictor failed, falling back to heuristics: {exc}")

    return _build_heuristic_predictions(
        initial_states, store, dynamics, verbose=verbose
    )


def _build_hybrid_predictions(
    initial_states: list[dict],
    store: ObservationStore,
    dynamics: WorldDynamics,
    *,
    verbose: bool = True,
) -> dict[int, np.ndarray]:
    H, W = store.height, store.width
    observations = _store_to_observations(store)
    contexts = _build_seed_contexts(initial_states, H, W)
    transition_model = _build_transition_model(store, contexts)
    inferred_params = estimate_params_from_observations(
        observations,
        initial_states,
        {s: store.settlement_snaps[s] for s in range(store.seeds_count)},
        verbose=verbose,
    )

    rollout_predictions: dict[int, np.ndarray] = {}
    for seed in range(store.seeds_count):
        init_state = initial_states[seed]
        init_grid = np.asarray(init_state["grid"], dtype=np.int64)
        rollout_predictions[seed] = monte_carlo(
            init_grid,
            init_state.get("settlements", []),
            inferred_params,
            n_runs=ROLLOUT_MC_RUNS,
            seed=ROLLOUT_BASE_SEED + seed * 1009,
        ).astype(np.float64, copy=False)

    rollout_predictions, rollout_trust, rollout_scale = _calibrate_rollouts(
        rollout_predictions, store
    )

    if verbose:
        print(
            "Hybrid predictor: "
            f"{ROLLOUT_MC_RUNS} simulations per seed, "
            f"rollout_trust={rollout_trust:.3f}, "
            f"class_scale={np.round(rollout_scale, 3).tolist()}"
        )

    predictions: dict[int, np.ndarray] = {}
    for seed in range(store.seeds_count):
        context = contexts[seed]
        init_grid = context["grid"]
        counts = store.counts[seed].astype(np.float64)
        n_samp = counts.sum(axis=2, keepdims=True)
        obs_mask = n_samp[:, :, 0] > 0
        pred = np.zeros((H, W, N_CLASSES), dtype=np.float64)

        for y in range(H):
            for x in range(W):
                init_code = int(init_grid[y, x])
                if init_code == OCEAN_CODE:
                    pred[y, x] = _ocean_distribution()
                    continue
                if init_code == MOUNTAIN_CODE:
                    pred[y, x] = _mountain_distribution()
                    continue

                heuristic = dynamics_adjusted_prior(
                    init_code=init_code,
                    dyn=dynamics,
                    is_near_settlement=bool(context["near_settle"][y, x]),
                    is_coastal=bool(context["coast"][y, x]),
                    n_forest_adj=int(context["fadj"][y, x]),
                    is_near_port=bool(context["has_port"][y, x]) or (
                        bool(context["coast"][y, x]) and bool(context["near_settle"][y, x])
                    ),
                    dist_to_coast=float(context["dist_coast"][y, x]),
                    local_settle_density=float(context["settle_den"][y, x]),
                )
                empirical_prior, support = _lookup_transition_prior(
                    transition_model,
                    init_code=init_code,
                    is_near_settlement=bool(context["near_settle"][y, x]),
                    is_coastal=bool(context["coast"][y, x]),
                    forest_bucket=int(context["fadj_bucket"][y, x]),
                )
                rollout = rollout_predictions[seed][y, x]

                if obs_mask[y, x]:
                    empirical = safe_normalize(counts[y, x] / float(n_samp[y, x, 0]))
                    weights = np.array([
                        min(0.88, 0.60 + 0.10 * min(float(n_samp[y, x, 0]), 2.0)),
                        0.22 + 0.18 * support,
                        0.08 + 0.14 * rollout_trust,
                        0.10 + 0.08 * (1.0 - support),
                    ])
                    pred[y, x] = _blend_distributions(
                        [empirical, empirical_prior, rollout, heuristic],
                        weights,
                    )
                else:
                    weights = np.array([
                        0.42 + 0.40 * support,
                        0.12 + ROLLOUT_MAX_WEIGHT * rollout_trust,
                        0.22 + 0.24 * (1.0 - support),
                    ])
                    cell_pred = _blend_distributions(
                        [empirical_prior, rollout, heuristic],
                        weights,
                    )
                    pred[y, x] = _conservative_unobserved_adjustment(
                        cell_pred,
                        empirical_prior=empirical_prior,
                        heuristic_prior=heuristic,
                        near_settlement=bool(context["near_settle"][y, x]),
                        is_coastal=bool(context["coast"][y, x]),
                        init_code=init_code,
                    )

        _force_static_cells(pred, init_grid)
        pred = _spatial_smooth(pred, obs_mask, init_grid, sigma=0.9)
        for y in range(H):
            for x in range(W):
                if obs_mask[y, x]:
                    continue
                init_code = int(init_grid[y, x])
                if init_code in (OCEAN_CODE, MOUNTAIN_CODE):
                    continue
                heuristic = dynamics_adjusted_prior(
                    init_code=init_code,
                    dyn=dynamics,
                    is_near_settlement=bool(context["near_settle"][y, x]),
                    is_coastal=bool(context["coast"][y, x]),
                    n_forest_adj=int(context["fadj"][y, x]),
                    is_near_port=bool(context["has_port"][y, x]) or (
                        bool(context["coast"][y, x]) and bool(context["near_settle"][y, x])
                    ),
                    dist_to_coast=float(context["dist_coast"][y, x]),
                    local_settle_density=float(context["settle_den"][y, x]),
                )
                empirical_prior, _ = _lookup_transition_prior(
                    transition_model,
                    init_code=init_code,
                    is_near_settlement=bool(context["near_settle"][y, x]),
                    is_coastal=bool(context["coast"][y, x]),
                    forest_bucket=int(context["fadj_bucket"][y, x]),
                )
                pred[y, x] = _conservative_unobserved_adjustment(
                    pred[y, x],
                    empirical_prior=empirical_prior,
                    heuristic_prior=heuristic,
                    near_settlement=bool(context["near_settle"][y, x]),
                    is_coastal=bool(context["coast"][y, x]),
                    init_code=init_code,
                )

        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=2, keepdims=True)
        predictions[seed] = pred.astype(np.float32)

        if verbose:
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


def _build_seed_contexts(
    initial_states: list[dict], height: int, width: int
) -> dict[int, dict[str, np.ndarray]]:
    contexts: dict[int, dict[str, np.ndarray]] = {}
    for seed, state in enumerate(initial_states):
        grid = np.asarray(state["grid"], dtype=np.int32)
        has_settle, has_port, near_settle = settlement_influence(
            state, height, width, radius=4
        )
        fadj = forest_adjacency(grid)
        contexts[seed] = {
            "grid": grid,
            "has_settle": has_settle,
            "has_port": has_port,
            "near_settle": near_settle,
            "coast": coastal_mask(grid),
            "dist_coast": distance_to_coast(grid),
            "settle_den": settlement_density(grid),
            "fadj": fadj,
            "fadj_bucket": np.minimum(fadj, 2).astype(np.int8),
        }
    return contexts


def _build_transition_model(
    store: ObservationStore,
    contexts: dict[int, dict[str, np.ndarray]],
) -> dict[str, object]:
    counts_by_key: dict[tuple, np.ndarray] = {}
    global_counts = np.zeros(N_CLASSES, dtype=np.float64)

    for seed in range(store.seeds_count):
        ctx = contexts[seed]
        counts = store.counts[seed].astype(np.float64)
        totals = counts.sum(axis=2)
        ys, xs = np.where(totals > 0)
        for y, x in zip(ys.tolist(), xs.tolist()):
            cell_counts = counts[y, x]
            global_counts += cell_counts
            keys = _transition_keys(
                init_code=int(ctx["grid"][y, x]),
                is_near_settlement=bool(ctx["near_settle"][y, x]),
                is_coastal=bool(ctx["coast"][y, x]),
                forest_bucket=int(ctx["fadj_bucket"][y, x]),
            )
            for key in keys[:-1]:
                bucket = counts_by_key.get(key)
                if bucket is None:
                    counts_by_key[key] = cell_counts.copy()
                else:
                    bucket += cell_counts

    if global_counts.sum() <= 0:
        global_counts += 1.0

    return {
        "counts_by_key": counts_by_key,
        "global_counts": global_counts,
        "global_prior": safe_normalize(global_counts),
    }


def _transition_keys(
    init_code: int,
    is_near_settlement: bool,
    is_coastal: bool,
    forest_bucket: int,
) -> list[tuple]:
    near = int(is_near_settlement)
    coast = int(is_coastal)
    fb = int(forest_bucket)
    return [
        ("full", init_code, near, coast, fb),
        ("context", init_code, near, coast),
        ("near", init_code, near),
        ("init", init_code),
        ("global",),
    ]


def _lookup_transition_prior(
    model: dict[str, object],
    *,
    init_code: int,
    is_near_settlement: bool,
    is_coastal: bool,
    forest_bucket: int,
) -> tuple[np.ndarray, float]:
    counts_by_key = model["counts_by_key"]
    global_counts = model["global_counts"]
    global_prior = model["global_prior"]

    dists: list[np.ndarray] = []
    weights: list[float] = []
    support_score = 0.0
    level_bases = [1.15, 0.85, 0.55, 0.35]
    level_scales = [50.0, 90.0, 150.0, 240.0]

    keys = _transition_keys(
        init_code=init_code,
        is_near_settlement=is_near_settlement,
        is_coastal=is_coastal,
        forest_bucket=forest_bucket,
    )
    for idx, key in enumerate(keys[:-1]):
        bucket = counts_by_key.get(key)
        if bucket is None:
            continue
        total = float(bucket.sum())
        if total <= 0:
            continue
        conf = 1.0 - math.exp(-total / level_scales[idx])
        pseudo = global_prior * max(2.0, 10.0 / math.sqrt(total + 1.0))
        dist = safe_normalize(bucket + pseudo)
        weight = level_bases[idx] * conf
        dists.append(dist)
        weights.append(weight)
        support_score += weight

    dists.append(safe_normalize(global_counts))
    weights.append(0.18)

    support = min(1.0, support_score / 2.2)
    return _blend_distributions(dists, np.asarray(weights, dtype=np.float64)), support


def _calibrate_rollouts(
    rollout_predictions: dict[int, np.ndarray],
    store: ObservationStore,
) -> tuple[dict[int, np.ndarray], float, np.ndarray]:
    empirical_totals = np.zeros(N_CLASSES, dtype=np.float64)
    rollout_totals = np.zeros(N_CLASSES, dtype=np.float64)
    weighted_kl_sum = 0.0
    weight_sum = 0.0

    for seed in rollout_predictions:
        rollout = rollout_predictions[seed].astype(np.float64, copy=False)
        counts = store.counts[seed].astype(np.float64)
        n_samp = counts.sum(axis=2)
        mask = n_samp > 0
        if not mask.any():
            continue

        empirical_totals += counts[mask].sum(axis=0)
        rollout_totals += (rollout[mask] * n_samp[mask, None]).sum(axis=0)

        empirical = np.divide(
            counts[mask],
            n_samp[mask, None],
            out=np.zeros_like(counts[mask]),
            where=n_samp[mask, None] > 0,
        )
        q = np.maximum(rollout[mask], PROB_FLOOR)
        q = q / q.sum(axis=1, keepdims=True)
        kls = np.sum(
            empirical * (np.log(np.maximum(empirical, 1e-9)) - np.log(q)),
            axis=1,
        )
        weighted_kl_sum += float((kls * n_samp[mask]).sum())
        weight_sum += float(n_samp[mask].sum())

    if weight_sum <= 0:
        return rollout_predictions, 0.0, np.ones(N_CLASSES, dtype=np.float64)

    class_scale = np.sqrt((empirical_totals + 1.0) / (rollout_totals + 1.0))
    class_scale = np.clip(class_scale, 0.65, 1.45)
    mean_kl = weighted_kl_sum / weight_sum
    trust = float(np.clip(math.exp(-1.8 * mean_kl), 0.12, 0.95))

    calibrated: dict[int, np.ndarray] = {}
    for seed, rollout in rollout_predictions.items():
        adjusted = rollout.astype(np.float64, copy=True) * class_scale.reshape(1, 1, -1)
        adjusted = adjusted / adjusted.sum(axis=2, keepdims=True)
        calibrated[seed] = adjusted

    return calibrated, trust, class_scale


def _blend_distributions(
    dists: list[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    w = np.maximum(weights.astype(np.float64), 0.0)
    if w.sum() <= 0:
        return np.full(N_CLASSES, 1.0 / N_CLASSES)
    w = w / w.sum()
    out = np.zeros(N_CLASSES, dtype=np.float64)
    for weight, dist in zip(w, dists):
        out += weight * dist
    return safe_normalize(out)


def _conservative_unobserved_adjustment(
    pred: np.ndarray,
    *,
    empirical_prior: np.ndarray,
    heuristic_prior: np.ndarray,
    near_settlement: bool,
    is_coastal: bool,
    init_code: int,
) -> np.ndarray:
    out = pred.astype(np.float64, copy=True)

    settlement_cap = max(
        float(empirical_prior[1] + empirical_prior[2]),
        float(heuristic_prior[1] + heuristic_prior[2]),
    )
    if near_settlement:
        settlement_cap = min(settlement_cap + 0.06, 0.35)
    else:
        settlement_cap = min(settlement_cap + 0.02, 0.20)

    current_settlement = float(out[1] + out[2])
    if current_settlement > settlement_cap:
        excess = current_settlement - settlement_cap
        scale = settlement_cap / max(current_settlement, 1e-9)
        out[1] *= scale
        out[2] *= scale
        redist = heuristic_prior.copy()
        redist[1] = 0.0
        redist[2] = 0.0
        redist = safe_normalize(redist)
        out += excess * redist

    port_cap = max(float(empirical_prior[2]), float(heuristic_prior[2]))
    if init_code == PORT_CODE:
        port_cap = min(max(port_cap, 0.22) + 0.10, 0.45)
    elif is_coastal and near_settlement:
        port_cap = min(port_cap + 0.04, 0.12)
    elif is_coastal:
        port_cap = min(port_cap + 0.02, 0.08)
    else:
        port_cap = min(port_cap + 0.01, 0.04)

    if out[2] > port_cap:
        excess = float(out[2] - port_cap)
        out[2] = port_cap
        redist = heuristic_prior.copy()
        redist[2] = 0.0
        redist = safe_normalize(redist)
        out += excess * redist

    ruin_cap = max(float(empirical_prior[3]), float(heuristic_prior[3]))
    if init_code == 3:
        ruin_cap = min(max(ruin_cap, 0.22) + 0.10, 0.45)
    elif init_code in (SETTLEMENT_CODE, PORT_CODE):
        ruin_cap = min(ruin_cap + 0.08, 0.30)
    else:
        ruin_cap = min(ruin_cap + (0.04 if near_settlement else 0.02), 0.14)

    if out[3] > ruin_cap:
        excess = float(out[3] - ruin_cap)
        out[3] = ruin_cap
        redist = heuristic_prior.copy()
        redist[3] = 0.0
        redist = safe_normalize(redist)
        out += excess * redist

    forest_cap = max(float(empirical_prior[4]), float(heuristic_prior[4]))
    if init_code == 4:
        forest_cap = min(forest_cap + 0.10, 0.75)
    else:
        forest_cap = min(forest_cap + 0.05, 0.35)

    if out[4] > forest_cap:
        excess = float(out[4] - forest_cap)
        out[4] = forest_cap
        redist = heuristic_prior.copy()
        redist[4] = 0.0
        redist = safe_normalize(redist)
        out += excess * redist

    return safe_normalize(out)


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
