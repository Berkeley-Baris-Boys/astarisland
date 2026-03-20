"""
Prediction builder.

Primary strategy:

1. Build an empirical-constrained predictor from current-round observations.
2. Smooth the settlement and ruin fields to better match the spatially
   coherent ground-truth probability maps.
3. On fully observed rounds, calibrate noisy single-sample cells against the
   repeated cells, then denoise the calibrated empirical field.
4. Re-anchor observed rare-class cells on partial-coverage rounds only.
5. Fall back to hybrid rollout or heuristics if the empirical path fails.
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
    RUIN_CODE,
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
EMPIRICAL_FINAL_SCALE = np.array([1.03, 0.97, 0.45, 0.40, 1.08, 0.85], dtype=np.float64)
EMPIRICAL_SETTLEMENT_SMOOTH_ALPHA = 0.60
EMPIRICAL_SETTLEMENT_SMOOTH_SIGMA = 1.4
EMPIRICAL_RUIN_SMOOTH_ALPHA = 0.55
EMPIRICAL_RUIN_SMOOTH_SIGMA = 1.6
EMPIRICAL_FOREST_SMOOTH_ALPHA = 0.15
EMPIRICAL_FOREST_SMOOTH_SIGMA = 0.8
EMPIRICAL_GLOBAL_SMOOTH_ALPHA = 0.08
EMPIRICAL_GLOBAL_SMOOTH_SIGMA = 1.2
EMPIRICAL_FULL_COVERAGE_FIELD_SIGMA = 10.0
EMPIRICAL_FULL_COVERAGE_FIELD_ALPHA_SINGLE = 0.70
EMPIRICAL_FULL_COVERAGE_FIELD_ALPHA_DOUBLE = 0.40
EMPIRICAL_FULL_COVERAGE_FIELD_ALPHA_MULTI = 0.25
EMPIRICAL_FULL_COVERAGE_FIELD_CONF_BIAS = 0.02
EMPIRICAL_FULL_COVERAGE_TEMPLATE_ALPHA_SINGLE = 0.24
EMPIRICAL_FULL_COVERAGE_TEMPLATE_ALPHA_DOUBLE = 0.06
EMPIRICAL_FULL_COVERAGE_TEMPLATE_ALPHA_MULTI = 0.00
EMPIRICAL_FULL_COVERAGE_ITER_SIGMA = 5.0
EMPIRICAL_FULL_COVERAGE_ITER_FOREST_SIGMA = 3.0
EMPIRICAL_FULL_COVERAGE_ITER_TAU = 1.4
EMPIRICAL_FULL_COVERAGE_ITER_STEPS = 5
EMPIRICAL_FULL_COVERAGE_ITER_SETTLE_BOOST = 1.0
EMPIRICAL_FULL_COVERAGE_ITER_RUIN_BOOST = 1.10
EMPIRICAL_FULL_COVERAGE_BLEND_SINGLE = 0.60
EMPIRICAL_FULL_COVERAGE_BLEND_DOUBLE = 0.20
EMPIRICAL_FULL_COVERAGE_BLEND_MULTI = 0.20
EMPIRICAL_OBS_SETTLEMENT_ANCHOR_BASE = 0.10
EMPIRICAL_OBS_SETTLEMENT_ANCHOR_BONUS = 0.20
EMPIRICAL_OBS_PORT_RUIN_ANCHOR_BASE = 0.20
EMPIRICAL_OBS_PORT_RUIN_ANCHOR_BONUS = 0.22
SETTLEMENT_INTENSITY_BLEND_ALPHA = 0.22
SETTLEMENT_INTENSITY_SIGMA = 2.2

try:
    from testing.simulator import estimate_params_from_observations, monte_carlo
except Exception:
    estimate_params_from_observations = None
    monte_carlo = None


def safe_normalize(v: np.ndarray, floor: float = PROB_FLOOR) -> np.ndarray:
    v = np.maximum(v.astype(np.float64), floor)
    s = v.sum()
    return v / s if s > 0 else np.full(N_CLASSES, 1.0 / N_CLASSES)


def safe_normalize_last_axis(
    arr: np.ndarray,
    floor: float = PROB_FLOOR,
) -> np.ndarray:
    out = np.maximum(arr.astype(np.float64, copy=False), floor)
    sums = out.sum(axis=-1, keepdims=True)
    return np.divide(
        out,
        sums,
        out=np.full_like(out, 1.0 / N_CLASSES),
        where=sums > 0,
    )


def build_predictions(
    initial_states: list[dict],
    store: ObservationStore,
    dynamics: WorldDynamics,
    *,
    verbose: bool = True,
) -> dict[int, np.ndarray]:
    """
    Build per-seed HxWx6 probability tensors.

    Preferred path is the empirical constrained predictor with semantic
    settlement/forest smoothing. Hybrid rollout and heuristics remain as
    fallbacks when that path fails.
    """
    try:
        return _build_empirical_constrained_predictions(
            initial_states, store, dynamics, verbose=verbose
        )
    except Exception as exc:
        if verbose:
            print(
                "Empirical constrained predictor failed, "
                f"falling back to hybrid: {exc}"
            )

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
        _enforce_hard_constraints(pred, init_grid)
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
        _enforce_hard_constraints(pred, ig)

        predictions[seed] = pred.astype(np.float32)

        if verbose:
            _print_seed_summary(seed, pred, obs_mask, n_samp)

    return predictions


def _build_empirical_constrained_predictions(
    initial_states: list[dict],
    store: ObservationStore,
    dynamics: WorldDynamics,
    *,
    verbose: bool = True,
) -> dict[int, np.ndarray]:
    """
    Empirical-only alternative.

    Uses current-round transition priors plus heuristic geography, with no
    simulator rollout contribution on unobserved cells. This is intended as a
    structurally different fallback when rollout-based predictions appear
    overconfident on rare classes like ports and ruins.
    """
    H, W = store.height, store.width
    contexts = _build_seed_contexts(initial_states, H, W)
    transition_model = _build_transition_model(store, contexts)
    predictions: dict[int, np.ndarray] = {}
    fully_observed_round = all(
        bool((store.counts[s].sum(axis=2) > 0).all())
        for s in range(store.seeds_count)
    )
    full_coverage_model = None
    if fully_observed_round:
        full_coverage_model = _build_full_coverage_calibration_model(store, contexts)

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

                if obs_mask[y, x]:
                    empirical = safe_normalize(counts[y, x] / float(n_samp[y, x, 0]))
                    weights = np.array([
                        min(0.92, 0.66 + 0.12 * min(float(n_samp[y, x, 0]), 2.0)),
                        0.22 + 0.22 * support,
                        0.10 + 0.08 * (1.0 - support),
                    ])
                    pred[y, x] = _blend_distributions(
                        [empirical, empirical_prior, heuristic],
                        weights,
                    )
                else:
                    weights = np.array([
                        0.58 + 0.24 * support,
                        0.42 - 0.24 * support,
                    ])
                    pred[y, x] = _blend_distributions(
                        [empirical_prior, heuristic],
                        weights,
                    )

                pred[y, x] = _conservative_unobserved_adjustment(
                    pred[y, x],
                    empirical_prior=empirical_prior,
                    heuristic_prior=heuristic,
                    near_settlement=bool(context["near_settle"][y, x]),
                    is_coastal=bool(context["coast"][y, x]),
                    init_code=init_code,
                )

        _force_static_cells(pred, init_grid)
        pred = _spatial_smooth(pred, obs_mask, init_grid, sigma=0.55)
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

        pred = _apply_final_class_calibration(
            pred,
            scale=EMPIRICAL_FINAL_SCALE,
        )
        pred = _apply_semantic_class_smoothing(
            pred,
            init_grid,
            settlement_alpha=EMPIRICAL_SETTLEMENT_SMOOTH_ALPHA,
            settlement_sigma=EMPIRICAL_SETTLEMENT_SMOOTH_SIGMA,
            ruin_alpha=EMPIRICAL_RUIN_SMOOTH_ALPHA,
            ruin_sigma=EMPIRICAL_RUIN_SMOOTH_SIGMA,
            forest_alpha=EMPIRICAL_FOREST_SMOOTH_ALPHA,
            forest_sigma=EMPIRICAL_FOREST_SMOOTH_SIGMA,
        )
        pred = _apply_global_probability_smoothing(
            pred,
            init_grid,
            alpha=EMPIRICAL_GLOBAL_SMOOTH_ALPHA,
            sigma=EMPIRICAL_GLOBAL_SMOOTH_SIGMA,
        )
        if fully_observed_round:
            wide_pred = _build_full_coverage_denoised_prediction(
                counts,
                init_grid,
                context["coast"],
                context["near_settle"],
                sigma=EMPIRICAL_FULL_COVERAGE_FIELD_SIGMA,
                alpha_single=EMPIRICAL_FULL_COVERAGE_FIELD_ALPHA_SINGLE,
                alpha_double=EMPIRICAL_FULL_COVERAGE_FIELD_ALPHA_DOUBLE,
                alpha_multi=EMPIRICAL_FULL_COVERAGE_FIELD_ALPHA_MULTI,
                conf_bias=EMPIRICAL_FULL_COVERAGE_FIELD_CONF_BIAS,
                template_model=full_coverage_model,
                template_alpha_single=EMPIRICAL_FULL_COVERAGE_TEMPLATE_ALPHA_SINGLE,
                template_alpha_double=EMPIRICAL_FULL_COVERAGE_TEMPLATE_ALPHA_DOUBLE,
                template_alpha_multi=EMPIRICAL_FULL_COVERAGE_TEMPLATE_ALPHA_MULTI,
            )
            iterative_pred = _build_full_coverage_iterative_prediction(
                counts,
                init_grid,
                sigma=EMPIRICAL_FULL_COVERAGE_ITER_SIGMA,
                forest_sigma=EMPIRICAL_FULL_COVERAGE_ITER_FOREST_SIGMA,
                tau=EMPIRICAL_FULL_COVERAGE_ITER_TAU,
                steps=EMPIRICAL_FULL_COVERAGE_ITER_STEPS,
                settlement_boost=EMPIRICAL_FULL_COVERAGE_ITER_SETTLE_BOOST,
                ruin_boost=EMPIRICAL_FULL_COVERAGE_ITER_RUIN_BOOST,
            )
            pred = _blend_full_coverage_predictions(
                wide_pred,
                iterative_pred,
                counts,
                alpha_single=EMPIRICAL_FULL_COVERAGE_BLEND_SINGLE,
                alpha_double=EMPIRICAL_FULL_COVERAGE_BLEND_DOUBLE,
                alpha_multi=EMPIRICAL_FULL_COVERAGE_BLEND_MULTI,
            )
        pred = _apply_settlement_intensity_prior(
            pred,
            init_grid,
            settlement_centers=context["has_settle"],
            alpha=SETTLEMENT_INTENSITY_BLEND_ALPHA,
            sigma=SETTLEMENT_INTENSITY_SIGMA,
        )
        _force_static_cells(pred, init_grid)
        pred = np.maximum(pred, PROB_FLOOR)
        pred = pred / pred.sum(axis=2, keepdims=True)
        _enforce_hard_constraints(pred, init_grid)
        predictions[seed] = pred.astype(np.float32)

        if verbose:
            _print_seed_summary(seed, pred, obs_mask, n_samp[:, :, 0])

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
        adjusted = safe_normalize_last_axis(adjusted)
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


def _enforce_hard_constraints(pred: np.ndarray, init_grid: np.ndarray) -> None:
    """
    Enforce exact terrain feasibility constraints.

    - Known mountain cells are one-hot Mountain.
    - All other cells have Mountain probability exactly 0 and are re-normalized
      over classes 0..4.
    - Port probability is exactly 0 on non-coastal cells and mountain cells.
    """
    mountain_mask = init_grid == MOUNTAIN_CODE
    non_mountain_mask = ~mountain_mask

    if mountain_mask.any():
        pred[mountain_mask] = 0.0
        pred[mountain_mask, MOUNTAIN_CODE] = 1.0

    if non_mountain_mask.any():
        pred[non_mountain_mask, MOUNTAIN_CODE] = 0.0
        non_mountain_probs = pred[non_mountain_mask, :MOUNTAIN_CODE]
        denom = non_mountain_probs.sum(axis=1, keepdims=True)
        valid = (denom[:, 0] > 0.0)
        if np.any(valid):
            non_mountain_probs[valid] /= denom[valid]
        if np.any(~valid):
            non_mountain_probs[~valid] = 0.0
            non_mountain_probs[~valid, 0] = 1.0
        pred[non_mountain_mask, :MOUNTAIN_CODE] = non_mountain_probs

    coast = coastal_mask(init_grid)
    port_impossible_mask = (~coast) | mountain_mask
    if port_impossible_mask.any():
        affected = pred[port_impossible_mask]
        affected[:, PORT_CODE] = 0.0
        denom = affected.sum(axis=1, keepdims=True)
        valid = denom[:, 0] > 0.0
        if np.any(valid):
            affected[valid] /= denom[valid]
        if np.any(~valid):
            affected[~valid] = 0.0
            affected[~valid, 0] = 1.0
        pred[port_impossible_mask] = affected


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


def _masked_gaussian_blur(
    field: np.ndarray,
    mask: np.ndarray,
    sigma: float,
) -> np.ndarray:
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        return field

    weight = gaussian_filter(mask.astype(np.float64), sigma=sigma)
    numer = gaussian_filter((field * mask).astype(np.float64), sigma=sigma)
    return np.divide(numer, np.maximum(weight, 1e-9))


def _apply_settlement_intensity_prior(
    pred: np.ndarray,
    init_grid: np.ndarray,
    *,
    settlement_centers: np.ndarray,
    alpha: float,
    sigma: float,
) -> np.ndarray:
    """
    Blend settlement mass with a smooth distance-decay field from seed centers.

    The generated field is normalized and re-scaled to the current total
    settlement mass so this only changes spatial shape, not global magnitude.
    """
    if alpha <= 0.0:
        return pred

    dynamic_mask = ~np.isin(init_grid, [OCEAN_CODE, MOUNTAIN_CODE])
    if not dynamic_mask.any():
        return pred

    centers = settlement_centers.astype(np.float64, copy=False) * dynamic_mask
    if centers.sum() <= 0.0:
        return pred

    out = pred.astype(np.float64, copy=True)
    settlement_total = out[:, :, SETTLEMENT_CODE] + out[:, :, PORT_CODE]
    port_ratio = np.divide(
        out[:, :, PORT_CODE],
        np.maximum(settlement_total, 1e-9),
    )

    intensity = _masked_gaussian_blur(centers, dynamic_mask, max(float(sigma), 0.6))
    intensity = np.maximum(intensity, 0.0) * dynamic_mask
    sum_intensity = float(intensity.sum())
    if sum_intensity <= 0.0:
        return out

    total_settlement_mass = float(settlement_total[dynamic_mask].sum())
    target = intensity * (total_settlement_mass / max(sum_intensity, 1e-9))
    target = np.clip(target, 0.0, 1.0)

    settlement_total = (
        (1.0 - float(alpha)) * settlement_total +
        float(alpha) * target
    )
    settlement_total = np.clip(settlement_total, 0.0, 1.0)

    out[:, :, PORT_CODE] = settlement_total * port_ratio
    out[:, :, SETTLEMENT_CODE] = np.maximum(
        settlement_total - out[:, :, PORT_CODE],
        PROB_FLOOR,
    )
    return safe_normalize_last_axis(out)


def _apply_semantic_class_smoothing(
    pred: np.ndarray,
    init_grid: np.ndarray,
    *,
    settlement_alpha: float,
    settlement_sigma: float,
    ruin_alpha: float,
    ruin_sigma: float,
    forest_alpha: float,
    forest_sigma: float,
) -> np.ndarray:
    dynamic_mask = ~np.isin(init_grid, [OCEAN_CODE, MOUNTAIN_CODE])
    if not dynamic_mask.any():
        return pred

    out = pred.astype(np.float64, copy=True)
    settlement_total = out[:, :, 1] + out[:, :, 2]
    port_ratio = np.divide(
        out[:, :, 2],
        np.maximum(settlement_total, 1e-9),
    )
    ruin = out[:, :, 3]
    forest = out[:, :, 4]

    if settlement_alpha > 0.0:
        settle_blur = _masked_gaussian_blur(
            settlement_total,
            dynamic_mask,
            settlement_sigma,
        )
        settlement_total = (
            (1.0 - settlement_alpha) * settlement_total +
            settlement_alpha * settle_blur
        )

    if forest_alpha > 0.0:
        forest_blur = _masked_gaussian_blur(
            forest,
            dynamic_mask,
            forest_sigma,
        )
        forest = (
            (1.0 - forest_alpha) * forest +
            forest_alpha * forest_blur
        )

    if ruin_alpha > 0.0:
        ruin_blur = _masked_gaussian_blur(
            ruin,
            dynamic_mask,
            ruin_sigma,
        )
        ruin = (
            (1.0 - ruin_alpha) * ruin +
            ruin_alpha * ruin_blur
        )

    out[:, :, 2] = settlement_total * port_ratio
    out[:, :, 1] = np.maximum(settlement_total - out[:, :, 2], PROB_FLOOR)
    out[:, :, 3] = np.maximum(ruin, PROB_FLOOR)
    out[:, :, 4] = np.maximum(forest, PROB_FLOOR)
    return safe_normalize_last_axis(out)


def _apply_global_probability_smoothing(
    pred: np.ndarray,
    init_grid: np.ndarray,
    *,
    alpha: float,
    sigma: float,
) -> np.ndarray:
    if alpha <= 0.0:
        return pred

    dynamic_mask = ~np.isin(init_grid, [OCEAN_CODE, MOUNTAIN_CODE])
    if not dynamic_mask.any():
        return pred

    out = pred.astype(np.float64, copy=True)
    blurred = np.zeros_like(out)
    for c in range(N_CLASSES):
        blurred[:, :, c] = _masked_gaussian_blur(
            out[:, :, c],
            dynamic_mask,
            sigma,
        )
    out = (1.0 - alpha) * out + alpha * blurred
    return safe_normalize_last_axis(out)


def _reanchor_observed_dynamic_cells(
    pred: np.ndarray,
    counts: np.ndarray,
    init_grid: np.ndarray,
    *,
    settlement_base: float,
    settlement_bonus: float,
    port_ruin_base: float,
    port_ruin_bonus: float,
) -> np.ndarray:
    """
    Restore part of the raw empirical signal on observed rare-class cells.

    Full-map smoothing is useful, but on fully covered rounds it can flatten
    observed settlement/port/ruin cells until Empty wins every argmax. This
    step only nudges cells whose observed majority is one of those classes.
    """
    totals = counts.sum(axis=2)
    obs_mask = totals > 0
    if not obs_mask.any():
        return pred

    dynamic_mask = ~np.isin(init_grid, [OCEAN_CODE, MOUNTAIN_CODE])
    target_mask = obs_mask & dynamic_mask
    if not target_mask.any():
        return pred

    empirical = np.divide(
        counts,
        totals[:, :, None],
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals[:, :, None] > 0,
    )
    majority = empirical.argmax(axis=2)
    support = np.clip(totals / 2.0, 0.0, 1.0)

    anchor = np.zeros_like(totals, dtype=np.float64)
    settle_mask = target_mask & (majority == SETTLEMENT_CODE)
    port_ruin_mask = target_mask & np.isin(majority, [PORT_CODE, 3])
    anchor[settle_mask] = settlement_base + settlement_bonus * support[settle_mask]
    anchor[port_ruin_mask] = port_ruin_base + port_ruin_bonus * support[port_ruin_mask]

    out = pred.astype(np.float64, copy=True)
    mask3 = anchor > 0.0
    out[mask3] = (
        (1.0 - anchor[mask3, None]) * out[mask3] +
        anchor[mask3, None] * empirical[mask3]
    )
    return safe_normalize_last_axis(out)


def _build_full_coverage_calibration_model(
    store: ObservationStore,
    contexts: dict[int, dict[str, np.ndarray]],
) -> dict[str, object]:
    """
    Learn how repeated cells denoise sparse observations in the current round.

    The 50-query strategy gives every cell at least one sample, then spends the
    spare budget on a handful of repeated windows. Those repeated cells are the
    best supervision available for how a noisy 1x/2x empirical observation
    should be corrected before spatial smoothing.
    """
    def _accumulate(
        table: dict[tuple[int, ...], dict[str, object]],
        key: tuple[int, ...],
        dist: np.ndarray,
        weight: float,
    ) -> None:
        entry = table.setdefault(key, {
            "sum": np.zeros(N_CLASSES, dtype=np.float64),
            "weight": 0.0,
            "cells": 0,
        })
        entry["sum"] += weight * dist
        entry["weight"] += weight
        entry["cells"] += 1

    full: dict[tuple[int, ...], dict[str, object]] = {}
    mid: dict[tuple[int, ...], dict[str, object]] = {}
    coarse: dict[tuple[int, ...], dict[str, object]] = {}
    global_sum = np.zeros(N_CLASSES, dtype=np.float64)
    global_weight = 0.0

    for seed in range(store.seeds_count):
        counts = store.counts[seed].astype(np.float64)
        totals = counts.sum(axis=2)
        empirical = np.divide(
            counts,
            totals[:, :, None],
            out=np.zeros_like(counts, dtype=np.float64),
            where=totals[:, :, None] > 0,
        )
        init_grid = contexts[seed]["grid"]
        coast = contexts[seed]["coast"]
        near_settle = contexts[seed]["near_settle"]
        dynamic_mask = ~np.isin(init_grid, [OCEAN_CODE, MOUNTAIN_CODE])
        majority = empirical.argmax(axis=2)

        ys, xs = np.nonzero((totals >= 2.0) & dynamic_mask)
        for y, x in zip(ys.tolist(), xs.tolist()):
            target = empirical[y, x]
            maj = int(majority[y, x])
            init_code = int(init_grid[y, x])
            weight = max(float(totals[y, x]) - 1.0, 1.0)
            _accumulate(
                full,
                (init_code, maj, int(bool(coast[y, x])), int(bool(near_settle[y, x]))),
                target,
                weight,
            )
            _accumulate(mid, (init_code, maj), target, weight)
            _accumulate(coarse, (maj,), target, weight)
            global_sum += weight * target
            global_weight += weight

    def _finalize(table: dict[tuple[int, ...], dict[str, object]]) -> dict[tuple[int, ...], dict[str, object]]:
        out: dict[tuple[int, ...], dict[str, object]] = {}
        for key, entry in table.items():
            weight = float(entry["weight"])
            if weight <= 0.0:
                continue
            out[key] = {
                "dist": safe_normalize(entry["sum"] / weight),
                "cells": int(entry["cells"]),
                "weight": weight,
            }
        return out

    global_prior = safe_normalize(global_sum / max(global_weight, 1e-9))
    return {
        "full": _finalize(full),
        "mid": _finalize(mid),
        "coarse": _finalize(coarse),
        "global": global_prior,
    }


def _lookup_full_coverage_template(
    template_model: dict[str, object] | None,
    *,
    init_code: int,
    majority_class: int,
    is_coastal: bool,
    near_settlement: bool,
) -> tuple[np.ndarray | None, float]:
    if template_model is None:
        return None, 0.0
    if majority_class not in (SETTLEMENT_CODE, PORT_CODE, RUIN_CODE):
        return None, 0.0

    full = template_model["full"]
    mid = template_model["mid"]
    coarse = template_model["coarse"]
    global_prior = template_model["global"]

    key_full = (init_code, majority_class, int(is_coastal), int(near_settlement))
    entry = full.get(key_full)
    if entry is not None and entry["cells"] >= 12:
        strength = min(1.0, 0.55 + entry["cells"] / 60.0)
        return entry["dist"], strength

    key_mid = (init_code, majority_class)
    entry = mid.get(key_mid)
    if entry is not None and entry["cells"] >= 18:
        strength = min(0.9, 0.40 + entry["cells"] / 110.0)
        return entry["dist"], strength

    key_coarse = (majority_class,)
    entry = coarse.get(key_coarse)
    if entry is not None and entry["cells"] >= 28:
        strength = min(0.75, 0.30 + entry["cells"] / 180.0)
        return entry["dist"], strength

    return global_prior, 0.18


def _build_full_coverage_denoised_prediction(
    counts: np.ndarray,
    init_grid: np.ndarray,
    coast: np.ndarray,
    near_settlement: np.ndarray,
    *,
    sigma: float,
    alpha_single: float,
    alpha_double: float,
    alpha_multi: float,
    conf_bias: float,
    template_model: dict[str, object] | None,
    template_alpha_single: float,
    template_alpha_double: float,
    template_alpha_multi: float,
) -> np.ndarray:
    """
    Build a denoised probability field directly from full-round observations.

    On full-coverage rounds, every cell has at least one direct observation.
    The raw per-cell empirical distributions are noisy, so this path first
    calibrates them against the repeated cells from the same round and then
    mixes the result with a wide confidence-weighted spatial field. Cells with
    only one sample get the strongest denoising; cells with repeated samples
    stay closer to their own empirical counts.
    """
    if max(alpha_single, alpha_double, alpha_multi) <= 0.0:
        return np.divide(
            counts,
            np.maximum(counts.sum(axis=2, keepdims=True), 1e-9),
            out=np.zeros_like(counts, dtype=np.float64),
            where=counts.sum(axis=2, keepdims=True) > 0,
        )

    totals = counts.sum(axis=2)
    obs_mask = totals > 0
    if not obs_mask.any():
        return np.zeros_like(counts, dtype=np.float64)

    dynamic_mask = obs_mask & ~np.isin(init_grid, [OCEAN_CODE, MOUNTAIN_CODE])
    if not dynamic_mask.any():
        return np.divide(
            counts,
            np.maximum(totals[:, :, None], 1e-9),
            out=np.zeros_like(counts, dtype=np.float64),
            where=totals[:, :, None] > 0,
        )

    empirical = np.divide(
        counts,
        totals[:, :, None],
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals[:, :, None] > 0,
    )
    majority = empirical.argmax(axis=2)
    template_alpha = np.where(
        totals <= 1,
        template_alpha_single,
        np.where(totals == 2, template_alpha_double, template_alpha_multi),
    ).astype(np.float64, copy=False)

    calibrated = empirical.astype(np.float64, copy=True)
    ys, xs = np.nonzero(dynamic_mask)
    for y, x in zip(ys.tolist(), xs.tolist()):
        template, strength = _lookup_full_coverage_template(
            template_model,
            init_code=int(init_grid[y, x]),
            majority_class=int(majority[y, x]),
            is_coastal=bool(coast[y, x]),
            near_settlement=bool(near_settlement[y, x]),
        )
        if template is None or strength <= 0.0:
            continue
        alpha = float(template_alpha[y, x]) * float(strength)
        calibrated[y, x] = (
            (1.0 - alpha) * calibrated[y, x] +
            alpha * template
        )

    calibrated = safe_normalize_last_axis(calibrated)
    top2 = np.partition(calibrated, -2, axis=2)[:, :, -2]
    margin = np.clip(calibrated.max(axis=2) - top2, 0.0, 1.0)
    support = np.clip(totals / 2.0, 0.0, 1.0)
    confidence = np.clip(conf_bias + 0.7 * margin + 0.3 * support, 0.0, 1.0)
    weights = confidence * dynamic_mask
    denom = _masked_gaussian_blur(weights, dynamic_mask, sigma)

    field = np.zeros_like(calibrated, dtype=np.float64)
    for c in range(N_CLASSES):
        numer = _masked_gaussian_blur(
            calibrated[:, :, c] * weights,
            dynamic_mask,
            sigma,
        )
        field[:, :, c] = np.divide(numer, np.maximum(denom, 1e-9))

    field = safe_normalize_last_axis(field)

    alpha = np.where(
        totals <= 1,
        alpha_single,
        np.where(totals == 2, alpha_double, alpha_multi),
    ).astype(np.float64, copy=False)

    out = (1.0 - alpha[:, :, None]) * calibrated + alpha[:, :, None] * field
    return safe_normalize_last_axis(out)


def _build_full_coverage_iterative_prediction(
    counts: np.ndarray,
    init_grid: np.ndarray,
    *,
    sigma: float,
    forest_sigma: float,
    tau: float,
    steps: int,
    settlement_boost: float,
    ruin_boost: float,
) -> np.ndarray:
    """
    Iteratively denoise the full-coverage empirical field with a smoother prior.

    This path is deliberately less global than the wide denoiser above. It
    works better when the single-sample map is noisy but still carries useful
    local structure, which is common on fully covered rounds with only a small
    repeat budget.
    """
    totals = counts.sum(axis=2)
    empirical = np.divide(
        counts,
        totals[:, :, None],
        out=np.zeros_like(counts, dtype=np.float64),
        where=totals[:, :, None] > 0,
    )
    pred = safe_normalize_last_axis(empirical)
    dynamic_mask = ~np.isin(init_grid, [OCEAN_CODE, MOUNTAIN_CODE])
    if not dynamic_mask.any():
        return pred

    steps = max(int(steps), 1)
    sigma = max(float(sigma), 0.5)
    forest_sigma = max(float(forest_sigma), 0.5)
    tau = max(float(tau), 0.0)

    for _ in range(steps):
        prior = np.zeros_like(pred, dtype=np.float64)
        class_sigmas = [sigma, sigma, sigma, sigma, forest_sigma, 1.0]
        for cls in range(N_CLASSES):
            prior[:, :, cls] = _masked_gaussian_blur(
                pred[:, :, cls],
                dynamic_mask,
                class_sigmas[cls],
            )
        prior[:, :, SETTLEMENT_CODE] *= settlement_boost
        prior[:, :, PORT_CODE] *= settlement_boost
        prior[:, :, RUIN_CODE] *= ruin_boost
        prior = safe_normalize_last_axis(prior)
        pred = counts.astype(np.float64) + tau * prior
        pred = safe_normalize_last_axis(pred)
        _force_static_cells(pred, init_grid)

    return safe_normalize_last_axis(pred)


def _blend_full_coverage_predictions(
    wide_pred: np.ndarray,
    iterative_pred: np.ndarray,
    counts: np.ndarray,
    *,
    alpha_single: float,
    alpha_double: float,
    alpha_multi: float,
) -> np.ndarray:
    totals = counts.sum(axis=2)
    alpha = np.where(
        totals <= 1,
        alpha_single,
        np.where(totals == 2, alpha_double, alpha_multi),
    ).astype(np.float64, copy=False)
    out = (
        alpha[:, :, None] * iterative_pred.astype(np.float64) +
        (1.0 - alpha[:, :, None]) * wide_pred.astype(np.float64)
    )
    return safe_normalize_last_axis(out)


def _apply_final_class_calibration(
    pred: np.ndarray,
    *,
    scale: np.ndarray,
) -> np.ndarray:
    out = pred.astype(np.float64, copy=True)
    out *= scale.reshape(1, 1, -1)
    return safe_normalize_last_axis(out)


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
