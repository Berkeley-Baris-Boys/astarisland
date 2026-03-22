from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import AstarConfig
from .features import build_all_features, make_bucket_keys
from .history import _round_detail_from_json
from .learned_prior import build_learned_prior_features
from .ood import OOD_REFERENCE_VERSION, build_ood_reference_from_archive
from .regime import compute_round_regime, infer_latent_summary_from_prediction
from .scoring import cell_entropy, cell_kl_divergence
from .types import CLASS_EMPTY, CLASS_FOREST, CLASS_MOUNTAIN, CLASS_NAMES, CLASS_PORT, CLASS_RUIN, CLASS_SETTLEMENT, NUM_CLASSES, SeedFeatures
from .utils import load_json, normalize_probabilities

EPS = 1e-12
RESIDUAL_CALIBRATOR_VERSION = 5


@dataclass
class ResidualCalibratorArtifact:
    feature_names: list[str]
    active_model: HistGradientBoostingRegressor
    forest_model: HistGradientBoostingRegressor
    settlement_model: HistGradientBoostingRegressor
    port_model: HistGradientBoostingRegressor
    ruin_model: HistGradientBoostingRegressor
    metadata: dict[str, Any]
    active_budget_model: HistGradientBoostingRegressor | None = None
    budget_feature_names: list[str] | None = None
    collapsed_active_model: HistGradientBoostingRegressor | None = None
    collapsed_active_feature_names: list[str] | None = None
    blend_model: HistGradientBoostingRegressor | None = None
    blend_feature_names: list[str] | None = None


def save_residual_calibrator_artifact(path: Path, artifact: ResidualCalibratorArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_residual_calibrator_artifact(path: Path) -> ResidualCalibratorArtifact:
    artifact = joblib.load(path)
    metadata = getattr(artifact, "metadata", {}) or {}
    version = metadata.get("version")
    if version != RESIDUAL_CALIBRATOR_VERSION:
        raise ValueError(
            f"Residual calibrator artifact at {path} has version {version!r}; "
            f"expected {RESIDUAL_CALIBRATOR_VERSION}. Rebuild the artifact."
        )
    return artifact


def build_active_budget_features(
    prediction: np.ndarray,
    seed_features: SeedFeatures,
    round_regime: dict[str, float] | None = None,
    *,
    observed_counts: np.ndarray | None = None,
    observation_counts: np.ndarray | None = None,
    bucket_target: np.ndarray | None = None,
    ood_signal_values: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if round_regime is None:
        round_regime = compute_round_regime(
            infer_latent_summary_from_prediction(prediction),
            AstarConfig().predictor,
        )

    buildable = seed_features.buildable_mask
    active = prediction[..., CLASS_SETTLEMENT] + prediction[..., CLASS_PORT] + prediction[..., CLASS_RUIN]
    forest = prediction[..., CLASS_FOREST]
    settlement = prediction[..., CLASS_SETTLEMENT]
    frontier = seed_features.frontier_mask & buildable
    coastal = seed_features.coastal_mask & buildable
    initial_settlement = seed_features.initial_settlement_mask & buildable
    if "dist_to_settlement" in seed_features.feature_names:
        dist_idx = seed_features.feature_names.index("dist_to_settlement")
        near_threshold = 2.0 / max(sum(prediction.shape[:2]), 1)
        near_settlement = buildable & (seed_features.feature_stack[..., dist_idx] <= near_threshold)
    else:
        near_settlement = frontier | initial_settlement

    def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(values[mask]))

    pred_active_mass = _masked_mean(active, buildable)
    pred_settlement_mass = _masked_mean(settlement, buildable)
    pred_forest_mass = _masked_mean(forest, buildable)
    pred_active_frontier = _masked_mean(active, frontier)
    pred_active_coastal = _masked_mean(active, coastal)
    pred_active_initial_settlement = _masked_mean(active, initial_settlement)
    pred_active_near_settlement = _masked_mean(active, near_settlement)

    if bucket_target is not None:
        bucket_active = bucket_target[..., CLASS_SETTLEMENT] + bucket_target[..., CLASS_PORT] + bucket_target[..., CLASS_RUIN]
        bucket_target_active_mass = _masked_mean(bucket_active, buildable)
    else:
        bucket_target_active_mass = float("nan")

    observed_active_rate = float("nan")
    observed_forest_share = float("nan")
    buildable_coverage_fraction = float("nan")
    buildable_repeat_fraction = float("nan")
    initial_settlement_observed_inactive_rate = float("nan")
    frontier_observed_active_rate = float("nan")
    coastal_observed_active_rate = float("nan")
    has_observation_context = 0.0

    if observed_counts is not None and observation_counts is not None:
        has_observation_context = 1.0
        observed_buildable = buildable & (observation_counts > 0)
        total_buildable = max(float(np.sum(buildable)), 1.0)
        buildable_coverage_fraction = float(np.sum(observed_buildable) / total_buildable)
        buildable_repeat_fraction = float(np.sum(buildable & (observation_counts >= 2)) / total_buildable)

        active_counts = (
            observed_counts[..., CLASS_SETTLEMENT]
            + observed_counts[..., CLASS_PORT]
            + observed_counts[..., CLASS_RUIN]
        )
        total_counts = np.sum(observed_counts, axis=-1)

        total_observed_mass = float(np.sum(total_counts[observed_buildable]))
        if total_observed_mass > 0.0:
            observed_active_rate = float(np.sum(active_counts[observed_buildable]) / total_observed_mass)
            observed_forest_share = float(np.sum(observed_counts[..., CLASS_FOREST][observed_buildable]) / total_observed_mass)

        observed_initial = observed_buildable & initial_settlement
        if np.any(observed_initial):
            initial_settlement_observed_inactive_rate = float(np.mean(active_counts[observed_initial] <= 0.0))

        observed_frontier = observed_buildable & frontier
        frontier_total = float(np.sum(total_counts[observed_frontier]))
        if frontier_total > 0.0:
            frontier_observed_active_rate = float(np.sum(active_counts[observed_frontier]) / frontier_total)

        observed_coastal = observed_buildable & coastal
        coastal_total = float(np.sum(total_counts[observed_coastal]))
        if coastal_total > 0.0:
            coastal_observed_active_rate = float(np.sum(active_counts[observed_coastal]) / coastal_total)

    ood_signal_values = ood_signal_values or {}
    arrays = [
        pred_active_mass,
        pred_settlement_mass,
        pred_forest_mass,
        pred_active_frontier,
        pred_active_coastal,
        pred_active_initial_settlement,
        pred_active_near_settlement,
        bucket_target_active_mass,
        observed_active_rate,
        observed_forest_share,
        buildable_coverage_fraction,
        buildable_repeat_fraction,
        initial_settlement_observed_inactive_rate,
        frontier_observed_active_rate,
        coastal_observed_active_rate,
        float(round_regime["settlement_signal"]),
        float(round_regime["forest_signal"]),
        float(round_regime["high_activity_factor"]),
        float(round_regime["low_activity_factor"]),
        float(round_regime["repeat_fraction"]),
        float(ood_signal_values.get("settlement_rate", np.nan)),
        float(ood_signal_values.get("forest_share_dynamic", np.nan)),
        float(ood_signal_values.get("port_share_given_active", np.nan)),
        float(ood_signal_values.get("observed_cells", np.nan)),
        has_observation_context,
        0.0 if bucket_target is None else 1.0,
    ]
    names = [
        "budget_pred_active_mass_buildable",
        "budget_pred_settlement_mass_buildable",
        "budget_pred_forest_mass_buildable",
        "budget_pred_active_frontier",
        "budget_pred_active_coastal",
        "budget_pred_active_initial_settlement",
        "budget_pred_active_near_settlement",
        "budget_bucket_target_active_mass_buildable",
        "budget_observed_active_rate_buildable",
        "budget_observed_forest_share_buildable",
        "budget_buildable_coverage_fraction",
        "budget_buildable_repeat_fraction",
        "budget_initial_settlement_observed_inactive_rate",
        "budget_frontier_observed_active_rate",
        "budget_coastal_observed_active_rate",
        "budget_round_regime_settlement_signal",
        "budget_round_regime_forest_signal",
        "budget_round_regime_high_activity_factor",
        "budget_round_regime_low_activity_factor",
        "budget_round_regime_repeat_fraction",
        "budget_ood_settlement_rate",
        "budget_ood_forest_share_dynamic",
        "budget_ood_port_share_given_active",
        "budget_ood_observed_cells",
        "budget_has_observation_context",
        "budget_has_bucket_target",
    ]
    return np.asarray(arrays, dtype=np.float64)[None, :], names


def predict_active_budget(
    artifact: ResidualCalibratorArtifact,
    prediction: np.ndarray,
    seed_features: SeedFeatures,
    round_regime: dict[str, float] | None = None,
    *,
    observed_counts: np.ndarray | None = None,
    observation_counts: np.ndarray | None = None,
    bucket_target: np.ndarray | None = None,
    ood_signal_values: dict[str, float] | None = None,
) -> tuple[float, dict[str, object] | None]:
    if artifact.active_budget_model is None or artifact.budget_feature_names is None:
        return float("nan"), None
    matrix, names = build_active_budget_features(
        prediction,
        seed_features,
        round_regime,
        observed_counts=observed_counts,
        observation_counts=observation_counts,
        bucket_target=bucket_target,
        ood_signal_values=ood_signal_values,
    )
    if names != artifact.budget_feature_names:
        raise ValueError("Active budget feature names do not match runtime construction.")
    budget = float(np.clip(artifact.active_budget_model.predict(matrix)[0], 0.0, 1.0))
    return budget, {
        "summary": {"predicted_active_budget": budget},
        "tensors": {"active_budget_features": matrix.reshape(-1)},
    }


def build_collapsed_active_features(
    prediction: np.ndarray,
    seed_features: SeedFeatures,
    round_regime: dict[str, float] | None = None,
    *,
    observed_counts: np.ndarray | None = None,
    observation_counts: np.ndarray | None = None,
    bucket_target: np.ndarray | None = None,
    ood_signal_values: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    if round_regime is None:
        round_regime = compute_round_regime(
            infer_latent_summary_from_prediction(prediction),
            AstarConfig().predictor,
        )

    buildable = seed_features.buildable_mask
    active = prediction[..., CLASS_SETTLEMENT] + prediction[..., CLASS_PORT] + prediction[..., CLASS_RUIN]
    frontier = seed_features.frontier_mask & buildable
    coastal = seed_features.coastal_mask & buildable
    initial_settlement = seed_features.initial_settlement_mask & buildable

    def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.mean(values[mask]))

    current_active_mass = _masked_mean(active, buildable)
    frontier_active_mass = _masked_mean(active, frontier)
    coastal_active_mass = _masked_mean(active, coastal)
    initial_settlement_active_mass = _masked_mean(active, initial_settlement)

    bucket_target_active_mass = float("nan")
    if bucket_target is not None:
        bucket_active = bucket_target[..., CLASS_SETTLEMENT] + bucket_target[..., CLASS_PORT] + bucket_target[..., CLASS_RUIN]
        bucket_target_active_mass = _masked_mean(bucket_active, buildable)

    observed_active_rate = float("nan")
    observed_forest_share = float("nan")
    buildable_coverage_fraction = float("nan")
    buildable_repeat_fraction = float("nan")
    frontier_observed_active_rate = float("nan")
    coastal_observed_active_rate = float("nan")
    initial_settlement_observed_active_rate = float("nan")
    has_observation_context = 0.0
    if observed_counts is not None and observation_counts is not None:
        has_observation_context = 1.0
        observed_buildable = buildable & (observation_counts > 0)
        total_buildable = max(float(np.sum(buildable)), 1.0)
        buildable_coverage_fraction = float(np.sum(observed_buildable) / total_buildable)
        buildable_repeat_fraction = float(np.sum(buildable & (observation_counts >= 2)) / total_buildable)

        active_counts = (
            observed_counts[..., CLASS_SETTLEMENT]
            + observed_counts[..., CLASS_PORT]
            + observed_counts[..., CLASS_RUIN]
        )
        total_counts = np.sum(observed_counts, axis=-1)
        total_observed_mass = float(np.sum(total_counts[observed_buildable]))
        if total_observed_mass > 0.0:
            observed_active_rate = float(np.sum(active_counts[observed_buildable]) / total_observed_mass)
            observed_forest_share = float(
                np.sum(observed_counts[..., CLASS_FOREST][observed_buildable]) / total_observed_mass
            )

        for mask, name in (
            (frontier, "frontier"),
            (coastal, "coastal"),
            (initial_settlement, "initial_settlement"),
        ):
            observed_mask = observed_buildable & mask
            denom = float(np.sum(total_counts[observed_mask]))
            value = float("nan")
            if denom > 0.0:
                value = float(np.sum(active_counts[observed_mask]) / denom)
            if name == "frontier":
                frontier_observed_active_rate = value
            elif name == "coastal":
                coastal_observed_active_rate = value
            else:
                initial_settlement_observed_active_rate = value

    ood_signal_values = ood_signal_values or {}
    arrays = [
        current_active_mass,
        bucket_target_active_mass,
        observed_active_rate,
        observed_forest_share,
        buildable_coverage_fraction,
        buildable_repeat_fraction,
        frontier_active_mass,
        coastal_active_mass,
        initial_settlement_active_mass,
        frontier_observed_active_rate,
        coastal_observed_active_rate,
        initial_settlement_observed_active_rate,
        float(round_regime["settlement_signal"]),
        float(round_regime["forest_signal"]),
        float(round_regime["high_activity_factor"]),
        float(round_regime["low_activity_factor"]),
        float(round_regime["repeat_fraction"]),
        float(ood_signal_values.get("settlement_rate", np.nan)),
        float(ood_signal_values.get("forest_share_dynamic", np.nan)),
        float(ood_signal_values.get("port_share_given_active", np.nan)),
        float(ood_signal_values.get("observed_cells", np.nan)),
        has_observation_context,
        0.0 if bucket_target is None else 1.0,
    ]
    names = [
        "collapsed_current_active_mass_buildable",
        "collapsed_bucket_target_active_mass_buildable",
        "collapsed_observed_active_rate_buildable",
        "collapsed_observed_forest_share_buildable",
        "collapsed_buildable_coverage_fraction",
        "collapsed_buildable_repeat_fraction",
        "collapsed_pred_active_frontier",
        "collapsed_pred_active_coastal",
        "collapsed_pred_active_initial_settlement",
        "collapsed_observed_active_frontier",
        "collapsed_observed_active_coastal",
        "collapsed_observed_active_initial_settlement",
        "collapsed_round_regime_settlement_signal",
        "collapsed_round_regime_forest_signal",
        "collapsed_round_regime_high_activity_factor",
        "collapsed_round_regime_low_activity_factor",
        "collapsed_round_regime_repeat_fraction",
        "collapsed_ood_settlement_rate",
        "collapsed_ood_forest_share_dynamic",
        "collapsed_ood_port_share_given_active",
        "collapsed_ood_observed_cells",
        "collapsed_has_observation_context",
        "collapsed_has_bucket_target",
    ]
    return np.asarray(arrays, dtype=np.float64)[None, :], names


def predict_collapsed_active_scale(
    artifact: ResidualCalibratorArtifact,
    prediction: np.ndarray,
    seed_features: SeedFeatures,
    round_regime: dict[str, float] | None = None,
    *,
    observed_counts: np.ndarray | None = None,
    observation_counts: np.ndarray | None = None,
    bucket_target: np.ndarray | None = None,
    ood_signal_values: dict[str, float] | None = None,
    scale_clip: tuple[float, float] = (0.0, 1.0),
) -> tuple[float, dict[str, object] | None]:
    model = getattr(artifact, "collapsed_active_model", None)
    feature_names = getattr(artifact, "collapsed_active_feature_names", None)
    if model is None or feature_names is None:
        return float("nan"), None
    matrix, names = build_collapsed_active_features(
        prediction,
        seed_features,
        round_regime,
        observed_counts=observed_counts,
        observation_counts=observation_counts,
        bucket_target=bucket_target,
        ood_signal_values=ood_signal_values,
    )
    if names != feature_names:
        raise ValueError("Collapsed-active feature names do not match runtime construction.")
    lo, hi = scale_clip
    scale = float(np.clip(model.predict(matrix)[0], lo, hi))
    return scale, {
        "summary": {"collapsed_active_predicted_scale": scale},
        "tensors": {"collapsed_active_features": matrix.reshape(-1)},
    }


def build_residual_features(
    prediction: np.ndarray,
    seed_features: SeedFeatures,
    round_regime: dict[str, float] | None = None,
) -> tuple[np.ndarray, list[str]]:
    structural, structural_names = build_learned_prior_features(seed_features)
    probs = prediction.reshape(-1, NUM_CLASSES)
    if round_regime is None:
        round_regime = compute_round_regime(
            infer_latent_summary_from_prediction(prediction),
            AstarConfig().predictor,
        )
    active = probs[:, CLASS_SETTLEMENT] + probs[:, CLASS_PORT] + probs[:, CLASS_RUIN]
    non_active = probs[:, CLASS_EMPTY] + probs[:, CLASS_FOREST]
    forest_share_non_active = probs[:, CLASS_FOREST] / np.maximum(non_active, EPS)
    settlement_share_active = probs[:, CLASS_SETTLEMENT] / np.maximum(active, EPS)
    port_share_active = probs[:, CLASS_PORT] / np.maximum(active, EPS)
    ruin_share_active = probs[:, CLASS_RUIN] / np.maximum(active, EPS)
    entropy = -np.sum(probs * np.log(np.maximum(probs, EPS)), axis=1)
    sorted_probs = np.sort(probs, axis=1)
    margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    log_probs = np.log(np.maximum(probs, EPS))

    extra_arrays = [
        probs[:, CLASS_EMPTY],
        probs[:, CLASS_SETTLEMENT],
        probs[:, CLASS_PORT],
        probs[:, CLASS_RUIN],
        probs[:, CLASS_FOREST],
        probs[:, CLASS_MOUNTAIN],
        log_probs[:, CLASS_EMPTY],
        log_probs[:, CLASS_SETTLEMENT],
        log_probs[:, CLASS_PORT],
        log_probs[:, CLASS_RUIN],
        log_probs[:, CLASS_FOREST],
        active,
        forest_share_non_active,
        settlement_share_active,
        port_share_active,
        ruin_share_active,
        entropy,
        margin,
    ]
    extra_names = [
        "pred_empty",
        "pred_settlement",
        "pred_port",
        "pred_ruin",
        "pred_forest",
        "pred_mountain",
        "log_pred_empty",
        "log_pred_settlement",
        "log_pred_port",
        "log_pred_ruin",
        "log_pred_forest",
        "pred_active",
        "pred_forest_share_non_active",
        "pred_settlement_share_active",
        "pred_port_share_active",
        "pred_ruin_share_active",
        "pred_entropy",
        "pred_margin",
    ]

    derived_map = {name: structural[:, idx] for idx, name in enumerate(structural_names)}
    extra_arrays.extend(
        [
            derived_map["coastal"] * probs[:, CLASS_PORT],
            derived_map["frontier"] * active,
            derived_map["conflict"] * probs[:, CLASS_RUIN],
            derived_map["initial_class_forest"] * probs[:, CLASS_FOREST],
            derived_map["initial_class_settlement"] * probs[:, CLASS_SETTLEMENT],
            derived_map["initial_class_port"] * probs[:, CLASS_PORT],
            derived_map["coastal"] * port_share_active,
            derived_map["border_support"] * probs[:, CLASS_PORT],
            derived_map["settlement_intensity"] * settlement_share_active,
            derived_map["settlement_intensity"] * ruin_share_active,
            derived_map["forest_density"] * probs[:, CLASS_FOREST],
            np.full(probs.shape[0], round_regime["settlement_signal"], dtype=np.float64),
            np.full(probs.shape[0], round_regime["forest_signal"], dtype=np.float64),
            np.full(probs.shape[0], round_regime["high_activity_factor"], dtype=np.float64),
            np.full(probs.shape[0], round_regime["low_activity_factor"], dtype=np.float64),
        ]
    )
    extra_names.extend(
        [
            "coastal_x_pred_port",
            "frontier_x_pred_active",
            "conflict_x_pred_ruin",
            "initial_forest_x_pred_forest",
            "initial_settlement_x_pred_settlement",
            "initial_port_x_pred_port",
            "coastal_x_pred_port_share_active",
            "border_support_x_pred_port",
            "settlement_intensity_x_pred_settlement_share_active",
            "settlement_intensity_x_pred_ruin_share_active",
            "forest_density_x_pred_forest",
            "round_regime_settlement_signal",
            "round_regime_forest_signal",
            "round_regime_high_activity_factor",
            "round_regime_low_activity_factor",
        ]
    )

    matrix = np.concatenate([structural, np.stack(extra_arrays, axis=1)], axis=1)
    return matrix, structural_names + extra_names


def build_residual_trust_gate_features(
    residual_features: np.ndarray,
    residual_feature_names: list[str],
    prediction_flat: np.ndarray,
    learned_flat: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Build trust gate features from already-computed residual features + learned/base disagreement.

    Args:
        residual_features: (N, F) matrix from build_residual_features.
        residual_feature_names: names matching residual_features columns.
        prediction_flat: (N, 6) base prediction probabilities.
        learned_flat: (N, 6) residual model output probabilities (already normalised).
    """
    pred = prediction_flat
    learned = learned_flat

    pred_active = pred[:, CLASS_SETTLEMENT] + pred[:, CLASS_PORT] + pred[:, CLASS_RUIN]
    pred_forest = pred[:, CLASS_FOREST]

    learned_active = learned[:, CLASS_SETTLEMENT] + learned[:, CLASS_PORT] + learned[:, CLASS_RUIN]
    learned_forest = learned[:, CLASS_FOREST]
    learned_settlement_share = learned[:, CLASS_SETTLEMENT] / np.maximum(learned_active, EPS)
    learned_port_share = learned[:, CLASS_PORT] / np.maximum(learned_active, EPS)
    learned_ruin_share = learned[:, CLASS_RUIN] / np.maximum(learned_active, EPS)

    pred_active_safe = np.maximum(pred_active, EPS)
    pred_settlement_share = pred[:, CLASS_SETTLEMENT] / pred_active_safe
    pred_port_share = pred[:, CLASS_PORT] / pred_active_safe
    pred_ruin_share = pred[:, CLASS_RUIN] / pred_active_safe

    learned_entropy = cell_entropy(learned)
    sorted_learned = np.sort(learned, axis=1)
    learned_margin = sorted_learned[:, -1] - sorted_learned[:, -2]

    delta_active = learned_active - pred_active
    delta_forest = learned_forest - pred_forest
    diff = learned - pred
    l1_diff = np.sum(np.abs(diff), axis=1)
    l2_diff = np.sqrt(np.sum(diff ** 2, axis=1))
    argmax_flip = (np.argmax(learned, axis=1) != np.argmax(pred, axis=1)).astype(np.float64)

    extra_arrays = [
        learned_active,
        learned_forest,
        learned_entropy,
        learned_margin,
        delta_active,
        np.abs(delta_active),
        delta_forest,
        l1_diff,
        l2_diff,
        argmax_flip,
        learned_settlement_share,
        learned_port_share,
        learned_ruin_share,
        learned_settlement_share - pred_settlement_share,
        learned_port_share - pred_port_share,
        learned_ruin_share - pred_ruin_share,
    ]
    extra_names = [
        "tg_learned_active",
        "tg_learned_forest",
        "tg_learned_entropy",
        "tg_learned_margin",
        "tg_delta_active",
        "tg_abs_delta_active",
        "tg_delta_forest",
        "tg_l1_diff_learned_base",
        "tg_l2_diff_learned_base",
        "tg_argmax_flip",
        "tg_learned_settlement_share",
        "tg_learned_port_share",
        "tg_learned_ruin_share",
        "tg_delta_settlement_share",
        "tg_delta_port_share",
        "tg_delta_ruin_share",
    ]
    matrix = np.concatenate([residual_features, np.stack(extra_arrays, axis=1)], axis=1)
    return matrix, residual_feature_names + extra_names


def optimal_residual_blend(
    base: np.ndarray,
    learned: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """Find per-cell optimal blend alpha minimising KL(gt, (1-a)*base + a*learned)."""
    alphas = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    best_alpha = np.ones(base.shape[0], dtype=np.float64)
    best_loss = np.full(base.shape[0], np.inf, dtype=np.float64)
    for alpha in alphas:
        blended = (1.0 - alpha) * base + alpha * learned
        blended = normalize_probabilities(blended, EPS)
        loss = cell_kl_divergence(ground_truth, blended)
        better = loss < best_loss
        best_loss[better] = loss[better]
        best_alpha[better] = alpha
    return best_alpha


def residual_trust_gate_sample_weight(
    base: np.ndarray,
    learned: np.ndarray,
    ground_truth: np.ndarray,
) -> np.ndarray:
    """Sample weights for trust gate training: upweight uncertain, high-disagreement cells."""
    entropy = cell_entropy(ground_truth)
    disagreement = np.sum(np.abs(learned - base), axis=1)
    return 0.05 + entropy * (0.2 + disagreement)


def apply_residual_calibrator(
    artifact: ResidualCalibratorArtifact,
    prediction: np.ndarray,
    seed_features: SeedFeatures,
    round_regime: dict[str, float] | None = None,
    *,
    blend: float | np.ndarray,
    min_probability: float,
    active_observed_cap: float,
    observation_counts: np.ndarray | None = None,
    observed_counts: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    features, feature_names = build_residual_features(prediction, seed_features, round_regime)
    if feature_names != artifact.feature_names:
        raise ValueError("Residual calibrator feature names do not match runtime construction.")

    active = np.clip(artifact.active_model.predict(features), 0.0, 1.0)
    forest_share = np.clip(artifact.forest_model.predict(features), 0.0, 1.0)
    settlement_share = np.clip(artifact.settlement_model.predict(features), 0.0, 1.0)
    port_share = np.clip(artifact.port_model.predict(features), 0.0, 1.0)
    ruin_share = np.clip(artifact.ruin_model.predict(features), 0.0, 1.0)

    active_types = np.stack([settlement_share, port_share, ruin_share], axis=1)
    active_types = np.maximum(active_types, min_probability)
    active_types /= np.maximum(active_types.sum(axis=1, keepdims=True), EPS)

    base = prediction.reshape(-1, NUM_CLASSES)
    mountain = base[:, CLASS_MOUNTAIN]
    non_mountain = np.clip(1.0 - mountain, 0.0, 1.0)
    active = np.clip(active, 0.0, non_mountain)
    non_active = np.clip(non_mountain - active, 0.0, 1.0)

    learned = np.zeros_like(base)
    learned[:, CLASS_EMPTY] = non_active * (1.0 - forest_share)
    learned[:, CLASS_FOREST] = non_active * forest_share
    learned[:, CLASS_SETTLEMENT] = active * active_types[:, 0]
    learned[:, CLASS_PORT] = active * active_types[:, 1]
    learned[:, CLASS_RUIN] = active * active_types[:, 2]
    learned[:, CLASS_MOUNTAIN] = mountain

    learned = normalize_probabilities(learned.reshape(prediction.shape), min_probability)

    # --- Compute blend map: trust gate (if available) or fallback to passed blend ---
    blend_gate = getattr(artifact, "blend_model", None)
    blend_feature_names_gate = getattr(artifact, "blend_feature_names", None)
    used_trust_gate = False
    if blend_gate is not None and blend_feature_names_gate is not None:
        gate_matrix, gate_names = build_residual_trust_gate_features(
            features, feature_names, base, learned.reshape(-1, NUM_CLASSES)
        )
        if gate_names != blend_feature_names_gate:
            raise ValueError("Residual trust gate feature names do not match runtime construction.")
        alpha = np.clip(blend_gate.predict(gate_matrix), 0.0, 1.0).reshape(prediction.shape[:2])
        # Post-hoc cap: directly observed active cells — trust the base prediction
        if observed_counts is not None:
            active_observed_mask = (
                observed_counts[..., CLASS_SETTLEMENT]
                + observed_counts[..., CLASS_PORT]
                + observed_counts[..., CLASS_RUIN]
            ) > 0
            alpha[active_observed_mask] = np.minimum(alpha[active_observed_mask], active_observed_cap)
        blend_map = alpha
        used_trust_gate = True
    else:
        if np.isscalar(blend):
            blend_map = np.full(prediction.shape[:2], float(blend), dtype=np.float64)
        else:
            blend_map = np.asarray(blend, dtype=np.float64)
            if blend_map.shape != prediction.shape[:2]:
                raise ValueError(f"Residual blend map shape {blend_map.shape} does not match prediction shape {prediction.shape[:2]}")
        blend_map = np.clip(blend_map, 0.0, 1.0)

    blended = normalize_probabilities(
        (1.0 - blend_map[..., None]) * prediction + blend_map[..., None] * learned,
        min_probability,
    )
    details = {
        "summary": {
            "blend_mean": float(np.mean(blend_map)),
            "blend_min": float(np.min(blend_map)),
            "blend_max": float(np.max(blend_map)),
            "used_trust_gate": used_trust_gate,
            "mean_pred_active": float(np.mean(active)),
            "mean_pred_forest_share_non_active": float(np.mean(forest_share)),
            "mean_pred_settlement_share_active": float(np.mean(active_types[:, 0])),
            "mean_pred_port_share_active": float(np.mean(active_types[:, 1])),
            "mean_pred_ruin_share_active": float(np.mean(active_types[:, 2])),
        },
        "tensors": {
            "residual_blend_map": blend_map,
            "residual_learned_prediction": learned,
            "residual_final_prediction": blended,
        },
    }
    return blended, details


def build_residual_calibrator_artifact_from_archive(
    history_dir: Path,
    *,
    holdout_round_number: int | None = None,
    max_iter: int = 300,
    max_depth: int = 4,
    min_samples_leaf: int = 300,
    learning_rate: float = 0.05,
    l2_regularization: float = 0.2,
) -> ResidualCalibratorArtifact:
    config = AstarConfig().predictor
    matrices: list[np.ndarray] = []
    budget_matrices: list[np.ndarray] = []
    collapsed_active_matrices: list[np.ndarray] = []
    active_targets: list[np.ndarray] = []
    forest_targets: list[np.ndarray] = []
    settlement_targets: list[np.ndarray] = []
    port_targets: list[np.ndarray] = []
    ruin_targets: list[np.ndarray] = []
    budget_targets: list[float] = []
    collapsed_active_targets: list[float] = []
    active_weights: list[np.ndarray] = []
    forest_weights: list[np.ndarray] = []
    type_weights: list[np.ndarray] = []
    budget_weights: list[float] = []
    collapsed_active_weights: list[float] = []
    base_preds: list[np.ndarray] = []
    ground_truths: list[np.ndarray] = []
    feature_names: list[str] | None = None
    budget_feature_names: list[str] | None = None
    collapsed_active_feature_names: list[str] | None = None
    used_rounds: list[dict[str, Any]] = []
    used_seeds = 0
    runtime_contexts = _discover_round_runtime_contexts(history_dir)

    for round_dir in sorted(history_dir.glob("round_*")):
        detail_path = round_dir / "round_detail.json"
        if not detail_path.exists():
            continue
        detail = _round_detail_from_json(load_json(detail_path))
        if holdout_round_number is not None and detail.round_number == holdout_round_number:
            continue
        features = build_all_features(detail.initial_states)
        runtime_context = runtime_contexts.get(detail.round_number)
        global_counts = None
        conditional_counts = None
        if runtime_context is not None:
            global_counts = runtime_context["class_counts"].sum(axis=(0, 1, 2))
            conditional_counts = runtime_context["conditional_counts"]
        round_used = False
        for seed_index, seed_features in features.items():
            analysis_path = round_dir / f"seed_{seed_index}" / "analysis.json"
            if not analysis_path.exists():
                continue
            payload = load_json(analysis_path)
            prediction = np.asarray(payload["prediction"], dtype=np.float64)
            ground_truth = np.asarray(payload["ground_truth"], dtype=np.float64)
            round_regime = compute_round_regime(
                infer_latent_summary_from_prediction(prediction),
                config,
            )
            matrix, names = build_residual_features(prediction, seed_features, round_regime)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("Residual feature name mismatch while building artifact.")
            bucket_target = None
            observed_counts = None
            observation_counts = None
            if runtime_context is not None and global_counts is not None and conditional_counts is not None:
                observed_counts = runtime_context["class_counts"][seed_index]
                observation_counts = runtime_context["observation_counts"][seed_index]
                bucket_target = _build_bucket_target_tensor(
                    seed_features,
                    conditional_counts=conditional_counts,
                    global_counts=global_counts,
                    min_probability=config.min_probability,
                )
            budget_matrix, budget_names = build_active_budget_features(
                prediction,
                seed_features,
                round_regime,
                observed_counts=observed_counts,
                observation_counts=observation_counts,
                bucket_target=bucket_target,
                ood_signal_values=_runtime_ood_signal_values(runtime_context, round_dir, prediction, round_regime),
            )
            if budget_feature_names is None:
                budget_feature_names = budget_names
            elif budget_feature_names != budget_names:
                raise ValueError("Active budget feature name mismatch while building artifact.")
            collapsed_matrix, collapsed_names = build_collapsed_active_features(
                prediction,
                seed_features,
                round_regime,
                observed_counts=observed_counts,
                observation_counts=observation_counts,
                bucket_target=bucket_target,
                ood_signal_values=_runtime_ood_signal_values(runtime_context, round_dir, prediction, round_regime),
            )
            if collapsed_active_feature_names is None:
                collapsed_active_feature_names = collapsed_names
            elif collapsed_active_feature_names != collapsed_names:
                raise ValueError("Collapsed-active feature name mismatch while building artifact.")
            gt = ground_truth.reshape(-1, NUM_CLASSES)
            entropy = -np.sum(np.where(gt > 0, gt * np.log(np.maximum(gt, EPS)), 0.0), axis=1)
            active_target = np.clip(gt[:, CLASS_SETTLEMENT] + gt[:, CLASS_PORT] + gt[:, CLASS_RUIN], 0.0, 1.0)
            non_active = np.clip(gt[:, CLASS_EMPTY] + gt[:, CLASS_FOREST], EPS, 1.0)
            forest_target = gt[:, CLASS_FOREST] / non_active
            active_mass = np.clip(active_target, EPS, 1.0)
            settlement_target = gt[:, CLASS_SETTLEMENT] / active_mass
            port_target = gt[:, CLASS_PORT] / active_mass
            ruin_target = gt[:, CLASS_RUIN] / active_mass

            matrices.append(matrix)
            budget_matrices.append(budget_matrix)
            collapsed_active_matrices.append(collapsed_matrix)
            active_targets.append(active_target)
            forest_targets.append(forest_target)
            settlement_targets.append(settlement_target)
            port_targets.append(port_target)
            ruin_targets.append(ruin_target)
            budget_target_value = float(np.mean(active_target[seed_features.buildable_mask.reshape(-1)]))
            pred_active_flat = prediction.reshape(-1, NUM_CLASSES)[:, CLASS_SETTLEMENT] + prediction.reshape(-1, NUM_CLASSES)[:, CLASS_PORT] + prediction.reshape(-1, NUM_CLASSES)[:, CLASS_RUIN]
            pred_budget_value = float(np.mean(pred_active_flat[seed_features.buildable_mask.reshape(-1)]))
            budget_targets.append(budget_target_value)
            collapsed_target_value = float(
                np.clip(
                    budget_target_value / max(pred_budget_value, EPS),
                    config.collapsed_active_calibrator_scale_clip_lo,
                    config.collapsed_active_calibrator_scale_clip_hi,
                )
            )
            collapsed_active_targets.append(collapsed_target_value)
            active_weights.append(0.15 + 1.85 * entropy)
            forest_weights.append(non_active * (0.1 + 1.5 * entropy))
            type_weights.append(active_target * (0.1 + 2.0 * entropy))
            budget_weights.append(
                1.0
                + float(round_regime["low_activity_factor"])
                + abs(budget_target_value - pred_budget_value)
            )
            collapsed_active_weights.append(
                1.0
                + 0.5 * float(round_regime["low_activity_factor"])
                + 0.75 * abs(collapsed_target_value - 1.0)
            )
            base_preds.append(prediction.reshape(-1, NUM_CLASSES))
            ground_truths.append(gt)
            used_seeds += 1
            round_used = True
        if round_used:
            used_rounds.append({"round_id": detail.round_id, "round_number": detail.round_number})

    if not matrices or feature_names is None:
        raise ValueError("No archived predictions available to build residual calibrator.")

    X = np.concatenate(matrices, axis=0)
    X_budget = np.concatenate(budget_matrices, axis=0)
    X_collapsed_active = np.concatenate(collapsed_active_matrices, axis=0)
    active_y = np.concatenate(active_targets, axis=0)
    forest_y = np.concatenate(forest_targets, axis=0)
    settlement_y = np.concatenate(settlement_targets, axis=0)
    port_y = np.concatenate(port_targets, axis=0)
    ruin_y = np.concatenate(ruin_targets, axis=0)
    budget_y = np.asarray(budget_targets, dtype=np.float64)
    collapsed_active_y = np.asarray(collapsed_active_targets, dtype=np.float64)
    active_w = np.concatenate(active_weights, axis=0)
    forest_w = np.concatenate(forest_weights, axis=0)
    type_w = np.concatenate(type_weights, axis=0)
    budget_w = np.asarray(budget_weights, dtype=np.float64)
    collapsed_active_w = np.asarray(collapsed_active_weights, dtype=np.float64)

    active_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)
    active_budget_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)
    collapsed_active_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)
    forest_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)
    settlement_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)
    port_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)
    ruin_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)

    active_model.fit(X, active_y, sample_weight=active_w)
    active_budget_model.fit(X_budget, budget_y, sample_weight=budget_w)
    collapsed_active_model.fit(X_collapsed_active, collapsed_active_y, sample_weight=collapsed_active_w)
    forest_model.fit(X, forest_y, sample_weight=forest_w)
    settlement_model.fit(X, settlement_y, sample_weight=type_w)
    port_model.fit(X, port_y, sample_weight=type_w)
    ruin_model.fit(X, ruin_y, sample_weight=type_w)

    # --- Build trust gate: learn per-cell optimal blend alpha ---
    gate_matrices: list[np.ndarray] = []
    gate_targets: list[np.ndarray] = []
    gate_weights: list[np.ndarray] = []
    blend_feature_names: list[str] | None = None

    for base_flat, gt_flat, feat_matrix in zip(base_preds, ground_truths, matrices):
        active_pred = np.clip(active_model.predict(feat_matrix), 0.0, 1.0)
        forest_pred = np.clip(forest_model.predict(feat_matrix), 0.0, 1.0)
        settlement_pred = np.clip(settlement_model.predict(feat_matrix), 0.0, 1.0)
        port_pred = np.clip(port_model.predict(feat_matrix), 0.0, 1.0)
        ruin_pred = np.clip(ruin_model.predict(feat_matrix), 0.0, 1.0)

        active_types = np.stack([settlement_pred, port_pred, ruin_pred], axis=1)
        active_types = np.maximum(active_types, EPS)
        active_types /= np.maximum(active_types.sum(axis=1, keepdims=True), EPS)

        mountain = base_flat[:, CLASS_MOUNTAIN]
        non_mountain = np.clip(1.0 - mountain, 0.0, 1.0)
        active_clamped = np.clip(active_pred, 0.0, non_mountain)
        non_active = np.clip(non_mountain - active_clamped, 0.0, 1.0)

        learned_flat = np.zeros_like(base_flat)
        learned_flat[:, CLASS_EMPTY] = non_active * (1.0 - forest_pred)
        learned_flat[:, CLASS_FOREST] = non_active * forest_pred
        learned_flat[:, CLASS_SETTLEMENT] = active_clamped * active_types[:, 0]
        learned_flat[:, CLASS_PORT] = active_clamped * active_types[:, 1]
        learned_flat[:, CLASS_RUIN] = active_clamped * active_types[:, 2]
        learned_flat[:, CLASS_MOUNTAIN] = mountain
        learned_flat = normalize_probabilities(learned_flat, EPS)

        gate_matrix, gate_names = build_residual_trust_gate_features(
            feat_matrix, feature_names, base_flat, learned_flat
        )
        if blend_feature_names is None:
            blend_feature_names = gate_names
        elif blend_feature_names != gate_names:
            raise ValueError("Trust gate feature name mismatch across seeds.")

        alpha_target = optimal_residual_blend(base_flat, learned_flat, gt_flat)
        weight = residual_trust_gate_sample_weight(base_flat, learned_flat, gt_flat)
        gate_matrices.append(gate_matrix)
        gate_targets.append(alpha_target)
        gate_weights.append(weight)

    blend_model: HistGradientBoostingRegressor | None = None
    if gate_matrices and blend_feature_names is not None:
        X_gate = np.concatenate(gate_matrices, axis=0)
        y_gate = np.concatenate(gate_targets, axis=0)
        w_gate = np.concatenate(gate_weights, axis=0)
        blend_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)
        blend_model.fit(X_gate, y_gate, sample_weight=w_gate)

    metadata = {
        "version": RESIDUAL_CALIBRATOR_VERSION,
        "num_rounds": len(used_rounds),
        "num_seeds": used_seeds,
        "rounds": used_rounds,
        "budget_training_rounds": used_rounds,
        "collapsed_active_training_rounds": used_rounds,
        "holdout_round_number": holdout_round_number,
        "max_iter": max_iter,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "learning_rate": learning_rate,
        "l2_regularization": l2_regularization,
        "predictor_regime_thresholds": {
            "settlement_rate_low": config.regime_settlement_rate_low,
            "settlement_rate_high": config.regime_settlement_rate_high,
            "forest_share_low": config.regime_forest_share_low,
            "forest_share_high": config.regime_forest_share_high,
        },
        "ood_reference_version": OOD_REFERENCE_VERSION,
        "ood_reference": build_ood_reference_from_archive(
            history_dir,
            holdout_round_number=holdout_round_number,
        ),
    }
    return ResidualCalibratorArtifact(
        feature_names=feature_names,
        active_model=active_model,
        forest_model=forest_model,
        settlement_model=settlement_model,
        port_model=port_model,
        ruin_model=ruin_model,
        metadata=metadata,
        active_budget_model=active_budget_model,
        budget_feature_names=budget_feature_names,
        collapsed_active_model=collapsed_active_model,
        collapsed_active_feature_names=collapsed_active_feature_names,
        blend_model=blend_model,
        blend_feature_names=blend_feature_names,
    )


def _discover_round_runtime_contexts(history_dir: Path) -> dict[int, dict[str, Any]]:
    if history_dir.name != "history":
        return {}
    artifacts_root = history_dir.parent
    contexts: dict[int, dict[str, Any]] = {}
    for metadata_path in artifacts_root.glob("*/metadata.json"):
        run_dir = metadata_path.parent
        if run_dir.name == "history":
            continue
        required = [
            run_dir / "class_counts.npy",
            run_dir / "observation_counts.npy",
            run_dir / "conditional_counts.json",
        ]
        if not all(path.exists() for path in required):
            continue
        try:
            payload = load_json(metadata_path)
        except Exception:
            continue
        round_number = payload.get("round_number")
        if round_number is None:
            continue
        current = contexts.get(int(round_number))
        if current is not None and current["mtime"] >= run_dir.stat().st_mtime:
            continue
        contexts[int(round_number)] = {
            "mtime": run_dir.stat().st_mtime,
            "metadata": payload,
            "class_counts": np.load(run_dir / "class_counts.npy"),
            "observation_counts": np.load(run_dir / "observation_counts.npy"),
            "conditional_counts": {
                key: np.asarray(value, dtype=np.float64)
                for key, value in load_json(run_dir / "conditional_counts.json").items()
            },
        }
    return contexts


def _build_bucket_target_tensor(
    seed_features: SeedFeatures,
    *,
    conditional_counts: dict[str, np.ndarray],
    global_counts: np.ndarray,
    min_probability: float,
) -> np.ndarray:
    bucket_keys = make_bucket_keys(seed_features)
    h, w = bucket_keys.shape
    bucket_target = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
    global_probs = global_counts / max(float(np.sum(global_counts)), EPS)
    bucket_target[:] = global_probs
    for key, counts in conditional_counts.items():
        mask = bucket_keys == int(key)
        if np.any(mask):
            bucket_target[mask] = counts / max(float(np.sum(counts)), EPS)
    return normalize_probabilities(bucket_target, min_probability)


def _runtime_ood_signal_values(
    runtime_context: dict[str, Any] | None,
    round_dir: Path,
    prediction: np.ndarray,
    round_regime: dict[str, float],
) -> dict[str, float]:
    if runtime_context is not None:
        latent = runtime_context.get("metadata", {}).get("latent_summary")
        if isinstance(latent, dict):
            return {
                "settlement_rate": float(latent.get("settlement_rate", np.nan)),
                "forest_share_dynamic": float(latent.get("forest_share_dynamic", np.nan)),
                "port_share_given_active": float(latent.get("port_share_given_active", np.nan)),
                "observed_cells": float(latent.get("observed_cells", np.nan)),
                "repeat_fraction": float(round_regime.get("repeat_fraction", np.nan)),
            }
    metadata_path = round_dir / "metadata.json"
    if metadata_path.exists():
        payload = load_json(metadata_path)
        latent = payload.get("latent_summary")
        if isinstance(latent, dict):
            return {
                "settlement_rate": float(latent.get("settlement_rate", np.nan)),
                "forest_share_dynamic": float(latent.get("forest_share_dynamic", np.nan)),
                "port_share_given_active": float(latent.get("port_share_given_active", np.nan)),
                "observed_cells": float(latent.get("observed_cells", np.nan)),
                "repeat_fraction": float(round_regime.get("repeat_fraction", np.nan)),
            }
    latent = infer_latent_summary_from_prediction(prediction)
    return {
        "settlement_rate": float(latent.get("settlement_rate", np.nan)),
        "forest_share_dynamic": float(latent.get("forest_share_dynamic", np.nan)),
        "port_share_given_active": float(latent.get("port_share_given_active", np.nan)),
        "observed_cells": float("nan"),
        "repeat_fraction": float(round_regime.get("repeat_fraction", np.nan)),
    }


def _make_regressor(
    max_iter: int,
    max_depth: int,
    min_samples_leaf: int,
    learning_rate: float,
    l2_regularization: float,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        early_stopping=False,
        random_state=0,
    )
