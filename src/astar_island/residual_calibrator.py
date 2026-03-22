from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import AstarConfig
from .history import _round_detail_from_json
from .learned_prior import build_learned_prior_features
from .regime import compute_round_regime, infer_latent_summary_from_prediction
from .scoring import cell_entropy, cell_kl_divergence
from .types import CLASS_EMPTY, CLASS_FOREST, CLASS_MOUNTAIN, CLASS_NAMES, CLASS_PORT, CLASS_RUIN, CLASS_SETTLEMENT, NUM_CLASSES, SeedFeatures
from .utils import load_json, normalize_probabilities

EPS = 1e-12
RESIDUAL_CALIBRATOR_VERSION = 3


@dataclass
class ResidualCalibratorArtifact:
    feature_names: list[str]
    active_model: HistGradientBoostingRegressor
    forest_model: HistGradientBoostingRegressor
    settlement_model: HistGradientBoostingRegressor
    port_model: HistGradientBoostingRegressor
    ruin_model: HistGradientBoostingRegressor
    metadata: dict[str, Any]
    blend_model: HistGradientBoostingRegressor | None = None
    blend_feature_names: list[str] | None = None


def save_residual_calibrator_artifact(path: Path, artifact: ResidualCalibratorArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_residual_calibrator_artifact(path: Path) -> ResidualCalibratorArtifact:
    return joblib.load(path)


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
    active_targets: list[np.ndarray] = []
    forest_targets: list[np.ndarray] = []
    settlement_targets: list[np.ndarray] = []
    port_targets: list[np.ndarray] = []
    ruin_targets: list[np.ndarray] = []
    active_weights: list[np.ndarray] = []
    forest_weights: list[np.ndarray] = []
    type_weights: list[np.ndarray] = []
    base_preds: list[np.ndarray] = []
    ground_truths: list[np.ndarray] = []
    feature_names: list[str] | None = None
    used_rounds: list[dict[str, Any]] = []
    used_seeds = 0

    for round_dir in sorted(history_dir.glob("round_*")):
        detail_path = round_dir / "round_detail.json"
        if not detail_path.exists():
            continue
        detail = _round_detail_from_json(load_json(detail_path))
        if holdout_round_number is not None and detail.round_number == holdout_round_number:
            continue
        from .features import build_all_features

        features = build_all_features(detail.initial_states)
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
            active_targets.append(active_target)
            forest_targets.append(forest_target)
            settlement_targets.append(settlement_target)
            port_targets.append(port_target)
            ruin_targets.append(ruin_target)
            active_weights.append(0.15 + 1.85 * entropy)
            forest_weights.append(non_active * (0.1 + 1.5 * entropy))
            type_weights.append(active_target * (0.1 + 2.0 * entropy))
            base_preds.append(prediction.reshape(-1, NUM_CLASSES))
            ground_truths.append(gt)
            used_seeds += 1
            round_used = True
        if round_used:
            used_rounds.append({"round_id": detail.round_id, "round_number": detail.round_number})

    if not matrices or feature_names is None:
        raise ValueError("No archived predictions available to build residual calibrator.")

    X = np.concatenate(matrices, axis=0)
    active_y = np.concatenate(active_targets, axis=0)
    forest_y = np.concatenate(forest_targets, axis=0)
    settlement_y = np.concatenate(settlement_targets, axis=0)
    port_y = np.concatenate(port_targets, axis=0)
    ruin_y = np.concatenate(ruin_targets, axis=0)
    active_w = np.concatenate(active_weights, axis=0)
    forest_w = np.concatenate(forest_weights, axis=0)
    type_w = np.concatenate(type_weights, axis=0)

    active_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)
    forest_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)
    settlement_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)
    port_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)
    ruin_model = _make_regressor(max_iter, max_depth, min_samples_leaf, learning_rate, l2_regularization)

    active_model.fit(X, active_y, sample_weight=active_w)
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
    }
    return ResidualCalibratorArtifact(
        feature_names=feature_names,
        active_model=active_model,
        forest_model=forest_model,
        settlement_model=settlement_model,
        port_model=port_model,
        ruin_model=ruin_model,
        metadata=metadata,
        blend_model=blend_model,
        blend_feature_names=blend_feature_names,
    )


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
