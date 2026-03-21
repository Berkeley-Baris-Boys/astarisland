from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .history import _round_detail_from_json
from .learned_prior import build_learned_prior_features
from .types import CLASS_EMPTY, CLASS_FOREST, CLASS_MOUNTAIN, CLASS_NAMES, CLASS_PORT, CLASS_RUIN, CLASS_SETTLEMENT, NUM_CLASSES, SeedFeatures
from .utils import load_json, normalize_probabilities

EPS = 1e-12
RESIDUAL_CALIBRATOR_VERSION = 1


@dataclass
class ResidualCalibratorArtifact:
    feature_names: list[str]
    active_model: HistGradientBoostingRegressor
    forest_model: HistGradientBoostingRegressor
    settlement_model: HistGradientBoostingRegressor
    port_model: HistGradientBoostingRegressor
    ruin_model: HistGradientBoostingRegressor
    metadata: dict[str, Any]


def save_residual_calibrator_artifact(path: Path, artifact: ResidualCalibratorArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_residual_calibrator_artifact(path: Path) -> ResidualCalibratorArtifact:
    return joblib.load(path)


def build_residual_features(prediction: np.ndarray, seed_features: SeedFeatures) -> tuple[np.ndarray, list[str]]:
    structural, structural_names = build_learned_prior_features(seed_features)
    probs = prediction.reshape(-1, NUM_CLASSES)
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
        ]
    )

    matrix = np.concatenate([structural, np.stack(extra_arrays, axis=1)], axis=1)
    return matrix, structural_names + extra_names


def apply_residual_calibrator(
    artifact: ResidualCalibratorArtifact,
    prediction: np.ndarray,
    seed_features: SeedFeatures,
    *,
    blend: float | np.ndarray,
    min_probability: float,
) -> tuple[np.ndarray, dict[str, object]]:
    features, feature_names = build_residual_features(prediction, seed_features)
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
    matrices: list[np.ndarray] = []
    active_targets: list[np.ndarray] = []
    forest_targets: list[np.ndarray] = []
    settlement_targets: list[np.ndarray] = []
    port_targets: list[np.ndarray] = []
    ruin_targets: list[np.ndarray] = []
    active_weights: list[np.ndarray] = []
    forest_weights: list[np.ndarray] = []
    type_weights: list[np.ndarray] = []
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
            matrix, names = build_residual_features(prediction, seed_features)
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
    }
    return ResidualCalibratorArtifact(
        feature_names=feature_names,
        active_model=active_model,
        forest_model=forest_model,
        settlement_model=settlement_model,
        port_model=port_model,
        ruin_model=ruin_model,
        metadata=metadata,
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
