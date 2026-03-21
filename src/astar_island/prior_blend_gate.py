from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .history import _round_detail_from_json
from .learned_prior import build_learned_prior_features, load_learned_prior_artifact
from .priors import load_historical_prior_artifact
from .scoring import cell_entropy, cell_kl_divergence
from .types import CLASS_EMPTY, CLASS_FOREST, CLASS_MOUNTAIN, CLASS_NAMES, CLASS_PORT, CLASS_RUIN, CLASS_SETTLEMENT, NUM_CLASSES, SeedFeatures
from .utils import load_json, normalize_probabilities

EPS = 1e-12
PRIOR_BLEND_GATE_VERSION = 1


@dataclass
class PriorBlendGateArtifact:
    feature_names: list[str]
    model: HistGradientBoostingRegressor
    metadata: dict[str, Any]


def save_prior_blend_gate_artifact(path: Path, artifact: PriorBlendGateArtifact) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_prior_blend_gate_artifact(path: Path) -> PriorBlendGateArtifact:
    return joblib.load(path)


def build_prior_blend_features(prior: np.ndarray, prediction: np.ndarray, seed_features: SeedFeatures) -> tuple[np.ndarray, list[str]]:
    structural, structural_names = build_learned_prior_features(seed_features)
    prior_flat = prior.reshape(-1, NUM_CLASSES)
    pred_flat = prediction.reshape(-1, NUM_CLASSES)

    prior_active = prior_flat[:, CLASS_SETTLEMENT] + prior_flat[:, CLASS_PORT] + prior_flat[:, CLASS_RUIN]
    pred_active = pred_flat[:, CLASS_SETTLEMENT] + pred_flat[:, CLASS_PORT] + pred_flat[:, CLASS_RUIN]
    prior_non_active = prior_flat[:, CLASS_EMPTY] + prior_flat[:, CLASS_FOREST]
    pred_non_active = pred_flat[:, CLASS_EMPTY] + pred_flat[:, CLASS_FOREST]
    prior_forest_share = prior_flat[:, CLASS_FOREST] / np.maximum(prior_non_active, EPS)
    pred_forest_share = pred_flat[:, CLASS_FOREST] / np.maximum(pred_non_active, EPS)
    prior_settlement_share = prior_flat[:, CLASS_SETTLEMENT] / np.maximum(prior_active, EPS)
    pred_settlement_share = pred_flat[:, CLASS_SETTLEMENT] / np.maximum(pred_active, EPS)
    prior_port_share = prior_flat[:, CLASS_PORT] / np.maximum(prior_active, EPS)
    pred_port_share = pred_flat[:, CLASS_PORT] / np.maximum(pred_active, EPS)
    prior_ruin_share = prior_flat[:, CLASS_RUIN] / np.maximum(prior_active, EPS)
    pred_ruin_share = pred_flat[:, CLASS_RUIN] / np.maximum(pred_active, EPS)
    prior_entropy = cell_entropy(prior_flat)
    pred_entropy = cell_entropy(pred_flat)
    prior_margin = _margin(prior_flat)
    pred_margin = _margin(pred_flat)
    abs_diff = np.abs(pred_flat - prior_flat)
    l1_diff = np.sum(abs_diff, axis=1)
    l2_diff = np.sqrt(np.sum((pred_flat - prior_flat) ** 2, axis=1))
    argmax_same = (np.argmax(pred_flat, axis=1) == np.argmax(prior_flat, axis=1)).astype(np.float64)
    structural_map = {name: structural[:, idx] for idx, name in enumerate(structural_names)}

    extra_arrays = [
        prior_flat[:, CLASS_EMPTY],
        prior_flat[:, CLASS_SETTLEMENT],
        prior_flat[:, CLASS_PORT],
        prior_flat[:, CLASS_RUIN],
        prior_flat[:, CLASS_FOREST],
        prior_flat[:, CLASS_MOUNTAIN],
        pred_flat[:, CLASS_EMPTY],
        pred_flat[:, CLASS_SETTLEMENT],
        pred_flat[:, CLASS_PORT],
        pred_flat[:, CLASS_RUIN],
        pred_flat[:, CLASS_FOREST],
        pred_flat[:, CLASS_MOUNTAIN],
        pred_flat[:, CLASS_EMPTY] - prior_flat[:, CLASS_EMPTY],
        pred_flat[:, CLASS_SETTLEMENT] - prior_flat[:, CLASS_SETTLEMENT],
        pred_flat[:, CLASS_PORT] - prior_flat[:, CLASS_PORT],
        pred_flat[:, CLASS_RUIN] - prior_flat[:, CLASS_RUIN],
        pred_flat[:, CLASS_FOREST] - prior_flat[:, CLASS_FOREST],
        pred_flat[:, CLASS_MOUNTAIN] - prior_flat[:, CLASS_MOUNTAIN],
        prior_active,
        pred_active,
        pred_active - prior_active,
        prior_forest_share,
        pred_forest_share,
        prior_settlement_share,
        pred_settlement_share,
        prior_port_share,
        pred_port_share,
        prior_ruin_share,
        pred_ruin_share,
        prior_entropy,
        pred_entropy,
        pred_entropy - prior_entropy,
        prior_margin,
        pred_margin,
        pred_margin - prior_margin,
        l1_diff,
        l2_diff,
        argmax_same,
        structural_map["coastal"] * pred_flat[:, CLASS_PORT],
        structural_map["frontier"] * pred_active,
        structural_map["conflict"] * pred_flat[:, CLASS_RUIN],
        structural_map["initial_class_forest"] * pred_flat[:, CLASS_FOREST],
        structural_map["initial_class_settlement"] * pred_flat[:, CLASS_SETTLEMENT],
        structural_map["border_support"] * pred_active,
        structural_map["settlement_intensity"] * pred_settlement_share,
        structural_map["forest_density"] * pred_flat[:, CLASS_FOREST],
        structural_map["coastal"] * (pred_flat[:, CLASS_PORT] - prior_flat[:, CLASS_PORT]),
        structural_map["frontier"] * (pred_active - prior_active),
    ]
    extra_names = [
        "prior_empty",
        "prior_settlement",
        "prior_port",
        "prior_ruin",
        "prior_forest",
        "prior_mountain",
        "pred_empty",
        "pred_settlement",
        "pred_port",
        "pred_ruin",
        "pred_forest",
        "pred_mountain",
        "delta_empty",
        "delta_settlement",
        "delta_port",
        "delta_ruin",
        "delta_forest",
        "delta_mountain",
        "prior_active",
        "pred_active",
        "delta_active",
        "prior_forest_share_non_active",
        "pred_forest_share_non_active",
        "prior_settlement_share_active",
        "pred_settlement_share_active",
        "prior_port_share_active",
        "pred_port_share_active",
        "prior_ruin_share_active",
        "pred_ruin_share_active",
        "prior_entropy",
        "pred_entropy",
        "delta_entropy",
        "prior_margin",
        "pred_margin",
        "delta_margin",
        "l1_diff",
        "l2_diff",
        "argmax_same",
        "coastal_x_pred_port",
        "frontier_x_pred_active",
        "conflict_x_pred_ruin",
        "initial_forest_x_pred_forest",
        "initial_settlement_x_pred_settlement",
        "border_support_x_pred_active",
        "settlement_intensity_x_pred_settlement_share_active",
        "forest_density_x_pred_forest",
        "coastal_x_delta_port",
        "frontier_x_delta_active",
    ]
    matrix = np.concatenate([structural, np.stack(extra_arrays, axis=1)], axis=1)
    return matrix, structural_names + extra_names


def apply_prior_blend_gate(
    artifact: PriorBlendGateArtifact,
    prior: np.ndarray,
    prediction: np.ndarray,
    seed_features: SeedFeatures,
    *,
    min_probability: float,
    strength: float = 1.0,
) -> tuple[np.ndarray, dict[str, object]]:
    features, feature_names = build_prior_blend_features(prior, prediction, seed_features)
    if feature_names != artifact.feature_names:
        raise ValueError("Prior blend gate feature names do not match runtime construction.")

    alpha = np.clip(artifact.model.predict(features), 0.0, 1.0).reshape(prediction.shape[:2])
    strength = float(np.clip(strength, 0.0, 1.0))
    alpha = 1.0 - strength * (1.0 - alpha)
    blended = normalize_probabilities(
        (1.0 - alpha[..., None]) * prior + alpha[..., None] * prediction,
        min_probability,
    )
    details = {
        "summary": {
            "alpha_mean": float(np.mean(alpha)),
            "alpha_min": float(np.min(alpha)),
            "alpha_max": float(np.max(alpha)),
            "strength": strength,
        },
        "tensors": {
            "prior_gate_alpha": alpha,
            "prior_gate_prediction": blended,
        },
    }
    return blended, details


def build_prior_blend_gate_artifact_from_archive(
    history_dir: Path,
    *,
    holdout_round_number: int | None = None,
    max_iter: int = 300,
    max_depth: int = 4,
    min_samples_leaf: int = 400,
    learning_rate: float = 0.05,
    l2_regularization: float = 0.2,
) -> PriorBlendGateArtifact:
    from .config import AstarConfig
    from .features import build_all_features
    from .predictor import Predictor

    config = AstarConfig()
    historical = load_historical_prior_artifact(config.predictor.historical_prior_path)
    learned = load_learned_prior_artifact(config.predictor.learned_prior_path)

    matrices: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
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
        features = build_all_features(detail.initial_states)
        predictor = Predictor(
            config.predictor,
            detail,
            features,
            historical_priors=historical,
            learned_prior=learned,
            residual_calibrator=None,
        )
        round_used = False
        for seed_index, seed_features in features.items():
            analysis_path = round_dir / f"seed_{seed_index}" / "analysis.json"
            if not analysis_path.exists():
                continue
            payload = load_json(analysis_path)
            prediction = np.asarray(payload["prediction"], dtype=np.float64)
            ground_truth = np.asarray(payload["ground_truth"], dtype=np.float64)
            latent = infer_latent_summary_from_prediction(prediction)
            prior = predictor._build_prior(seed_index, seed_features, None, latent)
            matrix, names = build_prior_blend_features(prior, prediction, seed_features)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("Prior blend gate feature name mismatch while building artifact.")
            target = optimal_prior_blend(prior.reshape(-1, NUM_CLASSES), prediction.reshape(-1, NUM_CLASSES), ground_truth.reshape(-1, NUM_CLASSES))
            weight = prior_blend_sample_weight(prior.reshape(-1, NUM_CLASSES), prediction.reshape(-1, NUM_CLASSES), ground_truth.reshape(-1, NUM_CLASSES))
            matrices.append(matrix)
            targets.append(target)
            weights.append(weight)
            used_seeds += 1
            round_used = True
        if round_used:
            used_rounds.append({"round_id": detail.round_id, "round_number": detail.round_number})

    if not matrices or feature_names is None:
        raise ValueError("No archived predictions available to build prior blend gate.")

    X = np.concatenate(matrices, axis=0)
    y = np.concatenate(targets, axis=0)
    sample_weight = np.concatenate(weights, axis=0)
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        early_stopping=False,
        random_state=0,
    )
    model.fit(X, y, sample_weight=sample_weight)

    metadata = {
        "version": PRIOR_BLEND_GATE_VERSION,
        "num_rounds": len(used_rounds),
        "num_seeds": used_seeds,
        "rounds": used_rounds,
        "holdout_round_number": holdout_round_number,
        "max_iter": max_iter,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "learning_rate": learning_rate,
        "l2_regularization": l2_regularization,
        "historical_prior_metadata": historical.metadata,
        "learned_prior_metadata": learned.metadata,
    }
    return PriorBlendGateArtifact(feature_names=feature_names, model=model, metadata=metadata)


def infer_latent_summary_from_prediction(prediction: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    totals = prediction.sum(axis=(0, 1))
    settlement_mass = float(totals[CLASS_SETTLEMENT] + totals[CLASS_PORT])
    dynamic_mass = float(totals[CLASS_SETTLEMENT] + totals[CLASS_PORT] + totals[CLASS_RUIN] + totals[CLASS_FOREST])
    total_mass = float(np.sum(totals))
    return {
        "observed_cells": 0.0,
        "settlement_rate": settlement_mass / max(total_mass, 1.0),
        "port_share_given_active": float(totals[CLASS_PORT]) / max(settlement_mass, 1.0),
        "ruin_share_given_active": float(totals[CLASS_RUIN]) / max(settlement_mass + totals[CLASS_RUIN], 1.0),
        "forest_share_dynamic": float(totals[CLASS_FOREST]) / max(dynamic_mass, 1.0),
        "mean_food": 0.0,
        "mean_wealth": 0.0,
        "mean_defense": 0.0,
        "mean_population": 0.0,
    }


def optimal_prior_blend(prior: np.ndarray, prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    alphas = np.linspace(0.0, 1.0, 21, dtype=np.float64)
    best_alpha = np.ones(prior.shape[0], dtype=np.float64)
    best_loss = np.full(prior.shape[0], np.inf, dtype=np.float64)
    for alpha in alphas:
        blended = (1.0 - alpha) * prior + alpha * prediction
        blended = normalize_probabilities(blended, 1e-6)
        loss = cell_kl_divergence(ground_truth, blended)
        better = loss < best_loss
        best_loss[better] = loss[better]
        best_alpha[better] = alpha
    return best_alpha


def prior_blend_sample_weight(prior: np.ndarray, prediction: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    entropy = cell_entropy(ground_truth)
    disagreement = np.sum(np.abs(prediction - prior), axis=1)
    return 0.05 + entropy * (0.25 + disagreement)


def _margin(probabilities: np.ndarray) -> np.ndarray:
    sorted_probs = np.sort(probabilities, axis=1)
    return sorted_probs[:, -1] - sorted_probs[:, -2]
