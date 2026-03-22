from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp

from .history import _round_detail_from_json
from .ood import OOD_REFERENCE_VERSION, build_ood_reference_from_archive
from .types import CLASS_EMPTY, CLASS_FOREST, CLASS_MOUNTAIN, CLASS_PORT, CLASS_RUIN, CLASS_SETTLEMENT, NUM_CLASSES, SeedFeatures
from .utils import load_json, normalize_probabilities, save_json

LEARNED_PRIOR_VERSION = 1
EPS = 1e-12


@dataclass
class BinaryLogisticModel:
    weights: np.ndarray
    bias: float

    def to_json(self) -> dict[str, Any]:
        return {"weights": self.weights.tolist(), "bias": float(self.bias)}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "BinaryLogisticModel":
        return cls(
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
        )


@dataclass
class SoftmaxModel:
    weights: np.ndarray
    bias: np.ndarray

    def to_json(self) -> dict[str, Any]:
        return {"weights": self.weights.tolist(), "bias": self.bias.tolist()}

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "SoftmaxModel":
        return cls(
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=np.asarray(payload["bias"], dtype=np.float64),
        )


@dataclass
class LearnedPriorArtifact:
    feature_names: list[str]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    active_model: BinaryLogisticModel
    forest_model: BinaryLogisticModel
    active_type_model: SoftmaxModel
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_names": self.feature_names,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "active_model": self.active_model.to_json(),
            "forest_model": self.forest_model.to_json(),
            "active_type_model": self.active_type_model.to_json(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "LearnedPriorArtifact":
        return cls(
            feature_names=list(payload["feature_names"]),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
            active_model=BinaryLogisticModel.from_json(payload["active_model"]),
            forest_model=BinaryLogisticModel.from_json(payload["forest_model"]),
            active_type_model=SoftmaxModel.from_json(payload["active_type_model"]),
            metadata=payload.get("metadata", {}),
        )


def save_learned_prior_artifact(path: Path, artifact: LearnedPriorArtifact) -> None:
    save_json(path, artifact.to_json())


def load_learned_prior_artifact(path: Path) -> LearnedPriorArtifact:
    return LearnedPriorArtifact.from_json(load_json(path))


def build_learned_prior_features(seed_features: SeedFeatures) -> tuple[np.ndarray, list[str]]:
    names = seed_features.feature_names
    stack = seed_features.feature_stack

    feature_arrays: list[np.ndarray] = [stack[..., idx] for idx in range(stack.shape[-1])]
    feature_names = names[:]

    initial_class_grid = seed_features.initial_class_grid
    for class_id, label in (
        (CLASS_EMPTY, "initial_class_empty"),
        (CLASS_SETTLEMENT, "initial_class_settlement"),
        (CLASS_PORT, "initial_class_port"),
        (CLASS_RUIN, "initial_class_ruin"),
        (CLASS_FOREST, "initial_class_forest"),
        (CLASS_MOUNTAIN, "initial_class_mountain"),
    ):
        feature_arrays.append((initial_class_grid == class_id).astype(np.float64))
        feature_names.append(label)

    settlement_intensity = stack[..., names.index("settlement_intensity")]
    port_intensity = stack[..., names.index("port_intensity")]
    settlement_density = stack[..., names.index("settlement_density")]
    forest_density = stack[..., names.index("forest_density")]
    border_distance = stack[..., names.index("border_distance")]
    coastal = seed_features.coastal_mask.astype(np.float64)
    frontier = seed_features.frontier_mask.astype(np.float64)
    conflict = seed_features.conflict_mask.astype(np.float64)
    reclaimable = seed_features.reclaimable_mask.astype(np.float64)
    initial_forest = (initial_class_grid == CLASS_FOREST).astype(np.float64)
    buildable = seed_features.buildable_mask.astype(np.float64)
    border_support = 1.0 - np.clip(border_distance, 0.0, 1.0)

    derived = {
        "border_support": border_support,
        "settlement_intensity_sq": settlement_intensity * settlement_intensity,
        "port_intensity_sq": port_intensity * port_intensity,
        "settlement_density_sq": settlement_density * settlement_density,
        "forest_density_sq": forest_density * forest_density,
        "coastal_x_settlement_intensity": coastal * settlement_intensity,
        "coastal_x_port_intensity": coastal * port_intensity,
        "frontier_x_settlement_intensity": frontier * settlement_intensity,
        "conflict_x_settlement_density": conflict * settlement_density,
        "initial_forest_x_forest_density": initial_forest * forest_density,
        "initial_forest_x_settlement_intensity": initial_forest * settlement_intensity,
        "reclaimable_x_settlement_intensity": reclaimable * settlement_intensity,
        "coastal_x_border_support": coastal * border_support,
        "buildable_x_settlement_intensity": buildable * settlement_intensity,
        "buildable_x_forest_density": buildable * forest_density,
        "settlement_x_forest_density": settlement_density * forest_density,
    }
    for label, value in derived.items():
        feature_arrays.append(value)
        feature_names.append(label)

    matrix = np.stack(feature_arrays, axis=-1).reshape(-1, len(feature_names))
    return matrix, feature_names


def predict_learned_prior(artifact: LearnedPriorArtifact, seed_features: SeedFeatures, min_probability: float) -> np.ndarray:
    features, names = build_learned_prior_features(seed_features)
    if names != artifact.feature_names:
        raise ValueError("Learned prior feature names do not match runtime feature construction.")
    standardized = (features - artifact.feature_mean) / artifact.feature_scale

    active_logit = standardized @ artifact.active_model.weights + artifact.active_model.bias
    active_prob = expit(active_logit)

    forest_logit = standardized @ artifact.forest_model.weights + artifact.forest_model.bias
    forest_prob_non_active = expit(forest_logit)

    active_type_logits = standardized @ artifact.active_type_model.weights + artifact.active_type_model.bias
    active_type_logits = active_type_logits - np.max(active_type_logits, axis=1, keepdims=True)
    active_type = np.exp(active_type_logits)
    active_type /= np.maximum(active_type.sum(axis=1, keepdims=True), EPS)

    non_active = np.clip(1.0 - active_prob, 0.0, 1.0)
    result = np.zeros((features.shape[0], NUM_CLASSES), dtype=np.float64)
    result[:, CLASS_EMPTY] = non_active * (1.0 - forest_prob_non_active)
    result[:, CLASS_FOREST] = non_active * forest_prob_non_active
    result[:, CLASS_SETTLEMENT] = active_prob * active_type[:, 0]
    result[:, CLASS_PORT] = active_prob * active_type[:, 1]
    result[:, CLASS_RUIN] = active_prob * active_type[:, 2]

    mountain_mask = seed_features.initial_class_grid.reshape(-1) == CLASS_MOUNTAIN
    if np.any(mountain_mask):
        result[mountain_mask] = min_probability
        result[mountain_mask, CLASS_MOUNTAIN] = 1.0

    result = result.reshape(seed_features.initial_class_grid.shape + (NUM_CLASSES,))
    return normalize_probabilities(result, min_probability)


def build_learned_prior_artifact_from_archive(
    history_dir: Path,
    *,
    holdout_round_number: int | None = None,
    l2_active: float = 0.6,
    l2_forest: float = 0.6,
    l2_active_type: float = 0.8,
    maxiter: int = 300,
) -> LearnedPriorArtifact:
    matrices: list[np.ndarray] = []
    gt_tensors: list[np.ndarray] = []
    used_rounds: list[dict[str, Any]] = []
    used_seeds = 0
    feature_names: list[str] | None = None

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
            gt_path = round_dir / f"seed_{seed_index}" / "ground_truth.npy"
            if not gt_path.exists():
                continue
            matrix, names = build_learned_prior_features(seed_features)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("Feature name mismatch while building learned prior artifact.")
            matrices.append(matrix)
            gt_tensors.append(np.load(gt_path).reshape(-1, NUM_CLASSES))
            used_seeds += 1
            round_used = True
        if round_used:
            used_rounds.append({"round_id": detail.round_id, "round_number": detail.round_number})

    if not matrices or feature_names is None:
        raise ValueError("No archived rounds available to build learned prior artifact.")

    X = np.concatenate(matrices, axis=0)
    gt = np.concatenate(gt_tensors, axis=0)
    feature_mean = X.mean(axis=0)
    feature_scale = np.maximum(X.std(axis=0), 1e-6)
    X_std = (X - feature_mean) / feature_scale

    entropy = -np.sum(np.where(gt > 0, gt * np.log(np.maximum(gt, EPS)), 0.0), axis=1)
    non_mountain = np.clip(1.0 - gt[:, CLASS_MOUNTAIN], 0.0, 1.0)
    active_target = np.clip(gt[:, CLASS_SETTLEMENT] + gt[:, CLASS_PORT] + gt[:, CLASS_RUIN], 0.0, 1.0)
    active_weight = 0.15 + 1.85 * entropy

    forest_residual = np.clip(gt[:, CLASS_EMPTY] + gt[:, CLASS_FOREST], EPS, 1.0)
    forest_target = gt[:, CLASS_FOREST] / forest_residual
    forest_weight = forest_residual * (0.1 + 1.5 * entropy) * non_mountain

    active_mass = np.clip(active_target, EPS, 1.0)
    active_type_target = gt[:, [CLASS_SETTLEMENT, CLASS_PORT, CLASS_RUIN]] / active_mass[:, None]
    active_type_weight = active_target * (0.1 + 2.0 * entropy)

    active_model = _fit_binary_logistic(X_std, active_target, active_weight, l2=l2_active, maxiter=maxiter)
    forest_model = _fit_binary_logistic(X_std, forest_target, forest_weight, l2=l2_forest, maxiter=maxiter)
    active_type_model = _fit_softmax(X_std, active_type_target, active_type_weight, l2=l2_active_type, maxiter=maxiter)

    metadata = {
        "version": LEARNED_PRIOR_VERSION,
        "num_rounds": len(used_rounds),
        "num_seeds": used_seeds,
        "rounds": used_rounds,
        "holdout_round_number": holdout_round_number,
        "l2_active": l2_active,
        "l2_forest": l2_forest,
        "l2_active_type": l2_active_type,
        "ood_reference_version": OOD_REFERENCE_VERSION,
        "ood_reference": build_ood_reference_from_archive(
            history_dir,
            holdout_round_number=holdout_round_number,
        ),
    }
    return LearnedPriorArtifact(
        feature_names=feature_names,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        active_model=active_model,
        forest_model=forest_model,
        active_type_model=active_type_model,
        metadata=metadata,
    )


def _fit_binary_logistic(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    l2: float,
    maxiter: int,
) -> BinaryLogisticModel:
    n_features = X.shape[1]
    total_weight = max(float(np.sum(sample_weight)), EPS)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        weights = theta[:n_features]
        bias = theta[n_features]
        logits = X @ weights + bias
        probs = expit(logits)
        loss = np.sum(sample_weight * (np.logaddexp(0.0, logits) - y * logits)) / total_weight
        loss += 0.5 * l2 * float(np.dot(weights, weights))
        diff = sample_weight * (probs - y) / total_weight
        grad_weights = X.T @ diff + l2 * weights
        grad_bias = float(np.sum(diff))
        grad = np.concatenate([grad_weights, np.array([grad_bias], dtype=np.float64)])
        return float(loss), grad

    theta0 = np.zeros(n_features + 1, dtype=np.float64)
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    theta = result.x
    return BinaryLogisticModel(weights=theta[:n_features], bias=float(theta[n_features]))


def _fit_softmax(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    l2: float,
    maxiter: int,
) -> SoftmaxModel:
    n_features = X.shape[1]
    n_classes = y.shape[1]
    total_weight = max(float(np.sum(sample_weight)), EPS)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        weights = theta[: n_features * n_classes].reshape(n_features, n_classes)
        bias = theta[n_features * n_classes :].reshape(1, n_classes)
        logits = X @ weights + bias
        log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
        probs = np.exp(log_probs)
        loss = -np.sum(sample_weight[:, None] * y * log_probs) / total_weight
        loss += 0.5 * l2 * float(np.sum(weights * weights))
        diff = sample_weight[:, None] * (probs - y) / total_weight
        grad_weights = X.T @ diff + l2 * weights
        grad_bias = np.sum(diff, axis=0)
        grad = np.concatenate([grad_weights.ravel(), grad_bias.ravel()])
        return float(loss), grad

    theta0 = np.zeros(n_features * n_classes + n_classes, dtype=np.float64)
    result = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    theta = result.x
    weights = theta[: n_features * n_classes].reshape(n_features, n_classes)
    bias = theta[n_features * n_classes :]
    return SoftmaxModel(weights=weights, bias=bias)
