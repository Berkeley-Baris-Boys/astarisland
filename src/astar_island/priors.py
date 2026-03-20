from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .features import build_all_features, make_bucket_keys
from .history import build_historical_prior_from_archive
from .types import NUM_CLASSES, RoundDetail
from .utils import load_json, save_json

LOGGER = logging.getLogger(__name__)


@dataclass
class HistoricalPriorArtifact:
    bucket_counts: dict[str, np.ndarray]
    initial_class_counts: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "bucket_counts": {key: value.tolist() for key, value in self.bucket_counts.items()},
            "initial_class_counts": {key: value.tolist() for key, value in self.initial_class_counts.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "HistoricalPriorArtifact":
        return cls(
            bucket_counts={key: np.asarray(value, dtype=np.float64) for key, value in payload.get("bucket_counts", {}).items()},
            initial_class_counts={
                key: np.asarray(value, dtype=np.float64) for key, value in payload.get("initial_class_counts", {}).items()
            },
            metadata=payload.get("metadata", {}),
        )


def save_historical_prior_artifact(path: Path, artifact: HistoricalPriorArtifact) -> None:
    save_json(path, artifact.to_json())


def load_historical_prior_artifact(path: Path) -> HistoricalPriorArtifact:
    return HistoricalPriorArtifact.from_json(load_json(path))


def extract_ground_truth_tensor(payload: Any) -> np.ndarray | None:
    tensor = _recursive_find_tensor(payload)
    if tensor is None:
        return None
    return np.asarray(tensor, dtype=np.float64)


def _recursive_find_tensor(payload: Any) -> Any | None:
    if isinstance(payload, dict):
        for preferred_key in ("ground_truth", "target", "truth", "probabilities"):
            if preferred_key in payload:
                found = _recursive_find_tensor(payload[preferred_key])
                if found is not None:
                    return found
        for value in payload.values():
            found = _recursive_find_tensor(value)
            if found is not None:
                return found
        return None
    if isinstance(payload, list):
        if _looks_like_probability_tensor(payload):
            return payload
        for value in payload:
            found = _recursive_find_tensor(value)
            if found is not None:
                return found
    return None


def _looks_like_probability_tensor(value: Any) -> bool:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except Exception:
        return False
    return arr.ndim == 3 and arr.shape[-1] == NUM_CLASSES


def build_historical_prior_artifact(api, round_ids: list[str] | None = None, max_rounds: int | None = None) -> HistoricalPriorArtifact:
    archive_dir = getattr(api.config, "history_dir", None)
    if archive_dir is not None and archive_dir.exists():
        bucket_counts, initial_class_counts, metadata = build_historical_prior_from_archive(archive_dir, max_rounds=max_rounds)
        if metadata["num_seeds"] > 0:
            artifact = HistoricalPriorArtifact(bucket_counts=bucket_counts, initial_class_counts=initial_class_counts, metadata=metadata)
            LOGGER.info("Built historical prior artifact from local archive using %s rounds and %s seeds", metadata["num_rounds"], metadata["num_seeds"])
            return artifact

    rounds = api.get_rounds(use_cache=False)
    if round_ids is None:
        eligible = [item for item in rounds if item.get("status") == "completed"]
        eligible.sort(key=lambda item: int(item.get("round_number", 0)), reverse=True)
        if max_rounds is not None:
            eligible = eligible[:max_rounds]
        round_ids = [item["id"] for item in eligible]

    bucket_counts: dict[str, np.ndarray] = {}
    initial_class_counts: dict[str, np.ndarray] = {}
    used_rounds: list[dict[str, Any]] = []
    used_seeds = 0

    for round_id in round_ids:
        try:
            detail: RoundDetail = api.get_round_details(round_id, use_cache=True)
        except Exception as exc:
            LOGGER.warning("Skipping round %s: failed to load details: %s", round_id, exc)
            continue
        features = build_all_features(detail.initial_states)
        round_used = False
        for seed_index, seed_features in features.items():
            try:
                analysis = api.get_analysis(round_id, seed_index)
            except Exception as exc:
                LOGGER.warning("Skipping round %s seed %s analysis: %s", round_id, seed_index, exc)
                continue
            ground_truth = extract_ground_truth_tensor(analysis)
            if ground_truth is None:
                LOGGER.warning("Skipping round %s seed %s: could not locate ground truth tensor", round_id, seed_index)
                continue
            if ground_truth.shape != (detail.map_height, detail.map_width, NUM_CLASSES):
                LOGGER.warning(
                    "Skipping round %s seed %s: expected %sx%sx%s, got %s",
                    round_id,
                    seed_index,
                    detail.map_height,
                    detail.map_width,
                    NUM_CLASSES,
                    ground_truth.shape,
                )
                continue
            round_used = True
            used_seeds += 1
            _accumulate_ground_truth(bucket_counts, initial_class_counts, seed_features.initial_class_grid, make_bucket_keys(seed_features), ground_truth)
        if round_used:
            used_rounds.append({"round_id": round_id, "round_number": detail.round_number})

    metadata = {
        "num_rounds": len(used_rounds),
        "num_seeds": used_seeds,
        "rounds": used_rounds,
    }
    artifact = HistoricalPriorArtifact(bucket_counts=bucket_counts, initial_class_counts=initial_class_counts, metadata=metadata)
    LOGGER.info("Built historical prior artifact from %s rounds and %s seeds", metadata["num_rounds"], used_seeds)
    return artifact


def _accumulate_ground_truth(
    bucket_counts: dict[str, np.ndarray],
    initial_class_counts: dict[str, np.ndarray],
    initial_class_grid: np.ndarray,
    bucket_keys: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    for key in np.unique(bucket_keys):
        mask = bucket_keys == key
        counts = ground_truth[mask].sum(axis=0)
        bucket_counts.setdefault(str(int(key)), np.zeros(NUM_CLASSES, dtype=np.float64))
        bucket_counts[str(int(key))] += counts
    for class_id in np.unique(initial_class_grid):
        mask = initial_class_grid == class_id
        counts = ground_truth[mask].sum(axis=0)
        initial_class_counts.setdefault(str(int(class_id)), np.zeros(NUM_CLASSES, dtype=np.float64))
        initial_class_counts[str(int(class_id))] += counts
