from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from .types import NUM_CLASSES


def setup_logging(log_dir: Path, run_name: str | None = None) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / (run_name or timestamp)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )
    return run_dir


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable(payload), sort_keys=True))
        handle.write("\n")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_cache_key(method: str, url: str, payload: dict[str, Any] | None = None) -> str:
    body = json.dumps(payload or {}, sort_keys=True, separators=(",", ":"))
    raw = f"{method}|{url}|{body}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def softmax(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    shifted = values / max(temperature, 1e-8)
    shifted = shifted - np.max(shifted)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def normalize_probabilities(prediction: np.ndarray, min_probability: float) -> np.ndarray:
    prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
    prediction = np.maximum(prediction, min_probability)
    denom = prediction.sum(axis=-1, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    return prediction / denom


def validate_prediction_tensor(prediction: np.ndarray, min_probability: float) -> None:
    if prediction.ndim != 3 or prediction.shape[-1] != NUM_CLASSES:
        raise ValueError(f"Expected HxWx{NUM_CLASSES}, got {prediction.shape}")
    if not np.all(np.isfinite(prediction)):
        raise ValueError("Prediction contains NaN or inf")
    if np.any(prediction < 0):
        raise ValueError("Prediction contains negative probabilities")
    sums = prediction.sum(axis=-1)
    if np.max(np.abs(sums - 1.0)) > 1e-3:
        raise ValueError("Prediction probabilities do not sum to 1 within tolerance")
    if np.any(prediction <= 0):
        raise ValueError("Prediction contains zero probabilities after smoothing")


def window_slices(x: int, y: int, w: int, h: int) -> tuple[slice, slice]:
    return slice(y, y + h), slice(x, x + w)


def entropy_from_counts(counts: np.ndarray) -> np.ndarray:
    totals = counts.sum(axis=-1, keepdims=True)
    probs = np.zeros_like(counts, dtype=np.float64)
    np.divide(counts, np.maximum(totals, 1e-12), out=probs, where=totals > 0)
    safe_probs = np.where(probs > 0, probs, 1.0)
    return -np.sum(probs * np.log(safe_probs), axis=-1)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
