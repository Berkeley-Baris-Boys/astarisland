from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .history import _round_detail_from_json
from .regime import infer_latent_summary_from_prediction, repeat_fraction_from_observation_counts
from .utils import load_json

OOD_REFERENCE_VERSION = 1
OOD_SIGNAL_NAMES = (
    "settlement_rate",
    "forest_share_dynamic",
    "port_share_given_active",
    "repeat_fraction",
    "observed_cells",
)
_FALLBACK_LATENT_SIGNALS = (
    "settlement_rate",
    "forest_share_dynamic",
    "port_share_given_active",
)


def build_ood_reference_from_archive(
    history_dir: Path,
    *,
    holdout_round_number: int | None = None,
) -> dict[str, Any]:
    samples: list[dict[str, float]] = []

    for round_dir in sorted(history_dir.glob("round_*")):
        detail_path = round_dir / "round_detail.json"
        if not detail_path.exists():
            continue
        detail = _round_detail_from_json(load_json(detail_path))
        if holdout_round_number is not None and detail.round_number == holdout_round_number:
            continue
        sample = _load_round_ood_sample(round_dir)
        if sample:
            samples.append(sample)

    signal_stats: dict[str, dict[str, float | int]] = {}
    for signal_name in OOD_SIGNAL_NAMES:
        values = [
            float(sample[signal_name])
            for sample in samples
            if signal_name in sample and np.isfinite(sample[signal_name])
        ]
        if not values:
            continue
        arr = np.asarray(values, dtype=np.float64)
        signal_stats[signal_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p10": float(np.percentile(arr, 10.0)),
            "p90": float(np.percentile(arr, 90.0)),
            "count": int(arr.size),
        }

    return {
        "version": OOD_REFERENCE_VERSION,
        "num_rounds": len(samples),
        "signals": signal_stats,
    }


def _load_round_ood_sample(round_dir: Path) -> dict[str, float]:
    sample: dict[str, float] = {}

    metadata_path = round_dir / "metadata.json"
    if metadata_path.exists():
        payload = load_json(metadata_path)
        latent = payload.get("latent_summary")
        if isinstance(latent, dict):
            for signal_name in ("settlement_rate", "forest_share_dynamic", "port_share_given_active", "observed_cells"):
                value = latent.get(signal_name)
                if value is not None:
                    sample[signal_name] = float(value)

    fallback_latent = _infer_round_latent_from_saved_predictions(round_dir)
    if fallback_latent is not None:
        for signal_name in _FALLBACK_LATENT_SIGNALS:
            sample.setdefault(signal_name, float(fallback_latent[signal_name]))

    observation_counts_path = round_dir / "observation_counts.npy"
    if observation_counts_path.exists():
        observation_counts = np.load(observation_counts_path)
        sample["repeat_fraction"] = float(repeat_fraction_from_observation_counts(observation_counts))

    return sample


def _infer_round_latent_from_saved_predictions(round_dir: Path) -> dict[str, float] | None:
    latents: list[dict[str, float]] = []
    for analysis_path in sorted(round_dir.glob("seed_*/analysis.json")):
        payload = load_json(analysis_path)
        prediction = payload.get("prediction")
        if prediction is None:
            continue
        latents.append(infer_latent_summary_from_prediction(np.asarray(prediction, dtype=np.float64)))

    if not latents:
        return None

    return {
        signal_name: float(np.mean([latent[signal_name] for latent in latents]))
        for signal_name in _FALLBACK_LATENT_SIGNALS
    }
