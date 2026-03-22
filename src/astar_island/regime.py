from __future__ import annotations

import numpy as np

from .config import PredictorConfig
from .types import CLASS_FOREST, CLASS_PORT, CLASS_RUIN, CLASS_SETTLEMENT


def range_signal(value: float, low: float, high: float) -> float:
    if high < low:
        low, high = high, low
    if high - low <= 1e-12:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def inverse_range_signal(value: float, low: float, high: float) -> float:
    return float(1.0 - range_signal(value, low, high))


def repeat_fraction_from_observation_counts(observation_counts: np.ndarray) -> float:
    observed_cells = float(np.sum(observation_counts > 0))
    return float(np.sum(observation_counts >= 2)) / max(observed_cells, 1.0)


def compute_round_regime(
    latent: dict[str, float],
    config: PredictorConfig,
    *,
    repeat_fraction: float = 0.0,
) -> dict[str, float]:
    settlement_signal = range_signal(
        latent["settlement_rate"],
        config.regime_settlement_rate_low,
        config.regime_settlement_rate_high,
    )
    forest_signal = inverse_range_signal(
        latent["forest_share_dynamic"],
        config.regime_forest_share_low,
        config.regime_forest_share_high,
    )
    repeat_signal = range_signal(
        repeat_fraction,
        config.regime_repeat_fraction_low,
        config.regime_repeat_fraction_high,
    )
    total_weight = (
        config.regime_settlement_weight
        + config.regime_forest_weight
        + config.regime_repeat_weight
    )
    high_activity_factor = (
        config.regime_settlement_weight * settlement_signal
        + config.regime_forest_weight * forest_signal
        + config.regime_repeat_weight * repeat_signal
    ) / max(total_weight, 1e-12)
    high_activity_factor = float(np.clip(high_activity_factor, 0.0, 1.0))
    return {
        "settlement_signal": float(settlement_signal),
        "forest_signal": float(forest_signal),
        "repeat_signal": float(repeat_signal),
        "repeat_fraction": float(repeat_fraction),
        "high_activity_factor": high_activity_factor,
        "low_activity_factor": float(1.0 - high_activity_factor),
    }


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


def regime_bucket(high_activity_factor: float) -> str:
    if high_activity_factor < 0.15:
        return "quiet"
    if high_activity_factor < 0.60:
        return "mixed"
    return "active"
