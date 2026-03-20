from __future__ import annotations

import logging

import numpy as np

from .config import PredictorConfig
from .features import make_bucket_keys
from .priors import HistoricalPriorArtifact
from .types import (
    CLASS_EMPTY,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_SETTLEMENT,
    NUM_CLASSES,
    RoundDetail,
    SeedFeatures,
    TERRAIN_FOREST,
    TERRAIN_MOUNTAIN,
    TERRAIN_OCEAN,
    TERRAIN_PORT,
    TERRAIN_PLAINS,
    TERRAIN_RUIN,
    TERRAIN_SETTLEMENT,
)
from .utils import normalize_probabilities, validate_prediction_tensor

LOGGER = logging.getLogger(__name__)


class Predictor:
    def __init__(
        self,
        config: PredictorConfig,
        round_detail: RoundDetail,
        seed_features: dict[int, SeedFeatures],
        historical_priors: HistoricalPriorArtifact | None = None,
    ):
        self.config = config
        self.round_detail = round_detail
        self.seed_features = seed_features
        self.historical_priors = historical_priors

    def predict_round(self, aggregator) -> dict[int, np.ndarray]:
        latent = aggregator.round_latent_summary()
        predictions: dict[int, np.ndarray] = {}
        for seed_index, features in self.seed_features.items():
            prediction = self.predict_seed(seed_index, features, aggregator, latent)
            validate_prediction_tensor(prediction, self.config.min_probability)
            predictions[seed_index] = prediction
        return predictions

    def predict_seed(self, seed_index: int, features: SeedFeatures, aggregator, latent: dict[str, float]) -> np.ndarray:
        prior = self._build_prior(seed_index, features, aggregator, latent)
        transfer = self._build_transfer(seed_index, features, aggregator)
        observed_counts = aggregator.class_counts[seed_index]
        observation_counts = aggregator.observation_counts[seed_index][..., None]
        alpha_prior = self.config.prior_strength_dynamic
        alpha_direct = self.config.direct_observation_strength
        combined = alpha_prior * prior + self.config.transfer_strength * transfer + alpha_direct * observed_counts
        combined = np.where(observation_counts > 0, combined + observed_counts, combined)
        prediction = normalize_probabilities(combined, self.config.min_probability)
        prediction = self._apply_settlement_intensity_prior(prediction, features)
        prediction = self._apply_physical_constraints(prediction, features)
        prediction = self._calibrate_confidence(prediction, observation_counts)
        prediction = normalize_probabilities(prediction, self.config.min_probability)
        return prediction

    def _build_prior(self, seed_index: int, features: SeedFeatures, aggregator, latent: dict[str, float]) -> np.ndarray:
        grid = self.round_detail.initial_states[seed_index].grid
        h, w = grid.shape
        prior = np.full((h, w, NUM_CLASSES), 0.02, dtype=np.float64)
        exp_settle = np.exp(-4.5 * features.feature_stack[..., features.feature_names.index("dist_to_settlement")])
        exp_coast = np.exp(-5.0 * features.feature_stack[..., features.feature_names.index("dist_to_coast")])
        frontier = features.frontier_mask.astype(np.float64)
        conflict = features.conflict_mask.astype(np.float64)
        reclaimable = features.reclaimable_mask.astype(np.float64)
        settlement_density = features.feature_stack[..., features.feature_names.index("settlement_density")]
        forest_density = features.feature_stack[..., features.feature_names.index("forest_density")]
        port_share = latent["port_share_given_active"]
        ruin_share = latent["ruin_share_given_active"]
        forest_share = max(latent["forest_share_dynamic"], 0.15)

        prior[..., CLASS_EMPTY] += 0.55 + 0.25 * (1.0 - frontier)
        prior[..., CLASS_SETTLEMENT] += 0.18 * exp_settle + 0.20 * frontier + 0.10 * settlement_density
        prior[..., CLASS_PORT] += features.coastal_mask.astype(np.float64) * (0.08 * exp_settle + 0.18 * exp_coast + 0.22 * port_share)
        prior[..., CLASS_RUIN] += 0.03 + 0.12 * conflict + 0.15 * reclaimable + 0.10 * ruin_share
        prior[..., CLASS_FOREST] += 0.05 + 0.35 * forest_density + 0.18 * forest_share

        prior[grid == TERRAIN_FOREST] += np.array([0.05, 0.02, 0.01, 0.04, 0.85, 0.0])
        prior[grid == TERRAIN_SETTLEMENT] += np.array([0.05, 0.65, 0.10, 0.22, 0.03, 0.0])
        prior[grid == TERRAIN_PORT] += np.array([0.05, 0.18, 0.60, 0.17, 0.02, 0.0])
        prior[grid == TERRAIN_RUIN] += np.array([0.15, 0.18, 0.08, 0.42, 0.17, 0.0])
        prior[grid == TERRAIN_PLAINS] += np.array([0.40, 0.06, 0.02, 0.04, 0.05, 0.0])
        prior[grid == TERRAIN_OCEAN] += np.array([0.95, 0.01, 0.01, 0.01, 0.01, 0.0])
        prior[grid == TERRAIN_MOUNTAIN] += np.array([0.01, 0.0, 0.0, 0.0, 0.0, 1.0])
        prior = self._apply_historical_priors(prior, features)
        return normalize_probabilities(prior, self.config.min_probability)

    def _build_transfer(self, seed_index: int, features: SeedFeatures, aggregator) -> np.ndarray:
        bucket_keys = make_bucket_keys(features)
        h, w = bucket_keys.shape
        transfer = np.full((h, w, NUM_CLASSES), 1.0, dtype=np.float64)
        global_counts = aggregator.class_counts.sum(axis=0)
        local_counts = aggregator.class_counts[seed_index]
        obs_counts = aggregator.observation_counts[seed_index]
        ys_obs, xs_obs = np.where(obs_counts > 0)
        if len(ys_obs) > 0:
            ys, xs = np.indices((h, w))
            for oy, ox in zip(ys_obs, xs_obs):
                dist2 = (ys - oy) ** 2 + (xs - ox) ** 2
                weight = np.exp(-dist2 / max(2 * self.config.local_kernel_sigma ** 2, 1e-6))
                transfer += weight[..., None] * local_counts[oy, ox]
        for key, counts in aggregator.conditional_counts.items():
            transfer[bucket_keys == int(key)] += counts
        transfer += 0.5 * global_counts
        return normalize_probabilities(transfer, self.config.min_probability)

    def _apply_settlement_intensity_prior(self, prediction: np.ndarray, features: SeedFeatures) -> np.ndarray:
        intensity = features.feature_stack[..., features.feature_names.index("settlement_intensity")]
        target_mass = prediction[..., CLASS_SETTLEMENT].sum()
        shaped = intensity.copy()
        if shaped.sum() > 0:
            shaped = shaped / shaped.sum() * target_mass
            blended = (
                (1.0 - self.config.settlement_intensity_blend) * prediction[..., CLASS_SETTLEMENT]
                + self.config.settlement_intensity_blend * shaped
            )
            prediction[..., CLASS_SETTLEMENT] = blended
        return prediction

    def _apply_historical_priors(self, prior: np.ndarray, features: SeedFeatures) -> np.ndarray:
        if self.historical_priors is None:
            return prior
        bucket_keys = make_bucket_keys(features)
        for key, counts in self.historical_priors.bucket_counts.items():
            mask = bucket_keys == int(key)
            if np.any(mask):
                counts = counts / max(float(np.sum(counts)), 1e-12)
                prior[mask] += self.config.historical_prior_strength * counts
        for class_id_str, counts in self.historical_priors.initial_class_counts.items():
            class_id = int(class_id_str)
            mask = features.initial_class_grid == class_id
            if np.any(mask):
                counts = counts / max(float(np.sum(counts)), 1e-12)
                prior[mask] += 0.5 * self.config.historical_prior_strength * counts
        return prior

    def _apply_physical_constraints(self, prediction: np.ndarray, features: SeedFeatures) -> np.ndarray:
        mountain_mask = features.initial_class_grid == CLASS_MOUNTAIN
        non_mountain = ~mountain_mask
        prediction[mountain_mask] = 0.0
        prediction[..., CLASS_MOUNTAIN][mountain_mask] = 1.0
        prediction[..., CLASS_MOUNTAIN][non_mountain] *= 0.05
        inland_port_mask = (~features.coastal_mask) | mountain_mask
        prediction[..., CLASS_PORT][inland_port_mask] *= 0.1
        ocean_mask = ~features.buildable_mask
        prediction[..., CLASS_SETTLEMENT][ocean_mask] *= 0.05
        prediction[..., CLASS_RUIN][ocean_mask] *= 0.05
        prediction[..., CLASS_FOREST][ocean_mask] *= 0.05
        return prediction

    def _calibrate_confidence(self, prediction: np.ndarray, observation_counts: np.ndarray) -> np.ndarray:
        confidence = np.clip(observation_counts / (observation_counts + 2.0), 0.0, 1.0)
        power = 1.0 + confidence * (self.config.confidence_sharpen_power - 1.0)
        prediction = np.power(prediction, power)
        return prediction
