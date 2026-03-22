from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import PredictorConfig
from .features import BUCKET_KEY_VERSION, make_bucket_keys
from .learned_prior import LearnedPriorArtifact, predict_learned_prior
from .prior_blend_gate import PriorBlendGateArtifact, apply_prior_blend_gate
from .priors import HistoricalPriorArtifact
from .regime import compute_round_regime, inverse_range_signal, range_signal, repeat_fraction_from_observation_counts
from .residual_calibrator import ResidualCalibratorArtifact, apply_residual_calibrator
from .types import (
    CLASS_EMPTY,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
    CLASS_NAMES,
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
        learned_prior: LearnedPriorArtifact | None = None,
        residual_calibrator: ResidualCalibratorArtifact | None = None,
        prior_blend_gate: PriorBlendGateArtifact | None = None,
    ):
        self.config = config
        self.round_detail = round_detail
        self.seed_features = seed_features
        self.historical_priors = historical_priors
        self.learned_prior = learned_prior
        self.residual_calibrator = residual_calibrator
        self.prior_blend_gate = prior_blend_gate
        self._warned_historical_bucket_mismatch = False

    def predict_round(self, aggregator) -> dict[int, np.ndarray]:
        predictions, _ = self._predict_round_internal(aggregator, collect_diagnostics=False)
        return predictions

    def predict_round_with_diagnostics(self, aggregator) -> tuple[dict[int, np.ndarray], dict[int, dict[str, object]]]:
        return self._predict_round_internal(aggregator, collect_diagnostics=True)

    def _predict_round_internal(
        self,
        aggregator,
        *,
        collect_diagnostics: bool,
    ) -> tuple[dict[int, np.ndarray], dict[int, dict[str, object]]]:
        latent = aggregator.round_latent_summary()
        round_regime = self._compute_round_regime(latent, aggregator)
        predictions: dict[int, np.ndarray] = {}
        diagnostics: dict[int, dict[str, object]] = {}
        for seed_index, features in self.seed_features.items():
            prediction, trace = self._predict_seed_internal(
                seed_index,
                features,
                aggregator,
                latent,
                round_regime,
                collect_diagnostics=collect_diagnostics,
            )
            validate_prediction_tensor(prediction, self.config.min_probability)
            predictions[seed_index] = prediction
            if trace is not None:
                diagnostics[seed_index] = trace
        return predictions, diagnostics

    def predict_seed(self, seed_index: int, features: SeedFeatures, aggregator, latent: dict[str, float]) -> np.ndarray:
        round_regime = self._compute_round_regime(latent, aggregator)
        prediction, _ = self._predict_seed_internal(
            seed_index,
            features,
            aggregator,
            latent,
            round_regime,
            collect_diagnostics=False,
        )
        return prediction

    def _predict_seed_internal(
        self,
        seed_index: int,
        features: SeedFeatures,
        aggregator,
        latent: dict[str, float],
        round_regime: dict[str, float],
        *,
        collect_diagnostics: bool,
    ) -> tuple[np.ndarray, dict[str, object] | None]:
        prior = self._build_prior(seed_index, features, aggregator, latent)
        observed_counts = aggregator.class_counts[seed_index]
        observation_counts = aggregator.observation_counts[seed_index]
        transfer, observation_details = self._build_transfer(
            seed_index,
            features,
            aggregator,
            prior,
            observed_counts,
            observation_counts,
            round_regime,
        )
        combined = transfer.copy()
        base_prediction = transfer.copy()
        prediction = self._apply_settlement_intensity_prior(base_prediction, features)
        post_settlement = prediction.copy()
        prediction = self._apply_structural_calibration(prediction, features, round_regime)
        post_structural = prediction.copy()
        rare_details = None
        if collect_diagnostics:
            prediction, rare_details = self._apply_rare_class_concentration(
                prediction,
                features,
                return_details=True,
            )
        else:
            prediction = self._apply_rare_class_concentration(
                prediction,
                features,
                return_details=False,
            )
        post_rare = prediction.copy()
        prediction = self._apply_physical_constraints(prediction, features)
        post_physical = prediction.copy()
        prediction = self._calibrate_confidence(prediction, observation_counts)
        prediction = normalize_probabilities(prediction, self.config.min_probability)
        post_confidence = prediction.copy()
        residual_details = None
        if collect_diagnostics:
            prediction, residual_details = self._apply_residual_calibrator(
                prediction,
                features,
                observed_counts,
                observation_counts,
                round_regime,
                return_details=True,
            )
        else:
            prediction = self._apply_residual_calibrator(
                prediction,
                features,
                observed_counts,
                observation_counts,
                round_regime,
                return_details=False,
            )
        prediction = self._apply_boundary_softening(prediction, features)
        prior_gate_details = None
        if collect_diagnostics:
            prediction, prior_gate_details = self._apply_prior_blend_gate(
                prior,
                prediction,
                features,
                round_regime,
                return_details=True,
            )
        else:
            prediction = self._apply_prior_blend_gate(
                prior,
                prediction,
                features,
                round_regime,
                return_details=False,
            )
        post_prior_gate = prediction.copy()
        prediction = self._apply_high_activity_active_concentration(
            prediction,
            features,
            round_regime,
        )
        post_active_concentration = prediction.copy()
        prediction = self._apply_active_dominance_separation(
            prediction,
            prior,
            features,
            observation_details["tensors"]["active_dominance_support"],
            round_regime,
        )
        prediction = self._apply_physical_constraints(prediction, features)
        prediction = normalize_probabilities(prediction, self.config.min_probability)
        prediction = self._apply_ruin_dampening(prediction)
        prediction = self._apply_global_villager_recalibration(
            prediction,
            observed_counts,
            observation_counts,
            features,
        )

        if not collect_diagnostics:
            return prediction, None

        learned_prior = None
        if self.learned_prior is not None:
            learned_prior = predict_learned_prior(self.learned_prior, features, self.config.min_probability)
        diagnostics = {
            "seed_index": seed_index,
            "observation_summary": self._summarize_observations(observed_counts, observation_counts),
            "round_regime": round_regime,
            "stage_summaries": {
                "prior": self._summarize_tensor(prior),
                "transfer": self._summarize_tensor(transfer),
                "base_prediction": self._summarize_tensor(base_prediction),
                "post_settlement": self._summarize_tensor(post_settlement),
                "post_structural": self._summarize_tensor(post_structural),
                "post_rare": self._summarize_tensor(post_rare),
                "post_physical": self._summarize_tensor(post_physical),
                "post_confidence": self._summarize_tensor(post_confidence),
                "post_prior_gate": self._summarize_tensor(post_prior_gate),
                "post_active_concentration": self._summarize_tensor(post_active_concentration),
                "final_prediction": self._summarize_tensor(prediction),
            },
            "tensors": {
                "prior": prior,
                "transfer": transfer,
                "combined": combined,
                "base_prediction": base_prediction,
                "post_settlement": post_settlement,
                "post_structural": post_structural,
                "post_rare": post_rare,
                "post_physical": post_physical,
                "post_confidence": post_confidence,
                "post_prior_gate": post_prior_gate,
                "post_active_concentration": post_active_concentration,
                "final_prediction": prediction,
            },
            "observation_model": observation_details["summary"],
        }
        diagnostics["tensors"].update(observation_details["tensors"])
        diagnostics["stage_summaries"]["observation_target"] = self._summarize_tensor(
            observation_details["tensors"]["observation_target"]
        )
        if learned_prior is not None:
            diagnostics["stage_summaries"]["learned_prior"] = self._summarize_tensor(learned_prior)
            diagnostics["tensors"]["learned_prior"] = learned_prior
        if rare_details is not None:
            diagnostics["rare_class"] = rare_details["summary"]
            diagnostics["tensors"].update(rare_details["tensors"])
        if residual_details is not None:
            diagnostics["residual_calibrator"] = residual_details["summary"]
            diagnostics["tensors"].update(residual_details["tensors"])
        if prior_gate_details is not None:
            diagnostics["prior_blend_gate"] = prior_gate_details["summary"]
            diagnostics["tensors"].update(prior_gate_details["tensors"])
        return prediction, diagnostics

    def _apply_global_villager_recalibration(
        self,
        prediction: np.ndarray,
        observed_counts: np.ndarray,
        observation_counts: np.ndarray,
        features: SeedFeatures,
    ) -> np.ndarray:
        """Scale CLASS_SETTLEMENT mass to close the gap between the observed Villager
        fraction and the predicted Villager fraction on the same observed buildable cells.

        Using the relative gap on *observed* cells (obs_rate / pred_on_obs) rather than
        comparing obs_rate to a fixed historical baseline removes the query-planner bias:
        if the planner focuses on high-Villager areas, both numerator and denominator are
        inflated equally and the ratio stays correct.
        """
        strength = self.config.villager_recalib_strength
        if strength <= 0.0:
            return prediction

        buildable = features.buildable_mask
        observed_buildable = (observation_counts > 0) & buildable

        if int(np.sum(observed_buildable)) < self.config.villager_recalib_min_obs_cells:
            return prediction

        obs_settlement = float(np.sum(observed_counts[observed_buildable, CLASS_SETTLEMENT]))
        obs_total = float(np.sum(observed_counts[observed_buildable]))
        if obs_total <= 0.0:
            return prediction
        obs_rate = obs_settlement / obs_total

        pred_rate = float(np.mean(prediction[observed_buildable, CLASS_SETTLEMENT]))
        if pred_rate <= 1e-6:
            return prediction

        raw_scale = obs_rate / pred_rate
        clipped = float(np.clip(raw_scale, self.config.villager_recalib_scale_clip_lo,
                                self.config.villager_recalib_scale_clip_hi))
        scale = 1.0 + strength * (clipped - 1.0)

        out = prediction.copy()
        out[..., CLASS_SETTLEMENT][buildable] *= scale
        return normalize_probabilities(out, self.config.min_probability)

    def _apply_boundary_softening(self, prediction: np.ndarray, features: SeedFeatures) -> np.ndarray:
        alpha = self.config.boundary_softening_alpha
        if alpha <= 0.0:
            return prediction
        from scipy.ndimage import binary_dilation
        si_idx = features.feature_names.index("settlement_intensity")
        settlement_intensity = features.feature_stack[..., si_idx]
        is_forest = features.feature_stack[..., features.feature_names.index("initial_forest")].astype(bool)
        non_forest_buildable = features.buildable_mask & ~is_forest
        forest_boundary = is_forest & binary_dilation(non_forest_buildable, iterations=1)
        intermediate_si = (settlement_intensity > 0.3) & (settlement_intensity < 0.7)
        mask = (intermediate_si | forest_boundary | features.frontier_mask) & features.buildable_mask
        out = prediction.copy().astype(np.float64)
        out[mask] += alpha
        return normalize_probabilities(out, self.config.min_probability)

    def _apply_ruin_dampening(self, prediction: np.ndarray) -> np.ndarray:
        scale = self.config.ruin_dampening_scale
        if scale >= 1.0:
            return prediction
        out = prediction.copy().astype(np.float64)
        removed = out[..., CLASS_RUIN] * (1.0 - scale)
        out[..., CLASS_RUIN] -= removed
        out[..., CLASS_EMPTY] += removed
        return normalize_probabilities(out, self.config.min_probability)

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
        prior = self._apply_learned_prior(prior, features)
        return normalize_probabilities(prior, self.config.min_probability)

    def _build_transfer(
        self,
        seed_index: int,
        features: SeedFeatures,
        aggregator,
        prior: np.ndarray,
        observed_counts: np.ndarray,
        observation_counts: np.ndarray,
        round_regime: dict[str, float],
    ) -> tuple[np.ndarray, dict[str, object]]:
        bucket_keys = make_bucket_keys(features)
        h, w = bucket_keys.shape
        regime = round_regime["high_activity_factor"]
        empirical = np.zeros_like(observed_counts, dtype=np.float64)
        np.divide(
            observed_counts,
            np.maximum(observation_counts[..., None], 1e-12),
            out=empirical,
            where=observation_counts[..., None] > 0,
        )

        bucket_target = np.zeros((h, w, NUM_CLASSES), dtype=np.float64)
        global_counts = aggregator.class_counts.sum(axis=(0, 1, 2))
        global_probs = global_counts / max(float(np.sum(global_counts)), 1e-12)
        if float(np.sum(global_probs)) <= 0.0:
            global_probs = np.mean(prior, axis=(0, 1))
        bucket_target[:] = global_probs
        for key, counts in aggregator.conditional_counts.items():
            mask = bucket_keys == int(key)
            if np.any(mask):
                bucket_target[mask] = counts / max(float(np.sum(counts)), 1e-12)
        bucket_target = normalize_probabilities(bucket_target, self.config.min_probability)

        obs_weight = observation_counts / (observation_counts + self.config.observation_support_temperature)
        obs_weight *= 1.0 + self.config.observation_repeat_bonus * np.clip(observation_counts - 1.0, 0.0, None)
        smooth_support = gaussian_filter(
            obs_weight,
            sigma=self.config.observation_smoothing_sigma,
            mode="nearest",
        )
        correction_support = smooth_support / (smooth_support + self.config.observation_support_temperature)

        smooth_empirical = np.zeros_like(empirical, dtype=np.float64)
        for class_id in range(NUM_CLASSES):
            smooth_empirical[..., class_id] = self._weighted_gaussian_average(
                empirical[..., class_id],
                obs_weight,
                sigma=self.config.observation_smoothing_sigma,
            )

        observation_target = np.where(
            smooth_support[..., None] > 1e-6,
            smooth_empirical,
            bucket_target,
        )
        bucket_blend = self._interp(
            self.config.observation_bucket_blend,
            self.config.observation_bucket_blend_high_activity,
            regime,
        )
        observation_target = normalize_probabilities(
            (1.0 - bucket_blend) * observation_target
            + bucket_blend * bucket_target,
            self.config.min_probability,
        )

        prior_active_mass = prior[..., CLASS_SETTLEMENT] + prior[..., CLASS_PORT] + prior[..., CLASS_RUIN]
        target_active_mass = (
            observation_target[..., CLASS_SETTLEMENT]
            + observation_target[..., CLASS_PORT]
            + observation_target[..., CLASS_RUIN]
        )
        prior_active_types = self._normalize_active_types(prior)
        target_active_types = self._normalize_active_types(observation_target, fallback=prior_active_types)

        nonactive_blend = self._interp(
            self.config.observation_nonactive_blend,
            self.config.observation_nonactive_blend_high_activity,
            regime,
        )
        nonactive_gate = np.clip(
            nonactive_blend * correction_support,
            0.0,
            1.0,
        )
        active_mass_blend = self._interp(
            self.config.observation_active_mass_blend,
            self.config.observation_active_mass_blend_high_activity,
            regime,
        )
        active_mass_gate = np.clip(
            active_mass_blend * np.sqrt(np.maximum(correction_support, 0.0)),
            0.0,
            1.0,
        )
        active_type_blend = self._interp(
            self.config.observation_active_type_blend,
            self.config.observation_active_type_blend_high_activity,
            regime,
        )
        active_type_gate = np.clip(
            active_type_blend * correction_support,
            0.0,
            1.0,
        )

        transfer = prior.copy().astype(np.float64)
        for class_id in (CLASS_EMPTY, CLASS_FOREST, CLASS_MOUNTAIN):
            transfer[..., class_id] = (1.0 - nonactive_gate) * prior[..., class_id] + nonactive_gate * observation_target[..., class_id]
        active_types = (
            (1.0 - active_type_gate)[..., None] * prior_active_types
            + active_type_gate[..., None] * target_active_types
        )
        active_types = self._normalize_active_types(active_types, fallback=prior_active_types)
        transfer_active_mass = (1.0 - active_mass_gate) * prior_active_mass + active_mass_gate * target_active_mass
        transfer[..., CLASS_SETTLEMENT] = transfer_active_mass * active_types[..., 0]
        transfer[..., CLASS_PORT] = transfer_active_mass * active_types[..., 1]
        transfer[..., CLASS_RUIN] = transfer_active_mass * active_types[..., 2]
        transfer = normalize_probabilities(transfer, self.config.min_probability)

        active_empirical = empirical[..., CLASS_SETTLEMENT] + empirical[..., CLASS_PORT] + empirical[..., CLASS_RUIN]
        active_signal = self._weighted_gaussian_average(
            active_empirical,
            obs_weight,
            sigma=self.config.active_dominance_sigma,
        )
        settlement_intensity = features.feature_stack[..., features.feature_names.index("settlement_intensity")]
        port_intensity = features.feature_stack[..., features.feature_names.index("port_intensity")]
        active_signal = np.maximum(active_signal, 0.6 * settlement_intensity + 0.4 * port_intensity)
        active_signal = np.maximum(active_signal, features.initial_settlement_mask.astype(np.float64))
        active_dominance_support = np.clip(
            active_signal / max(self.config.active_dominance_signal_scale, 1e-6),
            0.0,
            1.0,
        )
        details = {
            "summary": {
                "correction_support_mean": float(np.mean(correction_support)),
                "correction_support_p95": float(np.quantile(correction_support, 0.95)),
                "bucket_blend": float(bucket_blend),
                "nonactive_blend": float(nonactive_blend),
                "active_mass_blend": float(active_mass_blend),
                "active_type_blend": float(active_type_blend),
                "active_dominance_support_mean": float(np.mean(active_dominance_support)),
                "active_dominance_support_p95": float(np.quantile(active_dominance_support, 0.95)),
            },
            "tensors": {
                "bucket_target": bucket_target,
                "observation_target": observation_target,
                "correction_support": correction_support,
                "active_dominance_support": active_dominance_support,
            },
        }
        return transfer, details

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
        bucket_version = self.historical_priors.metadata.get("bucket_key_version")
        if bucket_version == BUCKET_KEY_VERSION:
            for key, counts in self.historical_priors.bucket_counts.items():
                mask = bucket_keys == int(key)
                if np.any(mask):
                    counts = counts / max(float(np.sum(counts)), 1e-12)
                    prior[mask] += self.config.historical_prior_strength * counts
        elif not self._warned_historical_bucket_mismatch:
            LOGGER.warning(
                "Historical bucket priors use key version %s but predictor expects %s; "
                "skipping bucket-conditioned historical priors until the artifact is rebuilt.",
                bucket_version,
                BUCKET_KEY_VERSION,
            )
            self._warned_historical_bucket_mismatch = True
        for class_id_str, counts in self.historical_priors.initial_class_counts.items():
            class_id = int(class_id_str)
            mask = features.initial_class_grid == class_id
            if np.any(mask):
                counts = counts / max(float(np.sum(counts)), 1e-12)
                prior[mask] += 0.5 * self.config.historical_prior_strength * counts
        return prior

    def _apply_structural_calibration(
        self,
        prediction: np.ndarray,
        features: SeedFeatures,
        round_regime: dict[str, float],
    ) -> np.ndarray:
        out = prediction.copy().astype(np.float64)
        regime = round_regime["high_activity_factor"]
        quiet_regime = self._quiet_dominance(round_regime)
        settlement_intensity = features.feature_stack[..., features.feature_names.index("settlement_intensity")]
        port_intensity = features.feature_stack[..., features.feature_names.index("port_intensity")]
        frontier = features.frontier_mask.astype(np.float64)
        buildable_empty = features.buildable_mask & (features.initial_class_grid == CLASS_EMPTY)

        support = np.clip(
            self.config.plains_support_intensity_weight * settlement_intensity
            + (1.0 - self.config.plains_support_intensity_weight) * frontier,
            0.0,
            1.0,
        )
        settlement_factor = np.clip(
            self._interp(
                self.config.plains_settlement_base,
                self.config.plains_settlement_base_high_activity,
                regime,
            )
            + self._interp(
                self.config.plains_settlement_gain,
                self.config.plains_settlement_gain_high_activity,
                regime,
            )
            * np.power(support, self.config.plains_settlement_power),
            0.25,
            2.5,
        )
        out[..., CLASS_SETTLEMENT][buildable_empty] *= settlement_factor[buildable_empty]

        empty_factor = np.clip(
            self._interp(
                self.config.plains_empty_base,
                self.config.plains_empty_base_high_activity,
                regime,
            )
            - self._interp(
                self.config.plains_empty_support_slope,
                self.config.plains_empty_support_slope_high_activity,
                regime,
            )
            * support,
            0.8,
            1.2,
        )
        out[..., CLASS_EMPTY][buildable_empty] *= empty_factor[buildable_empty]

        coastal_empty = buildable_empty & features.coastal_mask
        port_factor = np.clip(
            0.8
            + self._interp(
                self.config.coastal_port_support_gain,
                self.config.coastal_port_support_gain_high_activity,
                regime,
            )
            * np.maximum(support, port_intensity),
            0.75,
            2.0,
        )
        out[..., CLASS_PORT][coastal_empty] *= port_factor[coastal_empty]

        initial_forest = features.initial_class_grid == CLASS_FOREST
        out[..., CLASS_FOREST][initial_forest] *= self._interp(
            self.config.forest_retention_boost,
            self.config.forest_retention_boost_high_activity,
            regime,
        )
        out[..., CLASS_EMPTY][initial_forest] *= self._interp(
            self.config.forest_empty_suppression,
            self.config.forest_empty_suppression_high_activity,
            regime,
        )
        out[..., CLASS_SETTLEMENT][initial_forest] *= self._interp(
            self.config.forest_settlement_suppression,
            self.config.forest_settlement_suppression_high_activity,
            regime,
        )
        out[..., CLASS_PORT][initial_forest] *= self._interp(
            self.config.forest_port_suppression,
            self.config.forest_port_suppression_high_activity,
            regime,
        )
        out[..., CLASS_RUIN][initial_forest] *= self._interp(
            self.config.forest_ruin_suppression,
            self.config.forest_ruin_suppression_high_activity,
            regime,
        )

        initial_settlement = features.initial_class_grid == CLASS_SETTLEMENT
        out[..., CLASS_SETTLEMENT][initial_settlement] *= self._interp(
            self.config.initial_settlement_boost,
            self.config.initial_settlement_boost_high_activity,
            regime,
        )
        out[..., CLASS_EMPTY][initial_settlement] *= self._interp(
            self.config.initial_settlement_empty_suppression,
            self.config.initial_settlement_empty_suppression_high_activity,
            regime,
        )

        initial_port = features.initial_class_grid == CLASS_PORT
        out[..., CLASS_PORT][initial_port] *= self._interp(
            self.config.initial_port_boost,
            self.config.initial_port_boost_high_activity,
            regime,
        )
        out[..., CLASS_EMPTY][initial_port] *= self._interp(
            self.config.initial_port_empty_suppression,
            self.config.initial_port_empty_suppression_high_activity,
            regime,
        )

        if quiet_regime > 0.0:
            out[..., CLASS_EMPTY][buildable_empty] *= self._interp(1.0, self.config.quiet_plains_empty_boost, quiet_regime)
            out[..., CLASS_SETTLEMENT][buildable_empty] *= self._interp(
                1.0,
                self.config.quiet_plains_settlement_suppression,
                quiet_regime,
            )
            out[..., CLASS_PORT][buildable_empty] *= self._interp(
                1.0,
                self.config.quiet_plains_port_suppression,
                quiet_regime,
            )
            out[..., CLASS_RUIN][buildable_empty] *= self._interp(
                1.0,
                self.config.quiet_plains_ruin_suppression,
                quiet_regime,
            )
            out[..., CLASS_FOREST][initial_forest] *= self._interp(
                1.0,
                self.config.quiet_forest_retention_boost,
                quiet_regime,
            )
            out[..., CLASS_EMPTY][initial_forest] *= self._interp(
                1.0,
                self.config.quiet_forest_empty_suppression,
                quiet_regime,
            )
            out[..., CLASS_SETTLEMENT][initial_forest] *= self._interp(
                1.0,
                self.config.quiet_forest_settlement_suppression,
                quiet_regime,
            )
            out[..., CLASS_PORT][initial_forest] *= self._interp(
                1.0,
                self.config.quiet_forest_port_suppression,
                quiet_regime,
            )
            out[..., CLASS_RUIN][initial_forest] *= self._interp(
                1.0,
                self.config.quiet_forest_ruin_suppression,
                quiet_regime,
            )

        return normalize_probabilities(out, self.config.min_probability)

    def _apply_learned_prior(self, prior: np.ndarray, features: SeedFeatures) -> np.ndarray:
        if self.learned_prior is None:
            return prior
        learned = predict_learned_prior(self.learned_prior, features, self.config.min_probability)
        blended = (1.0 - self.config.learned_prior_blend) * prior + self.config.learned_prior_blend * learned
        return normalize_probabilities(blended, self.config.min_probability)

    def _apply_rare_class_concentration(
        self,
        prediction: np.ndarray,
        features: SeedFeatures,
        *,
        return_details: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
        out = prediction.copy().astype(np.float64)
        settlement_intensity = features.feature_stack[..., features.feature_names.index("settlement_intensity")]
        settlement_density = features.feature_stack[..., features.feature_names.index("settlement_density")]
        forest_density = features.feature_stack[..., features.feature_names.index("forest_density")]
        border_distance = features.feature_stack[..., features.feature_names.index("border_distance")]

        border_scale = max(float(np.max(border_distance)), 1e-6)
        border_support = 1.0 - np.clip(border_distance / border_scale, 0.0, 1.0)
        active_mass = out[..., CLASS_SETTLEMENT] + out[..., CLASS_PORT]

        port_support = (
            self.config.port_support_intensity_weight * settlement_intensity
            + self.config.port_support_frontier_weight * features.frontier_mask.astype(np.float64)
            + self.config.port_support_border_weight * border_support
            + self.config.port_support_predicted_active_weight * active_mass
        )
        port_support += self.config.port_support_initial_settlement_bonus * (features.initial_class_grid == CLASS_SETTLEMENT)
        port_support += self.config.port_support_initial_port_bonus * (features.initial_class_grid == CLASS_PORT)
        port_mask = features.coastal_mask & features.buildable_mask
        port_rate = float(np.mean(out[..., CLASS_PORT][port_mask])) if np.any(port_mask) else 0.0
        port_gate = np.clip(
            (port_rate - self.config.port_focus_rate_floor)
            / max(self.config.port_focus_rate_ceiling - self.config.port_focus_rate_floor, 1e-6),
            0.0,
            1.0,
        )
        port_blend = self.config.port_focus_min_blend + (
            self.config.port_focus_blend - self.config.port_focus_min_blend
        ) * port_gate
        out = self._reshape_class_mass(
            out,
            CLASS_PORT,
            port_support,
            blend=port_blend,
            power=self.config.port_focus_power,
            support_mask=port_mask,
        )

        interior_support = (~features.coastal_mask).astype(np.float64)
        ruin_support = (
            self.config.ruin_support_intensity_weight * settlement_intensity
            + self.config.ruin_support_frontier_weight * features.frontier_mask.astype(np.float64)
            + self.config.ruin_support_conflict_weight * features.conflict_mask.astype(np.float64)
            + self.config.ruin_support_density_weight * settlement_density
            + self.config.ruin_support_interior_weight * interior_support
            + self.config.ruin_support_predicted_settlement_weight * out[..., CLASS_SETTLEMENT]
            + self.config.ruin_support_predicted_ruin_weight * out[..., CLASS_RUIN]
            + self.config.ruin_support_forest_weight * forest_density
        )
        ruin_mask = features.buildable_mask
        out = self._reshape_class_mass(
            out,
            CLASS_RUIN,
            ruin_support,
            blend=self.config.ruin_focus_blend,
            power=self.config.ruin_focus_power,
            support_mask=ruin_mask,
        )
        if not return_details:
            return out
        details = {
            "summary": {
                "port_rate": port_rate,
                "port_gate": float(port_gate),
                "port_blend": float(port_blend),
                "port_support_summary": self._summarize_support(port_support, port_mask),
                "ruin_blend": float(self.config.ruin_focus_blend),
                "ruin_support_summary": self._summarize_support(ruin_support, ruin_mask),
            },
            "tensors": {
                "rare_port_support": port_support,
                "rare_ruin_support": ruin_support,
            },
        }
        return out, details

    def _reshape_class_mass(
        self,
        prediction: np.ndarray,
        class_id: int,
        support: np.ndarray,
        *,
        blend: float,
        power: float,
        support_mask: np.ndarray,
    ) -> np.ndarray:
        if blend <= 0.0:
            return prediction
        total_mass = float(prediction[..., class_id].sum())
        if total_mass <= 0.0 or not np.any(support_mask):
            return prediction

        shaped = np.maximum(support, self.config.rare_class_min_support).astype(np.float64)
        shaped = np.where(support_mask, shaped, 0.0)
        if np.sum(shaped) <= 0.0:
            return prediction

        shaped = np.power(shaped, max(power, 1e-6))
        if np.sum(shaped) <= 0.0:
            return prediction
        shaped = shaped / np.sum(shaped) * total_mass

        out = prediction.copy().astype(np.float64)
        out[..., class_id] = (1.0 - blend) * out[..., class_id] + blend * shaped
        return normalize_probabilities(out, self.config.min_probability)

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
        nonactive_power = 1.0 + confidence * (self.config.confidence_sharpen_power - 1.0)
        active_power = 1.0 + confidence * (self.config.active_confidence_sharpen_power - 1.0)
        out = prediction.copy().astype(np.float64)
        for class_id in (CLASS_EMPTY, CLASS_FOREST, CLASS_MOUNTAIN):
            out[..., class_id] = np.power(out[..., class_id], nonactive_power)
        for class_id in (CLASS_SETTLEMENT, CLASS_PORT, CLASS_RUIN):
            out[..., class_id] = np.power(out[..., class_id], active_power)
        return out

    def _apply_residual_calibrator(
        self,
        prediction: np.ndarray,
        features: SeedFeatures,
        observed_counts: np.ndarray,
        observation_counts: np.ndarray,
        round_regime: dict[str, float],
        *,
        return_details: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, object] | None]:
        if self.residual_calibrator is None:
            if return_details:
                return prediction, None
            return prediction
        blend_map = self._residual_blend_map(observed_counts, observation_counts, round_regime)
        calibrated, details = apply_residual_calibrator(
            self.residual_calibrator,
            prediction,
            features,
            round_regime,
            blend=blend_map,
            min_probability=self.config.min_probability,
            active_observed_cap=self.config.residual_trust_gate_active_observed_cap,
            observation_counts=observation_counts,
            observed_counts=observed_counts,
        )
        if return_details:
            return calibrated, details
        return calibrated

    def _apply_prior_blend_gate(
        self,
        prior: np.ndarray,
        prediction: np.ndarray,
        features: SeedFeatures,
        round_regime: dict[str, float],
        *,
        return_details: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, object] | None]:
        effective_strength = self.config.prior_blend_gate_strength * round_regime["low_activity_factor"]
        if round_regime["high_activity_factor"] < 0.15:
            effective_strength *= self.config.quiet_prior_blend_gate_multiplier
        if self.prior_blend_gate is None or effective_strength <= 0.0:
            if return_details:
                return prediction, None
            return prediction
        blended, details = apply_prior_blend_gate(
            self.prior_blend_gate,
            prior,
            prediction,
            features,
            round_regime=round_regime,
            min_probability=self.config.min_probability,
            strength=effective_strength,
        )
        if return_details:
            return blended, details
        return blended

    def _apply_high_activity_active_concentration(
        self,
        prediction: np.ndarray,
        features: SeedFeatures,
        round_regime: dict[str, float],
    ) -> np.ndarray:
        regime = round_regime["high_activity_factor"]
        if regime <= 0.0:
            return prediction
        out = prediction.copy().astype(np.float64)
        settlement_intensity = features.feature_stack[..., features.feature_names.index("settlement_intensity")]
        settlement_density = features.feature_stack[..., features.feature_names.index("settlement_density")]
        frontier = features.frontier_mask.astype(np.float64)
        initial_settlement = (features.initial_class_grid == CLASS_SETTLEMENT).astype(np.float64)
        settlement_support = (
            self.config.settlement_support_intensity_weight * settlement_intensity
            + self.config.settlement_support_frontier_weight * frontier
            + self.config.settlement_support_density_weight * settlement_density
            + self.config.settlement_support_prediction_weight * out[..., CLASS_SETTLEMENT]
            + self.config.settlement_support_initial_bonus * initial_settlement
        )
        settlement_blend = np.clip(
            regime * self.config.settlement_focus_blend_high_activity,
            0.0,
            1.0,
        )
        out = self._reshape_class_mass(
            out,
            CLASS_SETTLEMENT,
            settlement_support,
            blend=settlement_blend,
            power=self.config.settlement_focus_power_high_activity,
            support_mask=features.buildable_mask,
        )
        border_distance = features.feature_stack[..., features.feature_names.index("border_distance")]
        border_scale = max(float(np.max(border_distance)), 1e-6)
        border_support = 1.0 - np.clip(border_distance / border_scale, 0.0, 1.0)
        active_mass = out[..., CLASS_SETTLEMENT] + out[..., CLASS_PORT]
        port_support = (
            self.config.post_port_support_intensity_weight * settlement_intensity
            + self.config.post_port_support_frontier_weight * frontier
            + self.config.post_port_support_border_weight * border_support
            + self.config.post_port_support_prediction_weight * active_mass
        )
        port_support += self.config.post_port_support_initial_settlement_bonus * (
            features.initial_class_grid == CLASS_SETTLEMENT
        )
        port_support += self.config.post_port_support_initial_port_bonus * (features.initial_class_grid == CLASS_PORT)
        port_blend = np.clip(
            regime * self.config.post_port_focus_blend_high_activity,
            0.0,
            1.0,
        )
        return self._reshape_class_mass(
            out,
            CLASS_PORT,
            port_support,
            blend=port_blend,
            power=self.config.post_port_focus_power_high_activity,
            support_mask=features.coastal_mask & features.buildable_mask,
        )

    def _residual_blend_map(
        self,
        observed_counts: np.ndarray,
        observation_counts: np.ndarray,
        round_regime: dict[str, float],
    ) -> np.ndarray:
        base_blend = self._regime_endpoint_interp(
            self.config.residual_calibrator_low_activity_blend,
            self.config.residual_calibrator_blend,
            self.config.residual_calibrator_high_activity_blend,
            round_regime,
        )
        single_observed_blend = self._regime_endpoint_interp(
            self.config.residual_calibrator_low_activity_single_observed_blend,
            self.config.residual_calibrator_single_observed_blend,
            self.config.residual_calibrator_high_activity_single_observed_blend,
            round_regime,
        )
        repeated_observed_blend = self._regime_endpoint_interp(
            self.config.residual_calibrator_low_activity_repeated_observed_blend,
            self.config.residual_calibrator_repeated_observed_blend,
            self.config.residual_calibrator_high_activity_repeated_observed_blend,
            round_regime,
        )
        active_observed_blend = self._regime_endpoint_interp(
            self.config.residual_calibrator_low_activity_active_observed_blend,
            self.config.residual_calibrator_active_observed_blend,
            self.config.residual_calibrator_high_activity_active_observed_blend,
            round_regime,
        )
        blend_map = np.full(observation_counts.shape, base_blend, dtype=np.float64)
        blend_map[observation_counts == 1] = single_observed_blend
        blend_map[observation_counts >= 2] = repeated_observed_blend
        active_observed = (
            observed_counts[..., CLASS_SETTLEMENT]
            + observed_counts[..., CLASS_PORT]
            + observed_counts[..., CLASS_RUIN]
        ) > 0
        blend_map[active_observed] = active_observed_blend
        return blend_map

    def _apply_active_dominance_separation(
        self,
        prediction: np.ndarray,
        prior: np.ndarray,
        features: SeedFeatures,
        dominance_support: np.ndarray,
        round_regime: dict[str, float],
    ) -> np.ndarray:
        out = prediction.copy().astype(np.float64)
        active_mass = out[..., CLASS_SETTLEMENT] + out[..., CLASS_PORT] + out[..., CLASS_RUIN]
        nonactive_reference = np.maximum(out[..., CLASS_EMPTY], out[..., CLASS_FOREST])
        regime_relax = 1.0 + round_regime["high_activity_factor"] * self.config.active_dominance_regime_relaxation
        allowed_active = self.config.active_dominance_additive + nonactive_reference * (
            self.config.active_dominance_base_ratio
            + self.config.active_dominance_support_gain * dominance_support
        ) * regime_relax
        initial_active_like = np.isin(features.initial_class_grid, [CLASS_SETTLEMENT, CLASS_PORT, CLASS_RUIN])
        allowed_active[initial_active_like] += (
            self.config.active_dominance_initial_bonus
            + round_regime["high_activity_factor"] * self.config.active_dominance_regime_initial_bonus
        )
        allowed_active = np.clip(allowed_active, 0.0, 0.98)

        scale = np.ones_like(active_mass, dtype=np.float64)
        over_mask = active_mass > allowed_active
        scale[over_mask] = allowed_active[over_mask] / np.maximum(active_mass[over_mask], 1e-12)

        removed_mass = active_mass * (1.0 - scale)
        out[..., CLASS_SETTLEMENT] *= scale
        out[..., CLASS_PORT] *= scale
        out[..., CLASS_RUIN] *= scale

        redistribute_base = out[..., CLASS_EMPTY] + out[..., CLASS_FOREST]
        fallback_base = prior[..., CLASS_EMPTY] + prior[..., CLASS_FOREST]
        empty_share = np.zeros_like(redistribute_base, dtype=np.float64)
        np.divide(
            out[..., CLASS_EMPTY],
            np.maximum(redistribute_base, 1e-12),
            out=empty_share,
            where=redistribute_base > 1e-12,
        )
        fallback_empty_share = np.zeros_like(fallback_base, dtype=np.float64)
        np.divide(
            prior[..., CLASS_EMPTY],
            np.maximum(fallback_base, 1e-12),
            out=fallback_empty_share,
            where=fallback_base > 1e-12,
        )
        empty_share = np.where(redistribute_base > 1e-12, empty_share, fallback_empty_share)
        out[..., CLASS_EMPTY] += removed_mass * empty_share
        out[..., CLASS_FOREST] += removed_mass * (1.0 - empty_share)
        out = self._apply_settlement_dominance_allowance(out, features, round_regime)
        out = self._apply_port_dominance_allowance(out, features, round_regime)
        return out

    def _apply_settlement_dominance_allowance(
        self,
        prediction: np.ndarray,
        features: SeedFeatures,
        round_regime: dict[str, float],
    ) -> np.ndarray:
        regime = round_regime["high_activity_factor"]
        if regime <= 0.0 or self.config.active_dominance_settlement_bonus_high_activity <= 0.0:
            return prediction

        out = prediction.copy().astype(np.float64)
        settlement_intensity = features.feature_stack[..., features.feature_names.index("settlement_intensity")]
        settlement_density = features.feature_stack[..., features.feature_names.index("settlement_density")]
        frontier = features.frontier_mask.astype(np.float64)
        settlement_support = (
            self.config.settlement_support_intensity_weight * settlement_intensity
            + self.config.settlement_support_frontier_weight * frontier
            + self.config.settlement_support_density_weight * settlement_density
            + self.config.settlement_support_prediction_weight * out[..., CLASS_SETTLEMENT]
            + self.config.settlement_support_initial_bonus * (features.initial_class_grid == CLASS_SETTLEMENT)
        )
        settlement_support = np.clip(settlement_support, 0.0, 1.0)

        competitor_stack = np.stack(
            [
                out[..., CLASS_EMPTY],
                out[..., CLASS_PORT],
                out[..., CLASS_RUIN],
                out[..., CLASS_FOREST],
            ],
            axis=-1,
        )
        competitor_index = np.argmax(competitor_stack, axis=-1)
        competitor = np.take_along_axis(competitor_stack, competitor_index[..., None], axis=-1)[..., 0]
        deficit = competitor - out[..., CLASS_SETTLEMENT]
        candidate_mask = (
            features.buildable_mask
            & (out[..., CLASS_SETTLEMENT] >= self.config.active_dominance_settlement_min_probability)
            & (deficit > 0.0)
            & (deficit < self.config.active_dominance_settlement_margin_threshold)
        )
        max_shift = regime * self.config.active_dominance_settlement_bonus_high_activity * settlement_support
        desired_shift = 0.5 * deficit + 1e-4
        delta = np.where(candidate_mask, np.minimum(desired_shift, max_shift), 0.0)
        if not np.any(delta > 0.0):
            return prediction

        out[..., CLASS_SETTLEMENT] += delta
        for class_index, class_id in enumerate((CLASS_EMPTY, CLASS_PORT, CLASS_RUIN, CLASS_FOREST)):
            class_mask = candidate_mask & (competitor_index == class_index)
            out[..., class_id][class_mask] -= delta[class_mask]
        return out

    def _apply_port_dominance_allowance(
        self,
        prediction: np.ndarray,
        features: SeedFeatures,
        round_regime: dict[str, float],
    ) -> np.ndarray:
        regime = round_regime["high_activity_factor"]
        if regime <= 0.0 or self.config.active_dominance_port_bonus_high_activity <= 0.0:
            return prediction

        out = prediction.copy().astype(np.float64)
        settlement_intensity = features.feature_stack[..., features.feature_names.index("settlement_intensity")]
        border_distance = features.feature_stack[..., features.feature_names.index("border_distance")]
        frontier = features.frontier_mask.astype(np.float64)
        border_scale = max(float(np.max(border_distance)), 1e-6)
        border_support = 1.0 - np.clip(border_distance / border_scale, 0.0, 1.0)
        active_mass = out[..., CLASS_SETTLEMENT] + out[..., CLASS_PORT]
        port_support = (
            self.config.post_port_support_intensity_weight * settlement_intensity
            + self.config.post_port_support_frontier_weight * frontier
            + self.config.post_port_support_border_weight * border_support
            + self.config.post_port_support_prediction_weight * active_mass
        )
        port_support += self.config.post_port_support_initial_settlement_bonus * (
            features.initial_class_grid == CLASS_SETTLEMENT
        )
        port_support += self.config.post_port_support_initial_port_bonus * (features.initial_class_grid == CLASS_PORT)
        port_support = np.clip(port_support, 0.0, 1.0)

        competitor_stack = np.stack(
            [
                out[..., CLASS_EMPTY],
                out[..., CLASS_SETTLEMENT],
                out[..., CLASS_RUIN],
                out[..., CLASS_FOREST],
            ],
            axis=-1,
        )
        competitor_index = np.argmax(competitor_stack, axis=-1)
        competitor = np.take_along_axis(competitor_stack, competitor_index[..., None], axis=-1)[..., 0]
        deficit = competitor - out[..., CLASS_PORT]
        candidate_mask = (
            features.coastal_mask
            & features.buildable_mask
            & (out[..., CLASS_PORT] >= self.config.active_dominance_port_min_probability)
            & (deficit > 0.0)
            & (deficit < self.config.active_dominance_port_margin_threshold)
        )
        max_shift = regime * self.config.active_dominance_port_bonus_high_activity * port_support
        desired_shift = 0.5 * deficit + 1e-4
        delta = np.where(candidate_mask, np.minimum(desired_shift, max_shift), 0.0)
        if not np.any(delta > 0.0):
            return prediction

        out[..., CLASS_PORT] += delta
        for class_index, class_id in enumerate((CLASS_EMPTY, CLASS_SETTLEMENT, CLASS_RUIN, CLASS_FOREST)):
            class_mask = candidate_mask & (competitor_index == class_index)
            out[..., class_id][class_mask] -= delta[class_mask]
        return out

    def _weighted_gaussian_average(self, values: np.ndarray, weights: np.ndarray, *, sigma: float) -> np.ndarray:
        if sigma <= 0.0:
            out = np.zeros_like(values, dtype=np.float64)
            np.divide(values * weights, np.maximum(weights, 1e-12), out=out, where=weights > 1e-12)
            return out
        numerator = gaussian_filter(values * weights, sigma=sigma, mode="nearest")
        denominator = gaussian_filter(weights, sigma=sigma, mode="nearest")
        out = np.zeros_like(values, dtype=np.float64)
        np.divide(numerator, np.maximum(denominator, 1e-12), out=out, where=denominator > 1e-12)
        return out

    def _normalize_active_types(self, tensor: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
        if tensor.shape[-1] == NUM_CLASSES:
            active = tensor[..., [CLASS_SETTLEMENT, CLASS_PORT, CLASS_RUIN]].astype(np.float64)
        elif tensor.shape[-1] == 3:
            active = tensor.astype(np.float64)
        else:
            raise ValueError(f"Expected active-type tensor with 3 or {NUM_CLASSES} channels, got {tensor.shape}")
        totals = active.sum(axis=-1, keepdims=True)
        out = np.zeros_like(active, dtype=np.float64)
        np.divide(active, np.maximum(totals, 1e-12), out=out, where=totals > 1e-12)
        if fallback is None:
            fallback = np.zeros_like(out)
            fallback[..., 0] = 1.0
        return np.where(totals > 1e-12, out, fallback)

    def _summarize_tensor(self, tensor: np.ndarray) -> dict[str, object]:
        probs = normalize_probabilities(tensor, self.config.min_probability)
        entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=-1)
        argmax = np.argmax(probs, axis=-1)
        return {
            "class_mass": {name: float(probs[..., idx].sum()) for idx, name in enumerate(CLASS_NAMES)},
            "argmax_count": {name: int(np.sum(argmax == idx)) for idx, name in enumerate(CLASS_NAMES)},
            "mean_entropy": float(np.mean(entropy)),
            "max_entropy": float(np.max(entropy)),
            "min_probability": float(np.min(probs)),
        }

    def _summarize_support(self, support: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            return {"count": 0.0, "mean": 0.0, "max": 0.0, "p95": 0.0}
        values = support[mask]
        return {
            "count": float(values.size),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
            "p95": float(np.quantile(values, 0.95)),
        }

    def _summarize_observations(self, observed_counts: np.ndarray, observation_counts: np.ndarray) -> dict[str, object]:
        observed_mask = observation_counts > 0
        covered_fraction = float(np.mean(observed_mask))
        total_observations = float(np.sum(observation_counts))
        per_class = {name: float(observed_counts[..., idx].sum()) for idx, name in enumerate(CLASS_NAMES)}
        return {
            "covered_fraction": covered_fraction,
            "total_observations": total_observations,
            "repeat_cells": int(np.sum(observation_counts >= 2)),
            "per_class_observed_mass": per_class,
        }

    def _compute_round_regime(self, latent: dict[str, float], aggregator=None) -> dict[str, float]:
        if aggregator is None:
            repeat_fraction = 0.0
        else:
            repeat_fraction = repeat_fraction_from_observation_counts(aggregator.observation_counts)
        return compute_round_regime(latent, self.config, repeat_fraction=repeat_fraction)

    @staticmethod
    def _quiet_dominance(round_regime: dict[str, float]) -> float:
        return float(
            np.clip(
                round_regime["low_activity_factor"] - round_regime["high_activity_factor"],
                0.0,
                1.0,
            )
        )

    @classmethod
    def _regime_endpoint_interp(
        cls,
        quiet_value: float,
        base_value: float,
        high_value: float,
        round_regime: dict[str, float],
    ) -> float:
        high = round_regime["high_activity_factor"]
        quiet = round_regime["low_activity_factor"]
        if quiet > high:
            return cls._interp(base_value, quiet_value, quiet)
        return cls._interp(base_value, high_value, high)

    @staticmethod
    def _range_signal(value: float, low: float, high: float) -> float:
        return range_signal(value, low, high)

    @classmethod
    def _inverse_range_signal(cls, value: float, low: float, high: float) -> float:
        return inverse_range_signal(value, low, high)

    @staticmethod
    def _interp(low: float, high: float, t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        return float((1.0 - t) * low + t * high)
