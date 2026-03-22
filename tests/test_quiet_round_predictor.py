from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from astar_island.config import PredictorConfig
from astar_island.features import build_seed_features
from astar_island.prior_blend_gate import PriorBlendGateArtifact
from astar_island.predictor import Predictor
from astar_island.residual_calibrator import (
    ResidualCalibratorArtifact,
    build_active_budget_features,
    build_collapsed_active_features,
)
from astar_island.types import (
    CLASS_EMPTY,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_SETTLEMENT,
    InitialState,
    RoundDetail,
    Settlement,
    SeedFeatures,
    TERRAIN_EMPTY,
    TERRAIN_FOREST,
    TERRAIN_OCEAN,
    TERRAIN_PLAINS,
    TERRAIN_SETTLEMENT,
)


class _ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.value, dtype=np.float64)


def _make_predictor() -> Predictor:
    detail = RoundDetail(
        round_id="test",
        round_number=0,
        status="completed",
        map_width=2,
        map_height=2,
        seeds_count=1,
        initial_states=[],
    )
    return Predictor(PredictorConfig(), detail, {})


def _make_features() -> SeedFeatures:
    feature_names = [
        "settlement_intensity",
        "port_intensity",
        "settlement_density",
        "border_distance",
    ]
    feature_stack = np.zeros((2, 2, len(feature_names)), dtype=np.float64)
    feature_stack[0, 0, feature_names.index("settlement_intensity")] = 0.8
    feature_stack[0, 0, feature_names.index("port_intensity")] = 0.6
    feature_stack[0, 0, feature_names.index("settlement_density")] = 0.3
    feature_stack[0, 0, feature_names.index("border_distance")] = 0.1
    feature_stack[0, 1, feature_names.index("settlement_intensity")] = 0.2
    feature_stack[0, 1, feature_names.index("border_distance")] = 0.5
    feature_stack[1, :, feature_names.index("border_distance")] = 0.8
    return SeedFeatures(
        seed_index=0,
        feature_stack=feature_stack,
        feature_names=feature_names,
        buildable_mask=np.array([[True, True], [True, True]], dtype=bool),
        dynamic_prior_mask=np.array([[True, True], [True, True]], dtype=bool),
        coastal_mask=np.array([[True, False], [False, False]], dtype=bool),
        frontier_mask=np.array([[True, False], [False, False]], dtype=bool),
        conflict_mask=np.zeros((2, 2), dtype=bool),
        reclaimable_mask=np.zeros((2, 2), dtype=bool),
        initial_class_grid=np.array(
            [
                [CLASS_EMPTY, CLASS_FOREST],
                [CLASS_EMPTY, CLASS_EMPTY],
            ],
            dtype=np.int64,
        ),
        initial_settlement_mask=np.zeros((2, 2), dtype=bool),
    )


class QuietRoundPredictorTests(unittest.TestCase):
    def test_compute_round_regime_detects_quiet_and_active_rounds(self) -> None:
        predictor = _make_predictor()
        quiet = predictor._compute_round_regime(
            {
                "observed_cells": 6638.0,
                "settlement_rate": 0.09368191721132897,
                "port_share_given_active": 0.045454545454545456,
                "ruin_share_given_active": 0.08244422890397672,
                "forest_share_dynamic": 0.6847094801223241,
                "mean_food": 0.0,
                "mean_wealth": 0.0,
                "mean_defense": 0.0,
                "mean_population": 0.0,
            }
        )
        active = predictor._compute_round_regime(
            {
                "observed_cells": 6520.0,
                "settlement_rate": 0.27510398098633393,
                "port_share_given_active": 0.042116630669546434,
                "ruin_share_given_active": 0.10184287099903007,
                "forest_share_dynamic": 0.3586979058677172,
                "mean_food": 0.0,
                "mean_wealth": 0.0,
                "mean_defense": 0.0,
                "mean_population": 0.0,
            }
        )
        self.assertEqual(quiet["high_activity_factor"], 0.0)
        self.assertEqual(quiet["low_activity_factor"], 1.0)
        self.assertGreater(active["high_activity_factor"], 0.5)
        self.assertLess(active["low_activity_factor"], 0.5)

    def test_structural_calibration_pushes_quiet_plains_and_forest_in_expected_direction(self) -> None:
        predictor = _make_predictor()
        features = _make_features()
        prediction = np.array(
            [
                [
                    [0.55, 0.20, 0.08, 0.07, 0.08, 0.02],
                    [0.10, 0.12, 0.05, 0.05, 0.66, 0.02],
                ],
                [
                    [0.70, 0.12, 0.04, 0.04, 0.08, 0.02],
                    [0.70, 0.12, 0.04, 0.04, 0.08, 0.02],
                ],
            ],
            dtype=np.float64,
        )
        quiet_regime = {
            "settlement_signal": 0.0,
            "forest_signal": 0.0,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.0,
            "low_activity_factor": 1.0,
        }

        calibrated = predictor._apply_structural_calibration(prediction, features, quiet_regime)

        self.assertGreater(calibrated[0, 0, CLASS_EMPTY], prediction[0, 0, CLASS_EMPTY])
        self.assertLess(calibrated[0, 0, CLASS_SETTLEMENT], prediction[0, 0, CLASS_SETTLEMENT])
        self.assertLess(calibrated[0, 0, CLASS_PORT], prediction[0, 0, CLASS_PORT])
        self.assertLess(calibrated[0, 0, CLASS_RUIN], prediction[0, 0, CLASS_RUIN])

        self.assertGreater(calibrated[0, 1, CLASS_FOREST], prediction[0, 1, CLASS_FOREST])
        self.assertLess(calibrated[0, 1, CLASS_EMPTY], prediction[0, 1, CLASS_EMPTY])
        self.assertLess(calibrated[0, 1, CLASS_SETTLEMENT], prediction[0, 1, CLASS_SETTLEMENT])
        self.assertLess(calibrated[0, 1, CLASS_PORT], prediction[0, 1, CLASS_PORT])
        self.assertLess(calibrated[0, 1, CLASS_RUIN], prediction[0, 1, CLASS_RUIN])

    def test_residual_blend_map_quiet_round_is_less_aggressive(self) -> None:
        predictor = _make_predictor()
        observed_counts = np.zeros((2, 2, 6), dtype=np.float64)
        observed_counts[1, 1, CLASS_SETTLEMENT] = 1.0
        observation_counts = np.array([[0.0, 1.0], [2.0, 1.0]], dtype=np.float64)

        quiet = {
            "settlement_signal": 0.0,
            "forest_signal": 0.0,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.0,
            "low_activity_factor": 1.0,
        }
        mixed = {
            "settlement_signal": 0.5,
            "forest_signal": 0.5,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.5,
            "low_activity_factor": 0.5,
        }
        active = {
            "settlement_signal": 1.0,
            "forest_signal": 1.0,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 1.0,
            "low_activity_factor": 0.0,
        }

        quiet_map = predictor._residual_blend_map(observed_counts, observation_counts, quiet)
        mixed_map = predictor._residual_blend_map(observed_counts, observation_counts, mixed)
        active_map = predictor._residual_blend_map(observed_counts, observation_counts, active)

        self.assertLess(quiet_map[0, 0], mixed_map[0, 0])
        self.assertLess(quiet_map[0, 1], mixed_map[0, 1])
        self.assertEqual(active_map[0, 0], predictor.config.residual_calibrator_high_activity_blend)
        self.assertEqual(active_map[0, 1], predictor.config.residual_calibrator_high_activity_single_observed_blend)
        self.assertEqual(active_map[1, 1], predictor.config.residual_calibrator_high_activity_active_observed_blend)

    def test_prior_blend_gate_strength_is_reduced_only_for_quiet_rounds(self) -> None:
        config = PredictorConfig(prior_blend_gate_strength=1.0, quiet_prior_blend_gate_multiplier=0.5)
        predictor = Predictor(
            config,
            RoundDetail(
                round_id="test",
                round_number=0,
                status="completed",
                map_width=2,
                map_height=2,
                seeds_count=1,
                initial_states=[],
            ),
            {},
            prior_blend_gate=PriorBlendGateArtifact(feature_names=[], model=None, metadata={}),  # type: ignore[arg-type]
        )
        prior = np.full((2, 2, 6), 1.0 / 6.0, dtype=np.float64)
        prediction = prior.copy()
        features = _make_features()
        quiet_regime = {
            "settlement_signal": 0.0,
            "forest_signal": 0.0,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.0,
            "low_activity_factor": 1.0,
        }
        mixed_regime = {
            "settlement_signal": 0.5,
            "forest_signal": 0.5,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.2,
            "low_activity_factor": 0.8,
        }
        call_strengths: list[float] = []

        def _fake_apply_prior_blend_gate(*args, **kwargs):
            call_strengths.append(kwargs["strength"])
            return prediction, {"summary": {"strength": kwargs["strength"]}, "tensors": {}}

        with patch("astar_island.predictor.apply_prior_blend_gate", side_effect=_fake_apply_prior_blend_gate):
            predictor._apply_prior_blend_gate(prior, prediction, features, quiet_regime)
            predictor._apply_prior_blend_gate(prior, prediction, features, mixed_regime)

        self.assertEqual(call_strengths[0], 0.5)
        self.assertEqual(call_strengths[1], 0.8)

    def test_build_transfer_quiet_round_pulls_active_mass_toward_observation_target(self) -> None:
        detail = RoundDetail(
            round_id="test",
            round_number=0,
            status="completed",
            map_width=4,
            map_height=4,
            seeds_count=1,
            initial_states=[],
        )
        predictor = Predictor(PredictorConfig(), detail, {})
        initial_state = InitialState(
            grid=np.array(
                [
                    [TERRAIN_OCEAN, TERRAIN_OCEAN, TERRAIN_OCEAN, TERRAIN_OCEAN],
                    [TERRAIN_OCEAN, TERRAIN_SETTLEMENT, TERRAIN_PLAINS, TERRAIN_PLAINS],
                    [TERRAIN_OCEAN, TERRAIN_FOREST, TERRAIN_PLAINS, TERRAIN_PLAINS],
                    [TERRAIN_OCEAN, TERRAIN_PLAINS, TERRAIN_PLAINS, TERRAIN_PLAINS],
                ],
                dtype=np.int64,
            ),
            settlements=[Settlement(x=1, y=1, has_port=False, alive=True)],
        )
        features = build_seed_features(0, initial_state)
        predictor.seed_features = {0: features}

        observed_counts = np.zeros((4, 4, 6), dtype=np.float64)
        observation_counts = np.zeros((4, 4), dtype=np.float64)
        buildable = features.buildable_mask
        observation_counts[buildable] = 1.0
        observed_counts[..., CLASS_EMPTY][buildable] = 1.0
        observed_counts[2, 1, :] = 0.0
        observed_counts[2, 1, CLASS_FOREST] = 1.0

        aggregator = SimpleNamespace(
            class_counts=observed_counts[None, ...],
            observation_counts=observation_counts[None, ...],
            conditional_counts={},
        )
        prior = np.zeros((4, 4, 6), dtype=np.float64)
        prior[..., CLASS_EMPTY] = 0.20
        prior[..., CLASS_SETTLEMENT] = 0.55
        prior[..., CLASS_PORT] = 0.05
        prior[..., CLASS_RUIN] = 0.05
        prior[..., CLASS_FOREST] = 0.15

        quiet_regime = {
            "settlement_signal": 0.0,
            "forest_signal": 0.0,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.0,
            "low_activity_factor": 1.0,
        }
        transfer, details = predictor._build_transfer(
            0,
            features,
            aggregator,
            prior,
            observed_counts,
            observation_counts,
            quiet_regime,
        )

        prior_active = float(np.mean(prior[..., CLASS_SETTLEMENT] + prior[..., CLASS_PORT] + prior[..., CLASS_RUIN]))
        target_active = float(
            np.mean(
                details["tensors"]["observation_target"][..., CLASS_SETTLEMENT]
                + details["tensors"]["observation_target"][..., CLASS_PORT]
                + details["tensors"]["observation_target"][..., CLASS_RUIN]
            )
        )
        transfer_active = float(
            np.mean(transfer[..., CLASS_SETTLEMENT] + transfer[..., CLASS_PORT] + transfer[..., CLASS_RUIN])
        )

        self.assertLess(target_active, 0.05)
        self.assertLess(abs(transfer_active - target_active), abs(prior_active - target_active) * 0.7)
        self.assertLess(transfer_active, prior_active * 0.7)

    def test_prior_blend_gate_quiet_round_does_not_reinflate_active_mass(self) -> None:
        predictor = Predictor(
            PredictorConfig(prior_blend_gate_strength=1.0, quiet_prior_blend_gate_multiplier=0.5),
            RoundDetail(
                round_id="test",
                round_number=0,
                status="completed",
                map_width=2,
                map_height=2,
                seeds_count=1,
                initial_states=[],
            ),
            {},
            prior_blend_gate=PriorBlendGateArtifact(feature_names=[], model=None, metadata={}),  # type: ignore[arg-type]
        )
        features = _make_features()
        prior = np.array(
            [
                [
                    [0.30, 0.40, 0.08, 0.07, 0.13, 0.02],
                    [0.32, 0.36, 0.08, 0.07, 0.15, 0.02],
                ],
                [
                    [0.34, 0.34, 0.08, 0.08, 0.14, 0.02],
                    [0.33, 0.35, 0.08, 0.07, 0.15, 0.02],
                ],
            ],
            dtype=np.float64,
        )
        prediction = np.array(
            [
                [
                    [0.70, 0.06, 0.01, 0.01, 0.20, 0.02],
                    [0.69, 0.07, 0.01, 0.01, 0.20, 0.02],
                ],
                [
                    [0.68, 0.08, 0.01, 0.01, 0.20, 0.02],
                    [0.69, 0.07, 0.01, 0.01, 0.20, 0.02],
                ],
            ],
            dtype=np.float64,
        )
        inflated = np.array(
            [
                [
                    [0.56, 0.18, 0.03, 0.02, 0.19, 0.02],
                    [0.55, 0.19, 0.03, 0.02, 0.19, 0.02],
                ],
                [
                    [0.54, 0.20, 0.03, 0.02, 0.19, 0.02],
                    [0.55, 0.19, 0.03, 0.02, 0.19, 0.02],
                ],
            ],
            dtype=np.float64,
        )
        quiet_regime = {
            "settlement_signal": 0.0,
            "forest_signal": 0.0,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.0,
            "low_activity_factor": 1.0,
        }
        mixed_regime = {
            "settlement_signal": 0.5,
            "forest_signal": 0.5,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.4,
            "low_activity_factor": 0.6,
        }

        with patch(
            "astar_island.predictor.apply_prior_blend_gate",
            return_value=(inflated, {"summary": {"strength": 0.5}, "tensors": {}}),
        ):
            quiet_blended = predictor._apply_prior_blend_gate(prior, prediction, features, quiet_regime)
            mixed_blended = predictor._apply_prior_blend_gate(prior, prediction, features, mixed_regime)

        quiet_active = float(np.mean(quiet_blended[..., CLASS_SETTLEMENT] + quiet_blended[..., CLASS_PORT] + quiet_blended[..., CLASS_RUIN]))
        prediction_active = float(
            np.mean(prediction[..., CLASS_SETTLEMENT] + prediction[..., CLASS_PORT] + prediction[..., CLASS_RUIN])
        )
        inflated_active = float(
            np.mean(inflated[..., CLASS_SETTLEMENT] + inflated[..., CLASS_PORT] + inflated[..., CLASS_RUIN])
        )
        mixed_active = float(np.mean(mixed_blended[..., CLASS_SETTLEMENT] + mixed_blended[..., CLASS_PORT] + mixed_blended[..., CLASS_RUIN]))

        self.assertAlmostEqual(quiet_active, prediction_active)
        self.assertGreater(inflated_active, prediction_active)
        self.assertGreater(mixed_active, quiet_active)
        self.assertLess(mixed_active, inflated_active)

    def test_redistribute_to_active_budget_preserves_probability_and_active_type_shares(self) -> None:
        predictor = _make_predictor()
        features = _make_features()
        prediction = np.array(
            [
                [
                    [0.44, 0.22, 0.10, 0.08, 0.11, 0.05],
                    [0.56, 0.14, 0.05, 0.03, 0.20, 0.02],
                ],
                [
                    [0.52, 0.18, 0.07, 0.05, 0.16, 0.02],
                    [0.60, 0.10, 0.04, 0.04, 0.20, 0.02],
                ],
            ],
            dtype=np.float64,
        )
        active_before = prediction[..., CLASS_SETTLEMENT] + prediction[..., CLASS_PORT] + prediction[..., CLASS_RUIN]
        type_share_before = np.divide(
            prediction[..., CLASS_SETTLEMENT],
            np.maximum(active_before, 1e-12),
        )

        adjusted = predictor._redistribute_to_active_budget(
            prediction,
            features,
            target_active_mass=0.12,
            reference_nonactive=prediction,
        )

        np.testing.assert_allclose(np.sum(adjusted, axis=-1), 1.0, atol=1e-9)
        np.testing.assert_allclose(adjusted[..., CLASS_MOUNTAIN], prediction[..., CLASS_MOUNTAIN], atol=1e-9)
        active_after = adjusted[..., CLASS_SETTLEMENT] + adjusted[..., CLASS_PORT] + adjusted[..., CLASS_RUIN]
        type_share_after = np.divide(
            adjusted[..., CLASS_SETTLEMENT],
            np.maximum(active_after, 1e-12),
        )
        np.testing.assert_allclose(type_share_after, type_share_before, atol=1e-9)
        self.assertAlmostEqual(float(np.mean(active_after[features.buildable_mask])), 0.12, places=6)

    def test_active_budget_calibrator_quiet_round_reduces_active_mass(self) -> None:
        features = _make_features()
        prediction = np.array(
            [
                [
                    [0.42, 0.24, 0.10, 0.08, 0.14, 0.02],
                    [0.45, 0.22, 0.09, 0.08, 0.14, 0.02],
                ],
                [
                    [0.48, 0.20, 0.08, 0.06, 0.16, 0.02],
                    [0.50, 0.18, 0.08, 0.06, 0.16, 0.02],
                ],
            ],
            dtype=np.float64,
        )
        quiet_regime = {
            "settlement_signal": 0.0,
            "forest_signal": 0.0,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.0,
            "low_activity_factor": 1.0,
        }
        observed_counts = np.zeros((2, 2, 6), dtype=np.float64)
        observed_counts[..., CLASS_EMPTY] = 1.0
        observation_counts = np.ones((2, 2), dtype=np.float64)
        bucket_target = prediction.copy()
        bucket_target[..., CLASS_EMPTY] = 0.92
        bucket_target[..., CLASS_FOREST] = 0.06
        bucket_target[..., CLASS_SETTLEMENT] = 0.015
        bucket_target[..., CLASS_PORT] = 0.003
        bucket_target[..., CLASS_RUIN] = 0.002
        bucket_target[..., CLASS_MOUNTAIN] = 0.0
        budget_feature_names = build_active_budget_features(
            prediction,
            features,
            quiet_regime,
            observed_counts=observed_counts,
            observation_counts=observation_counts,
            bucket_target=bucket_target,
            ood_signal_values={"settlement_rate": 0.04, "forest_share_dynamic": 0.75, "port_share_given_active": 0.1, "observed_cells": 4.0},
        )[1]
        artifact = ResidualCalibratorArtifact(
            feature_names=["feat_0"],
            active_model=_ConstantModel(0.2),
            forest_model=_ConstantModel(0.5),
            settlement_model=_ConstantModel(0.6),
            port_model=_ConstantModel(0.2),
            ruin_model=_ConstantModel(0.2),
            metadata={"version": 5},
            active_budget_model=_ConstantModel(0.04),
            budget_feature_names=budget_feature_names,
        )
        predictor = Predictor(
            PredictorConfig(
                active_budget_enabled=True,
                active_budget_strength_low_activity=1.0,
                active_budget_strength_default=0.0,
                active_budget_strength_high_activity=0.0,
                active_budget_min_buildable_observed=1,
            ),
            RoundDetail(
                round_id="test",
                round_number=0,
                status="completed",
                map_width=2,
                map_height=2,
                seeds_count=1,
                initial_states=[],
            ),
            {},
            residual_calibrator=artifact,
        )

        adjusted, details = predictor._apply_active_budget_calibrator(
            prediction,
            features,
            observed_counts,
            observation_counts,
            bucket_target,
            quiet_regime,
            {"settlement_rate": 0.04, "forest_share_dynamic": 0.75, "port_share_given_active": 0.1, "observed_cells": 4.0},
            return_details=True,
        )

        active_before = float(np.mean((prediction[..., CLASS_SETTLEMENT] + prediction[..., CLASS_PORT] + prediction[..., CLASS_RUIN])[features.buildable_mask]))
        active_after = float(np.mean((adjusted[..., CLASS_SETTLEMENT] + adjusted[..., CLASS_PORT] + adjusted[..., CLASS_RUIN])[features.buildable_mask]))
        self.assertLess(active_after, active_before)
        self.assertAlmostEqual(details["summary"]["effective_target_active_budget"], 0.04, places=6)

    def test_active_budget_keeps_observed_active_cells_nonzero(self) -> None:
        features = _make_features()
        prediction = np.array(
            [
                [
                    [0.42, 0.24, 0.10, 0.08, 0.14, 0.02],
                    [0.45, 0.22, 0.09, 0.08, 0.14, 0.02],
                ],
                [
                    [0.48, 0.20, 0.08, 0.06, 0.16, 0.02],
                    [0.50, 0.18, 0.08, 0.06, 0.16, 0.02],
                ],
            ],
            dtype=np.float64,
        )
        adjusted = _make_predictor()._redistribute_to_active_budget(
            prediction,
            features,
            target_active_mass=0.01,
            observed_counts=np.array(
                [
                    [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                    [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                ],
                dtype=np.float64,
            ),
            reference_nonactive=prediction,
        )
        observed_active = float(
            adjusted[0, 0, CLASS_SETTLEMENT] + adjusted[0, 0, CLASS_PORT] + adjusted[0, 0, CLASS_RUIN]
        )
        self.assertGreater(observed_active, 0.0)

    def test_collapsed_active_calibrator_disabled_bypasses_stage(self) -> None:
        predictor = _make_predictor()
        features = _make_features()
        prediction = np.full((2, 2, 6), 1.0 / 6.0, dtype=np.float64)
        observed_counts = np.zeros((2, 2, 6), dtype=np.float64)
        observation_counts = np.ones((2, 2), dtype=np.float64)
        bucket_target = prediction.copy()
        quiet_regime = {
            "settlement_signal": 0.0,
            "forest_signal": 0.0,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.0,
            "low_activity_factor": 1.0,
        }

        adjusted, details = predictor._apply_collapsed_active_calibrator(
            prediction,
            features,
            observed_counts,
            observation_counts,
            bucket_target,
            quiet_regime,
            {"settlement_rate": 0.05},
            return_details=True,
        )

        np.testing.assert_allclose(adjusted, prediction)
        self.assertFalse(details["summary"]["collapsed_active_enabled"])

    def test_collapsed_active_calibrator_quiet_round_scales_down_active_mass(self) -> None:
        features = _make_features()
        prediction = np.array(
            [
                [
                    [0.42, 0.24, 0.10, 0.08, 0.14, 0.02],
                    [0.45, 0.22, 0.09, 0.08, 0.14, 0.02],
                ],
                [
                    [0.48, 0.20, 0.08, 0.06, 0.16, 0.02],
                    [0.50, 0.18, 0.08, 0.06, 0.16, 0.02],
                ],
            ],
            dtype=np.float64,
        )
        quiet_regime = {
            "settlement_signal": 0.0,
            "forest_signal": 0.0,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.0,
            "low_activity_factor": 1.0,
        }
        observed_counts = np.zeros((2, 2, 6), dtype=np.float64)
        observed_counts[..., CLASS_EMPTY] = 1.0
        observation_counts = np.ones((2, 2), dtype=np.float64)
        bucket_target = prediction.copy()
        bucket_target[..., CLASS_EMPTY] = 0.92
        bucket_target[..., CLASS_FOREST] = 0.06
        bucket_target[..., CLASS_SETTLEMENT] = 0.015
        bucket_target[..., CLASS_PORT] = 0.003
        bucket_target[..., CLASS_RUIN] = 0.002
        bucket_target[..., CLASS_MOUNTAIN] = 0.0
        collapsed_names = build_collapsed_active_features(
            prediction,
            features,
            quiet_regime,
            observed_counts=observed_counts,
            observation_counts=observation_counts,
            bucket_target=bucket_target,
            ood_signal_values={"settlement_rate": 0.04, "forest_share_dynamic": 0.75, "port_share_given_active": 0.1, "observed_cells": 4.0},
        )[1]
        artifact = ResidualCalibratorArtifact(
            feature_names=["feat_0"],
            active_model=_ConstantModel(0.2),
            forest_model=_ConstantModel(0.5),
            settlement_model=_ConstantModel(0.6),
            port_model=_ConstantModel(0.2),
            ruin_model=_ConstantModel(0.2),
            metadata={"version": 5},
            collapsed_active_model=_ConstantModel(0.7),
            collapsed_active_feature_names=collapsed_names,
        )
        predictor = Predictor(
            PredictorConfig(
                collapsed_active_calibrator_enabled=True,
                collapsed_active_calibrator_strength_low_activity=1.0,
                collapsed_active_calibrator_strength_default=0.0,
                collapsed_active_calibrator_strength_high_activity=0.0,
                collapsed_active_calibrator_min_buildable_observed=1,
            ),
            RoundDetail(
                round_id="test",
                round_number=0,
                status="completed",
                map_width=2,
                map_height=2,
                seeds_count=1,
                initial_states=[],
            ),
            {},
            residual_calibrator=artifact,
        )

        adjusted, details = predictor._apply_collapsed_active_calibrator(
            prediction,
            features,
            observed_counts,
            observation_counts,
            bucket_target,
            quiet_regime,
            {"settlement_rate": 0.04, "forest_share_dynamic": 0.75, "port_share_given_active": 0.1, "observed_cells": 4.0},
            return_details=True,
        )
        active_before = float(np.mean((prediction[..., CLASS_SETTLEMENT] + prediction[..., CLASS_PORT] + prediction[..., CLASS_RUIN])[features.buildable_mask]))
        active_after = float(np.mean((adjusted[..., CLASS_SETTLEMENT] + adjusted[..., CLASS_PORT] + adjusted[..., CLASS_RUIN])[features.buildable_mask]))
        self.assertLess(active_after, active_before)
        self.assertAlmostEqual(details["summary"]["collapsed_active_effective_scale"], 0.7, places=6)

    def test_collapsed_active_precedence_skips_legacy_budget_stage(self) -> None:
        predictor = Predictor(
            PredictorConfig(
                collapsed_active_calibrator_enabled=True,
                active_budget_enabled=True,
            ),
            RoundDetail(
                round_id="test",
                round_number=0,
                status="completed",
                map_width=2,
                map_height=2,
                seeds_count=1,
                initial_states=[],
            ),
            {},
        )
        features = _make_features()
        observed_counts = np.zeros((2, 2, 6), dtype=np.float64)
        observation_counts = np.ones((2, 2), dtype=np.float64)
        prediction = np.full((2, 2, 6), 1.0 / 6.0, dtype=np.float64)
        bucket_target = prediction.copy()
        quiet_regime = {
            "settlement_signal": 0.0,
            "forest_signal": 0.0,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.0,
            "low_activity_factor": 1.0,
        }
        with (
            patch.object(predictor, "_build_prior", return_value=prediction),
            patch.object(
                predictor,
                "_build_transfer",
                return_value=(prediction, {"summary": {}, "tensors": {"active_dominance_support": np.zeros((2, 2)), "bucket_target": prediction}}),
            ),
            patch.object(predictor, "_apply_settlement_intensity_prior", side_effect=lambda pred, *_args, **_kwargs: pred),
            patch.object(predictor, "_apply_structural_calibration", side_effect=lambda pred, *_args, **_kwargs: pred),
            patch.object(predictor, "_apply_rare_class_concentration", side_effect=lambda pred, *_args, **_kwargs: pred),
            patch.object(predictor, "_apply_physical_constraints", side_effect=lambda pred, *_args, **_kwargs: pred),
            patch.object(predictor, "_calibrate_confidence", side_effect=lambda pred, *_args, **_kwargs: pred),
            patch.object(predictor, "_apply_global_mass_matching", side_effect=lambda pred, *_args, **_kwargs: pred),
            patch.object(predictor, "_apply_collapsed_active_calibrator", return_value=prediction) as collapsed_mock,
            patch.object(predictor, "_apply_active_budget_calibrator", return_value=prediction) as budget_mock,
            patch.object(predictor, "_apply_residual_calibrator", return_value=prediction),
            patch.object(predictor, "_apply_prior_blend_gate", return_value=prediction),
            patch.object(predictor, "_apply_boundary_softening", side_effect=lambda pred, *_args, **_kwargs: pred),
            patch.object(predictor, "_apply_high_activity_active_concentration", side_effect=lambda pred, *_args, **_kwargs: pred),
            patch.object(predictor, "_apply_active_dominance_separation", side_effect=lambda pred, *_args, **_kwargs: pred),
            patch.object(predictor, "_apply_ruin_dampening", side_effect=lambda pred, *_args, **_kwargs: pred),
        ):
            predictor._predict_seed_internal(
                0,
                features,
                SimpleNamespace(
                    class_counts=np.zeros((1, 2, 2, 6)),
                    observation_counts=np.ones((1, 2, 2)),
                    conditional_counts={},
                ),
                {"settlement_rate": 0.05, "forest_share_dynamic": 0.7, "port_share_given_active": 0.1},
                quiet_regime,
                {"active_rate": 0.0, "settlement_share_given_active": 0.0, "forest_share_given_nonactive": 0.5, "empty_share_given_nonactive": 0.5},
                {"settlement_rate": 0.05},
                None,
                None,
                None,
                collect_diagnostics=False,
            )
        self.assertTrue(collapsed_mock.called)
        self.assertFalse(budget_mock.called)


if __name__ == "__main__":
    unittest.main()
