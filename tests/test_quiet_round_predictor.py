from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from astar_island.config import PredictorConfig
from astar_island.prior_blend_gate import PriorBlendGateArtifact
from astar_island.predictor import Predictor
from astar_island.types import (
    CLASS_EMPTY,
    CLASS_FOREST,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_SETTLEMENT,
    RoundDetail,
    SeedFeatures,
)


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


if __name__ == "__main__":
    unittest.main()
