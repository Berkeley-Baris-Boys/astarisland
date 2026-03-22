from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from astar_island.config import PredictorConfig
from astar_island.predictor import Predictor
from astar_island.types import (
    CLASS_EMPTY,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_SETTLEMENT,
    NUM_CLASSES,
    RoundDetail,
    SeedFeatures,
)


def _make_predictor(**config_overrides) -> Predictor:
    config = PredictorConfig(**config_overrides)
    detail = RoundDetail(
        round_id="test",
        round_number=0,
        status="completed",
        map_width=2,
        map_height=2,
        seeds_count=1,
        initial_states=[],
    )
    return Predictor(config, detail, {})


def _make_features(buildable_mask: np.ndarray | None = None) -> SeedFeatures:
    if buildable_mask is None:
        buildable_mask = np.array([[True, True], [True, False]], dtype=bool)
    h, w = buildable_mask.shape
    return SeedFeatures(
        seed_index=0,
        feature_stack=np.zeros((h, w, 1), dtype=np.float64),
        feature_names=["dummy"],
        buildable_mask=buildable_mask,
        dynamic_prior_mask=buildable_mask.copy(),
        coastal_mask=np.zeros((h, w), dtype=bool),
        frontier_mask=np.zeros((h, w), dtype=bool),
        conflict_mask=np.zeros((h, w), dtype=bool),
        reclaimable_mask=np.zeros((h, w), dtype=bool),
        initial_class_grid=np.zeros((h, w), dtype=np.int64),
        initial_settlement_mask=np.zeros((h, w), dtype=bool),
    )


def _base_prediction() -> np.ndarray:
    pred = np.array(
        [
            [
                [0.35, 0.20, 0.10, 0.10, 0.23, 0.02],
                [0.40, 0.15, 0.15, 0.10, 0.18, 0.02],
            ],
            [
                [0.30, 0.15, 0.10, 0.15, 0.28, 0.02],
                [0.50, 0.10, 0.10, 0.10, 0.18, 0.02],
            ],
        ],
        dtype=np.float64,
    )
    return pred


def _observation_counts() -> np.ndarray:
    return np.array([[2.0, 2.0], [2.0, 0.0]], dtype=np.float64)


class GlobalMassMatchingTests(unittest.TestCase):
    def test_strength_zero_is_noop(self) -> None:
        predictor = _make_predictor(mass_matching_strength=0.0)
        pred = _base_prediction()
        features = _make_features()
        observed_counts = np.zeros_like(pred)
        observation_counts = _observation_counts()

        result, details = predictor._apply_global_mass_matching(
            pred,
            observed_counts,
            observation_counts,
            features,
            round_anchor={
                "active_rate": 0.5,
                "settlement_share_given_active": 0.5,
                "forest_share_given_nonactive": 0.5,
                "empty_share_given_nonactive": 0.5,
            },
            return_details=True,
        )

        np.testing.assert_allclose(result, pred)
        self.assertFalse(details["summary"]["enabled"])

    def test_active_pass_increases_active_mass(self) -> None:
        predictor = _make_predictor(
            mass_matching_min_buildable_observed=1,
            mass_matching_min_active_observed=99,
            mass_matching_min_nonactive_observed=99,
        )
        pred = _base_prediction()
        features = _make_features()
        observation_counts = _observation_counts()
        observed_counts = np.zeros_like(pred)
        observed_counts[0, 0, CLASS_SETTLEMENT] = 2.0
        observed_counts[0, 1, CLASS_SETTLEMENT] = 2.0
        observed_counts[1, 0, CLASS_PORT] = 2.0

        before = float(np.sum(pred[..., CLASS_SETTLEMENT:CLASS_RUIN + 1][features.buildable_mask]))
        result, details = predictor._apply_global_mass_matching(
            pred,
            observed_counts,
            observation_counts,
            features,
            round_anchor={
                "active_rate": 0.75,
                "settlement_share_given_active": 0.5,
                "forest_share_given_nonactive": 0.5,
                "empty_share_given_nonactive": 0.5,
            },
            return_details=True,
        )
        after = float(np.sum(result[..., CLASS_SETTLEMENT:CLASS_RUIN + 1][features.buildable_mask]))

        self.assertGreater(after, before)
        self.assertTrue(details["summary"]["passes"]["active_applied"])
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-10)

    def test_settlement_pass_changes_share_without_changing_active_total(self) -> None:
        predictor = _make_predictor(
            mass_matching_min_buildable_observed=99,
            mass_matching_min_active_observed=1,
            mass_matching_min_nonactive_observed=99,
        )
        pred = _base_prediction()
        features = _make_features()
        observation_counts = _observation_counts()
        observed_counts = np.zeros_like(pred)
        observed_counts[0, 0, CLASS_SETTLEMENT] = 2.0
        observed_counts[0, 1, CLASS_SETTLEMENT] = 2.0
        observed_counts[1, 0, CLASS_PORT] = 2.0

        buildable = features.buildable_mask
        before_active = float(np.sum(pred[..., CLASS_SETTLEMENT:CLASS_RUIN + 1][buildable]))
        before_share = float(
            np.sum(pred[..., CLASS_SETTLEMENT][buildable]) /
            np.sum(pred[..., CLASS_SETTLEMENT:CLASS_RUIN + 1][buildable])
        )
        untouched = pred[1, 1].copy()

        result, details = predictor._apply_global_mass_matching(
            pred,
            observed_counts,
            observation_counts,
            features,
            round_anchor={
                "active_rate": 0.45,
                "settlement_share_given_active": 0.9,
                "forest_share_given_nonactive": 0.5,
                "empty_share_given_nonactive": 0.5,
            },
            return_details=True,
        )

        after_active = float(np.sum(result[..., CLASS_SETTLEMENT:CLASS_RUIN + 1][buildable]))
        after_share = float(
            np.sum(result[..., CLASS_SETTLEMENT][buildable]) /
            np.sum(result[..., CLASS_SETTLEMENT:CLASS_RUIN + 1][buildable])
        )

        self.assertGreater(after_share, before_share)
        self.assertAlmostEqual(after_active, before_active, places=8)
        np.testing.assert_allclose(result[1, 1], untouched, atol=1e-10)
        self.assertTrue(details["summary"]["passes"]["settlement_applied"])

    def test_forest_empty_pass_changes_nonactive_split_without_changing_nonactive_total(self) -> None:
        predictor = _make_predictor(
            mass_matching_enable_nonactive=True,
            mass_matching_min_buildable_observed=99,
            mass_matching_min_active_observed=99,
            mass_matching_min_nonactive_observed=1,
        )
        pred = _base_prediction()
        features = _make_features()
        observation_counts = _observation_counts()
        observed_counts = np.zeros_like(pred)
        observed_counts[0, 0, CLASS_FOREST] = 2.0
        observed_counts[0, 1, CLASS_FOREST] = 2.0
        observed_counts[1, 0, CLASS_EMPTY] = 2.0

        buildable = features.buildable_mask
        before_nonactive = float(np.sum(pred[..., [CLASS_EMPTY, CLASS_FOREST]][buildable]))
        before_forest_share = float(
            np.sum(pred[..., CLASS_FOREST][buildable]) /
            np.sum(pred[..., [CLASS_EMPTY, CLASS_FOREST]][buildable])
        )

        result, details = predictor._apply_global_mass_matching(
            pred,
            observed_counts,
            observation_counts,
            features,
            round_anchor={
                "active_rate": 0.45,
                "settlement_share_given_active": 0.5,
                "forest_share_given_nonactive": 0.8,
                "empty_share_given_nonactive": 0.2,
            },
            return_details=True,
        )

        after_nonactive = float(np.sum(result[..., [CLASS_EMPTY, CLASS_FOREST]][buildable]))
        after_forest_share = float(
            np.sum(result[..., CLASS_FOREST][buildable]) /
            np.sum(result[..., [CLASS_EMPTY, CLASS_FOREST]][buildable])
        )

        self.assertGreater(after_forest_share, before_forest_share)
        self.assertAlmostEqual(after_nonactive, before_nonactive, places=8)
        self.assertTrue(details["summary"]["passes"]["nonactive_applied"])

    def test_low_support_shrinks_toward_round_anchor(self) -> None:
        predictor = _make_predictor(
            mass_matching_min_buildable_observed=1,
            mass_matching_target_support_buildable=8,
            mass_matching_min_active_observed=99,
            mass_matching_min_nonactive_observed=99,
            mass_matching_active_scale_clip=(0.5, 2.0),
        )
        pred = _base_prediction()
        features = _make_features()
        observation_counts = np.array([[1.0, 1.0], [0.0, 0.0]], dtype=np.float64)
        observed_counts = np.zeros_like(pred)
        observed_counts[0, 0, CLASS_SETTLEMENT] = 1.0
        observed_counts[0, 1, CLASS_SETTLEMENT] = 1.0

        _, details = predictor._apply_global_mass_matching(
            pred,
            observed_counts,
            observation_counts,
            features,
            round_anchor={
                "active_rate": 0.2,
                "settlement_share_given_active": 0.5,
                "forest_share_given_nonactive": 0.5,
                "empty_share_given_nonactive": 0.5,
            },
            return_details=True,
        )

        expected_target = 0.25 * 1.0 + 0.75 * 0.2
        self.assertAlmostEqual(details["summary"]["blended_targets"]["active_rate"], expected_target, places=8)

    def test_low_support_can_skip_cleanly(self) -> None:
        predictor = _make_predictor(
            mass_matching_min_buildable_observed=5,
            mass_matching_min_active_observed=5,
            mass_matching_min_nonactive_observed=5,
        )
        pred = _base_prediction()
        features = _make_features()
        observation_counts = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        observed_counts = np.zeros_like(pred)
        observed_counts[0, 0, CLASS_SETTLEMENT] = 1.0

        result, details = predictor._apply_global_mass_matching(
            pred,
            observed_counts,
            observation_counts,
            features,
            round_anchor={
                "active_rate": 0.6,
                "settlement_share_given_active": 0.8,
                "forest_share_given_nonactive": 0.7,
                "empty_share_given_nonactive": 0.3,
            },
            return_details=True,
        )

        np.testing.assert_allclose(result, pred)
        self.assertFalse(details["summary"]["passes"]["active_applied"])
        self.assertFalse(details["summary"]["passes"]["settlement_applied"])
        self.assertFalse(details["summary"]["passes"]["nonactive_applied"])

    def test_nonactive_pass_disabled_by_default(self) -> None:
        predictor = _make_predictor(
            mass_matching_min_buildable_observed=99,
            mass_matching_min_active_observed=99,
            mass_matching_min_nonactive_observed=1,
        )
        pred = _base_prediction()
        features = _make_features()
        observation_counts = _observation_counts()
        observed_counts = np.zeros_like(pred)
        observed_counts[0, 0, CLASS_FOREST] = 2.0
        observed_counts[0, 1, CLASS_FOREST] = 2.0
        observed_counts[1, 0, CLASS_EMPTY] = 2.0

        result, details = predictor._apply_global_mass_matching(
            pred,
            observed_counts,
            observation_counts,
            features,
            round_anchor={
                "active_rate": 0.45,
                "settlement_share_given_active": 0.5,
                "forest_share_given_nonactive": 0.8,
                "empty_share_given_nonactive": 0.2,
            },
            return_details=True,
        )

        np.testing.assert_allclose(result, pred)
        self.assertFalse(details["summary"]["passes"]["nonactive_enabled"])
        self.assertFalse(details["summary"]["passes"]["nonactive_applied"])

    def test_predict_seed_internal_applies_mass_matching_before_residual_and_skips_old_villager_step(self) -> None:
        predictor = _make_predictor()
        features = _make_features(np.ones((2, 2), dtype=bool))
        base = np.full((2, 2, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
        observed_counts = np.zeros_like(base)
        observation_counts = np.ones((2, 2), dtype=np.float64)
        order: list[str] = []

        def _identity(pred, *args, **kwargs):
            return pred

        def _rare(pred, *args, **kwargs):
            if kwargs.get("return_details", False):
                return pred, {"summary": {}, "tensors": {}}
            return pred

        def _mass(pred, *args, **kwargs):
            order.append("mass")
            if kwargs.get("return_details", False):
                return pred, {"summary": {"enabled": True}}
            return pred

        def _residual(pred, *args, **kwargs):
            order.append("residual")
            if kwargs.get("return_details", False):
                return pred, {"summary": {}, "tensors": {}}
            return pred

        def _prior_gate(prior, pred, *args, **kwargs):
            if kwargs.get("return_details", False):
                return pred, {"summary": {}, "tensors": {}}
            return pred

        with (
            patch.object(predictor, "_build_prior", return_value=base.copy()),
            patch.object(
                predictor,
                "_build_transfer",
                return_value=(
                    base.copy(),
                    {
                        "summary": {},
                        "tensors": {
                            "observation_target": base.copy(),
                            "active_dominance_support": np.zeros((2, 2), dtype=np.float64),
                        },
                    },
                ),
            ),
            patch.object(predictor, "_apply_settlement_intensity_prior", side_effect=_identity),
            patch.object(predictor, "_apply_structural_calibration", side_effect=_identity),
            patch.object(predictor, "_apply_rare_class_concentration", side_effect=_rare),
            patch.object(predictor, "_apply_physical_constraints", side_effect=_identity),
            patch.object(predictor, "_calibrate_confidence", side_effect=_identity),
            patch.object(predictor, "_apply_global_mass_matching", side_effect=_mass),
            patch.object(predictor, "_apply_residual_calibrator", side_effect=_residual),
            patch.object(predictor, "_apply_boundary_softening", side_effect=_identity),
            patch.object(predictor, "_apply_prior_blend_gate", side_effect=_prior_gate),
            patch.object(predictor, "_apply_high_activity_active_concentration", side_effect=_identity),
            patch.object(predictor, "_apply_active_dominance_separation", side_effect=lambda pred, *args, **kwargs: pred),
            patch.object(predictor, "_apply_ruin_dampening", side_effect=_identity),
            patch.object(
                predictor,
                "_apply_global_villager_recalibration",
                side_effect=AssertionError("old villager recalibration should not be called"),
            ),
        ):
            _, diagnostics = predictor._predict_seed_internal(
                seed_index=0,
                features=features,
                aggregator=type(
                    "DummyAggregator",
                    (),
                    {
                        "class_counts": np.expand_dims(observed_counts, axis=0),
                        "observation_counts": np.expand_dims(observation_counts, axis=0),
                        "conditional_counts": {},
                    },
                )(),
                latent={},
                round_regime={
                    "settlement_signal": 0.0,
                    "forest_signal": 0.0,
                    "repeat_signal": 0.0,
                    "repeat_fraction": 0.0,
                    "high_activity_factor": 0.0,
                    "low_activity_factor": 1.0,
                },
                mass_matching_anchor={
                    "active_rate": 0.5,
                    "settlement_share_given_active": 0.5,
                    "forest_share_given_nonactive": 0.5,
                    "empty_share_given_nonactive": 0.5,
                },
                learned_prior_ood=None,
                residual_ood=None,
                collect_diagnostics=True,
            )

        self.assertEqual(order, ["mass", "residual"])
        self.assertIn("post_mass_matching", diagnostics["stage_summaries"])
        self.assertIn("post_mass_matching", diagnostics["tensors"])


if __name__ == "__main__":
    unittest.main()
