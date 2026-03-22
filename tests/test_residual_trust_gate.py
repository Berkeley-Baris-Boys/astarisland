from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from astar_island.residual_calibrator import (
    ResidualCalibratorArtifact,
    apply_residual_calibrator,
    build_residual_trust_gate_features,
    load_residual_calibrator_artifact,
    optimal_residual_blend,
    residual_trust_gate_sample_weight,
)
from astar_island.types import CLASS_PORT, CLASS_RUIN, CLASS_SETTLEMENT, NUM_CLASSES, SeedFeatures

_TG_EXTRA_COUNT = 16  # number of features added by build_residual_trust_gate_features


class ResidualTrustGateFeatureTests(unittest.TestCase):
    def _make_inputs(self, n: int = 16, f: int = 20) -> tuple:
        rng = np.random.default_rng(0)
        residual_features = rng.random((n, f))
        residual_names = [f"feat_{i}" for i in range(f)]
        pred_flat = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
        learned_flat = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
        return residual_features, residual_names, pred_flat, learned_flat

    def test_output_shape_and_name_count(self) -> None:
        n, f = 16, 20
        residual_features, residual_names, pred_flat, learned_flat = self._make_inputs(n, f)
        matrix, names = build_residual_trust_gate_features(
            residual_features, residual_names, pred_flat, learned_flat
        )
        self.assertEqual(matrix.shape, (n, f + _TG_EXTRA_COUNT))
        self.assertEqual(len(names), f + _TG_EXTRA_COUNT)
        self.assertEqual(len(names), matrix.shape[1])

    def test_new_feature_names_start_with_tg(self) -> None:
        f = 5
        residual_features, residual_names, pred_flat, learned_flat = self._make_inputs(f=f)
        _, names = build_residual_trust_gate_features(
            residual_features, residual_names, pred_flat, learned_flat
        )
        for name in names[f:]:
            self.assertTrue(name.startswith("tg_"), msg=f"Feature '{name}' does not start with 'tg_'")

    def test_original_residual_features_preserved(self) -> None:
        residual_features, residual_names, pred_flat, learned_flat = self._make_inputs()
        f = residual_features.shape[1]
        matrix, _ = build_residual_trust_gate_features(
            residual_features, residual_names, pred_flat, learned_flat
        )
        np.testing.assert_array_equal(matrix[:, :f], residual_features)

    def test_argmax_flip_feature_is_correct(self) -> None:
        n, f = 4, 3
        residual_features = np.zeros((n, f))
        residual_names = [f"feat_{i}" for i in range(f)]
        pred_flat = np.tile([0.6, 0.1, 0.1, 0.1, 0.05, 0.05], (n, 1)).astype(np.float64)
        learned_flat = pred_flat.copy()
        # cells 2 and 3 flip the argmax
        learned_flat[2] = [0.1, 0.6, 0.1, 0.1, 0.05, 0.05]
        learned_flat[3] = [0.1, 0.1, 0.6, 0.1, 0.05, 0.05]

        matrix, names = build_residual_trust_gate_features(
            residual_features, residual_names, pred_flat, learned_flat
        )
        flip_idx = names.index("tg_argmax_flip")
        self.assertEqual(matrix[0, flip_idx], 0.0)
        self.assertEqual(matrix[1, flip_idx], 0.0)
        self.assertEqual(matrix[2, flip_idx], 1.0)
        self.assertEqual(matrix[3, flip_idx], 1.0)

    def test_l1_diff_is_zero_when_learned_equals_pred(self) -> None:
        n, f = 8, 3
        residual_features = np.zeros((n, f))
        residual_names = [f"feat_{i}" for i in range(f)]
        pred_flat = np.random.dirichlet(np.ones(NUM_CLASSES), size=n)
        learned_flat = pred_flat.copy()

        matrix, names = build_residual_trust_gate_features(
            residual_features, residual_names, pred_flat, learned_flat
        )
        l1_idx = names.index("tg_l1_diff_learned_base")
        np.testing.assert_allclose(matrix[:, l1_idx], 0.0, atol=1e-12)


class OptimalResidualBlendTests(unittest.TestCase):
    def test_alpha_in_valid_range(self) -> None:
        rng = np.random.default_rng(42)
        n = 50
        base = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
        learned = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
        gt = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
        alpha = optimal_residual_blend(base, learned, gt)
        self.assertEqual(alpha.shape, (n,))
        self.assertTrue(np.all(alpha >= 0.0))
        self.assertTrue(np.all(alpha <= 1.0))

    def test_alpha_tends_to_1_when_learned_matches_gt(self) -> None:
        rng = np.random.default_rng(7)
        n = 10
        gt = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
        base = np.full((n, NUM_CLASSES), 1.0 / NUM_CLASSES)
        # learned == gt, base is uniform — optimal should strongly prefer alpha=1
        alpha = optimal_residual_blend(base, gt.copy(), gt)
        self.assertTrue(np.all(alpha >= 0.9), msg=f"Expected alpha≈1, got {alpha}")

    def test_alpha_tends_to_0_when_base_matches_gt(self) -> None:
        rng = np.random.default_rng(13)
        n = 10
        gt = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
        learned = np.full((n, NUM_CLASSES), 1.0 / NUM_CLASSES)
        # base == gt, learned is uniform — optimal should strongly prefer alpha=0
        alpha = optimal_residual_blend(gt.copy(), learned, gt)
        self.assertTrue(np.all(alpha <= 0.1), msg=f"Expected alpha≈0, got {alpha}")


class ResidualTrustGateSampleWeightTests(unittest.TestCase):
    def test_weights_are_positive(self) -> None:
        rng = np.random.default_rng(0)
        n = 30
        base = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
        learned = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
        gt = rng.dirichlet(np.ones(NUM_CLASSES), size=n)
        weights = residual_trust_gate_sample_weight(base, learned, gt)
        self.assertEqual(weights.shape, (n,))
        self.assertTrue(np.all(weights > 0.0))

    def test_high_disagreement_raises_weight(self) -> None:
        n = 4
        gt = np.full((n, NUM_CLASSES), 1.0 / NUM_CLASSES)
        base_close = gt.copy()
        learned_close = gt.copy()
        # high disagreement: learned is one-hot vs base uniform
        base_far = np.full((n, NUM_CLASSES), 1.0 / NUM_CLASSES)
        learned_far = np.zeros((n, NUM_CLASSES))
        learned_far[:, 0] = 1.0

        w_close = residual_trust_gate_sample_weight(base_close, learned_close, gt)
        w_far = residual_trust_gate_sample_weight(base_far, learned_far, gt)
        # high-disagreement cells should have larger weights
        self.assertTrue(np.all(w_far > w_close))


class _ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.full(features.shape[0], self.value, dtype=np.float64)


class ResidualTrustGateCapTests(unittest.TestCase):
    def test_active_observed_cap_uses_supplied_value(self) -> None:
        artifact = ResidualCalibratorArtifact(
            feature_names=["feat_0"],
            active_model=_ConstantModel(0.4),
            forest_model=_ConstantModel(0.5),
            settlement_model=_ConstantModel(0.6),
            port_model=_ConstantModel(0.2),
            ruin_model=_ConstantModel(0.2),
            metadata={},
            blend_model=_ConstantModel(0.9),
            blend_feature_names=["feat_0"] + [f"tg_{idx}" for idx in range(_TG_EXTRA_COUNT)],
        )
        prediction = np.full((2, 2, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
        observed_counts = np.zeros((2, 2, NUM_CLASSES), dtype=np.float64)
        observed_counts[0, 0, CLASS_SETTLEMENT] = 1.0
        observed_counts[0, 1, CLASS_PORT] = 1.0
        observed_counts[1, 0, CLASS_RUIN] = 1.0

        seed_features = SeedFeatures(
            seed_index=0,
            feature_stack=np.zeros((2, 2, 1), dtype=np.float64),
            feature_names=["dummy"],
            buildable_mask=np.ones((2, 2), dtype=bool),
            dynamic_prior_mask=np.ones((2, 2), dtype=bool),
            coastal_mask=np.zeros((2, 2), dtype=bool),
            frontier_mask=np.zeros((2, 2), dtype=bool),
            conflict_mask=np.zeros((2, 2), dtype=bool),
            reclaimable_mask=np.zeros((2, 2), dtype=bool),
            initial_class_grid=np.zeros((2, 2), dtype=np.int64),
            initial_settlement_mask=np.zeros((2, 2), dtype=bool),
        )

        residual_features = np.zeros((4, 1), dtype=np.float64)
        trust_gate_features = np.zeros((4, 1 + _TG_EXTRA_COUNT), dtype=np.float64)
        trust_gate_names = artifact.blend_feature_names
        with (
            patch("astar_island.residual_calibrator.build_residual_features", return_value=(residual_features, artifact.feature_names)),
            patch(
                "astar_island.residual_calibrator.build_residual_trust_gate_features",
                return_value=(trust_gate_features, trust_gate_names),
            ),
        ):
            _, details = apply_residual_calibrator(
                artifact,
                prediction,
                seed_features,
                blend=0.0,
                min_probability=1e-6,
                active_observed_cap=0.6,
                observed_counts=observed_counts,
            )

        blend_map = details["tensors"]["residual_blend_map"]
        self.assertEqual(blend_map[0, 0], 0.6)
        self.assertEqual(blend_map[0, 1], 0.6)
        self.assertEqual(blend_map[1, 0], 0.6)
        self.assertEqual(blend_map[1, 1], 0.9)

    def test_loading_stale_artifact_fails_fast(self) -> None:
        artifact = ResidualCalibratorArtifact(
            feature_names=["feat_0"],
            active_model=_ConstantModel(0.4),
            forest_model=_ConstantModel(0.5),
            settlement_model=_ConstantModel(0.6),
            port_model=_ConstantModel(0.2),
            ruin_model=_ConstantModel(0.2),
            metadata={"version": 3},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "residual.joblib"
            joblib.dump(artifact, path)
            with self.assertRaises(ValueError):
                load_residual_calibrator_artifact(path)


if __name__ == "__main__":
    unittest.main()
