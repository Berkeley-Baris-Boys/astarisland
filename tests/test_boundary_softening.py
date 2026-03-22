from __future__ import annotations

import unittest

import numpy as np

from astar_island.config import PredictorConfig
from astar_island.scoring import cell_entropy
from astar_island.types import NUM_CLASSES, SeedFeatures


def _make_predictor(alpha: float):
    from astar_island.predictor import Predictor
    cfg = PredictorConfig(boundary_softening_alpha=alpha)
    p = object.__new__(Predictor)
    p.config = cfg
    return p


def _make_features(h: int, w: int, *, frontier_mask: np.ndarray | None = None) -> SeedFeatures:
    """Build a minimal SeedFeatures for a grid of size h×w."""
    rng = np.random.default_rng(0)
    n_features = 19
    feature_stack = rng.random((h, w, n_features)).astype(np.float64)
    feature_names = [
        "bias", "buildable", "coastal", "initial_forest", "initial_settlement_like",
        "initial_mountain", "dist_to_settlement", "dist_to_coast", "dist_to_ruin",
        "settlement_density", "forest_density", "mountain_density", "coastal_density",
        "frontier", "conflict", "reclaimable", "settlement_intensity", "port_intensity",
        "border_distance",
    ]
    # Make settlement_intensity outside the (0.3, 0.7) band so only frontier / forest-boundary triggers
    feature_stack[..., feature_names.index("settlement_intensity")] = 0.9
    # No initial forest cells
    feature_stack[..., feature_names.index("initial_forest")] = 0.0

    buildable = np.ones((h, w), dtype=bool)
    if frontier_mask is None:
        frontier_mask = np.zeros((h, w), dtype=bool)

    return SeedFeatures(
        seed_index=0,
        feature_stack=feature_stack,
        feature_names=feature_names,
        buildable_mask=buildable,
        dynamic_prior_mask=buildable,
        coastal_mask=np.zeros((h, w), dtype=bool),
        frontier_mask=frontier_mask,
        conflict_mask=np.zeros((h, w), dtype=bool),
        reclaimable_mask=np.zeros((h, w), dtype=bool),
        initial_class_grid=np.zeros((h, w), dtype=np.int32),
        initial_settlement_mask=np.zeros((h, w), dtype=bool),
    )


def _uniform_prediction(h: int = 4, w: int = 4) -> np.ndarray:
    """Sharp prediction: almost all mass on class 0."""
    pred = np.full((h, w, NUM_CLASSES), 0.005, dtype=np.float64)
    pred[..., 0] = 0.975
    return pred


class BoundarySofteningTests(unittest.TestCase):
    def test_alpha_0_is_noop(self) -> None:
        p = _make_predictor(alpha=0.0)
        pred = _uniform_prediction()
        frontier = np.ones((4, 4), dtype=bool)
        features = _make_features(4, 4, frontier_mask=frontier)
        result = p._apply_boundary_softening(pred, features)
        self.assertIs(result, pred)

    def test_entropy_increases_in_mask_cells(self) -> None:
        p = _make_predictor(alpha=0.10)
        pred = _uniform_prediction(4, 4)
        frontier = np.zeros((4, 4), dtype=bool)
        frontier[0:2, :] = True  # top two rows in mask
        features = _make_features(4, 4, frontier_mask=frontier)
        before_entropy = cell_entropy(pred)
        result = p._apply_boundary_softening(pred, features)
        after_entropy = cell_entropy(result)
        # Masked cells must have strictly higher entropy
        self.assertTrue(
            np.all(after_entropy[frontier] > before_entropy[frontier]),
            msg=f"Entropy should increase in mask. before={before_entropy[frontier]}, after={after_entropy[frontier]}",
        )

    def test_non_mask_cells_unchanged(self) -> None:
        p = _make_predictor(alpha=0.10)
        pred = _uniform_prediction(4, 4)
        frontier = np.zeros((4, 4), dtype=bool)
        frontier[0:2, :] = True  # only top half in mask
        features = _make_features(4, 4, frontier_mask=frontier)
        result = p._apply_boundary_softening(pred, features)
        outside = ~frontier
        np.testing.assert_allclose(
            result[outside],
            pred[outside],
            atol=1e-10,
            err_msg="Cells outside the mask should be unchanged",
        )

    def test_probabilities_still_sum_to_1(self) -> None:
        p = _make_predictor(alpha=0.15)
        pred = _uniform_prediction(6, 6)
        frontier = np.ones((6, 6), dtype=bool)
        features = _make_features(6, 6, frontier_mask=frontier)
        result = p._apply_boundary_softening(pred, features)
        np.testing.assert_allclose(
            result.sum(axis=-1),
            1.0,
            atol=1e-10,
            err_msg="Each cell must still sum to 1 after softening",
        )


if __name__ == "__main__":
    unittest.main()
