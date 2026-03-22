from __future__ import annotations

import unittest

import numpy as np

from astar_island.config import PredictorConfig
from astar_island.types import CLASS_EMPTY, CLASS_RUIN, NUM_CLASSES


def _make_predictor_with_scale(scale: float):
    """Return a minimal Predictor-like object with only _apply_ruin_dampening available."""
    from astar_island.predictor import Predictor

    cfg = PredictorConfig(ruin_dampening_scale=scale)
    # Predictor.__init__ loads artifacts from disk; bypass by using object.__new__
    p = object.__new__(Predictor)
    p.config = cfg
    return p


def _uniform_prediction(h: int = 5, w: int = 5) -> np.ndarray:
    """Uniform distribution across all classes."""
    pred = np.full((h, w, NUM_CLASSES), 1.0 / NUM_CLASSES, dtype=np.float64)
    return pred


def _prediction_with_known_ruin(ruin_mass: float, h: int = 4, w: int = 4) -> np.ndarray:
    """Prediction where ruin probability is `ruin_mass` and remainder split across other classes."""
    other_mass = (1.0 - ruin_mass) / (NUM_CLASSES - 1)
    pred = np.full((h, w, NUM_CLASSES), other_mass, dtype=np.float64)
    pred[..., CLASS_RUIN] = ruin_mass
    return pred


class RuinDampeningTests(unittest.TestCase):
    def test_ruin_mass_reduced_by_scale(self) -> None:
        scale = 0.85
        p = _make_predictor_with_scale(scale)
        pred = _prediction_with_known_ruin(0.20)
        before_ruin = pred[..., CLASS_RUIN].sum()

        result = p._apply_ruin_dampening(pred)

        after_ruin = result[..., CLASS_RUIN].sum()
        expected_ruin = before_ruin * scale
        # After renormalization ruin mass may differ slightly, but it must be strictly lower
        self.assertLess(after_ruin, before_ruin)
        # Within 1% of expected (renormalization redistributes a tiny amount of the min_prob floor)
        np.testing.assert_allclose(after_ruin, expected_ruin, rtol=0.01)

    def test_removed_mass_goes_to_empty(self) -> None:
        scale = 0.85
        p = _make_predictor_with_scale(scale)
        pred = _prediction_with_known_ruin(0.20)
        before_empty = pred[..., CLASS_EMPTY].sum()
        before_ruin = pred[..., CLASS_RUIN].sum()

        result = p._apply_ruin_dampening(pred)

        after_empty = result[..., CLASS_EMPTY].sum()
        after_ruin = result[..., CLASS_RUIN].sum()
        removed = before_ruin - after_ruin
        gained = after_empty - before_empty
        # The mass removed from ruin should approximately equal the mass gained by empty
        np.testing.assert_allclose(gained, removed, rtol=0.01)
        # And total probability mass is preserved (each cell still sums to 1)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-10)

    def test_scale_1_is_noop(self) -> None:
        p = _make_predictor_with_scale(1.0)
        pred = _prediction_with_known_ruin(0.20)

        result = p._apply_ruin_dampening(pred)

        # Must be the exact same object (early return path)
        self.assertIs(result, pred)


if __name__ == "__main__":
    unittest.main()
