from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from astar_island.config import PredictorConfig
from astar_island.learned_prior import build_learned_prior_artifact_from_archive
from astar_island.predictor import Predictor
from astar_island.residual_calibrator import build_residual_calibrator_artifact_from_archive
from astar_island.types import (
    CLASS_EMPTY,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_SETTLEMENT,
    RoundDetail,
    SeedFeatures,
    TERRAIN_FOREST,
    TERRAIN_MOUNTAIN,
    TERRAIN_OCEAN,
    TERRAIN_PLAINS,
    TERRAIN_PORT,
    TERRAIN_RUIN,
    TERRAIN_SETTLEMENT,
)


def _make_round_detail() -> RoundDetail:
    grid = np.array(
        [
            [TERRAIN_PLAINS, TERRAIN_SETTLEMENT],
            [TERRAIN_PORT, TERRAIN_FOREST],
        ],
        dtype=np.int16,
    )
    return RoundDetail(
        round_id="test",
        round_number=1,
        status="completed",
        map_width=2,
        map_height=2,
        seeds_count=1,
        initial_states=[SimpleNamespace(grid=grid, settlements=[])],
    )


def _make_features() -> SeedFeatures:
    feature_names = [
        "settlement_intensity",
        "port_intensity",
        "settlement_density",
        "forest_density",
        "border_distance",
        "dist_to_settlement",
        "dist_to_coast",
    ]
    feature_stack = np.zeros((2, 2, len(feature_names)), dtype=np.float64)
    feature_stack[0, 0, feature_names.index("settlement_intensity")] = 0.2
    feature_stack[0, 1, feature_names.index("settlement_intensity")] = 0.9
    feature_stack[1, 0, feature_names.index("port_intensity")] = 0.8
    feature_stack[1, 1, feature_names.index("forest_density")] = 1.0
    feature_stack[..., feature_names.index("border_distance")] = np.array([[0.1, 0.4], [0.3, 0.8]])
    feature_stack[..., feature_names.index("dist_to_settlement")] = 0.5
    feature_stack[..., feature_names.index("dist_to_coast")] = 0.5
    return SeedFeatures(
        seed_index=0,
        feature_stack=feature_stack,
        feature_names=feature_names,
        buildable_mask=np.array([[True, True], [True, True]], dtype=bool),
        dynamic_prior_mask=np.array([[True, True], [True, True]], dtype=bool),
        coastal_mask=np.array([[False, True], [True, False]], dtype=bool),
        frontier_mask=np.array([[True, True], [False, False]], dtype=bool),
        conflict_mask=np.array([[False, False], [True, False]], dtype=bool),
        reclaimable_mask=np.array([[False, False], [True, True]], dtype=bool),
        initial_class_grid=np.array(
            [
                [CLASS_EMPTY, CLASS_SETTLEMENT],
                [CLASS_PORT, CLASS_FOREST],
            ],
            dtype=np.int64,
        ),
        initial_settlement_mask=np.array([[False, True], [False, False]], dtype=bool),
    )


def _make_predictor(**config_overrides: float | bool) -> Predictor:
    config = PredictorConfig(**config_overrides)
    detail = _make_round_detail()
    return Predictor(config, detail, {0: _make_features()})


def _ood_metadata() -> dict[str, object]:
    return {
        "ood_reference": {
            "signals": {
                "settlement_rate": {"mean": 0.20, "std": 0.02, "p10": 0.18, "p90": 0.22, "count": 4},
                "forest_share_dynamic": {"mean": 0.45, "std": 0.05, "p10": 0.40, "p90": 0.50, "count": 4},
                "port_share_given_active": {"mean": 0.10, "std": 0.02, "p10": 0.08, "p90": 0.12, "count": 4},
                "repeat_fraction": {"mean": 0.08, "std": 0.02, "p10": 0.06, "p90": 0.10, "count": 4},
                "observed_cells": {"mean": 100.0, "std": 10.0, "p10": 90.0, "p90": 110.0, "count": 4},
            }
        }
    }


class OodGatingTests(unittest.TestCase):
    def test_compute_artifact_ood_is_zero_for_in_band_values(self) -> None:
        predictor = _make_predictor()
        signal_values = {
            "settlement_rate": 0.20,
            "forest_share_dynamic": 0.46,
            "port_share_given_active": 0.10,
            "repeat_fraction": 0.08,
            "observed_cells": 100.0,
        }

        details = predictor._compute_artifact_ood(
            _ood_metadata(),
            signal_values,
            enabled=True,
            trigger=0.75,
            full=2.0,
            strength=0.75,
            base_blend=0.25,
        )

        self.assertIsNotNone(details)
        self.assertTrue(details["enabled"])
        self.assertEqual(details["ood_score"], 0.0)
        self.assertEqual(details["effective_blend"], 0.25)

    def test_compute_artifact_ood_rises_for_shifted_values(self) -> None:
        predictor = _make_predictor()
        signal_values = {
            "settlement_rate": 0.35,
            "forest_share_dynamic": 0.20,
            "port_share_given_active": 0.16,
            "repeat_fraction": 0.18,
            "observed_cells": 150.0,
        }

        details = predictor._compute_artifact_ood(
            _ood_metadata(),
            signal_values,
            enabled=True,
            trigger=0.75,
            full=2.0,
            strength=0.75,
            base_blend=0.25,
        )

        self.assertIsNotNone(details)
        self.assertGreater(details["ood_score"], 0.0)
        self.assertLess(details["effective_blend"], 0.25)

    def test_missing_reference_and_equal_trigger_full_do_not_nan(self) -> None:
        predictor = _make_predictor()
        signal_values = predictor._round_ood_signal_values(
            {
                "settlement_rate": 0.5,
                "forest_share_dynamic": 0.2,
                "port_share_given_active": 0.2,
                "observed_cells": 200.0,
            },
            {"repeat_fraction": 0.2},
        )

        disabled = predictor._compute_artifact_ood(
            {},
            signal_values,
            enabled=True,
            trigger=1.0,
            full=1.0,
            strength=0.5,
            base_blend=0.25,
        )
        self.assertFalse(disabled["enabled"])
        self.assertEqual(disabled["ood_score"], 0.0)

        finite = predictor._compute_artifact_ood(
            {
                "ood_reference": {
                    "signals": {
                        "settlement_rate": {"mean": 0.2, "std": 0.0, "p10": 0.2, "p90": 0.2, "count": 1}
                    }
                }
            },
            signal_values,
            enabled=True,
            trigger=1.0,
            full=1.0,
            strength=0.5,
            base_blend=0.25,
        )
        self.assertTrue(np.isfinite(finite["ood_score"]))

    def test_apply_learned_prior_uses_reduced_effective_blend_under_ood(self) -> None:
        predictor = _make_predictor()
        predictor.learned_prior = SimpleNamespace(metadata=_ood_metadata())  # type: ignore[assignment]
        features = _make_features()
        prior = np.full((2, 2, 6), 1.0 / 6.0, dtype=np.float64)
        learned = np.zeros_like(prior)
        learned[..., CLASS_SETTLEMENT] = 0.9
        learned[..., CLASS_EMPTY] = 0.1

        mild_ood = {"effective_blend": 0.25}
        strong_ood = {"effective_blend": 0.05}

        with patch("astar_island.predictor.predict_learned_prior", return_value=learned):
            mild = predictor._apply_learned_prior(prior, features, mild_ood)
            strong = predictor._apply_learned_prior(prior, features, strong_ood)

        self.assertGreater(mild[0, 0, CLASS_SETTLEMENT], strong[0, 0, CLASS_SETTLEMENT])

    def test_apply_residual_calibrator_reduces_blend_map_under_ood(self) -> None:
        predictor = _make_predictor()
        predictor.residual_calibrator = SimpleNamespace(metadata=_ood_metadata())  # type: ignore[assignment]
        features = _make_features()
        prediction = np.full((2, 2, 6), 1.0 / 6.0, dtype=np.float64)
        observed_counts = np.zeros((2, 2, 6), dtype=np.float64)
        observation_counts = np.array([[0.0, 1.0], [2.0, 1.0]], dtype=np.float64)
        round_regime = {
            "settlement_signal": 0.5,
            "forest_signal": 0.5,
            "repeat_signal": 0.0,
            "repeat_fraction": 0.0,
            "high_activity_factor": 0.5,
            "low_activity_factor": 0.5,
        }
        captured: list[np.ndarray] = []

        def _fake_apply(*args, **kwargs):
            captured.append(np.asarray(kwargs["blend"], dtype=np.float64))
            return prediction, {"summary": {}, "tensors": {}}

        with patch("astar_island.predictor.apply_residual_calibrator", side_effect=_fake_apply):
            predictor._apply_residual_calibrator(
                prediction,
                features,
                observed_counts,
                observation_counts,
                round_regime,
                {"enabled": True, "ood_score": 1.0, "attenuation_factor": 0.4},
            )

        base_mean = float(np.mean(predictor._residual_blend_map(observed_counts, observation_counts, round_regime)))
        effective_mean = float(np.mean(captured[0]))
        self.assertLess(effective_mean, base_mean)

    def test_predict_round_diagnostics_include_ood_sections(self) -> None:
        predictor = Predictor(
            PredictorConfig(
                learned_prior_ood_enabled=True,
                residual_ood_enabled=True,
            ),
            _make_round_detail(),
            {0: _make_features()},
            learned_prior=SimpleNamespace(metadata=_ood_metadata()),  # type: ignore[arg-type]
            residual_calibrator=SimpleNamespace(metadata=_ood_metadata()),  # type: ignore[arg-type]
        )
        base = np.full((2, 2, 6), 1.0 / 6.0, dtype=np.float64)
        observed_counts = np.zeros((1, 2, 2, 6), dtype=np.float64)
        observation_counts = np.ones((1, 2, 2), dtype=np.float64)
        aggregator = SimpleNamespace(
            class_counts=observed_counts,
            observation_counts=observation_counts,
            round_latent_summary=lambda: {
                "observed_cells": 200.0,
                "settlement_rate": 0.34,
                "port_share_given_active": 0.15,
                "ruin_share_given_active": 0.05,
                "forest_share_dynamic": 0.24,
                "mean_food": 0.0,
                "mean_wealth": 0.0,
                "mean_defense": 0.0,
                "mean_population": 0.0,
            },
        )

        with (
            patch.object(predictor, "_build_prior", return_value=base),
            patch.object(
                predictor,
                "_build_transfer",
                return_value=(
                    base,
                    {
                        "summary": {},
                        "tensors": {
                            "observation_target": base,
                            "active_dominance_support": np.zeros((2, 2), dtype=np.float64),
                        },
                    },
                ),
            ),
            patch.object(predictor, "_apply_settlement_intensity_prior", side_effect=lambda pred, *_: pred),
            patch.object(predictor, "_apply_structural_calibration", side_effect=lambda pred, *_: pred),
            patch.object(predictor, "_apply_rare_class_concentration", side_effect=lambda pred, *args, **kwargs: (pred, None) if kwargs.get("return_details") else pred),
            patch.object(predictor, "_apply_physical_constraints", side_effect=lambda pred, *_: pred),
            patch.object(predictor, "_calibrate_confidence", side_effect=lambda pred, *_: pred),
            patch.object(predictor, "_apply_global_mass_matching", side_effect=lambda pred, *args, **kwargs: (pred, {"summary": {}}) if kwargs.get("return_details") else pred),
            patch.object(
                predictor,
                "_apply_residual_calibrator",
                side_effect=lambda pred, *args, **kwargs: (
                    pred,
                    {"summary": {"base_blend_mean": 0.2, "effective_blend_mean": 0.1}, "tensors": {}},
                )
                if kwargs.get("return_details")
                else pred,
            ),
            patch.object(predictor, "_apply_boundary_softening", side_effect=lambda pred, *_: pred),
            patch.object(predictor, "_apply_prior_blend_gate", side_effect=lambda prior, pred, *args, **kwargs: (pred, None) if kwargs.get("return_details") else pred),
            patch.object(predictor, "_apply_high_activity_active_concentration", side_effect=lambda pred, *_: pred),
            patch.object(predictor, "_apply_active_dominance_separation", side_effect=lambda pred, *args, **kwargs: pred),
            patch.object(predictor, "_apply_ruin_dampening", side_effect=lambda pred, *_: pred),
            patch("astar_island.predictor.predict_learned_prior", return_value=base),
        ):
            _, diagnostics = predictor.predict_round_with_diagnostics(aggregator)

        payload = diagnostics[0]
        self.assertIn("learned_prior_ood", payload)
        self.assertIn("residual_ood", payload)
        self.assertGreater(payload["learned_prior_ood"]["ood_score"], 0.0)
        self.assertGreater(payload["residual_ood"]["ood_score"], 0.0)


class OodReferenceBuilderTests(unittest.TestCase):
    def _write_round_fixture(self, history_dir: Path, *, include_metadata: bool) -> None:
        round_dir = history_dir / "round_01_fixture"
        seed_dir = round_dir / "seed_0"
        seed_dir.mkdir(parents=True, exist_ok=True)

        grid = np.array(
            [
                [TERRAIN_PLAINS, TERRAIN_SETTLEMENT, TERRAIN_PORT, TERRAIN_FOREST],
                [TERRAIN_RUIN, TERRAIN_MOUNTAIN, TERRAIN_PLAINS, TERRAIN_OCEAN],
                [TERRAIN_FOREST, TERRAIN_SETTLEMENT, TERRAIN_PLAINS, TERRAIN_PORT],
                [TERRAIN_PLAINS, TERRAIN_RUIN, TERRAIN_FOREST, TERRAIN_PLAINS],
            ],
            dtype=np.int16,
        )
        round_detail = {
            "id": "fixture",
            "round_number": 1,
            "status": "completed",
            "map_width": 4,
            "map_height": 4,
            "seeds_count": 1,
            "initial_states": [{"grid": grid.tolist(), "settlements": []}],
        }
        (round_dir / "round_detail.json").write_text(json.dumps(round_detail))

        class_grid = np.array(
            [
                [CLASS_EMPTY, CLASS_SETTLEMENT, CLASS_PORT, CLASS_FOREST],
                [CLASS_RUIN, CLASS_MOUNTAIN, CLASS_EMPTY, CLASS_EMPTY],
                [CLASS_FOREST, CLASS_SETTLEMENT, CLASS_EMPTY, CLASS_PORT],
                [CLASS_EMPTY, CLASS_RUIN, CLASS_FOREST, CLASS_EMPTY],
            ],
            dtype=np.int64,
        )
        ground_truth = np.eye(6, dtype=np.float64)[class_grid]
        prediction = ground_truth * 0.8 + (1.0 - ground_truth) * 0.04
        prediction /= np.sum(prediction, axis=-1, keepdims=True)

        np.save(seed_dir / "ground_truth.npy", ground_truth)
        (seed_dir / "analysis.json").write_text(
            json.dumps({"prediction": prediction.tolist(), "ground_truth": ground_truth.tolist(), "score": 90.0})
        )

        if include_metadata:
            (round_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "latent_summary": {
                            "observed_cells": 64.0,
                            "settlement_rate": 0.28,
                            "port_share_given_active": 0.12,
                            "ruin_share_given_active": 0.08,
                            "forest_share_dynamic": 0.36,
                        }
                    }
                )
            )
            observation_counts = np.array(
                [[[1.0, 2.0, 1.0, 1.0], [1.0, 1.0, 2.0, 1.0], [1.0, 1.0, 1.0, 2.0], [1.0, 1.0, 1.0, 1.0]]],
                dtype=np.float64,
            )
            np.save(round_dir / "observation_counts.npy", observation_counts)

    def test_artifact_builders_write_ood_reference_when_metadata_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            self._write_round_fixture(history_dir, include_metadata=True)

            learned = build_learned_prior_artifact_from_archive(history_dir, maxiter=5)
            residual = build_residual_calibrator_artifact_from_archive(
                history_dir,
                max_iter=5,
                max_depth=2,
                min_samples_leaf=1,
                learning_rate=0.1,
                l2_regularization=0.0,
            )

            learned_signals = learned.metadata["ood_reference"]["signals"]
            residual_signals = residual.metadata["ood_reference"]["signals"]
            self.assertIn("observed_cells", learned_signals)
            self.assertIn("repeat_fraction", residual_signals)

    def test_artifact_builders_fallback_without_metadata_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_dir = Path(tmpdir)
            self._write_round_fixture(history_dir, include_metadata=False)

            learned = build_learned_prior_artifact_from_archive(history_dir, maxiter=5)
            residual = build_residual_calibrator_artifact_from_archive(
                history_dir,
                max_iter=5,
                max_depth=2,
                min_samples_leaf=1,
                learning_rate=0.1,
                l2_regularization=0.0,
            )

            learned_signals = learned.metadata["ood_reference"]["signals"]
            residual_signals = residual.metadata["ood_reference"]["signals"]
            self.assertIn("settlement_rate", learned_signals)
            self.assertNotIn("observed_cells", learned_signals)
            self.assertNotIn("repeat_fraction", learned_signals)
            self.assertNotIn("repeat_fraction", residual_signals)


if __name__ == "__main__":
    unittest.main()
