#!/usr/bin/env python3
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api import APIError
from main import run_query_phase, load_active_round
from metrics import MetricsLogger, fetch_and_log_analysis
from predictor import build_predictions
from state import ObservationStore
from world_dynamics import estimate_world_dynamics


class _AlwaysFailAPI:
    def __init__(self) -> None:
        self.calls = 0

    def simulate(self, *args, **kwargs):
        self.calls += 1
        raise APIError("simulated failure")


class _RoundAPI:
    def get_active_round(self):
        return {"id": "r1", "round_number": 7, "round_weight": 1.25}

    def get_round_detail(self, round_id: str):
        self.last_round_id = round_id
        return {
            "map_width": 3,
            "map_height": 2,
            "seeds_count": 1,
            "initial_states": [{"grid": [[0, 0, 0], [0, 0, 0]], "settlements": []}],
        }


class _PartialAnalysisAPI:
    def get_analysis(self, round_id: str, seed: int):
        if seed == 0:
            raise APIError("not ready")
        return {
            "ground_truth": [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            "score": 99.0,
        }


class _CaptureLogger:
    def __init__(self) -> None:
        self.logged = None

    def log_ground_truth(self, round_id, ground_truth, predictions):
        self.logged = (round_id, ground_truth, predictions)


class RegressionTests(unittest.TestCase):
    def test_world_dynamics_survivor_averages_are_aligned(self):
        store = ObservationStore(round_id="r", width=2, height=2, seeds_count=1)
        store.settlement_snaps[0] = [{
            "query_index": 1,
            "viewport": {"x": 0, "y": 0, "w": 2, "h": 2},
            "settlements": [
                {"x": 0, "y": 0, "alive": False, "food": None, "population": None},
                {"x": 1, "y": 0, "alive": True, "food": 4.0, "population": 2.0},
                {"x": 0, "y": 1, "alive": False, "food": 10.0, "population": 8.0},
            ],
        }]
        init_states = [{"grid": [[0, 0], [0, 0]], "settlements": []}]

        dyn = estimate_world_dynamics(store, init_states)

        self.assertAlmostEqual(dyn.avg_food_survivors, 4.0, places=6)
        self.assertAlmostEqual(dyn.avg_pop_survivors, 2.0, places=6)

    def test_run_query_phase_stops_after_repeated_reserve_errors(self):
        store = ObservationStore(round_id="r", width=5, height=5, seeds_count=1)
        initial_states = [{"grid": [[0] * 5 for _ in range(5)], "settlements": []}]
        api = _AlwaysFailAPI()

        run_query_phase(api, store, initial_states, budget=5)

        self.assertEqual(store.queries_used, 0)
        # 1 core attempt + capped reserve failures
        self.assertLessEqual(api.calls, 10)

    def test_load_active_round_returns_round_number(self):
        api = _RoundAPI()
        round_id, round_number, w, h, seeds, weight, states = load_active_round(api)
        self.assertEqual(round_id, "r1")
        self.assertEqual(round_number, 7)
        self.assertEqual((w, h, seeds), (3, 2, 1))
        self.assertEqual(weight, 1.25)
        self.assertEqual(len(states), 1)

    def test_corrupt_metrics_file_is_backed_up(self):
        with tempfile.TemporaryDirectory() as td:
            metrics_path = Path(td) / "metrics.json"
            metrics_path.write_text("{bad json", encoding="utf-8")

            logger = MetricsLogger(path=str(metrics_path))

            self.assertEqual(logger._data, {})
            self.assertFalse(metrics_path.exists())
            backups = list(Path(td).glob("metrics.json.corrupt.*.bak"))
            self.assertTrue(backups)

    def test_fetch_analysis_continues_with_partial_seed_data(self):
        logger = _CaptureLogger()
        predictions = {1: np.full((1, 1, 6), 1 / 6, dtype=np.float32)}

        gt = fetch_and_log_analysis(
            _PartialAnalysisAPI(),
            round_id="round-x",
            seeds_count=2,
            predictions=predictions,
            logger=logger,
        )

        self.assertIsNotNone(gt)
        assert gt is not None
        self.assertIn(1, gt)
        self.assertNotIn(0, gt)
        self.assertIsNotNone(logger.logged)

    def test_ingest_query_index_is_consistent_between_logs(self):
        store = ObservationStore(round_id="r", width=1, height=1, seeds_count=1)
        result = {
            "grid": [[0]],
            "settlements": [{"x": 0, "y": 0, "alive": True}],
            "viewport": {"x": 0, "y": 0, "w": 1, "h": 1},
        }
        store.ingest(seed=0, vx=0, vy=0, result=result)

        self.assertEqual(store.query_log[0]["query_index"], 1)
        self.assertEqual(store.settlement_snaps[0][0]["query_index"], 1)

    def test_predictor_errors_are_not_silently_swallowed(self):
        with patch(
            "predictor._build_empirical_constrained_predictions",
            side_effect=RuntimeError("predictor bug"),
        ):
            with self.assertRaises(RuntimeError):
                build_predictions(
                    initial_states=[],
                    store=SimpleNamespace(),
                    dynamics=SimpleNamespace(),
                    verbose=False,
                )


if __name__ == "__main__":
    unittest.main()
