from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from astar_island.aggregator import ObservationAggregator
from astar_island.config import AstarConfig
from astar_island.features import build_all_features
from astar_island.history import _round_detail_from_json
from astar_island.learned_prior import load_learned_prior_artifact
from astar_island.predictor import Predictor
from astar_island.priors import load_historical_prior_artifact
from astar_island.regime import compute_round_regime, regime_bucket, repeat_fraction_from_observation_counts
from astar_island.residual_calibrator import (
    ResidualCalibratorArtifact,
    build_residual_calibrator_artifact_from_archive,
)
from astar_island.scoring import score_prediction
from astar_island.utils import save_json

SCENARIOS = (
    "baseline",
    "quiet_patch",
    "collapsed_active_only",
    "collapsed_active_plus_residual",
    "budget_only",
    "budget_plus_residual",
)


def _discover_run_dirs(artifacts_dir: Path, rounds: set[int]) -> dict[int, Path]:
    matches: dict[int, list[Path]] = {}
    for metadata_path in artifacts_dir.glob("*/metadata.json"):
        if metadata_path.parent.name == "history":
            continue
        try:
            payload = json.loads(metadata_path.read_text())
        except json.JSONDecodeError:
            continue
        round_number = payload.get("round_number")
        if round_number not in rounds:
            continue
        run_dir = metadata_path.parent
        required = [
            run_dir / "round_detail.json",
            run_dir / "class_counts.npy",
            run_dir / "observation_counts.npy",
            run_dir / "conditional_counts.json",
        ]
        if all(path.exists() for path in required):
            matches.setdefault(int(round_number), []).append(run_dir)
    resolved: dict[int, Path] = {}
    for round_number, candidates in matches.items():
        resolved[round_number] = max(candidates, key=lambda path: path.stat().st_mtime)
    return resolved


def _load_artifacts(config: AstarConfig):
    historical = None
    learned = None
    if config.predictor.historical_prior_path.exists():
        historical = load_historical_prior_artifact(config.predictor.historical_prior_path)
    if config.predictor.learned_prior_path.exists():
        learned = load_learned_prior_artifact(config.predictor.learned_prior_path)
    return historical, learned


def _make_budget_only_artifact(artifact: ResidualCalibratorArtifact) -> ResidualCalibratorArtifact:
    return replace(artifact, blend_model=None, blend_feature_names=None)


def _configure_for_scenario(
    base_config: AstarConfig,
    scenario: str,
) -> tuple[AstarConfig, bool]:
    config = copy.deepcopy(base_config)
    residual_enabled = True
    if scenario == "baseline":
        config.predictor.active_budget_enabled = False
        config.predictor.collapsed_active_calibrator_enabled = False
        residual_enabled = False
    elif scenario == "quiet_patch":
        config.predictor.active_budget_enabled = False
        config.predictor.collapsed_active_calibrator_enabled = False
    elif scenario == "collapsed_active_only":
        config.predictor.active_budget_enabled = False
        config.predictor.collapsed_active_calibrator_enabled = True
        config.predictor.residual_calibrator_blend = 0.0
        config.predictor.residual_calibrator_low_activity_blend = 0.0
        config.predictor.residual_calibrator_high_activity_blend = 0.0
        config.predictor.residual_calibrator_single_observed_blend = 0.0
        config.predictor.residual_calibrator_repeated_observed_blend = 0.0
        config.predictor.residual_calibrator_active_observed_blend = 0.0
        config.predictor.residual_calibrator_low_activity_single_observed_blend = 0.0
        config.predictor.residual_calibrator_low_activity_repeated_observed_blend = 0.0
        config.predictor.residual_calibrator_low_activity_active_observed_blend = 0.0
        config.predictor.residual_calibrator_high_activity_single_observed_blend = 0.0
        config.predictor.residual_calibrator_high_activity_repeated_observed_blend = 0.0
        config.predictor.residual_calibrator_high_activity_active_observed_blend = 0.0
    elif scenario == "collapsed_active_plus_residual":
        config.predictor.active_budget_enabled = False
        config.predictor.collapsed_active_calibrator_enabled = True
    elif scenario == "budget_only":
        config.predictor.active_budget_enabled = True
        config.predictor.collapsed_active_calibrator_enabled = False
        config.predictor.residual_calibrator_blend = 0.0
        config.predictor.residual_calibrator_low_activity_blend = 0.0
        config.predictor.residual_calibrator_high_activity_blend = 0.0
        config.predictor.residual_calibrator_single_observed_blend = 0.0
        config.predictor.residual_calibrator_repeated_observed_blend = 0.0
        config.predictor.residual_calibrator_active_observed_blend = 0.0
        config.predictor.residual_calibrator_low_activity_single_observed_blend = 0.0
        config.predictor.residual_calibrator_low_activity_repeated_observed_blend = 0.0
        config.predictor.residual_calibrator_low_activity_active_observed_blend = 0.0
        config.predictor.residual_calibrator_high_activity_single_observed_blend = 0.0
        config.predictor.residual_calibrator_high_activity_repeated_observed_blend = 0.0
        config.predictor.residual_calibrator_high_activity_active_observed_blend = 0.0
    elif scenario == "budget_plus_residual":
        config.predictor.active_budget_enabled = True
        config.predictor.collapsed_active_calibrator_enabled = False
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return config, residual_enabled


def _buildable_mass_errors(prediction: np.ndarray, ground_truth: np.ndarray, buildable_mask: np.ndarray) -> dict[str, float]:
    buildable_count = float(np.sum(buildable_mask))
    if buildable_count <= 0.0:
        return {"active": 0.0, "settlement": 0.0}
    return {
        "active": float(
            abs(
                np.sum(prediction[..., 1:4][buildable_mask])
                - np.sum(ground_truth[..., 1:4][buildable_mask])
            )
            / buildable_count
        ),
        "settlement": float(
            abs(np.sum(prediction[..., 1][buildable_mask]) - np.sum(ground_truth[..., 1][buildable_mask]))
            / buildable_count
        ),
    }


def _active_ece(prediction: np.ndarray, ground_truth: np.ndarray, buildable_mask: np.ndarray, bins: int = 10) -> float:
    pred_active = prediction[..., 1] + prediction[..., 2] + prediction[..., 3]
    true_active = ground_truth[..., 1] + ground_truth[..., 2] + ground_truth[..., 3]
    pred_flat = pred_active[buildable_mask].reshape(-1)
    true_flat = true_active[buildable_mask].reshape(-1)
    if pred_flat.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for idx in range(bins):
        lo = edges[idx]
        hi = edges[idx + 1]
        if idx == bins - 1:
            mask = (pred_flat >= lo) & (pred_flat <= hi)
        else:
            mask = (pred_flat >= lo) & (pred_flat < hi)
        if not np.any(mask):
            continue
        total += float(np.mean(mask) * abs(np.mean(pred_flat[mask]) - np.mean(true_flat[mask])))
    return total


def _mask_overprediction(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    mask: np.ndarray,
) -> float:
    if not np.any(mask):
        return 0.0
    pred_active = prediction[..., 1] + prediction[..., 2] + prediction[..., 3]
    gt_active = ground_truth[..., 1] + ground_truth[..., 2] + ground_truth[..., 3]
    return float(np.mean(pred_active[mask] - gt_active[mask]))


def _scenario_artifact(
    scenario: str,
    artifact: ResidualCalibratorArtifact | None,
) -> ResidualCalibratorArtifact | None:
    if artifact is None:
        return None
    if scenario == "baseline":
        return None
    if scenario == "budget_only":
        return _make_budget_only_artifact(artifact)
    return artifact


def evaluate_round(
    run_dir: Path,
    history_dir: Path,
    *,
    artifact: ResidualCalibratorArtifact,
) -> dict[str, object]:
    base_config = AstarConfig()
    historical, learned = _load_artifacts(base_config)
    detail = _round_detail_from_json(json.loads((run_dir / "round_detail.json").read_text()))
    features = build_all_features(detail.initial_states, settlement_sigma=base_config.predictor.settlement_sigma)
    aggregator = ObservationAggregator(detail, features)
    aggregator.class_counts = np.load(run_dir / "class_counts.npy")
    aggregator.observation_counts = np.load(run_dir / "observation_counts.npy")
    aggregator.conditional_counts = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in json.loads((run_dir / "conditional_counts.json").read_text()).items()
    }

    history_matches = list(history_dir.glob(f"round_{detail.round_number:02d}_*"))
    if not history_matches:
        raise FileNotFoundError(f"Missing archived round for {detail.round_number}")
    history_round_dir = history_matches[0]

    round_regime = compute_round_regime(
        aggregator.round_latent_summary(),
        base_config.predictor,
        repeat_fraction=repeat_fraction_from_observation_counts(aggregator.observation_counts),
    )
    result: dict[str, object] = {
        "round_number": detail.round_number,
        "round_id": detail.round_id,
        "run_dir": str(run_dir),
        "round_bucket": regime_bucket(round_regime["high_activity_factor"]),
        "scenarios": {},
    }

    for scenario in SCENARIOS:
        config, _ = _configure_for_scenario(base_config, scenario)
        predictor = Predictor(
            config.predictor,
            detail,
            features,
            historical_priors=historical,
            learned_prior=learned,
            residual_calibrator=_scenario_artifact(scenario, artifact),
        )
        predictions = predictor.predict_round(aggregator)
        scores: list[float] = []
        active_errors: list[float] = []
        settlement_errors: list[float] = []
        active_eces: list[float] = []
        frontier_over: list[float] = []
        initial_settlement_over: list[float] = []
        near_settlement_over: list[float] = []
        per_seed: dict[str, float] = {}
        for seed_index, seed_features in features.items():
            gt = np.load(history_round_dir / f"seed_{seed_index}" / "ground_truth.npy").astype(np.float64)
            pred = predictions[seed_index]
            score = float(score_prediction(gt, pred))
            scores.append(score)
            per_seed[str(seed_index)] = score
            errors = _buildable_mass_errors(pred, gt, seed_features.buildable_mask)
            active_errors.append(errors["active"])
            settlement_errors.append(errors["settlement"])
            active_eces.append(_active_ece(pred, gt, seed_features.buildable_mask))
            dist_idx = seed_features.feature_names.index("dist_to_settlement")
            near_threshold = 2.0 / max(detail.map_width + detail.map_height, 1)
            near_settlement_mask = seed_features.buildable_mask & (
                seed_features.feature_stack[..., dist_idx] <= near_threshold
            )
            frontier_over.append(_mask_overprediction(pred, gt, seed_features.frontier_mask & seed_features.buildable_mask))
            initial_settlement_over.append(_mask_overprediction(pred, gt, seed_features.initial_settlement_mask & seed_features.buildable_mask))
            near_settlement_over.append(_mask_overprediction(pred, gt, near_settlement_mask))
        result["scenarios"][scenario] = {
            "score_mean": float(np.mean(scores)),
            "active_mass_error_mean": float(np.mean(active_errors)),
            "settlement_mass_error_mean": float(np.mean(settlement_errors)),
            "active_ece_mean": float(np.mean(active_eces)),
            "frontier_active_overprediction_mean": float(np.mean(frontier_over)),
            "initial_settlement_active_overprediction_mean": float(np.mean(initial_settlement_over)),
            "near_settlement_active_overprediction_mean": float(np.mean(near_settlement_over)),
            "per_seed_scores": per_seed,
        }
    quiet_patch_score = result["scenarios"]["quiet_patch"]["score_mean"]  # type: ignore[index]
    for scenario in SCENARIOS:
        result["scenarios"][scenario]["score_delta_vs_quiet_patch"] = float(  # type: ignore[index]
            result["scenarios"][scenario]["score_mean"] - quiet_patch_score  # type: ignore[index]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-round-out evaluation for the active-budget stage.")
    parser.add_argument("--history-dir", type=Path, default=Path("artifacts/history"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--rounds", type=int, nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.2)
    args = parser.parse_args()

    round_dirs = sorted(args.history_dir.glob("round_*"))
    round_numbers = {
        _round_detail_from_json(json.loads((round_dir / "round_detail.json").read_text())).round_number
        for round_dir in round_dirs
    }
    if args.rounds:
        round_numbers &= set(args.rounds)
    run_dirs = _discover_run_dirs(args.artifacts_dir, round_numbers)
    missing = sorted(round_numbers - set(run_dirs))
    if missing:
        raise SystemExit(f"Missing replay run directories for rounds: {missing}")

    results: list[dict[str, object]] = []
    for round_number in sorted(round_numbers):
        artifact = build_residual_calibrator_artifact_from_archive(
            args.history_dir,
            holdout_round_number=round_number,
            max_iter=args.max_iter,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
        )
        result = evaluate_round(
            run_dirs[round_number],
            args.history_dir,
            artifact=artifact,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "round_number": result["round_number"],
                    "round_bucket": result["round_bucket"],
                    "scores": {
                        scenario: result["scenarios"][scenario]["score_mean"]  # type: ignore[index]
                        for scenario in SCENARIOS
                    },
                },
                sort_keys=True,
            )
        )

    overall = {
        scenario: {
            "score_mean": float(np.mean([item["scenarios"][scenario]["score_mean"] for item in results])),
            "active_mass_error_mean": float(np.mean([item["scenarios"][scenario]["active_mass_error_mean"] for item in results])),
            "settlement_mass_error_mean": float(np.mean([item["scenarios"][scenario]["settlement_mass_error_mean"] for item in results])),
            "active_ece_mean": float(np.mean([item["scenarios"][scenario]["active_ece_mean"] for item in results])),
            "score_delta_vs_quiet_patch": float(
                np.mean([item["scenarios"][scenario]["score_delta_vs_quiet_patch"] for item in results])
            ),
        }
        for scenario in SCENARIOS
    }
    bucket_summary: dict[str, dict[str, float]] = {}
    for bucket in ("quiet", "mixed", "active"):
        bucket_items = [item for item in results if item["round_bucket"] == bucket]
        if not bucket_items:
            continue
        bucket_summary[bucket] = {
            scenario: float(np.mean([item["scenarios"][scenario]["score_mean"] for item in bucket_items]))
            for scenario in SCENARIOS
        }
    payload = {
        "rounds": results,
        "overall": overall,
        "bucket_summary": bucket_summary,
        "params": {
            "history_dir": str(args.history_dir),
            "artifacts_dir": str(args.artifacts_dir),
            "rounds": sorted(round_numbers),
            "max_iter": args.max_iter,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "learning_rate": args.learning_rate,
            "l2_regularization": args.l2_regularization,
        },
    }
    if args.output is not None:
        save_json(args.output, payload)
    print(json.dumps({"overall": overall, "bucket_summary": bucket_summary}, sort_keys=True))


if __name__ == "__main__":
    main()
