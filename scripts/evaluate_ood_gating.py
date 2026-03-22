from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from astar_island.aggregator import ObservationAggregator
from astar_island.config import AstarConfig
from astar_island.features import build_all_features
from astar_island.history import _round_detail_from_json
from astar_island.learned_prior import build_learned_prior_artifact_from_archive
from astar_island.predictor import Predictor
from astar_island.prior_blend_gate import load_prior_blend_gate_artifact
from astar_island.priors import load_historical_prior_artifact
from astar_island.residual_calibrator import build_residual_calibrator_artifact_from_archive
from astar_island.scoring import score_prediction

TARGET_ROUNDS = (15, 16, 17, 18)


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
    return {round_number: max(candidates, key=lambda path: path.stat().st_mtime) for round_number, candidates in matches.items()}


def _load_static_artifacts(config: AstarConfig):
    historical = None
    prior_blend_gate = None
    if config.predictor.historical_prior_path.exists():
        historical = load_historical_prior_artifact(config.predictor.historical_prior_path)
    if config.predictor.prior_blend_gate_strength > 0.0 and config.predictor.prior_blend_gate_path.exists():
        prior_blend_gate = load_prior_blend_gate_artifact(config.predictor.prior_blend_gate_path)
    return historical, prior_blend_gate


def _evaluate_run(
    run_dir: Path,
    history_dir: Path,
    *,
    learned_prior_ood_enabled: bool,
    residual_ood_enabled: bool,
) -> dict[str, object]:
    config = AstarConfig()
    config.predictor.learned_prior_ood_enabled = learned_prior_ood_enabled
    config.predictor.residual_ood_enabled = residual_ood_enabled
    historical, prior_blend_gate = _load_static_artifacts(config)

    detail = _round_detail_from_json(json.loads((run_dir / "round_detail.json").read_text()))
    features = build_all_features(
        detail.initial_states,
        settlement_sigma=config.predictor.settlement_sigma,
    )
    aggregator = ObservationAggregator(detail, features)
    aggregator.class_counts = np.load(run_dir / "class_counts.npy")
    aggregator.observation_counts = np.load(run_dir / "observation_counts.npy")
    aggregator.conditional_counts = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in json.loads((run_dir / "conditional_counts.json").read_text()).items()
    }

    learned_prior = build_learned_prior_artifact_from_archive(
        history_dir,
        holdout_round_number=detail.round_number,
        maxiter=300,
    )
    residual_calibrator = build_residual_calibrator_artifact_from_archive(
        history_dir,
        holdout_round_number=detail.round_number,
        max_iter=300,
        max_depth=4,
        min_samples_leaf=300,
        learning_rate=0.05,
        l2_regularization=0.2,
    )

    predictor = Predictor(
        config.predictor,
        detail,
        features,
        historical_priors=historical,
        learned_prior=learned_prior,
        residual_calibrator=residual_calibrator,
        prior_blend_gate=prior_blend_gate,
    )
    predictions, diagnostics = predictor.predict_round_with_diagnostics(aggregator)

    history_matches = list(history_dir.glob(f"round_{detail.round_number:02d}_*"))
    if not history_matches:
        raise FileNotFoundError(f"Could not find archived history for round {detail.round_number}")
    history_round_dir = history_matches[0]

    score_values: list[float] = []
    learned_blends: list[float] = []
    residual_blends: list[float] = []
    learned_ood_scores: list[float] = []
    residual_ood_scores: list[float] = []
    for seed_index in range(detail.seeds_count):
        gt = np.load(history_round_dir / f"seed_{seed_index}" / "ground_truth.npy").astype(np.float64)
        score_values.append(float(score_prediction(gt, predictions[seed_index])))
        learned_diag = diagnostics[seed_index].get("learned_prior_ood")
        residual_diag = diagnostics[seed_index].get("residual_ood")
        if isinstance(learned_diag, dict):
            learned_blends.append(float(learned_diag.get("effective_blend", learned_diag.get("base_blend", 0.0))))
            learned_ood_scores.append(float(learned_diag.get("ood_score", 0.0)))
        if isinstance(residual_diag, dict):
            residual_blends.append(float(residual_diag.get("effective_blend_mean", residual_diag.get("base_blend_mean", 0.0))))
            residual_ood_scores.append(float(residual_diag.get("ood_score", 0.0)))

    return {
        "round_number": detail.round_number,
        "score": float(np.mean(score_values)),
        "learned_prior_effective_blend": float(np.mean(learned_blends)) if learned_blends else 0.0,
        "residual_effective_blend_mean": float(np.mean(residual_blends)) if residual_blends else 0.0,
        "learned_prior_ood_score": float(np.mean(learned_ood_scores)) if learned_ood_scores else 0.0,
        "residual_ood_score": float(np.mean(residual_ood_scores)) if residual_ood_scores else 0.0,
        "run_dir": str(run_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay R15-R18 with OOD-aware learned prior / residual attenuation.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--history-dir", type=Path, default=Path("artifacts/history"))
    parser.add_argument("--rounds", type=int, nargs="*", default=list(TARGET_ROUNDS))
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rounds = set(args.rounds)
    run_dirs = _discover_run_dirs(args.artifacts_dir, rounds)
    missing = sorted(rounds - set(run_dirs))
    if missing:
        raise SystemExit(f"Missing run directories for rounds: {missing}")

    scenarios = {
        "baseline": (False, False),
        "learned_prior_ood_only": (True, False),
        "residual_ood_only": (False, True),
        "both": (True, True),
    }

    per_round: list[dict[str, object]] = []
    for round_number in sorted(rounds):
        scenario_results = {
            name: _evaluate_run(
                run_dirs[round_number],
                args.history_dir,
                learned_prior_ood_enabled=flags[0],
                residual_ood_enabled=flags[1],
            )
            for name, flags in scenarios.items()
        }
        baseline = scenario_results["baseline"]
        per_round.append(
            {
                "round_number": round_number,
                "scenarios": scenario_results,
                "score_delta_vs_baseline": {
                    name: float(result["score"] - baseline["score"])
                    for name, result in scenario_results.items()
                    if name != "baseline"
                },
                "run_dir": baseline["run_dir"],
            }
        )

    payload = {
        "rounds": per_round,
        "overall": {
            "baseline_score": float(np.mean([item["scenarios"]["baseline"]["score"] for item in per_round])),
            "learned_prior_ood_only_score": float(np.mean([item["scenarios"]["learned_prior_ood_only"]["score"] for item in per_round])),
            "residual_ood_only_score": float(np.mean([item["scenarios"]["residual_ood_only"]["score"] for item in per_round])),
            "both_score": float(np.mean([item["scenarios"]["both"]["score"] for item in per_round])),
            "score_delta_vs_baseline": {
                "learned_prior_ood_only": float(np.mean([item["score_delta_vs_baseline"]["learned_prior_ood_only"] for item in per_round])),
                "residual_ood_only": float(np.mean([item["score_delta_vs_baseline"]["residual_ood_only"] for item in per_round])),
                "both": float(np.mean([item["score_delta_vs_baseline"]["both"] for item in per_round])),
            },
        },
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
