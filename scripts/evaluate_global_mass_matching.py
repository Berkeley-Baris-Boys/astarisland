from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from astar_island.aggregator import ObservationAggregator
from astar_island.config import AstarConfig
from astar_island.features import build_all_features
from astar_island.history import _round_detail_from_json
from astar_island.learned_prior import load_learned_prior_artifact
from astar_island.predictor import Predictor
from astar_island.prior_blend_gate import load_prior_blend_gate_artifact
from astar_island.priors import load_historical_prior_artifact
from astar_island.residual_calibrator import load_residual_calibrator_artifact
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
    resolved: dict[int, Path] = {}
    for round_number, candidates in matches.items():
        resolved[round_number] = max(candidates, key=lambda path: path.stat().st_mtime)
    return resolved


def _load_predictor_artifacts(config: AstarConfig):
    historical = None
    learned = None
    residual = None
    prior_blend_gate = None
    if config.predictor.historical_prior_path.exists():
        historical = load_historical_prior_artifact(config.predictor.historical_prior_path)
    if config.predictor.learned_prior_path.exists():
        learned = load_learned_prior_artifact(config.predictor.learned_prior_path)
    if config.predictor.residual_calibrator_path.exists():
        residual = load_residual_calibrator_artifact(config.predictor.residual_calibrator_path)
    if config.predictor.prior_blend_gate_strength > 0.0 and config.predictor.prior_blend_gate_path.exists():
        prior_blend_gate = load_prior_blend_gate_artifact(config.predictor.prior_blend_gate_path)
    return historical, learned, residual, prior_blend_gate


def _buildable_mass_errors(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    buildable_mask: np.ndarray,
) -> dict[str, float]:
    buildable_count = float(np.sum(buildable_mask))
    if buildable_count <= 0.0:
        return {"active": 0.0, "settlement": 0.0, "forest": 0.0, "empty": 0.0}
    return {
        "active": float(
            abs(
                np.sum(prediction[..., 1:4][buildable_mask])
                - np.sum(ground_truth[..., 1:4][buildable_mask])
            )
            / buildable_count
        ),
        "settlement": float(abs(np.sum(prediction[..., 1][buildable_mask]) - np.sum(ground_truth[..., 1][buildable_mask])) / buildable_count),
        "forest": float(abs(np.sum(prediction[..., 4][buildable_mask]) - np.sum(ground_truth[..., 4][buildable_mask])) / buildable_count),
        "empty": float(abs(np.sum(prediction[..., 0][buildable_mask]) - np.sum(ground_truth[..., 0][buildable_mask])) / buildable_count),
    }


def _evaluate_run(
    run_dir: Path,
    history_dir: Path,
    *,
    mass_matching_strength: float,
    mass_matching_enable_nonactive: bool | None = None,
) -> dict[str, object]:
    config = AstarConfig()
    config.predictor.mass_matching_strength = mass_matching_strength
    if mass_matching_enable_nonactive is not None:
        config.predictor.mass_matching_enable_nonactive = mass_matching_enable_nonactive
    historical, learned, residual, prior_blend_gate = _load_predictor_artifacts(config)

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

    predictor = Predictor(
        config.predictor,
        detail,
        features,
        historical_priors=historical,
        learned_prior=learned,
        residual_calibrator=residual,
        prior_blend_gate=prior_blend_gate,
    )
    predictions, diagnostics = predictor.predict_round_with_diagnostics(aggregator)

    history_matches = list(history_dir.glob(f"round_{detail.round_number:02d}_*"))
    if not history_matches:
        raise FileNotFoundError(f"Could not find archived history for round {detail.round_number}")
    history_round_dir = history_matches[0]

    score_values: list[float] = []
    stage_scores: dict[str, list[float]] = {
        "post_confidence": [],
        "post_mass_matching": [],
        "final_prediction": [],
    }
    mass_errors: dict[str, list[float]] = {
        "active": [],
        "settlement": [],
        "forest": [],
        "empty": [],
    }

    for seed_index in range(detail.seeds_count):
        gt = np.load(history_round_dir / f"seed_{seed_index}" / "ground_truth.npy").astype(np.float64)
        pred = predictions[seed_index]
        score_values.append(float(score_prediction(gt, pred)))
        for stage_name in stage_scores:
            stage_scores[stage_name].append(float(score_prediction(gt, diagnostics[seed_index]["tensors"][stage_name])))
        seed_errors = _buildable_mass_errors(pred, gt, features[seed_index].buildable_mask)
        for key, value in seed_errors.items():
            mass_errors[key].append(value)

    return {
        "round_number": detail.round_number,
        "run_dir": str(run_dir),
        "score": float(np.mean(score_values)),
        "stage_scores": {key: float(np.mean(values)) for key, values in stage_scores.items()},
        "mass_error": {key: float(np.mean(values)) for key, values in mass_errors.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay R15-R18 with and without global mass matching.")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--history-dir", type=Path, default=Path("artifacts/history"))
    parser.add_argument("--rounds", type=int, nargs="*", default=list(TARGET_ROUNDS))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--candidate-strength", type=float, default=AstarConfig().predictor.mass_matching_strength)
    parser.add_argument(
        "--candidate-enable-nonactive",
        action="store_true",
        help="Enable the forest/empty non-active pass for the candidate run.",
    )
    args = parser.parse_args()

    rounds = set(args.rounds)
    run_dirs = _discover_run_dirs(args.artifacts_dir, rounds)
    missing = sorted(rounds - set(run_dirs))
    if missing:
        raise SystemExit(f"Missing run directories for rounds: {missing}")

    per_round: list[dict[str, object]] = []
    for round_number in sorted(rounds):
        baseline = _evaluate_run(run_dirs[round_number], args.history_dir, mass_matching_strength=0.0)
        candidate = _evaluate_run(
            run_dirs[round_number],
            args.history_dir,
            mass_matching_strength=args.candidate_strength,
            mass_matching_enable_nonactive=args.candidate_enable_nonactive,
        )
        per_round.append(
            {
                "round_number": round_number,
                "baseline_score": baseline["score"],
                "candidate_score": candidate["score"],
                "score_delta": float(candidate["score"] - baseline["score"]),
                "baseline_stage_scores": baseline["stage_scores"],
                "candidate_stage_scores": candidate["stage_scores"],
                "baseline_mass_error": baseline["mass_error"],
                "candidate_mass_error": candidate["mass_error"],
                "mass_error_delta": {
                    key: float(candidate["mass_error"][key] - baseline["mass_error"][key])
                    for key in baseline["mass_error"]
                },
                "run_dir": baseline["run_dir"],
            }
        )

    payload = {
        "rounds": per_round,
        "overall": {
            "baseline_score": float(np.mean([item["baseline_score"] for item in per_round])),
            "candidate_score": float(np.mean([item["candidate_score"] for item in per_round])),
            "score_delta": float(np.mean([item["score_delta"] for item in per_round])),
            "mass_error_delta": {
                key: float(np.mean([item["mass_error_delta"][key] for item in per_round]))
                for key in ("active", "settlement", "forest", "empty")
            },
        },
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
