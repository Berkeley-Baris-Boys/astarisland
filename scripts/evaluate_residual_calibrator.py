from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from astar_island.config import AstarConfig
from astar_island.features import build_all_features
from astar_island.history import _round_detail_from_json
from astar_island.residual_calibrator import apply_residual_calibrator, build_residual_calibrator_artifact_from_archive
from astar_island.scoring import score_prediction
from astar_island.utils import load_json, normalize_probabilities, save_json


def parse_blends(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("At least one blend value is required.")
    return values


def evaluate_round(
    round_dir: Path,
    *,
    history_dir: Path,
    blends: list[float],
    max_iter: int,
    max_depth: int,
    min_samples_leaf: int,
    learning_rate: float,
    l2_regularization: float,
) -> dict[str, object]:
    detail = _round_detail_from_json(load_json(round_dir / "round_detail.json"))
    features = build_all_features(detail.initial_states)
    artifact = build_residual_calibrator_artifact_from_archive(
        history_dir,
        holdout_round_number=detail.round_number,
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
    )

    base_scores: list[float] = []
    blend_scores: dict[float, list[float]] = {blend: [] for blend in blends}
    seed_results: dict[str, object] = {}
    config = AstarConfig()

    for seed_index, seed_features in features.items():
        analysis_path = round_dir / f"seed_{seed_index}" / "analysis.json"
        if not analysis_path.exists():
            continue
        payload = load_json(analysis_path)
        prediction = np.asarray(payload["prediction"], dtype=np.float64)
        ground_truth = np.asarray(payload["ground_truth"], dtype=np.float64)
        base_score = score_prediction(ground_truth, prediction)
        base_scores.append(base_score)
        per_seed = {
            "base_score": base_score,
            "official_score": float(payload.get("score", base_score)),
            "blends": {},
        }

        for blend in blends:
            calibrated, details = apply_residual_calibrator(
                artifact,
                prediction,
                seed_features,
                blend=blend,
                min_probability=config.predictor.min_probability,
            )
            calibrated = normalize_probabilities(calibrated, config.predictor.min_probability)
            calibrated_score = score_prediction(ground_truth, calibrated)
            blend_scores[blend].append(calibrated_score)
            per_seed["blends"][str(blend)] = {
                "score": calibrated_score,
                "summary": details["summary"],
            }
        seed_results[str(seed_index)] = per_seed

    summary = {
        "round_id": detail.round_id,
        "round_number": detail.round_number,
        "base_mean": float(np.mean(base_scores)) if base_scores else 0.0,
        "blend_means": {
            str(blend): float(np.mean(scores)) if scores else 0.0 for blend, scores in blend_scores.items()
        },
        "num_seeds": len(base_scores),
        "artifact_metadata": artifact.metadata,
        "seeds": seed_results,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-round-out evaluation for the residual calibrator.")
    parser.add_argument("--history-dir", type=Path, default=None, help="Archive directory. Defaults to ASTAR_ISLAND_HISTORY_DIR or config.")
    parser.add_argument("--rounds", type=int, nargs="*", default=None, help="Optional round numbers to evaluate.")
    parser.add_argument("--blends", type=str, default="0.05,0.10,0.15,0.20,0.25,0.35", help="Comma-separated blend values.")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    config = AstarConfig()
    history_dir = config.history_dir if args.history_dir is None else args.history_dir
    blends = parse_blends(args.blends)

    round_dirs = sorted(history_dir.glob("round_*"))
    if args.rounds:
        wanted = set(args.rounds)
        round_dirs = [round_dir for round_dir in round_dirs if int(round_dir.name.split("_")[1]) in wanted]

    results = []
    for round_dir in round_dirs:
        result = evaluate_round(
            round_dir,
            history_dir=history_dir,
            blends=blends,
            max_iter=args.max_iter,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            learning_rate=args.learning_rate,
            l2_regularization=args.l2_regularization,
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "round_number": result["round_number"],
                    "base_mean": result["base_mean"],
                    "blend_means": result["blend_means"],
                },
                sort_keys=True,
            )
        )

    overall = {
        "base_mean": float(np.mean([item["base_mean"] for item in results])) if results else 0.0,
        "blend_means": {
            str(blend): float(np.mean([item["blend_means"][str(blend)] for item in results])) if results else 0.0
            for blend in blends
        },
    }
    payload = {
        "history_dir": str(history_dir),
        "rounds": results,
        "overall": overall,
        "params": {
            "blends": blends,
            "max_iter": args.max_iter,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "learning_rate": args.learning_rate,
            "l2_regularization": args.l2_regularization,
        },
    }
    if args.output is not None:
        save_json(args.output, payload)
    print(json.dumps({"overall": overall}, sort_keys=True))


if __name__ == "__main__":
    main()
