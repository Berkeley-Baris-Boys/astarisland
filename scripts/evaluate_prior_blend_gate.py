from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from astar_island.config import AstarConfig
from astar_island.features import build_all_features
from astar_island.history import _round_detail_from_json
from astar_island.learned_prior import load_learned_prior_artifact
from astar_island.predictor import Predictor
from astar_island.prior_blend_gate import apply_prior_blend_gate, build_prior_blend_gate_artifact_from_archive, infer_latent_summary_from_prediction
from astar_island.priors import load_historical_prior_artifact
from astar_island.scoring import score_prediction
from astar_island.utils import load_json, save_json


def evaluate_round(
    round_dir: Path,
    *,
    history_dir: Path,
    max_iter: int,
    max_depth: int,
    min_samples_leaf: int,
    learning_rate: float,
    l2_regularization: float,
) -> dict[str, object]:
    config = AstarConfig()
    detail = _round_detail_from_json(load_json(round_dir / "round_detail.json"))
    features = build_all_features(detail.initial_states)
    historical = load_historical_prior_artifact(config.predictor.historical_prior_path)
    learned = load_learned_prior_artifact(config.predictor.learned_prior_path)
    predictor = Predictor(
        config.predictor,
        detail,
        features,
        historical_priors=historical,
        learned_prior=learned,
        residual_calibrator=None,
    )
    artifact = build_prior_blend_gate_artifact_from_archive(
        history_dir,
        holdout_round_number=detail.round_number,
        max_iter=max_iter,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
    )

    prior_scores: list[float] = []
    base_scores: list[float] = []
    gated_scores: list[float] = []
    seed_results: dict[str, object] = {}

    for seed_index, seed_features in features.items():
        analysis_path = round_dir / f"seed_{seed_index}" / "analysis.json"
        if not analysis_path.exists():
            continue
        payload = load_json(analysis_path)
        prediction = np.asarray(payload["prediction"], dtype=np.float64)
        ground_truth = np.asarray(payload["ground_truth"], dtype=np.float64)
        latent = infer_latent_summary_from_prediction(prediction)
        prior = predictor._build_prior(seed_index, seed_features, None, latent)
        gated, details = apply_prior_blend_gate(
            artifact,
            prior,
            prediction,
            seed_features,
            min_probability=config.predictor.min_probability,
        )
        prior_score = score_prediction(ground_truth, prior)
        base_score = score_prediction(ground_truth, prediction)
        gated_score = score_prediction(ground_truth, gated)
        prior_scores.append(prior_score)
        base_scores.append(base_score)
        gated_scores.append(gated_score)
        seed_results[str(seed_index)] = {
            "prior_score": prior_score,
            "base_score": base_score,
            "official_score": float(payload.get("score", base_score)),
            "gated_score": gated_score,
            "gate_summary": details["summary"],
        }

    return {
        "round_id": detail.round_id,
        "round_number": detail.round_number,
        "prior_mean": float(np.mean(prior_scores)) if prior_scores else 0.0,
        "base_mean": float(np.mean(base_scores)) if base_scores else 0.0,
        "gated_mean": float(np.mean(gated_scores)) if gated_scores else 0.0,
        "num_seeds": len(base_scores),
        "artifact_metadata": artifact.metadata,
        "seeds": seed_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Leave-one-round-out evaluation for the prior blend gate.")
    parser.add_argument("--history-dir", type=Path, default=None, help="Archive directory. Defaults to config history dir.")
    parser.add_argument("--rounds", type=int, nargs="*", default=None, help="Optional round numbers to evaluate.")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.2)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    config = AstarConfig()
    history_dir = config.history_dir if args.history_dir is None else args.history_dir

    round_dirs = sorted(history_dir.glob("round_*"))
    if args.rounds:
        wanted = set(args.rounds)
        round_dirs = [round_dir for round_dir in round_dirs if int(round_dir.name.split("_")[1]) in wanted]

    results = []
    for round_dir in round_dirs:
        result = evaluate_round(
            round_dir,
            history_dir=history_dir,
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
                    "prior_mean": result["prior_mean"],
                    "base_mean": result["base_mean"],
                    "gated_mean": result["gated_mean"],
                },
                sort_keys=True,
            )
        )

    payload = {
        "history_dir": str(history_dir),
        "rounds": results,
        "overall": {
            "prior_mean": float(np.mean([item["prior_mean"] for item in results])) if results else 0.0,
            "base_mean": float(np.mean([item["base_mean"] for item in results])) if results else 0.0,
            "gated_mean": float(np.mean([item["gated_mean"] for item in results])) if results else 0.0,
        },
        "params": {
            "max_iter": args.max_iter,
            "max_depth": args.max_depth,
            "min_samples_leaf": args.min_samples_leaf,
            "learning_rate": args.learning_rate,
            "l2_regularization": args.l2_regularization,
        },
    }
    if args.output is not None:
        save_json(args.output, payload)
    print(json.dumps({"overall": payload["overall"]}, sort_keys=True))


if __name__ == "__main__":
    main()
