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
from astar_island.regime import compute_round_regime, regime_bucket, repeat_fraction_from_observation_counts
from astar_island.residual_calibrator import load_residual_calibrator_artifact
from astar_island.scoring import score_prediction


def _resolve_round_dir(history_dir: Path, round_number: int | None, round_id: str | None) -> Path | None:
    if round_number is not None and round_id is not None:
        candidate = history_dir / f"round_{round_number:02d}_{round_id}"
        if candidate.exists():
            return candidate
    if round_id is not None:
        matches = list(history_dir.glob(f"round_*_{round_id}"))
        if matches:
            return matches[0]
    if round_number is not None:
        matches = list(history_dir.glob(f"round_{round_number:02d}_*"))
        if matches:
            return matches[0]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved run directory using the current predictor.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("artifacts/history"),
        help="Directory containing archived official round analyses.",
    )
    parser.add_argument(
        "--no-diagnostics",
        action="store_true",
        help="Skip predictor diagnostics and replay only the final predictions.",
    )
    args = parser.parse_args()

    required = [
        args.run_dir / "round_detail.json",
        args.run_dir / "metadata.json",
        args.run_dir / "class_counts.npy",
        args.run_dir / "observation_counts.npy",
        args.run_dir / "conditional_counts.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Saved run is incomplete. Missing: {missing_str}")

    config = AstarConfig()
    detail = _round_detail_from_json(json.loads((args.run_dir / "round_detail.json").read_text()))
    features = build_all_features(
        detail.initial_states,
        settlement_sigma=config.predictor.settlement_sigma,
    )
    aggregator = ObservationAggregator(detail, features)
    aggregator.class_counts = np.load(args.run_dir / "class_counts.npy")
    aggregator.observation_counts = np.load(args.run_dir / "observation_counts.npy")
    aggregator.conditional_counts = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in json.loads((args.run_dir / "conditional_counts.json").read_text()).items()
    }

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

    predictor = Predictor(
        config.predictor,
        detail,
        features,
        historical_priors=historical,
        learned_prior=learned,
        residual_calibrator=residual,
        prior_blend_gate=prior_blend_gate,
    )
    latent = aggregator.round_latent_summary()
    round_regime = compute_round_regime(
        latent,
        config.predictor,
        repeat_fraction=repeat_fraction_from_observation_counts(aggregator.observation_counts),
    )
    if args.no_diagnostics:
        predictions = predictor.predict_round(aggregator)
        diagnostics: dict[int, dict[str, object]] = {}
    else:
        predictions, diagnostics = predictor.predict_round_with_diagnostics(aggregator)

    metadata = json.loads((args.run_dir / "metadata.json").read_text())
    round_dir = _resolve_round_dir(
        args.history_dir,
        metadata.get("round_number"),
        metadata.get("round_id"),
    )
    result: dict[str, object] = {
        "run_dir": str(args.run_dir),
        "round_number": metadata.get("round_number"),
        "round_id": metadata.get("round_id"),
        "history_round_dir": str(round_dir) if round_dir is not None else None,
        "round_regime": round_regime,
        "round_bucket": regime_bucket(round_regime["high_activity_factor"]),
    }

    if round_dir is None:
        print(json.dumps(result, indent=2))
        return

    scores: list[float] = []
    per_seed: dict[str, float] = {}
    for seed in range(detail.seeds_count):
        gt_path = round_dir / f"seed_{seed}" / "ground_truth.npy"
        if not gt_path.exists():
            continue
        ground_truth = np.load(gt_path)
        score = float(score_prediction(ground_truth, predictions[seed]))
        scores.append(score)
        per_seed[str(seed)] = score
    result["per_seed_scores"] = per_seed
    result["score_avg"] = float(np.mean(scores)) if scores else None

    if diagnostics:
        stage_scores: dict[str, float] = {}
        per_seed_round_regime: dict[str, object] = {}
        for seed, payload in diagnostics.items():
            per_seed_round_regime[str(seed)] = payload.get("round_regime", {})
        result["per_seed_round_regime"] = per_seed_round_regime
        common_stages = sorted(
            set.intersection(*[set(payload["tensors"].keys()) for payload in diagnostics.values()])  # type: ignore[index]
        )
        for stage in common_stages:
            values: list[float] = []
            for seed in range(detail.seeds_count):
                gt_path = round_dir / f"seed_{seed}" / "ground_truth.npy"
                if not gt_path.exists():
                    continue
                ground_truth = np.load(gt_path)
                tensor = diagnostics[seed]["tensors"][stage]  # type: ignore[index]
                if np.shape(tensor) != np.shape(ground_truth):
                    continue
                values.append(float(score_prediction(ground_truth, tensor)))
            if values:
                stage_scores[stage] = float(np.mean(values))
        result["stage_scores"] = stage_scores

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
