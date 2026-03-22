from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency

    class _FallbackTqdm:
        """Minimal tqdm stand-in so callers can use set_postfix without tqdm installed."""

        def __init__(self, iterable=None, **_kwargs: object) -> None:
            if iterable is None:
                raise TypeError("tqdm fallback requires an iterable")
            self._iterable = iterable

        def __iter__(self):
            return iter(self._iterable)

        def set_postfix(self, *args: object, **kwargs: object) -> None:
            pass

        def update(self, *args: object, **kwargs: object) -> None:
            pass

    def tqdm(iterable=None, **kwargs: object) -> _FallbackTqdm:
        return _FallbackTqdm(iterable, **kwargs)

from astar_island.aggregator import ObservationAggregator
from astar_island.config import AstarConfig, PredictorConfig
from astar_island.features import build_all_features
from astar_island.history import _round_detail_from_json
from astar_island.learned_prior import load_learned_prior_artifact
from astar_island.predictor import Predictor
from astar_island.prior_blend_gate import load_prior_blend_gate_artifact
from astar_island.priors import load_historical_prior_artifact
from astar_island.regime import compute_round_regime, regime_bucket, repeat_fraction_from_observation_counts
from astar_island.residual_calibrator import load_residual_calibrator_artifact
from astar_island.scoring import score_collapsed_prediction, score_prediction
from astar_island.types import CLASS_FOREST, CLASS_SETTLEMENT, TERRAIN_FOREST, TERRAIN_PLAINS
from astar_island.utils import load_json, save_json


DEFAULT_SWEEPS: dict[str, list[float]] = {
    "settlement_sigma": [1.6, 2.2],
    "observation_smoothing_sigma": [1.4, 2.2],
    "settlement_intensity_blend": [0.22, 0.32],
    "settlement_focus_blend_high_activity": [0.35, 0.55],
    "plains_settlement_gain": [0.5, 0.8],
}


@dataclass(frozen=True)
class ReplayBundle:
    run_dir: Path
    round_id: str | None
    round_number: int | None
    history_round_dir: Path
    detail: object
    class_counts: np.ndarray
    observation_counts: np.ndarray
    conditional_counts: dict[str, np.ndarray]


@dataclass(frozen=True)
class ArtifactSet:
    historical: object | None
    learned: object | None
    residual: object | None
    prior_blend_gate: object | None


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


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_sweep(raw: str, base_predictor: PredictorConfig) -> tuple[str, list[object]]:
    if "=" not in raw:
        raise ValueError(f"Expected NAME=v1,v2,..., got: {raw}")
    name, values_raw = raw.split("=", 1)
    name = name.strip()
    if not hasattr(base_predictor, name):
        raise ValueError(f"Unknown PredictorConfig field: {name}")
    current = getattr(base_predictor, name)
    values: list[object] = []
    for item in parse_csv(values_raw):
        if isinstance(current, bool):
            lowered = item.lower()
            if lowered in {"1", "true", "yes", "on"}:
                values.append(True)
            elif lowered in {"0", "false", "no", "off"}:
                values.append(False)
            else:
                raise ValueError(f"Invalid boolean value for {name}: {item}")
        elif isinstance(current, int) and not isinstance(current, bool):
            values.append(int(item))
        elif isinstance(current, float):
            values.append(float(item))
        else:
            values.append(item)
    if not values:
        raise ValueError(f"No values provided for {name}")
    return name, values


def expand_candidates(
    base_predictor: PredictorConfig,
    sweeps: dict[str, list[object]],
    *,
    max_combinations: int,
) -> list[dict[str, object]]:
    names = sorted(sweeps)
    combinations = list(itertools.product(*(sweeps[name] for name in names)))
    if len(combinations) > max_combinations:
        raise ValueError(
            f"Refusing to evaluate {len(combinations)} combinations; increase --max-combinations if this is intentional."
        )
    return [dict(zip(names, combo)) for combo in combinations]


def load_bundle(run_dir: Path, history_dir: Path) -> ReplayBundle:
    required = [
        run_dir / "round_detail.json",
        run_dir / "metadata.json",
        run_dir / "class_counts.npy",
        run_dir / "observation_counts.npy",
        run_dir / "conditional_counts.json",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        missing_str = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Saved run is incomplete. Missing: {missing_str}")

    detail = _round_detail_from_json(load_json(run_dir / "round_detail.json"))
    metadata = load_json(run_dir / "metadata.json")
    round_dir = _resolve_round_dir(
        history_dir,
        metadata.get("round_number"),
        metadata.get("round_id"),
    )
    if round_dir is None:
        raise FileNotFoundError(
            f"Could not resolve archived round directory for run {run_dir} using history dir {history_dir}"
        )
    return ReplayBundle(
        run_dir=run_dir,
        round_id=metadata.get("round_id"),
        round_number=metadata.get("round_number"),
        history_round_dir=round_dir,
        detail=detail,
        class_counts=np.load(run_dir / "class_counts.npy"),
        observation_counts=np.load(run_dir / "observation_counts.npy"),
        conditional_counts={
            key: np.asarray(value, dtype=np.float64)
            for key, value in load_json(run_dir / "conditional_counts.json").items()
        },
    )


def build_aggregator(bundle: ReplayBundle, features: dict[int, object]) -> ObservationAggregator:
    aggregator = ObservationAggregator(bundle.detail, features)
    aggregator.class_counts = bundle.class_counts.copy()
    aggregator.observation_counts = bundle.observation_counts.copy()
    aggregator.conditional_counts = {
        key: value.copy()
        for key, value in bundle.conditional_counts.items()
    }
    return aggregator


def load_artifacts(config: AstarConfig, args: argparse.Namespace) -> ArtifactSet:
    historical = None
    learned = None
    residual = None
    prior_blend_gate = None
    if not args.disable_historical and config.predictor.historical_prior_path.exists():
        historical = load_historical_prior_artifact(config.predictor.historical_prior_path)
    if not args.disable_learned and config.predictor.learned_prior_path.exists():
        learned = load_learned_prior_artifact(config.predictor.learned_prior_path)
    if not args.disable_residual and config.predictor.residual_calibrator_path.exists():
        residual = load_residual_calibrator_artifact(config.predictor.residual_calibrator_path)
    if (
        not args.disable_prior_gate
        and config.predictor.prior_blend_gate_strength > 0.0
        and config.predictor.prior_blend_gate_path.exists()
    ):
        prior_blend_gate = load_prior_blend_gate_artifact(config.predictor.prior_blend_gate_path)
    return ArtifactSet(
        historical=historical,
        learned=learned,
        residual=residual,
        prior_blend_gate=prior_blend_gate,
    )


def evaluate_candidate(
    bundles: list[ReplayBundle],
    predictor_config: PredictorConfig,
    artifacts: ArtifactSet,
    *,
    features_cache: dict[tuple[Path, float], dict[int, object]],
    show_progress: bool = False,
) -> dict[str, object]:
    overall_scores: list[float] = []
    settlement_scores: list[float] = []
    forest_scores: list[float] = []
    bucket_scores: dict[str, list[float]] = {"quiet": [], "mixed": [], "active": []}
    terrain_scores: dict[str, list[float]] = {"plains": [], "forest": []}
    per_run: list[dict[str, object]] = []

    bundle_iter = bundles
    if show_progress:
        bundle_iter = tqdm(
            bundles,
            desc="Runs",
            leave=False,
        )
    for bundle in bundle_iter:
        cache_key = (bundle.run_dir, float(predictor_config.settlement_sigma))
        features = features_cache.get(cache_key)
        if features is None:
            features = build_all_features(
                bundle.detail.initial_states,
                settlement_sigma=predictor_config.settlement_sigma,
            )
            features_cache[cache_key] = features
        aggregator = build_aggregator(bundle, features)
        predictor = Predictor(
            predictor_config,
            bundle.detail,
            features,
            historical_priors=artifacts.historical,
            learned_prior=artifacts.learned,
            residual_calibrator=artifacts.residual,
            prior_blend_gate=artifacts.prior_blend_gate,
        )
        predictions = predictor.predict_round(aggregator)
        latent = aggregator.round_latent_summary()
        round_regime = compute_round_regime(
            latent,
            predictor_config,
            repeat_fraction=repeat_fraction_from_observation_counts(aggregator.observation_counts),
        )
        bucket = regime_bucket(round_regime["high_activity_factor"])

        run_overall: list[float] = []
        run_settlement: list[float] = []
        run_forest: list[float] = []
        run_plains_scores: list[float] = []
        run_forest_terrain_scores: list[float] = []
        for seed_index in range(bundle.detail.seeds_count):
            gt_path = bundle.history_round_dir / f"seed_{seed_index}" / "ground_truth.npy"
            if not gt_path.exists():
                continue
            ground_truth = np.load(gt_path)
            prediction = predictions[seed_index]
            run_overall.append(float(score_prediction(ground_truth, prediction)))
            run_settlement.append(
                float(score_collapsed_prediction(ground_truth, prediction, CLASS_SETTLEMENT))
            )
            run_forest.append(
                float(score_collapsed_prediction(ground_truth, prediction, CLASS_FOREST))
            )
            initial_grid = bundle.detail.initial_states[seed_index].grid
            plains_mask = initial_grid == TERRAIN_PLAINS
            forest_mask = initial_grid == TERRAIN_FOREST
            if np.any(plains_mask):
                run_plains_scores.append(float(score_prediction(ground_truth[plains_mask], prediction[plains_mask])))
            if np.any(forest_mask):
                run_forest_terrain_scores.append(float(score_prediction(ground_truth[forest_mask], prediction[forest_mask])))

        if run_overall:
            overall_scores.extend(run_overall)
            settlement_scores.extend(run_settlement)
            forest_scores.extend(run_forest)
            bucket_scores[bucket].extend(run_overall)
            terrain_scores["plains"].extend(run_plains_scores)
            terrain_scores["forest"].extend(run_forest_terrain_scores)
            per_run.append(
                {
                    "run_dir": str(bundle.run_dir),
                    "round_id": bundle.round_id,
                    "round_number": bundle.round_number,
                    "round_bucket": bucket,
                    "round_regime": round_regime,
                    "overall_mean": float(np.mean(run_overall)),
                    "settlement_mean": float(np.mean(run_settlement)),
                    "forest_mean": float(np.mean(run_forest)),
                    "plains_mean": float(np.mean(run_plains_scores)) if run_plains_scores else None,
                    "forest_terrain_mean": float(np.mean(run_forest_terrain_scores)) if run_forest_terrain_scores else None,
                    "num_scored_seeds": len(run_overall),
                }
            )

    return {
        "overall_mean": float(np.mean(overall_scores)) if overall_scores else 0.0,
        "settlement_mean": float(np.mean(settlement_scores)) if settlement_scores else 0.0,
        "forest_mean": float(np.mean(forest_scores)) if forest_scores else 0.0,
        "quiet_mean": float(np.mean(bucket_scores["quiet"])) if bucket_scores["quiet"] else None,
        "quiet_worst": float(np.min(bucket_scores["quiet"])) if bucket_scores["quiet"] else None,
        "mixed_mean": float(np.mean(bucket_scores["mixed"])) if bucket_scores["mixed"] else None,
        "active_mean": float(np.mean(bucket_scores["active"])) if bucket_scores["active"] else None,
        "terrain_means": {
            "plains": float(np.mean(terrain_scores["plains"])) if terrain_scores["plains"] else None,
            "forest": float(np.mean(terrain_scores["forest"])) if terrain_scores["forest"] else None,
        },
        "num_scored_seeds": len(overall_scores),
        "runs": per_run,
    }


def summarize_candidate(
    overrides: dict[str, object],
    metrics: dict[str, object],
    baseline: dict[str, object],
) -> dict[str, object]:
    overall_mean = float(metrics["overall_mean"])
    settlement_mean = float(metrics["settlement_mean"])
    forest_mean = float(metrics["forest_mean"])
    return {
        "overrides": overrides,
        "overall_mean": overall_mean,
        "settlement_mean": settlement_mean,
        "forest_mean": forest_mean,
        "quiet_mean": metrics["quiet_mean"],
        "quiet_worst": metrics["quiet_worst"],
        "mixed_mean": metrics["mixed_mean"],
        "active_mean": metrics["active_mean"],
        "terrain_means": metrics["terrain_means"],
        "overall_delta": overall_mean - float(baseline["overall_mean"]),
        "settlement_delta": settlement_mean - float(baseline["settlement_mean"]),
        "forest_delta": forest_mean - float(baseline["forest_mean"]),
        "num_scored_seeds": int(metrics["num_scored_seeds"]),
        "runs": metrics["runs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay saved runs across a grid of predictor settings and rank candidates by overall and settlement-only score."
    )
    parser.add_argument("run_dirs", type=Path, nargs="+", help="Saved run directories with replay bundles.")
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=Path("artifacts/history"),
        help="Directory containing archived official round analyses.",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=[],
        help="PredictorConfig sweep in the form name=v1,v2,... May be repeated.",
    )
    parser.add_argument("--max-combinations", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--disable-historical", action="store_true")
    parser.add_argument("--disable-learned", action="store_true")
    parser.add_argument("--disable-residual", action="store_true")
    parser.add_argument("--disable-prior-gate", action="store_true")
    args = parser.parse_args()

    config = AstarConfig()
    bundles = [load_bundle(run_dir, args.history_dir) for run_dir in args.run_dirs]
    artifacts = load_artifacts(config, args)

    if args.sweep:
        sweeps = dict(parse_sweep(item, config.predictor) for item in args.sweep)
    else:
        sweeps = dict(DEFAULT_SWEEPS)

    candidates = expand_candidates(
        config.predictor,
        sweeps,
        max_combinations=args.max_combinations,
    )
    features_cache: dict[tuple[Path, float], dict[int, object]] = {}

    baseline_metrics = evaluate_candidate(
        bundles,
        config.predictor,
        artifacts,
        features_cache=features_cache,
        show_progress=False,
    )
    baseline = {
        "overrides": {},
        "overall_mean": float(baseline_metrics["overall_mean"]),
        "settlement_mean": float(baseline_metrics["settlement_mean"]),
        "forest_mean": float(baseline_metrics["forest_mean"]),
        "quiet_mean": baseline_metrics["quiet_mean"],
        "quiet_worst": baseline_metrics["quiet_worst"],
        "mixed_mean": baseline_metrics["mixed_mean"],
        "active_mean": baseline_metrics["active_mean"],
        "terrain_means": baseline_metrics["terrain_means"],
        "num_scored_seeds": int(baseline_metrics["num_scored_seeds"]),
    }
    print(json.dumps({"baseline": baseline}, sort_keys=True))

    results: list[dict[str, object]] = []
    candidate_iter = tqdm(
        candidates,
        desc="Candidates",
    )
    for overrides in candidate_iter:
        predictor_config = replace(config.predictor, **overrides)
        candidate_iter.set_postfix(
            {
                "sigma": f"{predictor_config.settlement_sigma:.2f}",
                "obs": f"{predictor_config.observation_smoothing_sigma:.2f}",
            }
        )
        metrics = evaluate_candidate(
            bundles,
            predictor_config,
            artifacts,
            features_cache=features_cache,
            show_progress=len(bundles) > 1,
        )
        summary = summarize_candidate(overrides, metrics, baseline)
        results.append(summary)
        print(
            json.dumps(
                {
                    "overrides": overrides,
                    "overall_mean": summary["overall_mean"],
                    "settlement_mean": summary["settlement_mean"],
                    "forest_mean": summary["forest_mean"],
                    "quiet_mean": summary["quiet_mean"],
                    "quiet_worst": summary["quiet_worst"],
                    "active_mean": summary["active_mean"],
                    "terrain_means": summary["terrain_means"],
                    "overall_delta": summary["overall_delta"],
                    "settlement_delta": summary["settlement_delta"],
                    "forest_delta": summary["forest_delta"],
                },
                sort_keys=True,
            )
        )

    by_settlement = sorted(
        results,
        key=lambda item: (item["settlement_mean"], item["overall_mean"]),
        reverse=True,
    )
    by_overall = sorted(
        results,
        key=lambda item: (item["overall_mean"], item["settlement_mean"]),
        reverse=True,
    )
    payload = {
        "history_dir": str(args.history_dir),
        "run_dirs": [str(path) for path in args.run_dirs],
        "artifact_usage": {
            "historical": artifacts.historical is not None,
            "learned": artifacts.learned is not None,
            "residual": artifacts.residual is not None,
            "prior_blend_gate": artifacts.prior_blend_gate is not None,
        },
        "baseline": baseline,
        "sweeps": {name: list(values) for name, values in sweeps.items()},
        "num_candidates": len(candidates),
        "top_by_settlement": by_settlement[: args.top_k],
        "top_by_overall": by_overall[: args.top_k],
        "all_results": results,
    }
    if args.output is not None:
        save_json(args.output, payload)
    print(
        json.dumps(
            {
                "top_by_settlement": payload["top_by_settlement"][: min(args.top_k, 3)],
                "top_by_overall": payload["top_by_overall"][: min(args.top_k, 3)],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
