from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .aggregator import ObservationAggregator
from .api import AstarIslandAPI
from .config import AstarConfig
from .features import build_all_features
from .learned_prior import load_learned_prior_artifact
from .predictor import Predictor
from .prior_blend_gate import load_prior_blend_gate_artifact
from .priors import load_historical_prior_artifact
from .query_planner import QueryPlanner
from .residual_calibrator import load_residual_calibrator_artifact
from .utils import append_jsonl, save_json, setup_logging, to_jsonable

LOGGER = logging.getLogger(__name__)


def run_active_round(config: AstarConfig, *, submit: bool = True, make_plots: bool = True) -> Path:
    if make_plots:
        from .visualize import save_class_probability_maps, save_grid_image, save_heatmap

    run_dir = setup_logging(config.log_dir)
    api = AstarIslandAPI(config)
    active = api.get_active_round()
    if not active:
        raise RuntimeError("No active round found.")
    round_id = active["id"]
    detail = api.get_round_details(round_id, use_cache=False)
    LOGGER.info("Active round %s, number %s", detail.round_id, detail.round_number)
    save_json(run_dir / "round_detail.json", detail.raw)
    save_json(run_dir / "active_round.json", active)
    features = build_all_features(detail.initial_states)
    aggregator = ObservationAggregator(detail, features)
    planner = QueryPlanner(config.planner, detail, features)
    historical_priors = None
    learned_prior = None
    residual_calibrator = None
    prior_blend_gate = None
    observations_log: list[dict[str, object]] = []
    query_events_path = run_dir / "query_events.jsonl"
    if config.predictor.historical_prior_path.exists():
        historical_priors = load_historical_prior_artifact(config.predictor.historical_prior_path)
        LOGGER.info(
            "Loaded historical priors from %s using %s rounds and %s seeds",
            config.predictor.historical_prior_path,
            historical_priors.metadata.get("num_rounds", 0),
            historical_priors.metadata.get("num_seeds", 0),
        )
    else:
        LOGGER.info("No historical prior artifact found at %s; continuing without it", config.predictor.historical_prior_path)
    if config.predictor.learned_prior_path.exists():
        learned_prior = load_learned_prior_artifact(config.predictor.learned_prior_path)
        LOGGER.info(
            "Loaded learned prior from %s using %s rounds and %s seeds",
            config.predictor.learned_prior_path,
            learned_prior.metadata.get("num_rounds", 0),
            learned_prior.metadata.get("num_seeds", 0),
        )
    else:
        LOGGER.info("No learned prior artifact found at %s; continuing without it", config.predictor.learned_prior_path)
    if config.predictor.residual_calibrator_path.exists():
        residual_calibrator = load_residual_calibrator_artifact(config.predictor.residual_calibrator_path)
        LOGGER.info(
            "Loaded residual calibrator from %s using %s rounds and %s seeds",
            config.predictor.residual_calibrator_path,
            residual_calibrator.metadata.get("num_rounds", 0),
            residual_calibrator.metadata.get("num_seeds", 0),
        )
    else:
        LOGGER.info(
            "No residual calibrator artifact found at %s; continuing without it",
            config.predictor.residual_calibrator_path,
        )
    if config.predictor.prior_blend_gate_strength > 0.0:
        if config.predictor.prior_blend_gate_path.exists():
            prior_blend_gate = load_prior_blend_gate_artifact(config.predictor.prior_blend_gate_path)
            LOGGER.info(
                "Loaded prior blend gate from %s using %s rounds and %s seeds at strength %.3f",
                config.predictor.prior_blend_gate_path,
                prior_blend_gate.metadata.get("num_rounds", 0),
                prior_blend_gate.metadata.get("num_seeds", 0),
                config.predictor.prior_blend_gate_strength,
            )
        else:
            LOGGER.info(
                "No prior blend gate artifact found at %s; continuing without it",
                config.predictor.prior_blend_gate_path,
            )
    elif config.predictor.prior_blend_gate_path.exists():
        LOGGER.info(
            "Prior blend gate artifact exists at %s but is disabled by strength %.3f",
            config.predictor.prior_blend_gate_path,
            config.predictor.prior_blend_gate_strength,
        )

    for _ in range(config.planner.max_queries):
        step = planner.next_step(aggregator)
        aggregator.add_plan_step(step)
        LOGGER.info(
            "Query %s/%s seed=%s phase=%s viewport=(%s,%s,%s,%s) score=%.4f reason=%s",
            len(aggregator.query_history),
            config.planner.max_queries,
            step.seed_index,
            step.phase,
            step.viewport.x,
            step.viewport.y,
            step.viewport.w,
            step.viewport.h,
            step.score,
            step.reason,
        )
        observation = api.simulate(
            round_id=detail.round_id,
            seed_index=step.seed_index,
            viewport_x=step.viewport.x,
            viewport_y=step.viewport.y,
            viewport_w=step.viewport.w,
            viewport_h=step.viewport.h,
        )
        aggregator.add_observation(observation)
        event = {
            "query_index": len(aggregator.query_history),
            "planner": asdict(step),
            "observation": {
                "seed_index": observation.seed_index,
                "viewport": asdict(observation.viewport),
                "queries_used": observation.queries_used,
                "queries_max": observation.queries_max,
                "grid": to_jsonable(observation.grid),
                "class_grid": to_jsonable(observation.class_grid),
                "settlements": [asdict(item) for item in observation.settlements],
                "raw": to_jsonable(observation.raw),
            },
        }
        observations_log.append(event)
        append_jsonl(query_events_path, event)
        if observation.queries_used >= observation.queries_max:
            break

    predictor = Predictor(
        config.predictor,
        detail,
        features,
        historical_priors=historical_priors,
        learned_prior=learned_prior,
        residual_calibrator=residual_calibrator,
        prior_blend_gate=prior_blend_gate,
    )
    predictions, diagnostics = predictor.predict_round_with_diagnostics(aggregator)

    metadata = {
        "round_id": detail.round_id,
        "round_number": detail.round_number,
        "latent_summary": aggregator.round_latent_summary(),
        "query_history": [asdict(item) for item in aggregator.query_history],
        "historical_prior_metadata": historical_priors.metadata if historical_priors is not None else None,
        "learned_prior_metadata": learned_prior.metadata if learned_prior is not None else None,
        "residual_calibrator_metadata": residual_calibrator.metadata if residual_calibrator is not None else None,
        "prior_blend_gate_metadata": prior_blend_gate.metadata if prior_blend_gate is not None else None,
        "config": to_jsonable(_redact_config_snapshot(asdict(config))),
    }
    save_json(run_dir / "metadata.json", metadata)
    save_json(run_dir / "observations.json", observations_log)
    np.save(run_dir / "class_counts.npy", aggregator.class_counts)
    np.save(run_dir / "observation_counts.npy", aggregator.observation_counts)
    save_json(
        run_dir / "conditional_counts.json",
        {key: value.tolist() for key, value in aggregator.conditional_counts.items()},
    )
    _save_prediction_diagnostics(run_dir, diagnostics)

    for seed_index, prediction in predictions.items():
        np.save(run_dir / f"prediction_seed_{seed_index}.npy", prediction)
        if make_plots:
            save_grid_image(
                features[seed_index].initial_class_grid,
                run_dir / f"seed_{seed_index}_initial.png",
                f"Seed {seed_index} initial classes",
            )
            save_heatmap(
                aggregator.observation_counts[seed_index],
                run_dir / f"seed_{seed_index}_observations.png",
                f"Seed {seed_index} observation count",
            )
            save_class_probability_maps(prediction, run_dir, f"seed_{seed_index}")
        if submit:
            response = api.submit(detail.round_id, seed_index, prediction.tolist())
            LOGGER.info("Submission response for seed %s: %s", seed_index, json.dumps(response))
            append_jsonl(
                run_dir / "submission_events.jsonl",
                {"seed_index": seed_index, "response": response},
            )
    _verify_replay_bundle(
        run_dir,
        round_id=detail.round_id,
        round_number=detail.round_number,
        seeds_count=detail.seeds_count,
        query_count=len(aggregator.query_history),
        observation_events=len(observations_log),
        diagnostics=diagnostics,
    )
    return run_dir


def _save_prediction_diagnostics(run_dir: Path, diagnostics: dict[int, dict[str, object]]) -> None:
    diagnostics_dir = run_dir / "diagnostics"
    summary_index: dict[str, object] = {}
    for seed_index, payload in diagnostics.items():
        seed_dir = diagnostics_dir / f"seed_{seed_index}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        tensors = payload.get("tensors", {})
        for name, tensor in tensors.items():
            np.save(seed_dir / f"{name}.npy", tensor)
        summary = {key: value for key, value in payload.items() if key != "tensors"}
        save_json(seed_dir / "summary.json", summary)
        summary_index[str(seed_index)] = summary
    save_json(diagnostics_dir / "index.json", summary_index)


def _verify_replay_bundle(
    run_dir: Path,
    *,
    round_id: str,
    round_number: int,
    seeds_count: int,
    query_count: int,
    observation_events: int,
    diagnostics: dict[int, dict[str, object]],
) -> None:
    diagnostic_seed_ids = {str(key) for key in diagnostics.keys()}
    required = [
        run_dir / "round_detail.json",
        run_dir / "active_round.json",
        run_dir / "metadata.json",
        run_dir / "observations.json",
        run_dir / "query_events.jsonl",
        run_dir / "class_counts.npy",
        run_dir / "observation_counts.npy",
        run_dir / "conditional_counts.json",
        run_dir / "diagnostics" / "index.json",
    ]
    for seed_index in range(seeds_count):
        required.append(run_dir / f"prediction_seed_{seed_index}.npy")
        if str(seed_index) in diagnostic_seed_ids:
            required.append(run_dir / "diagnostics" / f"seed_{seed_index}" / "summary.json")

    missing = [str(path) for path in required if not path.exists()]
    manifest = {
        "round_id": round_id,
        "round_number": round_number,
        "seeds_count": seeds_count,
        "query_count": query_count,
        "observation_events": observation_events,
        "required_files": [str(path.relative_to(run_dir)) for path in required],
        "missing_files": missing,
        "complete": len(missing) == 0,
    }
    save_json(run_dir / "replay_manifest.json", manifest)
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Replay bundle incomplete for {run_dir}: missing {missing_str}")


def _redact_config_snapshot(config_dict: dict[str, object]) -> dict[str, object]:
    redacted = dict(config_dict)
    if "token" in redacted and redacted["token"] is not None:
        redacted["token"] = "<redacted>"
    return redacted
