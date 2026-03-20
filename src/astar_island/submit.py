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
from .predictor import Predictor
from .priors import load_historical_prior_artifact
from .query_planner import QueryPlanner
from .utils import append_jsonl, save_json, setup_logging, to_jsonable
from .visualize import save_class_probability_maps, save_grid_image, save_heatmap

LOGGER = logging.getLogger(__name__)


def run_active_round(config: AstarConfig, *, submit: bool = True, make_plots: bool = True) -> Path:
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

    predictor = Predictor(config.predictor, detail, features, historical_priors=historical_priors)
    predictions = predictor.predict_round(aggregator)

    metadata = {
        "round_id": detail.round_id,
        "round_number": detail.round_number,
        "latent_summary": aggregator.round_latent_summary(),
        "query_history": [asdict(item) for item in aggregator.query_history],
        "historical_prior_metadata": historical_priors.metadata if historical_priors is not None else None,
    }
    save_json(run_dir / "metadata.json", metadata)
    save_json(run_dir / "observations.json", observations_log)
    np.save(run_dir / "class_counts.npy", aggregator.class_counts)
    np.save(run_dir / "observation_counts.npy", aggregator.observation_counts)
    save_json(
        run_dir / "conditional_counts.json",
        {key: value.tolist() for key, value in aggregator.conditional_counts.items()},
    )

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
    return run_dir
