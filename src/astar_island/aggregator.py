from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from .features import make_bucket_keys
from .types import (
    CLASS_FOREST,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_SETTLEMENT,
    NUM_CLASSES,
    QueryPlanStep,
    RoundDetail,
    RoundEvidence,
    SeedFeatures,
    SimulationObservation,
)
from .utils import entropy_from_counts, window_slices

LOGGER = logging.getLogger(__name__)


class ObservationAggregator:
    def __init__(self, round_detail: RoundDetail, seed_features: dict[int, SeedFeatures]):
        self.round_detail = round_detail
        self.seed_features = seed_features
        seeds = round_detail.seeds_count
        h, w = round_detail.map_height, round_detail.map_width
        self.class_counts = np.zeros((seeds, h, w, NUM_CLASSES), dtype=np.float64)
        self.observation_counts = np.zeros((seeds, h, w), dtype=np.float64)
        self.settlement_stats: dict[str, list[float]] = defaultdict(list)
        self.conditional_counts: dict[str, np.ndarray] = {}
        self.query_history: list[QueryPlanStep] = []

    def add_observation(self, observation: SimulationObservation) -> None:
        y_slice, x_slice = window_slices(
            observation.viewport.x,
            observation.viewport.y,
            observation.viewport.w,
            observation.viewport.h,
        )
        counts_view = np.eye(NUM_CLASSES, dtype=np.float64)[observation.class_grid]
        self.class_counts[observation.seed_index, y_slice, x_slice] += counts_view
        self.observation_counts[observation.seed_index, y_slice, x_slice] += 1.0
        self._update_conditional_counts(observation)
        self._update_settlement_stats(observation)

    def add_plan_step(self, step: QueryPlanStep) -> None:
        self.query_history.append(step)

    def _update_conditional_counts(self, observation: SimulationObservation) -> None:
        seed_keys = make_bucket_keys(self.seed_features[observation.seed_index])
        y_slice, x_slice = window_slices(
            observation.viewport.x,
            observation.viewport.y,
            observation.viewport.w,
            observation.viewport.h,
        )
        local_keys = seed_keys[y_slice, x_slice]
        for key in np.unique(local_keys):
            mask = local_keys == key
            counts = np.bincount(observation.class_grid[mask].ravel(), minlength=NUM_CLASSES).astype(np.float64)
            slot = self.conditional_counts.setdefault(str(int(key)), np.zeros(NUM_CLASSES, dtype=np.float64))
            slot += counts

    def _update_settlement_stats(self, observation: SimulationObservation) -> None:
        for settlement in observation.settlements:
            if settlement.population is not None:
                self.settlement_stats["population"].append(settlement.population)
            if settlement.food is not None:
                self.settlement_stats["food"].append(settlement.food)
            if settlement.wealth is not None:
                self.settlement_stats["wealth"].append(settlement.wealth)
            if settlement.defense is not None:
                self.settlement_stats["defense"].append(settlement.defense)
            self.settlement_stats["has_port"].append(1.0 if settlement.has_port else 0.0)
            self.settlement_stats["alive"].append(1.0 if settlement.alive else 0.0)

    def get_round_evidence(self) -> RoundEvidence:
        return RoundEvidence(
            round_id=self.round_detail.round_id,
            class_counts=self.class_counts.copy(),
            observation_counts=self.observation_counts.copy(),
            observed_mask=self.observation_counts > 0,
            settlement_stats={key: value[:] for key, value in self.settlement_stats.items()},
            conditional_counts={key: value.copy() for key, value in self.conditional_counts.items()},
            query_history=self.query_history[:],
        )

    def empirical_distribution(self, seed_index: int) -> np.ndarray:
        counts = self.class_counts[seed_index]
        totals = counts.sum(axis=-1, keepdims=True)
        return np.divide(counts, np.maximum(totals, 1e-12), where=totals > 0)

    def observed_entropy(self, seed_index: int) -> np.ndarray:
        return entropy_from_counts(self.class_counts[seed_index])

    def round_latent_summary(self) -> dict[str, float]:
        counts = self.class_counts.sum(axis=(0, 1, 2))
        total = float(np.sum(counts))
        observed_cells = float(np.sum(self.observation_counts > 0))
        settlement_mass = float(counts[CLASS_SETTLEMENT] + counts[CLASS_PORT])
        dynamic_mass = float(counts[CLASS_SETTLEMENT] + counts[CLASS_PORT] + counts[CLASS_RUIN] + counts[CLASS_FOREST])
        summary = {
            "observed_cells": observed_cells,
            "settlement_rate": settlement_mass / max(total, 1.0),
            "port_share_given_active": float(counts[CLASS_PORT]) / max(settlement_mass, 1.0),
            "ruin_share_given_active": float(counts[CLASS_RUIN]) / max(settlement_mass + counts[CLASS_RUIN], 1.0),
            "forest_share_dynamic": float(counts[CLASS_FOREST]) / max(dynamic_mass, 1.0),
            "mean_food": float(np.mean(self.settlement_stats.get("food", [0.0]))),
            "mean_wealth": float(np.mean(self.settlement_stats.get("wealth", [0.0]))),
            "mean_defense": float(np.mean(self.settlement_stats.get("defense", [0.0]))),
            "mean_population": float(np.mean(self.settlement_stats.get("population", [0.0]))),
        }
        LOGGER.info("Round latent summary: %s", summary)
        return summary
