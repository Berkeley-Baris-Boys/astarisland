from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

NUM_CLASSES = 6
GRID_SIZE = 40

TERRAIN_EMPTY = 0
TERRAIN_SETTLEMENT = 1
TERRAIN_PORT = 2
TERRAIN_RUIN = 3
TERRAIN_FOREST = 4
TERRAIN_MOUNTAIN = 5
TERRAIN_OCEAN = 10
TERRAIN_PLAINS = 11

CLASS_EMPTY = 0
CLASS_SETTLEMENT = 1
CLASS_PORT = 2
CLASS_RUIN = 3
CLASS_FOREST = 4
CLASS_MOUNTAIN = 5

TERRAIN_TO_CLASS = {
    TERRAIN_EMPTY: CLASS_EMPTY,
    TERRAIN_SETTLEMENT: CLASS_SETTLEMENT,
    TERRAIN_PORT: CLASS_PORT,
    TERRAIN_RUIN: CLASS_RUIN,
    TERRAIN_FOREST: CLASS_FOREST,
    TERRAIN_MOUNTAIN: CLASS_MOUNTAIN,
    TERRAIN_OCEAN: CLASS_EMPTY,
    TERRAIN_PLAINS: CLASS_EMPTY,
}

CLASS_NAMES = ["empty", "settlement", "port", "ruin", "forest", "mountain"]


def terrain_grid_to_class_grid(grid: np.ndarray) -> np.ndarray:
    mapper = np.vectorize(lambda x: TERRAIN_TO_CLASS[int(x)], otypes=[np.int64])
    return mapper(grid)


@dataclass
class Settlement:
    x: int
    y: int
    has_port: bool
    alive: bool
    population: float | None = None
    food: float | None = None
    wealth: float | None = None
    defense: float | None = None
    owner_id: int | None = None
    tech_level: float | None = None
    longships: float | None = None

    @classmethod
    def from_api(cls, payload: dict[str, Any]) -> "Settlement":
        return cls(
            x=int(payload["x"]),
            y=int(payload["y"]),
            has_port=bool(payload.get("has_port", False)),
            alive=bool(payload.get("alive", True)),
            population=_maybe_float(payload.get("population")),
            food=_maybe_float(payload.get("food")),
            wealth=_maybe_float(payload.get("wealth")),
            defense=_maybe_float(payload.get("defense")),
            owner_id=payload.get("owner_id"),
            tech_level=_maybe_float(payload.get("tech_level")),
            longships=_maybe_float(payload.get("longships_owned")),
        )


def _maybe_float(value: Any) -> float | None:
    return None if value is None else float(value)


@dataclass
class InitialState:
    grid: np.ndarray
    settlements: list[Settlement]


@dataclass
class RoundDetail:
    round_id: str
    round_number: int
    status: str
    map_width: int
    map_height: int
    seeds_count: int
    initial_states: list[InitialState]
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class Viewport:
    x: int
    y: int
    w: int
    h: int

    def area(self) -> int:
        return self.w * self.h


@dataclass
class SimulationObservation:
    round_id: str
    seed_index: int
    viewport: Viewport
    grid: np.ndarray
    class_grid: np.ndarray
    settlements: list[Settlement]
    queries_used: int
    queries_max: int
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryPlanStep:
    seed_index: int
    viewport: Viewport
    reason: str
    score: float
    phase: str


@dataclass
class RoundBudget:
    round_id: str
    queries_used: int
    queries_max: int
    active: bool


@dataclass
class SeedFeatures:
    seed_index: int
    feature_stack: np.ndarray
    feature_names: list[str]
    buildable_mask: np.ndarray
    dynamic_prior_mask: np.ndarray
    coastal_mask: np.ndarray
    frontier_mask: np.ndarray
    conflict_mask: np.ndarray
    reclaimable_mask: np.ndarray
    initial_class_grid: np.ndarray
    initial_settlement_mask: np.ndarray


@dataclass
class RoundEvidence:
    round_id: str
    class_counts: np.ndarray
    observation_counts: np.ndarray
    observed_mask: np.ndarray
    settlement_stats: dict[str, list[float]]
    conditional_counts: dict[str, np.ndarray]
    query_history: list[QueryPlanStep]


@dataclass
class PredictionBundle:
    seed_predictions: dict[int, np.ndarray]
    metadata: dict[str, Any]
