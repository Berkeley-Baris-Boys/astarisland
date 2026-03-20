"""
World state, observation storage, and query strategy.

ObservationStore is the single source of truth during a round run.
It accumulates raw observations and settlement snapshots from every API call,
persists them to disk after each query, and exposes derived statistics for the
predictor and parameter estimator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    CODE_TO_CLASS, N_CLASSES, TOTAL_BUDGET, MAX_VIEWPORT,
    OBSERVATIONS_FILE, OCEAN_CODE, MOUNTAIN_CODE,
    SETTLEMENT_CODE, PORT_CODE, RUIN_CODE, FOREST_CODE, PLAINS_CODE,
)


# ── Terrain helpers ───────────────────────────────────────────────────────────

def code_to_class(raw_code: int) -> int:
    return CODE_TO_CLASS.get(raw_code, 0)


def make_class_grid(raw_grid: np.ndarray) -> np.ndarray:
    """Convert H×W raw-code grid to H×W prediction-class grid."""
    lookup = np.zeros(256, dtype=np.int8)
    for code, cls in CODE_TO_CLASS.items():
        if 0 <= code < 256:
            lookup[code] = cls
    # Clip raw values to [0,255] to be safe
    safe = np.clip(raw_grid, 0, 255).astype(np.uint8)
    return lookup[safe]


# ── Grid spatial helpers ──────────────────────────────────────────────────────

def coastal_mask(grid: np.ndarray) -> np.ndarray:
    """True for land cells adjacent to ocean (potential port locations)."""
    ocean = (grid == OCEAN_CODE)
    H, W = grid.shape
    adj = np.zeros((H, W), dtype=bool)
    adj[:-1, :] |= ocean[1:, :]
    adj[1:, :]  |= ocean[:-1, :]
    adj[:, :-1] |= ocean[:, 1:]
    adj[:, 1:]  |= ocean[:, :-1]
    return adj & ~ocean


def distance_to_coast(grid: np.ndarray) -> np.ndarray:
    """
    Euclidean distance (in cells) from each cell to the nearest ocean cell.
    Used as a continuous geographic feature — cells closer to coast are more
    likely to develop ports and be involved in trade/raiding.
    """
    try:
        from scipy.ndimage import distance_transform_edt
        ocean = (grid == OCEAN_CODE)
        return distance_transform_edt(~ocean).astype(np.float32)
    except ImportError:
        # Fallback: approximate with coastal_mask (1 = coastal, 0 = inland)
        coast = coastal_mask(grid)
        return (~coast).astype(np.float32)


def settlement_density(grid: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    Number of initial settlements within `radius` cells of each cell.
    High density areas tend to have more conflict and expansion dynamics.
    """
    settle = np.isin(grid, [SETTLEMENT_CODE, PORT_CODE]).astype(np.float32)
    try:
        from scipy.ndimage import uniform_filter
        # uniform_filter approximates counts within a square neighbourhood
        size = 2 * radius + 1
        density = uniform_filter(settle, size=size, mode="constant") * (size * size)
    except ImportError:
        density = settle  # fallback: just the cell itself
    return density.astype(np.float32)


def forest_adjacency(grid: np.ndarray) -> np.ndarray:
    """Number of forest cells (code 4) adjacent to each cell (8-neighbours)."""
    forest = (grid == 4).astype(np.int16)
    H, W = grid.shape
    adj = np.zeros((H, W), dtype=np.int16)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            r0, r1 = max(0, -dy), min(H, H - dy)
            c0, c1 = max(0, -dx), min(W, W - dx)
            adj[r0:r1, c0:c1] += forest[r0 + dy: r1 + dy, c0 + dx: c1 + dx]
    return adj


def settlement_influence(
    initial_state: dict, height: int, width: int, radius: int = 3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return three boolean masks from the initial state:
    - has_settlement[y,x]  → initial settlement location
    - has_port[y,x]        → initial port location
    - near_settlement[y,x] → within `radius` cells of any initial settlement
    """
    has_settle = np.zeros((height, width), dtype=bool)
    has_port   = np.zeros((height, width), dtype=bool)
    near       = np.zeros((height, width), dtype=bool)
    for s in initial_state.get("settlements", []):
        if not isinstance(s, dict):
            continue
        x, y = int(s.get("x", -1)), int(s.get("y", -1))
        if not (0 <= x < width and 0 <= y < height):
            continue
        has_settle[y, x] = True
        if s.get("has_port", False):
            has_port[y, x] = True
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        near[y0:y1, x0:x1] = True
    return has_settle, has_port, near


# ── Query planning ────────────────────────────────────────────────────────────

def tiling_offsets(size: int, window: int) -> list[int]:
    """Non-overlapping tile start positions that cover [0, size)."""
    if size <= window:
        return [0]
    offsets = list(range(0, size - window, window))
    offsets.append(size - window)   # ensure last tile reaches the edge
    return sorted(set(offsets))


def _seed_richness(initial_state: dict, width: int, height: int) -> float:
    """
    Score a seed by how "diagnostically rich" it is for regime inference.
    Rich = many initial settlements AND complex coastline.
    PDF recommendation: spend 25-30 queries on the 1-2 richest seeds.
    """
    grid = np.asarray(initial_state["grid"], dtype=np.int32)
    settlements = initial_state.get("settlements", [])
    n_settle = len(settlements)
    # Coastal cells = land cells adjacent to ocean
    coast = coastal_mask(grid)
    n_coast = int(coast.sum())
    return float(n_settle) * 2.0 + float(n_coast) * 0.3


def rank_viewports_by_interest(
    tiles: list[tuple[int, int, int, int]],
    initial_state: dict,
) -> list[tuple[int, int, int, int]]:
    """
    Sort tiles by "interestingness" based on initial map content.
    Priority (from PDF):
    1. Many initial settlements in window
    2. Much coastline (ports, longships, trade)
    3. Mix of forest + plains (food production + expansion opportunities)
    """
    grid = np.asarray(initial_state["grid"], dtype=np.int32)
    coast = coastal_mask(grid)
    forest = (grid == FOREST_CODE)
    plains = np.isin(grid, [0, PLAINS_CODE])
    settle_mask = np.isin(grid, [SETTLEMENT_CODE, PORT_CODE])

    scored: list[tuple[float, tuple]] = []
    for vp in tiles:
        vx, vy, vw, vh = vp
        region_settle  = float(settle_mask[vy:vy+vh, vx:vx+vw].sum())
        region_coast   = float(coast[vy:vy+vh, vx:vx+vw].sum())
        region_forest  = float(forest[vy:vy+vh, vx:vx+vw].sum())
        region_plains  = float(plains[vy:vy+vh, vx:vx+vw].sum())
        mix_bonus = min(region_forest, region_plains) * 0.3  # diversity
        score = region_settle * 3.0 + region_coast * 0.4 + mix_bonus
        scored.append((score, vp))

    scored.sort(key=lambda x: -x[0])
    return [vp for _, vp in scored]


def plan_core_queries(
    width: int,
    height: int,
    seeds_count: int,
    initial_states: Optional[list[dict]] = None,
    budget: int = TOTAL_BUDGET,
) -> list[tuple[int, int, int, int, int]]:
    """
    Plan queries with guaranteed full coverage first.

    For the standard 40x40 / 15x15 setup this means:
    - 45 queries for a single full pass over all 5 seeds
    - remaining queries reserved for adaptive re-sampling

    Seeds are still ordered by richness so that, if a run is interrupted early,
    the most informative windows are queried first.
    """
    xs = tiling_offsets(width, MAX_VIEWPORT)
    ys = tiling_offsets(height, MAX_VIEWPORT)
    all_tiles = [
        (x, y, min(MAX_VIEWPORT, width - x), min(MAX_VIEWPORT, height - y))
        for y in ys for x in xs
    ]
    n_tiles = len(all_tiles)  # typically 9 for 40×40

    if initial_states is None:
        # Fallback: even distribution, interleaved
        plan: list[tuple[int, int, int, int, int]] = []
        for tile_idx in range(n_tiles):
            for seed in range(seeds_count):
                x, y, w, h = all_tiles[tile_idx]
                plan.append((seed, x, y, w, h))
                if len(plan) >= min(budget, seeds_count * n_tiles):
                    return plan
        return plan

    richness = [
        (s, _seed_richness(initial_states[s], width, height))
        for s in range(seeds_count)
    ]
    richness.sort(key=lambda x: -x[1])
    ranked_seeds = [s for s, _ in richness]
    coverage_budget = min(budget, seeds_count * n_tiles)
    reserve_budget = max(0, budget - coverage_budget)
    print(
        f"Query plan: full coverage={coverage_budget} "
        f"reserve={reserve_budget} ranked_seeds={ranked_seeds}"
    )

    ordered_tiles = {
        seed: rank_viewports_by_interest(all_tiles, initial_states[seed])
        for seed in range(seeds_count)
    }

    plan: list[tuple[int, int, int, int, int]] = []
    for tile_idx in range(n_tiles):
        for seed in ranked_seeds:
            vp = ordered_tiles[seed][tile_idx]
            plan.append((seed, vp[0], vp[1], vp[2], vp[3]))
            if len(plan) >= coverage_budget:
                return plan

    return plan


def all_viewports(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """All valid 15×15 viewport positions (for reserve query selection)."""
    xs = tiling_offsets(width, MAX_VIEWPORT)
    ys = tiling_offsets(height, MAX_VIEWPORT)
    return [
        (x, y, min(MAX_VIEWPORT, width - x), min(MAX_VIEWPORT, height - y))
        for y in ys for x in xs
    ]


# ── Observation store ─────────────────────────────────────────────────────────

@dataclass
class SettlementSnapshot:
    """One settlement's state as returned by the API."""
    x: int
    y: int
    population: Optional[float]
    food: Optional[float]
    wealth: Optional[float]
    defense: Optional[float]
    has_port: bool
    alive: bool
    owner_id: Optional[int]

    @classmethod
    def from_dict(cls, d: dict) -> "SettlementSnapshot":
        return cls(
            x=int(d.get("x", -1)),
            y=int(d.get("y", -1)),
            population=d.get("population"),
            food=d.get("food"),
            wealth=d.get("wealth"),
            defense=d.get("defense"),
            has_port=bool(d.get("has_port", False)),
            alive=bool(d.get("alive", True)),
            owner_id=d.get("owner_id"),
        )

    def to_dict(self) -> dict:
        return {
            "x": self.x, "y": self.y,
            "population": self.population, "food": self.food,
            "wealth": self.wealth, "defense": self.defense,
            "has_port": self.has_port, "alive": self.alive,
            "owner_id": self.owner_id,
        }


@dataclass
class ObservationStore:
    """
    Accumulates all observations for a round.

    - latest[seed][y][x]  : last observed raw terrain code (or None)
    - counts[seed][y,x,c] : number of times class c was observed at (y,x)
    - settlement_snaps[seed] : list of (query_idx, vp_dict, [SettlementSnapshot])
    - query_log           : log of all executed queries
    - hidden_params       : estimated hidden parameters (populated after querying)
    """
    round_id: str
    width: int
    height: int
    seeds_count: int

    latest: dict[int, list[list[Optional[int]]]] = field(default_factory=dict)
    counts: dict[int, np.ndarray] = field(default_factory=dict)
    settlement_snaps: dict[int, list] = field(default_factory=dict)
    query_log: list[dict] = field(default_factory=list)
    hidden_params: dict[str, float] = field(default_factory=dict)
    queries_used: int = 0

    def __post_init__(self) -> None:
        for s in range(self.seeds_count):
            if s not in self.latest:
                self.latest[s] = [[None] * self.width for _ in range(self.height)]
            if s not in self.counts:
                self.counts[s] = np.zeros((self.height, self.width, N_CLASSES), dtype=np.int32)
            if s not in self.settlement_snaps:
                self.settlement_snaps[s] = []

    # ── Ingest an API simulate() response ────────────────────────────────────

    def ingest(
        self,
        seed: int,
        vx: int,
        vy: int,
        result: dict,
    ) -> int:
        """
        Record one viewport observation.

        Returns the number of newly-seen (previously None) cells.
        """
        grid = result.get("grid")
        if not isinstance(grid, list):
            raise ValueError(f"simulate response missing 'grid': {result}")

        query_index = self.queries_used + 1
        new_cells = 0
        for dy, row in enumerate(grid):
            for dx, raw_val in enumerate(row):
                if not isinstance(raw_val, int):
                    continue
                y, x = vy + dy, vx + dx
                if not (0 <= y < self.height and 0 <= x < self.width):
                    continue
                if self.latest[seed][y][x] is None:
                    new_cells += 1
                self.latest[seed][y][x] = raw_val
                cls = code_to_class(raw_val)
                self.counts[seed][y, x, cls] += 1

        # Settlement snapshots
        snaps_raw = result.get("settlements", [])
        if isinstance(snaps_raw, list):
            snaps = [
                SettlementSnapshot.from_dict(s)
                for s in snaps_raw
                if isinstance(s, dict)
            ]
            self.settlement_snaps[seed].append({
                "query_index": query_index,
                "viewport": {"x": vx, "y": vy,
                             "w": result.get("viewport", {}).get("w", MAX_VIEWPORT),
                             "h": result.get("viewport", {}).get("h", MAX_VIEWPORT)},
                "settlements": [s.to_dict() for s in snaps],
            })

        self.queries_used += 1
        self.query_log.append({
            "query_index": query_index,
            "seed": seed,
            "viewport": {"x": vx, "y": vy},
            "new_cells": new_cells,
        })
        return new_cells

    # ── Coverage & statistics ─────────────────────────────────────────────────

    def coverage(self, seed: int) -> float:
        samples = self.counts[seed].sum(axis=2)
        total = samples.size
        if total == 0:
            return 0.0
        seen = int((samples > 0).sum())
        return seen / total

    def observed_mask(self, seed: int) -> np.ndarray:
        """Boolean H×W mask: True where cell has been observed at least once."""
        return self.counts[seed].sum(axis=2) > 0

    def n_samples(self, seed: int) -> np.ndarray:
        """H×W int array of how many times each cell has been sampled."""
        return self.counts[seed].sum(axis=2)

    def settlement_stats(self, seed: int) -> dict[tuple[int, int], list[dict]]:
        """Per-position list of settlement stat dicts observed across all queries."""
        stats: dict[tuple[int, int], list[dict]] = {}
        for snap in self.settlement_snaps[seed]:
            for s in snap["settlements"]:
                key = (s["x"], s["y"])
                stats.setdefault(key, []).append(s)
        return stats

    def alive_rate(self, seed: int) -> float:
        """Fraction of observed settlements that were alive."""
        total = alive = 0
        for snap in self.settlement_snaps[seed]:
            for s in snap["settlements"]:
                total += 1
                if s.get("alive", True):
                    alive += 1
        return alive / total if total > 0 else 1.0

    def avg_food(self, seed: int) -> Optional[float]:
        foods = [
            s["food"]
            for snap in self.settlement_snaps[seed]
            for s in snap["settlements"]
            if s.get("food") is not None
        ]
        return float(np.mean(foods)) if foods else None

    def best_reserve_viewport(
        self,
        seed: int,
        viewports: list[tuple[int, int, int, int]],
        visited: dict[tuple, int],
    ) -> tuple[int, int, int, int]:
        """
        Pick the viewport with the highest average empirical cell entropy for
        re-sampling.  Penalise already-visited viewports to encourage coverage
        of the full variance landscape.

        Single-sample cells that were observed as Settlement, Port, or Ruin
        receive a priority bonus — those are the cells with the highest true
        uncertainty and the most to gain from a second independent observation.
        """
        obs_mask = self.observed_mask(seed)
        cnt = self.counts[seed]   # H×W×6

        # Cell entropy from count distribution
        totals = cnt.sum(axis=2, keepdims=True).astype(np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.where(totals > 0, cnt / totals, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            entropy_map = -np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0), axis=2)
        # Unobserved cells get baseline entropy
        entropy_map[~obs_mask] = np.log(N_CLASSES)

        # Resample bonus: single-observation cells that showed a dynamic class
        # (Settlement/Port/Ruin) carry the most unresolved uncertainty, so boost
        # their apparent entropy to attract reserve queries.
        _RESAMPLE_BONUS = {
            SETTLEMENT_CODE: 0.50,
            PORT_CODE:       0.60,
            RUIN_CODE:       0.45,
        }
        n_obs = cnt.sum(axis=2)  # H×W
        single_mask = (n_obs == 1)
        if single_mask.any():
            richness = np.zeros((self.height, self.width), dtype=np.float32)
            for y, row in enumerate(self.latest[seed]):
                for x, val in enumerate(row):
                    if val is not None and single_mask[y, x]:
                        bonus = _RESAMPLE_BONUS.get(int(val), 0.0)
                        if bonus > 0.0:
                            richness[y, x] = bonus
            entropy_map = entropy_map + richness

        best_score, best_vp = -1.0, viewports[0]
        for vp in viewports:
            vx, vy, vw, vh = vp
            region = entropy_map[vy:vy + vh, vx:vx + vw]
            unseen_frac = float((~obs_mask[vy:vy + vh, vx:vx + vw]).mean())
            penalty = visited.get(vp, 0) * 0.15
            score = float(region.mean()) + 2.0 * unseen_frac - penalty
            if score > best_score:
                best_score, best_vp = score, vp
        return best_vp

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = OBSERVATIONS_FILE) -> None:
        """Save entire store to JSON (crash-safe checkpoint)."""
        data = {
            "round_id":     self.round_id,
            "queries_used": self.queries_used,
            "width":        self.width,
            "height":       self.height,
            "seeds_count":  self.seeds_count,
            "hidden_params": self.hidden_params,
            "query_log":    self.query_log,
            "latest": {str(s): self.latest[s] for s in range(self.seeds_count)},
            "counts": {str(s): self.counts[s].tolist() for s in range(self.seeds_count)},
            "settlement_snaps": {str(s): self.settlement_snaps[s] for s in range(self.seeds_count)},
        }
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str = OBSERVATIONS_FILE) -> "ObservationStore":
        """Resume from a previously saved checkpoint."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(
            round_id=data["round_id"],
            width=int(data["width"]),
            height=int(data["height"]),
            seeds_count=int(data["seeds_count"]),
        )
        store.queries_used  = int(data.get("queries_used", 0))
        store.hidden_params = data.get("hidden_params", {})
        store.query_log     = data.get("query_log", [])

        for s in range(store.seeds_count):
            key = str(s)
            store.latest[s] = data["latest"][key]
            store.counts[s] = np.array(data["counts"][key], dtype=np.int32)
            store.settlement_snaps[s] = data.get("settlement_snaps", {}).get(key, [])

        return store

    # ── Transition analysis (for parameter estimation) ────────────────────────

    def transition_summary(
        self, initial_states: list[dict]
    ) -> dict[tuple[int, int], int]:
        """
        Count (initial_code, final_class) transitions across all observed cells.
        Used by the parameter estimator.
        """
        transitions: dict[tuple[int, int], int] = {}
        for s in range(self.seeds_count):
            init_grid = np.asarray(initial_states[s]["grid"], dtype=np.int32)
            for y, row in enumerate(self.latest[s]):
                for x, raw_val in enumerate(row):
                    if raw_val is None:
                        continue
                    init_code = int(init_grid[y, x])
                    final_cls  = code_to_class(raw_val)
                    key = (init_code, final_cls)
                    transitions[key] = transitions.get(key, 0) + 1
        return transitions

    def print_summary(self) -> None:
        print(f"\nObservations: {self.queries_used}/{TOTAL_BUDGET} queries used")
        for s in range(self.seeds_count):
            cov  = 100.0 * self.coverage(s)
            nsam = int(self.n_samples(s).sum())
            ar   = self.alive_rate(s)
            af   = self.avg_food(s)
            food_str = f", avg_food={af:.2f}" if af is not None else ""
            print(f"  Seed {s}: {cov:.1f}% coverage, {nsam} samples, "
                  f"alive_rate={ar:.1%}{food_str}")
