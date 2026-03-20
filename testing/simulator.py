"""
Norse world simulator for Astar Island.

Implements all 5 simulation phases described in the task mechanics:
  Growth → Conflict → Trade → Winter → Environment

Run standalone to sanity-check:
  python simulator.py
"""

from __future__ import annotations

import dataclasses
import random
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Terrain codes (match the API)
# ---------------------------------------------------------------------------
OCEAN      = 10
PLAINS     = 11
EMPTY      = 0
SETTLEMENT = 1
PORT       = 2
RUIN       = 3
FOREST     = 4
MOUNTAIN   = 5

STATIC_TERRAIN = {OCEAN, MOUNTAIN}

# class index for prediction tensor
TERRAIN_TO_CLASS = {
    OCEAN: 0, PLAINS: 0, EMPTY: 0,
    SETTLEMENT: 1, PORT: 2, RUIN: 3,
    FOREST: 4, MOUNTAIN: 5,
}


# ---------------------------------------------------------------------------
# Simulator parameters  (hidden per round, shared across seeds)
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class SimParams:
    # Growth
    forest_food_bonus: float = 2.0      # food gained per adjacent forest per year
    base_food_production: float = 3.0   # food every settlement produces regardless
    growth_threshold: float = 15.0      # food needed to grow population
    expansion_threshold: float = 12.0   # food needed to found a new settlement
    expansion_prob: float = 0.40        # probability of expanding when threshold met
    expansion_max_range: int = 3        # max Chebyshev distance for founding new settlement
    port_formation_prob: float = 0.20   # prob of forming port if coastal + food > threshold

    # Conflict
    raid_food_threshold: float = 5.0    # food below which a settlement raids
    raid_damage: float = 8.0            # food stolen per successful raid
    raid_range: float = 6.0             # max Chebyshev distance for land raids
    longship_range_bonus: float = 6.0   # extra range for settlements with longships
    longship_prob: float = 0.15         # prob of building longship if wealthy
    conquest_prob: float = 0.10         # prob raided settlement switches faction

    # Trade
    trade_range: float = 8.0            # max distance for port-to-port trade
    trade_food_bonus: float = 3.0       # food gained per trade partner per year
    trade_wealth_bonus: float = 2.0     # wealth gained per trade partner

    # Winter
    base_winter_cost: float = 5.0       # food lost per winter
    winter_severity: float = 0.3        # std dev of winter cost (as fraction of base)
    pop_winter_scale: float = 0.10      # extra food cost per unit of population
    starvation_threshold: float = 0.0   # food below this → collapse risk
    starvation_collapse_prob: float = 0.6
    collapse_to_forest_prob: float = 0.35  # prob collapse → Forest if adjacent to forest
    direct_collapse_prob: float = 0.55     # of remaining: prob collapse → Plains vs Ruin

    # Environment
    forest_reclaim_prob: float = 0.03   # prob ruin→forest per check (capped, not multiplied)
    forest_spread_prob: float = 0.001   # prob plains→forest if forest adjacent (per year)
    forest_depletion_prob: float = 0.003 # prob forest→plains per adjacent settlement (per year)
    ruin_rebuild_prob: float = 0.15     # prob ruin→settlement if thriving settle nearby
    ruin_decay_prob: float = 0.10       # prob ruin→empty/plains if isolated

    # Initial settlement stats
    init_population: float = 10.0
    init_food: float = 30.0
    init_wealth: float = 10.0
    init_defense: float = 5.0

    def randomise(self, rng: np.random.Generator, noise: float = 0.25) -> "SimParams":
        """Return a copy with all float fields multiplied by ~N(1, noise)."""
        d = dataclasses.asdict(self)
        for k, v in d.items():
            if isinstance(v, float) and v > 0:
                d[k] = float(np.clip(v * rng.normal(1.0, noise), 0.01, v * 4))
            # int fields (like expansion_max_range) are kept unchanged
        return SimParams(**d)


# ---------------------------------------------------------------------------
# Settlement entity
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Settlement:
    x: int
    y: int
    owner_id: int
    population: float
    food: float
    wealth: float
    defense: float
    tech_level: float = 1.0
    has_port: bool = False
    has_longship: bool = False
    alive: bool = True

    def effective_raid_range(self, params: SimParams) -> float:
        base = params.raid_range
        if self.has_longship:
            base += params.longship_range_bonus
        return base


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def chebyshev(x1: int, y1: int, x2: int, y2: int) -> float:
    return float(max(abs(x1 - x2), abs(y1 - y2)))


def adjacent_cells(x: int, y: int, width: int, height: int) -> List[Tuple[int, int]]:
    cells = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                cells.append((nx, ny))
    return cells


def is_coastal(x: int, y: int, grid: np.ndarray, width: int, height: int) -> bool:
    for nx, ny in adjacent_cells(x, y, width, height):
        if grid[ny, nx] == OCEAN:
            return True
    # Also treat border cells as coastal
    return x == 0 or x == width - 1 or y == 0 or y == height - 1


def count_adjacent_terrain(
    x: int, y: int, terrain: int, grid: np.ndarray, width: int, height: int
) -> int:
    return sum(
        1 for nx, ny in adjacent_cells(x, y, width, height)
        if grid[ny, nx] == terrain
    )


def find_empty_adjacent(
    x: int, y: int, grid: np.ndarray, width: int, height: int,
    rng: np.random.Generator,
) -> Optional[Tuple[int, int]]:
    """Find a random empty/plains/forest cell adjacent to (x, y)."""
    candidates = [
        (nx, ny)
        for nx, ny in adjacent_cells(x, y, width, height)
        if grid[ny, nx] in (EMPTY, PLAINS, RUIN, FOREST)
    ]
    if not candidates:
        return None
    idx = rng.integers(len(candidates))
    return candidates[idx]


def find_expansion_cell(
    x: int, y: int, grid: np.ndarray, width: int, height: int,
    max_range: int, rng: np.random.Generator,
) -> Optional[Tuple[int, int]]:
    """Find a suitable cell within max_range Chebyshev distance for founding a settlement."""
    expandable = {EMPTY, PLAINS, RUIN, FOREST}
    candidates: List[Tuple[int, int]] = []
    for dy in range(-max_range, max_range + 1):
        for dx in range(-max_range, max_range + 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and grid[ny, nx] in expandable:
                candidates.append((nx, ny))
    if not candidates:
        return None
    idx = rng.integers(len(candidates))
    return candidates[idx]


# ---------------------------------------------------------------------------
# Simulation phases
# ---------------------------------------------------------------------------

def phase_growth(
    settlements: List[Settlement],
    grid: np.ndarray,
    params: SimParams,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> None:
    """Settlements produce food, may grow, expand, form ports, build longships."""
    new_settlements: List[Settlement] = []

    for s in settlements:
        if not s.alive:
            continue

        # Food production
        forests = count_adjacent_terrain(s.x, s.y, FOREST, grid, width, height)
        s.food += params.base_food_production + forests * params.forest_food_bonus

        # Population growth
        if s.food >= params.growth_threshold:
            s.population += rng.uniform(0.5, 1.5)
            s.food -= params.growth_threshold * 0.3

        # Port formation
        if (not s.has_port
                and s.food >= params.growth_threshold
                and is_coastal(s.x, s.y, grid, width, height)
                and rng.random() < params.port_formation_prob):
            s.has_port = True
            grid[s.y, s.x] = PORT

        # Longship
        if (not s.has_longship
                and s.has_port
                and s.wealth >= 15
                and rng.random() < params.longship_prob):
            s.has_longship = True

        # Expansion: found new settlement (searches up to expansion_max_range)
        if (s.food >= params.expansion_threshold
                and s.population >= 5
                and rng.random() < params.expansion_prob):
            cell = find_expansion_cell(
                s.x, s.y, grid, width, height, params.expansion_max_range, rng,
            )
            if cell is not None:
                nx, ny = cell
                child = Settlement(
                    x=nx, y=ny,
                    owner_id=s.owner_id,
                    population=s.population * 0.3,
                    food=s.food * 0.3,
                    wealth=s.wealth * 0.2,
                    defense=params.init_defense,
                )
                s.population *= 0.7
                s.food *= 0.7
                s.wealth *= 0.8
                grid[ny, nx] = SETTLEMENT
                new_settlements.append(child)

    settlements.extend(new_settlements)


def phase_conflict(
    settlements: List[Settlement],
    grid: np.ndarray,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """Desperate settlements raid neighbours."""
    alive = [s for s in settlements if s.alive]

    for raider in alive:
        if raider.food >= params.raid_food_threshold:
            continue

        # Find targets in range
        raid_range = raider.effective_raid_range(params)
        targets = [
            t for t in alive
            if t is not raider
            and t.owner_id != raider.owner_id
            and chebyshev(raider.x, raider.y, t.x, t.y) <= raid_range
        ]
        if not targets:
            continue

        # Pick weakest target
        target = min(targets, key=lambda t: t.defense + t.population)

        # Raid outcome
        success_prob = 0.5 + 0.1 * (raider.population - target.defense) / max(target.population, 1)
        success_prob = float(np.clip(success_prob, 0.1, 0.9))

        if rng.random() < success_prob:
            stolen = min(params.raid_damage, target.food)
            raider.food += stolen
            raider.wealth += stolen * 0.5
            target.food -= stolen
            target.defense = max(0, target.defense - 1)

            # Conquest
            if rng.random() < params.conquest_prob:
                target.owner_id = raider.owner_id


def phase_trade(
    settlements: List[Settlement],
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """Ports within range trade with each other."""
    ports = [s for s in settlements if s.alive and s.has_port]

    for port in ports:
        partners = [
            p for p in ports
            if p is not port
            and chebyshev(port.x, port.y, p.x, p.y) <= params.trade_range
        ]
        if not partners:
            continue

        n = len(partners)
        port.food   += params.trade_food_bonus   * n
        port.wealth += params.trade_wealth_bonus * n
        # Tech diffusion
        if partners:
            max_tech = max(p.tech_level for p in partners)
            port.tech_level += 0.05 * (max_tech - port.tech_level)


def phase_winter(
    settlements: List[Settlement],
    grid: np.ndarray,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """Winter costs food (scaled by population); starvation collapses settlements."""
    for s in settlements:
        if not s.alive:
            continue

        severity = rng.normal(1.0, params.winter_severity)
        severity = max(0.1, severity)
        cost = params.base_winter_cost * severity + s.population * params.pop_winter_scale
        s.food -= cost

        if s.food <= params.starvation_threshold:
            if rng.random() < params.starvation_collapse_prob:
                _collapse(s, grid, params, rng)


def _collapse(
    s: Settlement, grid: np.ndarray, params: SimParams, rng: np.random.Generator,
) -> None:
    s.alive = False
    h, w = grid.shape
    forest_adj = count_adjacent_terrain(s.x, s.y, FOREST, grid, w, h)
    if forest_adj > 0 and rng.random() < params.collapse_to_forest_prob:
        grid[s.y, s.x] = FOREST
    elif rng.random() < params.direct_collapse_prob:
        grid[s.y, s.x] = PLAINS
    else:
        grid[s.y, s.x] = RUIN


def phase_environment(
    settlements: List[Settlement],
    grid: np.ndarray,
    params: SimParams,
    width: int,
    height: int,
    rng: np.random.Generator,
) -> None:
    """Ruin decay, forest spread, forest depletion by settlements."""
    height_g, width_g = grid.shape

    # --- Ruin processing ---
    ruin_cells = list(zip(*np.where(grid == RUIN)))
    for ry, rx in ruin_cells:
        # Forest reclaim (single check, not multiplied by adjacency count)
        forest_adj = count_adjacent_terrain(rx, ry, FOREST, grid, width_g, height_g)
        if forest_adj > 0 and rng.random() < params.forest_reclaim_prob:
            grid[ry, rx] = FOREST
            continue

        # Settlement rebuild from nearby thriving settlement
        rebuild = False
        for s in settlements:
            if s.alive and s.food >= params.expansion_threshold * 0.8:
                if chebyshev(rx, ry, s.x, s.y) <= 3:
                    if rng.random() < params.ruin_rebuild_prob:
                        child = Settlement(
                            x=rx, y=ry,
                            owner_id=s.owner_id,
                            population=s.population * 0.2,
                            food=s.food * 0.2,
                            wealth=s.wealth * 0.1,
                            defense=params.init_defense * 0.5,
                            has_port=is_coastal(rx, ry, grid, width_g, height_g),
                        )
                        grid[ry, rx] = PORT if child.has_port else SETTLEMENT
                        settlements.append(child)
                        rebuild = True
                        break
        if rebuild:
            continue

        # Decay to plains
        if rng.random() < params.ruin_decay_prob:
            grid[ry, rx] = PLAINS

    # --- Forest spread: plains/empty near forest may become forest (vectorized) ---
    forest_mask = grid == FOREST
    has_forest_neighbor = np.zeros_like(forest_mask)
    if height_g > 1:
        has_forest_neighbor[:-1, :] |= forest_mask[1:, :]
        has_forest_neighbor[1:, :]  |= forest_mask[:-1, :]
    if width_g > 1:
        has_forest_neighbor[:, :-1] |= forest_mask[:, 1:]
        has_forest_neighbor[:, 1:]  |= forest_mask[:, :-1]
    plains_mask = (grid == PLAINS) | (grid == EMPTY)
    spread_candidates = plains_mask & has_forest_neighbor
    n_spread = int(spread_candidates.sum())
    if n_spread > 0:
        rolls = rng.random(n_spread)
        ys, xs = np.where(spread_candidates)
        grid[ys[rolls < params.forest_spread_prob], xs[rolls < params.forest_spread_prob]] = FOREST

    # --- Forest depletion: active settlements consume adjacent forest (vectorized) ---
    alive_list = [(s.x, s.y) for s in settlements if s.alive]
    if alive_list:
        settle_mask = np.zeros((height_g, width_g), dtype=np.int32)
        for sx, sy in alive_list:
            settle_mask[sy, sx] = 1
        settle_neighbor_count = np.zeros((height_g, width_g), dtype=np.int32)
        if height_g > 1:
            settle_neighbor_count[:-1, :] += settle_mask[1:, :]
            settle_neighbor_count[1:, :]  += settle_mask[:-1, :]
        if width_g > 1:
            settle_neighbor_count[:, :-1] += settle_mask[:, 1:]
            settle_neighbor_count[:, 1:]  += settle_mask[:, :-1]
        forest_mask = grid == FOREST
        deplete_candidates = forest_mask & (settle_neighbor_count > 0)
        n_deplete = int(deplete_candidates.sum())
        if n_deplete > 0:
            ys, xs = np.where(deplete_candidates)
            probs = params.forest_depletion_prob * settle_neighbor_count[ys, xs].astype(np.float64)
            rolls = rng.random(n_deplete)
            grid[ys[rolls < probs], xs[rolls < probs]] = PLAINS


# ---------------------------------------------------------------------------
# Single full simulation run
# ---------------------------------------------------------------------------

def run_simulation(
    initial_grid: np.ndarray,
    initial_settlements_data: List[Dict],
    params: SimParams,
    rng: np.random.Generator,
    n_years: int = 50,
) -> np.ndarray:
    """
    Run one 50-year simulation from the given initial state.

    Args:
        initial_grid: (H, W) int array of terrain codes
        initial_settlements_data: list of settlement dicts from the API
            each has keys: x, y, has_port, alive
        params: hidden simulator parameters
        rng: numpy random generator (for reproducibility)
        n_years: number of simulation steps

    Returns:
        (H, W) int array of final terrain codes
    """
    grid = initial_grid.copy()
    height, width = grid.shape

    # Initialise settlements from API data
    settlements: List[Settlement] = []
    for i, sd in enumerate(initial_settlements_data):
        if not sd.get("alive", True):
            continue
        s = Settlement(
            x=int(sd["x"]),
            y=int(sd["y"]),
            owner_id=i,   # each starts as its own faction
            population=params.init_population,
            food=params.init_food,
            wealth=params.init_wealth,
            defense=params.init_defense,
            has_port=bool(sd.get("has_port", False)),
        )
        # Sync grid
        grid[s.y, s.x] = PORT if s.has_port else SETTLEMENT
        settlements.append(s)

    # Run years
    for _ in range(n_years):
        phase_growth(settlements, grid, params, width, height, rng)
        phase_conflict(settlements, grid, params, rng)
        phase_trade(settlements, params, rng)
        phase_winter(settlements, grid, params, rng)
        phase_environment(settlements, grid, params, width, height, rng)

    return grid


def run_simulation_with_snapshots(
    initial_grid: np.ndarray,
    initial_settlements_data: List[Dict],
    params: SimParams,
    rng: np.random.Generator,
    n_years: int = 50,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Run one 50-year simulation and return (final_grid, settlement_snapshots).

    settlement_snapshots: list of dicts with x, y, alive, has_port (API format).
    """
    grid = initial_grid.copy()
    height, width = grid.shape

    settlements: List[Settlement] = []
    for i, sd in enumerate(initial_settlements_data):
        if not sd.get("alive", True):
            continue
        s = Settlement(
            x=int(sd["x"]),
            y=int(sd["y"]),
            owner_id=i,
            population=params.init_population,
            food=params.init_food,
            wealth=params.init_wealth,
            defense=params.init_defense,
            has_port=bool(sd.get("has_port", False)),
        )
        grid[s.y, s.x] = PORT if s.has_port else SETTLEMENT
        settlements.append(s)

    for _ in range(n_years):
        phase_growth(settlements, grid, params, width, height, rng)
        phase_conflict(settlements, grid, params, rng)
        phase_trade(settlements, params, rng)
        phase_winter(settlements, grid, params, rng)
        phase_environment(settlements, grid, params, width, height, rng)

    snapshots = [
        {"x": s.x, "y": s.y, "alive": s.alive, "has_port": s.has_port}
        for s in settlements
    ]
    return grid, snapshots


def simulate_viewport_query(
    initial_grid: np.ndarray,
    initial_settlements_data: List[Dict],
    params: SimParams,
    seed: int,
    viewport_x: int,
    viewport_y: int,
    viewport_w: int,
    viewport_h: int,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Simulate one viewport query (like API /simulate).

    Returns (viewport_grid, settlements_in_viewport).
    viewport_grid: (viewport_h, viewport_w) raw terrain codes.
    settlements_in_viewport: settlements whose (x,y) falls in viewport.
    """
    rng = np.random.default_rng(seed)
    run_params = params.randomise(rng, noise=0.0)
    final_grid, all_snapshots = run_simulation_with_snapshots(
        initial_grid, initial_settlements_data, run_params, rng, n_years=50
    )
    # Extract viewport region
    vp_grid = final_grid[
        viewport_y : viewport_y + viewport_h,
        viewport_x : viewport_x + viewport_w,
    ].copy()
    # Filter settlements in viewport
    in_viewport = [
        s for s in all_snapshots
        if viewport_x <= s["x"] < viewport_x + viewport_w
        and viewport_y <= s["y"] < viewport_y + viewport_h
    ]
    return vp_grid, in_viewport


# ---------------------------------------------------------------------------
# Monte Carlo: run many simulations, collect class frequencies per cell
# ---------------------------------------------------------------------------

def monte_carlo(
    initial_grid: np.ndarray,
    initial_settlements_data: List[Dict],
    params: SimParams,
    n_runs: int = 300,
    n_years: int = 50,
    param_noise: float = 0.0,   # if >0, randomise params slightly each run
    seed: int = 42,
) -> np.ndarray:
    """
    Run n_runs simulations and return empirical class frequency tensor.

    Returns:
        (H, W, 6) float array of class probabilities, already floored + normalised
    """
    height, width = initial_grid.shape
    counts = np.zeros((height, width, 6), dtype=np.float64)

    master_rng = np.random.default_rng(seed)

    for i in range(n_runs):
        run_rng = np.random.default_rng(master_rng.integers(0, 2**32))
        run_params = params.randomise(run_rng, noise=param_noise) if param_noise > 0 else params
        final_grid = run_simulation(
            initial_grid, initial_settlements_data, run_params, run_rng, n_years
        )
        flat = final_grid.ravel().astype(np.intp)
        lookup = np.zeros(max(12, int(flat.max()) + 1), dtype=np.intp)
        for code, ci in TERRAIN_TO_CLASS.items():
            if 0 <= code < len(lookup):
                lookup[code] = ci
        classes = lookup[flat].reshape(height, width)
        for c in range(6):
            counts[:, :, c] += (classes == c).astype(np.float64)

    # Convert to probabilities with floor
    probs = counts + 0.1   # Dirichlet-style smoothing (avoids zeros)
    probs = probs / probs.sum(axis=2, keepdims=True)
    return probs


# ---------------------------------------------------------------------------
# Parameter estimation from observations
# ---------------------------------------------------------------------------

def estimate_params_from_observations(
    observations: Dict[int, np.ndarray],      # seed → (H,W) int array, -1=unobserved
    initial_states: List[Dict],
    settlement_snapshots: Dict[int, List],    # seed → list of snapshot dicts
    base_params: Optional[SimParams] = None,
    verbose: bool = True,
) -> SimParams:
    """
    Estimate hidden parameters from observed cell outcomes and settlement stats.

    Uses simple signal extraction:
      - High ruin rate   → high winter_severity + high base_winter_cost
      - High port rate   → high port_formation_prob
      - Dense expansion  → high expansion_prob + low expansion_threshold
      - Low avg food     → high base_winter_cost
      - Low avg pop      → high winter_severity or aggression
    """
    if base_params is None:
        base_params = SimParams()

    # Aggregate observed terrain outcomes
    total_obs = 0
    settlement_obs = 0
    ruin_obs = 0
    port_obs = 0
    forest_obs = 0

    for seed, obs_grid in observations.items():
        init_grid_raw = np.array(initial_states[seed]["grid"])
        for y in range(obs_grid.shape[0]):
            for x in range(obs_grid.shape[1]):
                v = int(obs_grid[y, x])
                if v < 0:
                    continue
                total_obs += 1
                if v == SETTLEMENT:  settlement_obs += 1
                elif v == RUIN:      ruin_obs += 1
                elif v == PORT:      port_obs += 1
                elif v == FOREST:    forest_obs += 1

    if total_obs == 0:
        return base_params

    ruin_rate = ruin_obs / max(settlement_obs + ruin_obs + port_obs, 1)
    port_rate = port_obs / max(settlement_obs + port_obs + 1, 1)
    expansion_density = (settlement_obs + port_obs) / max(total_obs, 1)

    # Aggregate settlement stats from snapshots.
    # NOTE: the API returns food, population, wealth, defense all normalised
    # to roughly 0.0–1.0 (not raw internal simulator values).  All thresholds
    # below are calibrated to this 0–1 range.
    all_food, all_pop, all_wealth = [], [], []
    alive_count = dead_count = 0
    for seed_snaps in settlement_snapshots.values():
        for snap in seed_snaps:
            for s in snap.get("settlements", []):
                if not isinstance(s, dict):
                    continue
                if s.get("alive", True):
                    alive_count += 1
                    # Clamp to [0, 1] in case of minor API drift
                    if s.get("food") is not None:
                        all_food.append(float(np.clip(s["food"], 0, 1)))
                    if s.get("population") is not None:
                        all_pop.append(float(np.clip(s["population"], 0, 1)))
                    if s.get("wealth") is not None:
                        all_wealth.append(float(np.clip(s["wealth"], 0, 1)))
                else:
                    dead_count += 1

    # Defaults at midpoint of 0–1 range when no data is available
    mean_food   = float(np.mean(all_food))   if all_food   else 0.5
    mean_pop    = float(np.mean(all_pop))    if all_pop    else 0.5
    mean_wealth = float(np.mean(all_wealth)) if all_wealth else 0.5

    # Survival ratio: fraction of observed settlements still alive
    # (separate signal from ruin_rate which is terrain-based)
    total_snapped = alive_count + dead_count
    survival_rate = alive_count / max(total_snapped, 1)

    # --- Parameter adjustments based on signals ---
    # All thresholds are against 0–1 normalised stats.
    p = dataclasses.replace(base_params)   # start from base

    # Winter / aggression severity
    # Signals: high ruin_rate OR low mean_food OR low survival_rate → harsh world
    harshness = 0.4 * ruin_rate + 0.3 * (1 - mean_food) + 0.3 * (1 - survival_rate)
    if harshness > 0.55:
        p = dataclasses.replace(p,
            winter_severity=min(p.winter_severity * 1.6, 0.85),
            base_winter_cost=min(p.base_winter_cost * 1.4, 22.0),
            starvation_collapse_prob=min(p.starvation_collapse_prob + 0.20, 0.90),
            raid_food_threshold=min(p.raid_food_threshold * 1.3, 12.0),
        )
    elif harshness < 0.25:
        p = dataclasses.replace(p,
            winter_severity=max(p.winter_severity * 0.65, 0.08),
            base_winter_cost=max(p.base_winter_cost * 0.75, 2.0),
            starvation_collapse_prob=max(p.starvation_collapse_prob - 0.15, 0.25),
        )

    # Food / growth abundance
    # High mean_food + high mean_wealth → generous production or mild winters
    prosperity = 0.6 * mean_food + 0.4 * mean_wealth
    if prosperity > 0.65:
        p = dataclasses.replace(p,
            forest_food_bonus=min(p.forest_food_bonus * 1.3, 5.0),
            base_food_production=min(p.base_food_production * 1.25, 3.0),
            trade_food_bonus=min(p.trade_food_bonus * 1.2, 8.0),
        )
    elif prosperity < 0.35:
        p = dataclasses.replace(p,
            forest_food_bonus=max(p.forest_food_bonus * 0.8, 0.5),
            base_food_production=max(p.base_food_production * 0.8, 0.3),
        )

    # Port / trade activity
    if port_rate > 0.30:
        p = dataclasses.replace(p,
            port_formation_prob=min(p.port_formation_prob * 1.5, 0.60),
            trade_food_bonus=min(p.trade_food_bonus * 1.3, 8.0),
            trade_range=min(p.trade_range * 1.2, 14.0),
        )
    elif port_rate < 0.05:
        p = dataclasses.replace(p,
            port_formation_prob=max(p.port_formation_prob * 0.55, 0.04),
            trade_food_bonus=max(p.trade_food_bonus * 0.8, 1.0),
        )

    # Expansion aggressiveness
    if expansion_density > 0.15:
        p = dataclasses.replace(p,
            expansion_prob=min(p.expansion_prob * 1.4, 0.60),
            expansion_threshold=max(p.expansion_threshold * 0.80, 10.0),
        )
    elif expansion_density < 0.04:
        p = dataclasses.replace(p,
            expansion_prob=max(p.expansion_prob * 0.60, 0.04),
            expansion_threshold=min(p.expansion_threshold * 1.30, 40.0),
        )

    # Population health: high mean_pop → settlements are growing well
    if mean_pop > 0.65:
        p = dataclasses.replace(p,
            growth_threshold=max(p.growth_threshold * 0.85, 8.0),
        )
    elif mean_pop < 0.30:
        p = dataclasses.replace(p,
            growth_threshold=min(p.growth_threshold * 1.20, 25.0),
            starvation_collapse_prob=min(p.starvation_collapse_prob + 0.10, 0.90),
        )

    if verbose:
        print(
            f"Param estimates — terrain: ruin_rate={ruin_rate:.2f} "
            f"port_rate={port_rate:.2f} expansion_density={expansion_density:.2f}"
        )
        print(
            f"  settlement stats (0–1 scale): food={mean_food:.2f} "
            f"pop={mean_pop:.2f} wealth={mean_wealth:.2f} "
            f"survival={survival_rate:.2f} (alive={alive_count} dead={dead_count})"
        )
        print(f"  → harshness={harshness:.2f} prosperity={prosperity:.2f}")
        print(
            f"  inferred: winter_severity={p.winter_severity:.2f} "
            f"base_winter_cost={p.base_winter_cost:.1f} "
            f"expansion_prob={p.expansion_prob:.2f} "
            f"port_formation_prob={p.port_formation_prob:.2f} "
            f"starvation_collapse_prob={p.starvation_collapse_prob:.2f}"
        )

    return p


# ---------------------------------------------------------------------------
# Sanity-check / smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running simulator smoke test…")

    # Build a tiny 20×20 test world
    H, W = 20, 20
    grid = np.full((H, W), PLAINS, dtype=np.int64)

    # Ocean border
    grid[0, :] = OCEAN
    grid[-1, :] = OCEAN
    grid[:, 0] = OCEAN
    grid[:, -1] = OCEAN

    # Some mountains and forests
    grid[5, 10] = MOUNTAIN
    grid[8, 8]  = FOREST
    grid[9, 8]  = FOREST
    grid[9, 9]  = FOREST

    # Two initial settlements
    grid[5, 5] = SETTLEMENT
    grid[12, 12] = SETTLEMENT

    settlements_data = [
        {"x": 5,  "y": 5,  "has_port": False, "alive": True},
        {"x": 12, "y": 12, "has_port": False, "alive": True},
    ]

    params = SimParams()
    rng = np.random.default_rng(0)

    print("  Single run…", end=" ")
    final = run_simulation(grid, settlements_data, params, rng, n_years=50)
    classes, counts_arr = np.unique(
        [TERRAIN_TO_CLASS.get(int(v), 0) for v in final.ravel()], return_counts=True
    )
    for c, n in zip(classes, counts_arr):
        names = ["Empty","Settlement","Port","Ruin","Forest","Mountain"]
        print(f"{names[c]}={n}", end=" ")
    print()

    print("  Monte Carlo (50 runs)…", end=" ")
    probs = monte_carlo(grid, settlements_data, params, n_runs=50, seed=1)
    assert probs.shape == (H, W, 6)
    assert abs(probs.sum(axis=2) - 1.0).max() < 1e-5
    assert probs.min() > 0
    print(f"OK  shape={probs.shape}  min={probs.min():.4f}")

    print("  Parameter estimation…", end=" ")
    obs = {0: np.full((H, W), -1, dtype=np.int32)}
    obs[0][5, 5]   = RUIN
    obs[0][12, 12] = SETTLEMENT
    obs[0][12, 13] = PORT
    estimated = estimate_params_from_observations(
        obs,
        [{"grid": grid.tolist(), "settlements": settlements_data}],
        {0: []},
    )
    print(f"OK  winter_severity={estimated.winter_severity:.2f}")

    print("\nAll smoke tests passed.")
