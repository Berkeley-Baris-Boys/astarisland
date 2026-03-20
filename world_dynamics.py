"""
World dynamics estimator — learns the round's hidden parameters empirically
from observations across all 5 seeds (which share the same hidden parameters).

Instead of simulating the world, we estimate a "dynamics vector" that
captures how hostile/forested/trade-active the world is this round,
then use that to adjust terrain transition priors for unobserved cells.

Core insight: All 5 seeds share the same hidden parameters.
Pooling observations across seeds gives us a better estimate of round dynamics
than looking at any single seed in isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import (
    N_CLASSES, SETTLEMENT_CODE, PORT_CODE, RUIN_CODE, FOREST_CODE,
    PLAINS_CODE, OCEAN_CODE, MOUNTAIN_CODE,
)
from state import ObservationStore, forest_adjacency, code_to_class


@dataclass
class WorldDynamics:
    """
    Empirically estimated round dynamics, pooled across all 5 seeds.

    All values in [0, 1] unless noted. Higher = more intense.
    """
    # How often do initial settlements survive? (1.0 = all survive, 0.0 = all ruined)
    settlement_survival_rate: float = 0.8

    # What fraction of initial settlements became Ruins?
    settlement_ruin_rate: float = 0.1

    # What fraction of initial settlements became Empty/Plains? (abandoned/absorbed)
    settlement_abandoned_rate: float = 0.05

    # Did settlements expand? (fraction of Plains/Empty near settlements that became settled)
    expansion_rate: float = 0.05

    # Are forests growing? (fraction of empty land adjacent to initial forest that became forest)
    forest_growth_signal: float = 0.0

    # Average food level of surviving settlements (normalised to ~1.0 as baseline)
    avg_food_survivors: float = 1.5

    # Average population of surviving settlements
    avg_pop_survivors: float = 2.5

    # Fraction of surviving settlements with a port
    port_fraction: float = 0.3

    # Average wealth of ports (higher → active trade)
    avg_port_wealth: float = 1.0

    # Factionalization: how many distinct owner_ids are active?
    # High → many factions fighting = high conflict. Low → consolidated world.
    n_factions: float = 1.0
    # owner_id entropy across observed settlements (higher = more fragmented)
    owner_entropy: float = 0.0

    # n_observations used to compute these estimates
    n_settlement_obs: int = 0
    n_transition_obs: int = 0

    def is_hostile(self) -> bool:
        return (self.settlement_ruin_rate > 0.2
                or self.settlement_survival_rate < 0.6
                or self.owner_entropy > 1.2)

    def is_harsh_winter(self) -> bool:
        return self.avg_food_survivors < 0.8 or self.settlement_survival_rate < 0.5

    def is_expansive(self) -> bool:
        return self.expansion_rate > 0.08 or self.settlement_survival_rate > 0.9

    def is_forested(self) -> bool:
        return self.forest_growth_signal > 0.05

    def summary(self) -> str:
        return (
            f"survival={self.settlement_survival_rate:.1%} "
            f"ruin={self.settlement_ruin_rate:.1%} "
            f"expansion={self.expansion_rate:.1%} "
            f"forest_growth={self.forest_growth_signal:.1%} "
            f"avg_food={self.avg_food_survivors:.2f} "
            f"avg_pop={self.avg_pop_survivors:.2f} "
            f"port_wealth={self.avg_port_wealth:.2f} "
            f"factions={self.n_factions:.0f} owner_entropy={self.owner_entropy:.2f} "
            f"[n_settle={self.n_settlement_obs}, n_trans={self.n_transition_obs}]"
        )


def estimate_world_dynamics(
    store: ObservationStore,
    initial_states: list[dict],
) -> WorldDynamics:
    """
    Pool observations across all seeds to estimate round-level dynamics.

    Uses:
    - Settlement alive/dead snapshots → survival and ruin rates
    - Terrain transitions (initial_code → final_code) → expansion and forest growth
    - Settlement stats (food, pop, wealth) → winter severity and trade activity
    """
    dyn = WorldDynamics()

    # ── Settlement stats (pooled across all seeds) ────────────────────────────
    alive_list: list[bool] = []
    survivor_food_list: list[float] = []
    survivor_pop_list: list[float] = []
    port_wealth_list: list[float] = []
    port_alive: list[bool] = []

    for s in range(store.seeds_count):
        # Use the last observation of each settlement position (most up-to-date)
        last_obs: dict[tuple[int, int], dict] = {}
        for snap in store.settlement_snaps[s]:
            for settle in snap["settlements"]:
                key = (settle["x"], settle["y"])
                last_obs[key] = settle

        for settle in last_obs.values():
            alive = bool(settle.get("alive", True))
            alive_list.append(alive)
            if alive and settle.get("food") is not None:
                survivor_food_list.append(float(settle["food"]))
            if alive and settle.get("population") is not None:
                survivor_pop_list.append(float(settle["population"]))
            if settle.get("has_port", False):
                port_alive.append(alive)
                if settle.get("wealth") is not None and alive:
                    port_wealth_list.append(float(settle["wealth"]))

    dyn.n_settlement_obs = len(alive_list)

    if alive_list:
        dyn.settlement_survival_rate = float(np.mean(alive_list))

    if survivor_food_list:
        dyn.avg_food_survivors = float(np.mean(survivor_food_list))
    if survivor_pop_list:
        dyn.avg_pop_survivors = float(np.mean(survivor_pop_list))
    if port_wealth_list:
        dyn.avg_port_wealth = float(np.mean(port_wealth_list))
    if port_alive:
        dyn.port_fraction = float(np.mean(port_alive))

    # ── Terrain transitions (pooled across all seeds) ─────────────────────────
    ruin_from_settle  = 0
    empty_from_settle = 0
    settle_total_obs  = 0
    settled_from_empty = 0
    empty_total_obs    = 0
    forest_from_adj    = 0
    forest_adj_total   = 0

    for s in range(store.seeds_count):
        ig = np.asarray(initial_states[s]["grid"], dtype=np.int32)
        fadj = forest_adjacency(ig)

        for y, row in enumerate(store.latest[s]):
            for x, raw_val in enumerate(row):
                if raw_val is None:
                    continue
                init_code  = int(ig[y, x])
                final_code = int(raw_val)
                final_cls  = code_to_class(final_code)
                dyn.n_transition_obs += 1

                if init_code in (SETTLEMENT_CODE, PORT_CODE):
                    settle_total_obs += 1
                    if final_cls == 3:    # Ruin
                        ruin_from_settle += 1
                    elif final_cls == 0:  # Empty/Plains
                        empty_from_settle += 1

                elif init_code in (0, PLAINS_CODE):
                    empty_total_obs += 1
                    if final_cls in (1, 2):  # Settlement or Port
                        settled_from_empty += 1

                # Forest growth: was adjacent-to-forest empty land now a forest?
                if init_code in (0, PLAINS_CODE, RUIN_CODE) and fadj[y, x] > 0:
                    forest_adj_total += 1
                    if final_code == FOREST_CODE:
                        forest_from_adj += 1

    if settle_total_obs > 0:
        dyn.settlement_ruin_rate      = ruin_from_settle / settle_total_obs
        dyn.settlement_abandoned_rate = empty_from_settle / settle_total_obs
        # Cross-check survival rate from transitions vs from snapshots
        trans_survival = 1.0 - dyn.settlement_ruin_rate - dyn.settlement_abandoned_rate
        if dyn.n_settlement_obs > 0:
            # Blend: transitions can see all cells, snapshots give richer stats
            w_trans = min(1.0, settle_total_obs / 20.0)
            w_snap  = 1.0 - w_trans
            dyn.settlement_survival_rate = (
                w_snap * dyn.settlement_survival_rate + w_trans * trans_survival
            )
        else:
            dyn.settlement_survival_rate = trans_survival

    if empty_total_obs > 5:
        dyn.expansion_rate = settled_from_empty / empty_total_obs

    if forest_adj_total > 5:
        dyn.forest_growth_signal = forest_from_adj / forest_adj_total

    # ── Factionalization: owner_id diversity across alive settlements ─────────
    # High diversity = many competing factions = high-conflict world
    import collections
    owner_ids: list[int] = []
    for s in range(store.seeds_count):
        for snap in store.settlement_snaps[s]:
            for settle in snap["settlements"]:
                oid = settle.get("owner_id")
                if oid is not None and settle.get("alive", True):
                    owner_ids.append(int(oid))

    if owner_ids:
        cnts = np.array(list(collections.Counter(owner_ids).values()), dtype=np.float32)
        probs = cnts / cnts.sum()
        dyn.n_factions    = float(len(cnts))
        dyn.owner_entropy = float(-np.sum(probs * np.log(probs + 1e-12)))

    print(f"World dynamics: {dyn.summary()}")
    return dyn


# ── Adjusted priors using world dynamics ─────────────────────────────────────

def dynamics_adjusted_prior(
    init_code: int,
    dyn: WorldDynamics,
    is_near_settlement: bool,
    is_coastal: bool,
    n_forest_adj: int,
    is_near_port: bool = False,
    dist_to_coast: float = 10.0,
    local_settle_density: float = 0.0,
) -> np.ndarray:
    """
    Return a terrain prior adjusted for this round's observed dynamics.

    Base prior is tuned from rounds 2–3 empirical transitions.
    World dynamics shift the distribution based on what we actually see.
    """
    # ── Base priors (calibrated from R2/R3 ground truth) ─────────────────────
    BASE: dict[int, np.ndarray] = {
        OCEAN_CODE:    np.array([0.96, 0.01, 0.01, 0.01, 0.01, 0.00]),
        MOUNTAIN_CODE: np.array([0.00, 0.00, 0.00, 0.00, 0.00, 1.00]),
        PLAINS_CODE:   np.array([0.82, 0.13, 0.01, 0.01, 0.02, 0.01]),
        0:             np.array([0.68, 0.11, 0.02, 0.05, 0.14, 0.00]),
        SETTLEMENT_CODE: np.array([0.35, 0.48, 0.05, 0.08, 0.04, 0.00]),
        PORT_CODE:       np.array([0.28, 0.18, 0.36, 0.12, 0.06, 0.00]),
        RUIN_CODE:       np.array([0.20, 0.10, 0.02, 0.35, 0.30, 0.03]),
        FOREST_CODE:     np.array([0.04, 0.05, 0.01, 0.01, 0.88, 0.01]),
    }
    p = BASE.get(init_code, np.array([0.75, 0.10, 0.02, 0.05, 0.08, 0.00])).copy()

    # ── Apply world dynamics adjustments ─────────────────────────────────────

    # Initial settlement cells: shift toward observed survival/ruin rates
    if init_code == SETTLEMENT_CODE:
        p[1] = dyn.settlement_survival_rate * 0.75   # Settlement
        p[2] = dyn.settlement_survival_rate * 0.15   # Port (some develop)
        p[3] = dyn.settlement_ruin_rate * 1.0         # Ruin
        p[0] = dyn.settlement_abandoned_rate + 0.05  # Empty
        p[4] = 0.05                                   # Forest

    elif init_code == PORT_CODE:
        p[2] = dyn.settlement_survival_rate * 0.65
        p[1] = dyn.settlement_survival_rate * 0.15
        p[3] = dyn.settlement_ruin_rate * 1.0
        p[0] = dyn.settlement_abandoned_rate + 0.05
        p[4] = 0.05

    # Empty/Plains: driven by expansion rate + proximity signals
    elif init_code in (0, PLAINS_CODE):
        # Continuous distance-to-coast boost for port development
        coastal_proximity = max(0.0, 1.0 - dist_to_coast / 8.0)

        # Settlement density boosts expansion probability
        density_boost = min(1.0, local_settle_density / 3.0) * 0.15

        if is_near_settlement:
            expansion_boost = dyn.expansion_rate * 2.0 + density_boost
            p[1] += expansion_boost * 0.70   # Settlement
            p[2] += expansion_boost * 0.30 * coastal_proximity  # Port
            p[0] -= expansion_boost * 0.85
            p[3] += expansion_boost * 0.15   # Some expansions collapse too

        # Factionalization raises ruin rate even for empty land (collateral)
        if dyn.is_hostile():
            hostility = min(0.3, dyn.owner_entropy * 0.05)
            p[3] += hostility
            p[0] -= hostility

        # Forest growth signal
        if n_forest_adj > 0 and dyn.is_forested():
            forest_boost = dyn.forest_growth_signal * n_forest_adj * 0.4
            p[4] = min(0.85, p[4] + forest_boost)
            p[0] = max(0.05, p[0] - forest_boost)

    # Ruin cells: reclamation vs forest vs persistence
    elif init_code == RUIN_CODE:
        if dyn.is_expansive() and is_near_settlement:
            p[1] += 0.10
            p[2] += 0.04 if is_coastal else 0.0
            p[3] -= 0.08
        if dyn.is_forested():
            p[4] += dyn.forest_growth_signal * 0.4
            p[3] -= dyn.forest_growth_signal * 0.3

    # Normalise + floor
    p = np.maximum(p, 0.01)
    return p / p.sum()
