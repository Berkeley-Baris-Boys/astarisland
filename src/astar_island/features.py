from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .types import (
    CLASS_EMPTY,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_SETTLEMENT,
    InitialState,
    SeedFeatures,
    TERRAIN_FOREST,
    TERRAIN_MOUNTAIN,
    TERRAIN_OCEAN,
    TERRAIN_PORT,
    TERRAIN_RUIN,
    TERRAIN_SETTLEMENT,
    terrain_grid_to_class_grid,
)

BUCKET_KEY_VERSION = 2


def _neighbor_sum(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    h, w = mask.shape
    result = np.zeros((h, w), dtype=np.float64)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            ys = slice(max(0, dy), min(h, h + dy))
            xs = slice(max(0, dx), min(w, w + dx))
            yt = slice(max(0, -dy), min(h, h - dy))
            xt = slice(max(0, -dx), min(w, w - dx))
            result[yt, xt] += mask[ys, xs]
    return result


def _distance_to_mask(mask: np.ndarray, max_distance: int = 80) -> np.ndarray:
    points = np.argwhere(mask > 0)
    h, w = mask.shape
    if len(points) == 0:
        return np.full((h, w), float(max_distance), dtype=np.float64)
    ys, xs = np.indices((h, w))
    best = np.full((h, w), np.inf, dtype=np.float64)
    for py, px in points:
        best = np.minimum(best, np.abs(ys - py) + np.abs(xs - px))
    return np.minimum(best, max_distance)


def _gaussian_intensity(mask: np.ndarray, sigma: float) -> np.ndarray:
    h, w = mask.shape
    ys, xs = np.indices((h, w))
    points = np.argwhere(mask > 0)
    intensity = np.zeros((h, w), dtype=np.float64)
    if len(points) == 0:
        return intensity
    denom = 2.0 * sigma * sigma
    for py, px in points:
        dist2 = (ys - py) ** 2 + (xs - px) ** 2
        intensity += np.exp(-dist2 / max(denom, 1e-6))
    if intensity.max() > 0:
        intensity /= intensity.max()
    return intensity


def build_seed_features(
    seed_index: int,
    initial_state: InitialState,
    *,
    settlement_sigma: float = 2.2,
) -> SeedFeatures:
    grid = initial_state.grid
    class_grid = terrain_grid_to_class_grid(grid)
    h, w = grid.shape
    is_ocean = grid == TERRAIN_OCEAN
    is_mountain = grid == TERRAIN_MOUNTAIN
    is_forest = grid == TERRAIN_FOREST
    is_settlement_like = np.isin(grid, [TERRAIN_SETTLEMENT, TERRAIN_PORT, TERRAIN_RUIN])
    is_buildable = ~(is_ocean | is_mountain)
    initial_settlement_mask = np.zeros((h, w), dtype=bool)
    initial_port_mask = np.zeros((h, w), dtype=bool)
    for settlement in initial_state.settlements:
        initial_settlement_mask[settlement.y, settlement.x] = True
        if settlement.has_port:
            initial_port_mask[settlement.y, settlement.x] = True
    coastal_mask = is_buildable & (_neighbor_sum(is_ocean.astype(float), radius=1) > 0)
    dist_to_settlement = _distance_to_mask(initial_settlement_mask)
    dist_to_coast = _distance_to_mask(coastal_mask)
    dist_to_ruin = _distance_to_mask(grid == TERRAIN_RUIN)
    settlement_density = _neighbor_sum(initial_settlement_mask.astype(float), radius=3) / 48.0
    forest_density = _neighbor_sum(is_forest.astype(float), radius=2) / 24.0
    mountain_density = _neighbor_sum(is_mountain.astype(float), radius=2) / 24.0
    coastal_density = _neighbor_sum(coastal_mask.astype(float), radius=2) / 24.0
    frontier_mask = is_buildable & (dist_to_settlement <= 3) & ~initial_settlement_mask
    conflict_mask = frontier_mask & (settlement_density >= np.quantile(settlement_density[is_buildable], 0.75) if np.any(is_buildable) else False)
    reclaimable_mask = is_buildable & ((grid == TERRAIN_RUIN) | (dist_to_ruin <= 2))
    dynamic_prior_mask = is_buildable & (~is_ocean)
    settlement_intensity = _gaussian_intensity(
        initial_settlement_mask.astype(float),
        sigma=max(float(settlement_sigma), 1e-6),
    )
    port_intensity = _gaussian_intensity(initial_port_mask.astype(float), sigma=2.0)
    ys, xs = np.indices((h, w))
    border_distance = np.minimum.reduce([ys, xs, h - 1 - ys, w - 1 - xs]).astype(np.float64)

    feature_names = [
        "bias",
        "buildable",
        "coastal",
        "initial_forest",
        "initial_settlement_like",
        "initial_mountain",
        "dist_to_settlement",
        "dist_to_coast",
        "dist_to_ruin",
        "settlement_density",
        "forest_density",
        "mountain_density",
        "coastal_density",
        "frontier",
        "conflict",
        "reclaimable",
        "settlement_intensity",
        "port_intensity",
        "border_distance",
    ]
    feature_stack = np.stack(
        [
            np.ones((h, w), dtype=np.float64),
            is_buildable.astype(np.float64),
            coastal_mask.astype(np.float64),
            is_forest.astype(np.float64),
            is_settlement_like.astype(np.float64),
            is_mountain.astype(np.float64),
            dist_to_settlement / max(h + w, 1),
            dist_to_coast / max(h + w, 1),
            dist_to_ruin / max(h + w, 1),
            settlement_density,
            forest_density,
            mountain_density,
            coastal_density,
            frontier_mask.astype(np.float64),
            conflict_mask.astype(np.float64),
            reclaimable_mask.astype(np.float64),
            settlement_intensity,
            port_intensity,
            border_distance / max(min(h, w), 1),
        ],
        axis=-1,
    )
    return SeedFeatures(
        seed_index=seed_index,
        feature_stack=feature_stack,
        feature_names=feature_names,
        buildable_mask=is_buildable,
        dynamic_prior_mask=dynamic_prior_mask,
        coastal_mask=coastal_mask,
        frontier_mask=frontier_mask,
        conflict_mask=conflict_mask,
        reclaimable_mask=reclaimable_mask,
        initial_class_grid=class_grid,
        initial_settlement_mask=initial_settlement_mask,
    )


def build_all_features(
    initial_states: list[InitialState],
    *,
    settlement_sigma: float = 2.2,
) -> dict[int, SeedFeatures]:
    return {
        seed_index: build_seed_features(
            seed_index,
            state,
            settlement_sigma=settlement_sigma,
        )
        for seed_index, state in enumerate(initial_states)
    }


def make_bucket_keys(seed_features: SeedFeatures) -> np.ndarray:
    features = seed_features.feature_stack
    dist_settle = np.digitize(features[..., seed_features.feature_names.index("dist_to_settlement")], [0.05, 0.1, 0.2])
    dist_coast = np.digitize(features[..., seed_features.feature_names.index("dist_to_coast")], [0.05, 0.1, 0.2])
    settle_density = np.digitize(features[..., seed_features.feature_names.index("settlement_density")], [0.1, 0.25, 0.5])
    forest_density = np.digitize(features[..., seed_features.feature_names.index("forest_density")], [0.1, 0.25, 0.5])
    keys = seed_features.initial_class_grid.astype(np.int32)
    for component, base in (
        (seed_features.coastal_mask.astype(np.int32), 2),
        (seed_features.frontier_mask.astype(np.int32), 2),
        (seed_features.conflict_mask.astype(np.int32), 2),
        (dist_settle.astype(np.int32), 4),
        (dist_coast.astype(np.int32), 4),
        (settle_density.astype(np.int32), 4),
        (forest_density.astype(np.int32), 4),
    ):
        keys = keys * base + component
    return keys
