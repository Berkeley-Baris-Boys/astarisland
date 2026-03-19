#!/usr/bin/env python3
"""Solver for the Astar Island challenge (NM i AI 2026)."""
# pip install requests numpy scipy

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests
from scipy.ndimage import gaussian_filter

BASE_URL = "https://api.ainm.no"
CLASS_COUNT = 6
MAX_CLASS = 16
MAX_VIEWPORT = 15
TOTAL_BUDGET = 50


class SolverError(RuntimeError):
    """Raised for recoverable solver failures."""


def request_json(
    session: requests.Session,
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    retry_5xx: bool = True,
) -> Any:
    """Send a JSON request with one retry on 5xx/network errors."""
    url = f"{BASE_URL}{path}"
    attempts = 2 if retry_5xx else 1

    for attempt in range(attempts):
        try:
            response = session.request(method=method, url=url, json=payload, timeout=30)
        except requests.RequestException as exc:
            if attempt + 1 < attempts:
                print(f"Network error on {method} {path}: {exc}. Retrying in 2s...")
                time.sleep(2.0)
                continue
            raise SolverError(f"Network error on {method} {path}: {exc}") from exc

        status = response.status_code

        if status >= 500:
            if attempt + 1 < attempts:
                print(f"HTTP {status} on {method} {path}. Retrying in 2s...")
                time.sleep(2.0)
                continue
            raise SolverError(f"HTTP {status} on {method} {path}: {response.text}")

        if status >= 400:
            if status == 429:
                raise SolverError(f"Rate limited (429) on {method} {path}: {response.text}")
            raise SolverError(f"HTTP {status} on {method} {path}: {response.text}")

        try:
            return response.json()
        except ValueError as exc:
            print(f"JSON parse error on {method} {path}. Raw response follows:")
            print(response.text)
            raise SolverError(f"Could not parse JSON for {method} {path}") from exc

    raise SolverError(f"Unreachable request failure for {method} {path}")


def normalize_rounds_payload(rounds_payload: Any) -> List[Dict[str, Any]]:
    if isinstance(rounds_payload, list):
        return [r for r in rounds_payload if isinstance(r, dict)]
    if isinstance(rounds_payload, dict):
        for key in ("rounds", "data", "items"):
            value = rounds_payload.get(key)
            if isinstance(value, list):
                return [r for r in value if isinstance(r, dict)]
    raise SolverError(f"Unexpected rounds payload format: {type(rounds_payload)}")


def load_round(
    session: requests.Session,
) -> Tuple[str, int, int, int, List[Dict[str, Any]]]:
    """Load active round metadata and initial states."""
    rounds_payload = request_json(session, "GET", "/astar-island/rounds")
    rounds = normalize_rounds_payload(rounds_payload)

    active = next((r for r in rounds if str(r.get("status", "")).lower() == "active"), None)
    if active is None:
        raise SolverError("No active round found in /astar-island/rounds")

    if "id" not in active:
        raise SolverError("Active round payload has no 'id'")

    round_id = str(active["id"])
    detail = request_json(session, "GET", f"/astar-island/rounds/{round_id}")

    try:
        width = int(detail["map_width"])
        height = int(detail["map_height"])
        seeds_count = int(detail["seeds_count"])
        initial_states = detail["initial_states"]
    except KeyError as exc:
        raise SolverError(f"Round detail missing expected key: {exc}") from exc

    if not isinstance(initial_states, list) or len(initial_states) != seeds_count:
        raise SolverError("initial_states is missing or does not match seeds_count")

    round_number = active.get("round_number", detail.get("round_number", round_id))
    print(f"Active round: {round_number}")
    print(f"Map size: {width}x{height}")
    print(f"Seeds: {seeds_count}")

    return round_id, width, height, seeds_count, initial_states


def tiling_offsets(size: int, window: int) -> List[int]:
    if size <= window:
        return [0]
    offsets = list(range(0, size - window + 1, window))
    last = size - window
    if offsets[-1] != last:
        offsets.append(last)
    return offsets


def plan_queries(
    width: int,
    height: int,
    seeds_count: int,
    budget: int = TOTAL_BUDGET,
) -> List[Tuple[int, int, int, int, int]]:
    """Plan core coverage queries (reserve budget used adaptively later)."""
    x_offsets = tiling_offsets(width, MAX_VIEWPORT)
    y_offsets = tiling_offsets(height, MAX_VIEWPORT)

    positions: List[Tuple[int, int, int, int]] = []
    for y in y_offsets:
        for x in x_offsets:
            w = min(MAX_VIEWPORT, width - x)
            h = min(MAX_VIEWPORT, height - y)
            positions.append((x, y, w, h))

    query_plan: List[Tuple[int, int, int, int, int]] = []
    for seed_index in range(seeds_count):
        for x, y, w, h in positions:
            query_plan.append((seed_index, x, y, w, h))

    if len(query_plan) > budget:
        query_plan = query_plan[:budget]

    reserve = max(0, budget - len(query_plan))
    print(
        f"Planned {len(query_plan)} core queries "
        f"({len(positions)} per seed), reserve budget: {reserve}"
    )
    return query_plan


def initialize_observations(
    width: int,
    height: int,
    seeds_count: int,
) -> Dict[int, List[List[Optional[int]]]]:
    observations: Dict[int, List[List[Optional[int]]]] = {}
    for seed in range(seeds_count):
        observations[seed] = [[None for _ in range(width)] for _ in range(height)]
    return observations


def observed_mask_from_latest(latest_grid: List[List[Optional[int]]]) -> np.ndarray:
    h = len(latest_grid)
    w = len(latest_grid[0]) if h else 0
    mask = np.zeros((h, w), dtype=bool)
    for y, row in enumerate(latest_grid):
        for x, value in enumerate(row):
            mask[y, x] = value is not None
    return mask


def compute_transition_matrix(
    initial_states: Sequence[Dict[str, Any]],
    counts: Dict[int, np.ndarray],
) -> np.ndarray:
    """Compute cross-seed transition matrix with additive smoothing."""
    transition_counts = np.full((CLASS_COUNT, CLASS_COUNT), 0.5, dtype=np.float64)
    np.fill_diagonal(transition_counts, 8.0)

    for seed_index, state in enumerate(initial_states):
        init_grid = np.asarray(state["grid"], dtype=np.int64)
        seed_counts = counts[seed_index]

        for init_class in range(CLASS_COUNT):
            mask = init_grid == init_class
            if not np.any(mask):
                continue
            transition_counts[init_class] += seed_counts[mask][:, :CLASS_COUNT].sum(axis=0)

    row_sums = transition_counts.sum(axis=1, keepdims=True)
    return transition_counts / row_sums


def seed_transition_entropy(initial_grid: np.ndarray, counts: np.ndarray) -> float:
    entropies: List[float] = []
    weights: List[float] = []

    for init_class in range(CLASS_COUNT):
        mask = initial_grid == init_class
        if not np.any(mask):
            continue

        class_counts = counts[mask][:, :CLASS_COUNT].sum(axis=0).astype(np.float64)
        total = float(class_counts.sum())
        if total <= 0:
            continue

        probs = class_counts / total
        entropy = float(-(probs * np.log(probs + 1e-12)).sum() / np.log(CLASS_COUNT))
        entropies.append(entropy)
        weights.append(total)

    if not weights:
        return 1.0
    return float(np.average(np.asarray(entropies), weights=np.asarray(weights)))


def choose_reserve_query(
    initial_states: Sequence[Dict[str, Any]],
    observations: Dict[int, List[List[Optional[int]]]],
    counts: Dict[int, np.ndarray],
    base_positions: List[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int, int]:
    """Choose one additional query after the core scan.

    Priority:
    1) Seeds with lowest direct coverage
    2) Otherwise seeds with highest transition entropy
    3) Within target seed, viewport with highest uncertainty score
    """
    seeds = sorted(observations.keys())
    height = len(next(iter(observations.values())))
    width = len(next(iter(observations.values()))[0])
    total_cells = width * height

    coverage: Dict[int, float] = {}
    masks: Dict[int, np.ndarray] = {}
    for seed in seeds:
        mask = observed_mask_from_latest(observations[seed])
        masks[seed] = mask
        coverage[seed] = float(mask.sum()) / float(total_cells)

    min_coverage = min(coverage.values())
    if min_coverage < 1.0:
        candidate_seeds = [s for s in seeds if coverage[s] == min_coverage]
        target_seed = candidate_seeds[0]
    else:
        entropy_by_seed: Dict[int, float] = {}
        for seed in seeds:
            init_grid = np.asarray(initial_states[seed]["grid"], dtype=np.int64)
            entropy_by_seed[seed] = seed_transition_entropy(init_grid, counts[seed])
        target_seed = max(entropy_by_seed, key=entropy_by_seed.get)

    global_transition = compute_transition_matrix(initial_states, counts)
    row_entropies = -np.sum(global_transition * np.log(global_transition + 1e-12), axis=1)
    row_entropies = row_entropies / np.log(CLASS_COUNT)

    init_grid_target = np.asarray(initial_states[target_seed]["grid"], dtype=np.int64)
    sample_totals = counts[target_seed].sum(axis=2).astype(np.float64)
    uncertainty = row_entropies[init_grid_target]

    # Prefer less-sampled cells.
    uncertainty = uncertainty / (1.0 + sample_totals)

    target_mask = masks[target_seed]

    best_score = -1.0
    best_position = base_positions[0]

    for x, y, w, h in base_positions:
        region_uncertainty = uncertainty[y : y + h, x : x + w].mean()
        region_unseen = float((~target_mask[y : y + h, x : x + w]).mean())

        if min_coverage < 1.0:
            score = 2.0 * region_unseen + region_uncertainty
        else:
            score = region_uncertainty

        if score > best_score:
            best_score = score
            best_position = (x, y, w, h)

    x, y, w, h = best_position
    return target_seed, x, y, w, h


def discover_terrain_classes(observations: Dict[int, List[List[Optional[int]]]]) -> None:
    all_values = set()
    for grid in observations.values():
        for row in grid:
            for v in row:
                if v is not None:
                    all_values.add(v)
    print(f"Discovered terrain values: {sorted(all_values)}")


def infer_hidden_params(
    observations: Dict[int, List[List[Optional[int]]]],
    initial_states: Sequence[Dict[str, Any]],
    project: str = "ai-nm26osl-1722",
    location: str = "us-central1",
) -> Dict[str, float]:
    """Infer hidden simulator parameters using Gemini via Vertex AI."""
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except Exception:
        print("Vertex AI SDK unavailable; skipping hidden-parameter inference")
        return {}

    try:
        vertexai.init(project=project, location=location)
        model = GenerativeModel("gemini-2.0-flash-001")
    except Exception as exc:
        print(f"Vertex AI init/model error; skipping hidden-parameter inference: {exc}")
        return {}

    summary_lines: List[str] = []
    for seed in sorted(observations.keys()):
        grid = observations[seed]
        observed: List[Tuple[int, int, int]] = []
        for y, row in enumerate(grid):
            for x, value in enumerate(row):
                if value is None:
                    continue
                observed.append((y, x, int(value)))

        init = initial_states[seed]["grid"]
        transitions: Dict[Tuple[int, int], int] = {}
        for y, x, value in observed:
            init_class = int(init[y][x])
            key = (init_class, value)
            transitions[key] = transitions.get(key, 0) + 1

        compact = ", ".join(
            f"{a}->{b}:{cnt}" for (a, b), cnt in sorted(transitions.items(), key=lambda item: item[0])
        )
        summary_lines.append(f"Seed {seed} transitions: {compact}")

    prompt = f"""You are analyzing a Norse civilization simulator.
The simulator runs for 50 years with hidden parameters controlling:
- faction_aggression (how often settlements attack each other)
- forest_growth_rate (how fast forests spread and reclaim land)
- winter_severity (how often harsh winters kill settlements)
- trade_activity (how often ports and trade routes form)

Terrain classes: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain

Here are observed terrain transitions (initial_class -> final_class after 50 years):
{chr(10).join(summary_lines)}

Based on these transition patterns, estimate the hidden parameters as a JSON dict.
For example, high Settlement->Ruin rate implies high faction_aggression or winter_severity.
High Empty->Forest rate implies high forest_growth_rate.

Respond with ONLY a JSON object like:
{{"faction_aggression": 0.7, "forest_growth_rate": 0.4, "winter_severity": 0.3, "trade_activity": 0.5}}
Values are 0.0 (low) to 1.0 (high)."""

    try:
        response = model.generate_content(prompt)
        response_text = getattr(response, "text", "") or str(response)
        match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not match:
            print("Gemini response did not contain JSON; skipping hidden-parameter inference")
            return {}

        parsed = json.loads(match.group())
        if not isinstance(parsed, dict):
            return {}

        result: Dict[str, float] = {}
        for key in (
            "faction_aggression",
            "forest_growth_rate",
            "winter_severity",
            "trade_activity",
        ):
            value = parsed.get(key, 0.0)
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                value_f = 0.0
            result[key] = float(np.clip(value_f, 0.0, 1.0))
        return result
    except Exception as exc:
        print(f"Gemini inference failed; continuing without hidden params: {exc}")
        return {}


def run_queries(
    session: requests.Session,
    round_id: str,
    query_plan: List[Tuple[int, int, int, int, int]],
    observations: Dict[int, List[List[Optional[int]]]],
    initial_states: Sequence[Dict[str, Any]],
    gcp_project: str = "ai-nm26osl-1722",
    budget: int = TOTAL_BUDGET,
) -> Dict[str, Any]:
    """Run simulation queries and store observations and settlement snapshots."""
    if not observations:
        raise SolverError("Observations are empty")

    seeds = sorted(observations.keys())
    height = len(observations[seeds[0]])
    width = len(observations[seeds[0]][0]) if height else 0

    counts: Dict[int, np.ndarray] = {
        seed: np.zeros((height, width, MAX_CLASS), dtype=np.int32) for seed in seeds
    }
    settlements: Dict[int, List[Dict[str, Any]]] = {seed: [] for seed in seeds}
    query_log: List[Dict[str, Any]] = []
    hidden_params: Dict[str, float] = {}
    hidden_params_inferred = False

    base_positions = sorted(
        {(x, y, w, h) for _, x, y, w, h in query_plan}, key=lambda p: (p[1], p[0])
    )

    queries_used = 0

    def execute_query(seed_index: int, x: int, y: int, w: int, h: int) -> int:
        nonlocal queries_used, hidden_params, hidden_params_inferred
        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": x,
            "viewport_y": y,
            "viewport_w": w,
            "viewport_h": h,
        }

        result = request_json(session, "POST", "/astar-island/simulate", payload)

        grid = result.get("grid")
        if not isinstance(grid, list):
            raise SolverError(f"simulate response missing 'grid': {result}")

        new_cells = 0
        for dy, row in enumerate(grid):
            if not isinstance(row, list):
                raise SolverError("simulate response grid row is not a list")
            for dx, value in enumerate(row):
                if not isinstance(value, int) or value < 0:
                    raise SolverError(f"Unexpected terrain value at ({dx}, {dy}): {value}")

                yy = y + dy
                xx = x + dx
                if yy >= height or xx >= width:
                    continue

                if observations[seed_index][yy][xx] is None:
                    new_cells += 1
                observations[seed_index][yy][xx] = value
                counts[seed_index][yy, xx, value] += 1

        settlements_snapshot = result.get("settlements", [])
        if isinstance(settlements_snapshot, list):
            settlements[seed_index].append(
                {
                    "query_index": queries_used,
                    "viewport": {"x": x, "y": y, "w": w, "h": h},
                    "settlements": settlements_snapshot,
                }
            )

        queries_used += 1
        print(
            f"Query {queries_used}/{budget}: seed={seed_index} "
            f"viewport=({x},{y},{w},{h}) -> observed {new_cells} new cells"
        )

        if False and not hidden_params_inferred and queries_used >= min(25, budget):
            print("Inferring hidden simulator parameters from first-pass observations...")
            hidden_params = infer_hidden_params(
                observations=observations,
                initial_states=initial_states,
                project=gcp_project,
            )
            if hidden_params:
                print(f"Inferred hidden params: {hidden_params}")
            hidden_params_inferred = True

        query_log.append(
            {
                "query_index": queries_used,
                "seed_index": seed_index,
                "viewport": {"x": x, "y": y, "w": w, "h": h},
                "new_cells": new_cells,
            }
        )

        if queries_used < budget:
            time.sleep(0.05)

        return new_cells

    # Execute core coverage plan first.
    for seed_index, x, y, w, h in query_plan:
        if queries_used >= budget:
            break
        execute_query(seed_index, x, y, w, h)

    # Spend reserve queries adaptively after initial scan.
    MAX_RESERVE = 5
    reserve_used = 0
    while queries_used < budget and reserve_used < MAX_RESERVE:
        seed_index, x, y, w, h = choose_reserve_query(
            initial_states=initial_states,
            observations=observations,
            counts=counts,
            base_positions=base_positions,
        )
        execute_query(seed_index, x, y, w, h)
        reserve_used += 1

    serializable_counts = {str(seed): counts[seed].tolist() for seed in seeds}
    serializable_observations = {str(seed): observations[seed] for seed in seeds}
    serializable_settlements = {str(seed): settlements[seed] for seed in seeds}

    discover_terrain_classes(observations)

    observations_path = Path("observations.json")
    observations_payload = {
        "round_id": round_id,
        "queries_used": queries_used,
        "latest": serializable_observations,
        "counts": serializable_counts,
        "settlements": serializable_settlements,
        "hidden_params": hidden_params,
        "queries": query_log,
    }
    observations_path.write_text(json.dumps(observations_payload, indent=2), encoding="utf-8")
    print(f"Saved observations to {observations_path.resolve()}")

    return {
        "latest": observations,
        "counts": counts,
        "settlements": settlements,
        "hidden_params": hidden_params,
        "queries": query_log,
        "queries_used": queries_used,
    }


def empirical_class_prior(
    counts: Dict[int, np.ndarray],
    initial_states: Sequence[Dict[str, Any]],
) -> np.ndarray:
    class_counts = np.ones(CLASS_COUNT, dtype=np.float64)
    observed_total = 0.0

    for arr in counts.values():
        c = arr.sum(axis=(0, 1)).astype(np.float64)[:CLASS_COUNT]
        class_counts += c
        observed_total += float(c.sum())

    if observed_total == 0.0:
        class_counts = np.ones(CLASS_COUNT, dtype=np.float64)
        for state in initial_states:
            grid = np.asarray(state["grid"], dtype=np.int64)
            class_counts += np.bincount(grid.ravel(), minlength=CLASS_COUNT).astype(np.float64)

    return class_counts / class_counts.sum()


def compute_forest_adjacency(initial_grid: np.ndarray) -> np.ndarray:
    height, width = initial_grid.shape
    forest = initial_grid == 4
    forest_adj = np.zeros((height, width), dtype=bool)

    ys, xs = np.where(forest)
    for y, x in zip(ys.tolist(), xs.tolist()):
        y0 = max(0, y - 1)
        y1 = min(height, y + 2)
        x0 = max(0, x - 1)
        x1 = min(width, x + 2)
        forest_adj[y0:y1, x0:x1] = True

    # Adjacency should describe neighbors, not necessarily the forest tile itself.
    forest_adj[forest] = False
    return forest_adj


def compute_settlement_maps(
    initial_state: Dict[str, Any],
    width: int,
    height: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    settlements = initial_state.get("settlements", [])

    has_settlement = np.zeros((height, width), dtype=bool)
    has_port = np.zeros((height, width), dtype=bool)
    near_settlement = np.zeros((height, width), dtype=bool)

    if not isinstance(settlements, list):
        return has_settlement, has_port, near_settlement

    for s in settlements:
        if not isinstance(s, dict):
            continue

        x = int(s.get("x", -1))
        y = int(s.get("y", -1))
        if not (0 <= x < width and 0 <= y < height):
            continue

        has_settlement[y, x] = True
        has_port[y, x] = bool(s.get("has_port", False))

        y0 = max(0, y - 2)
        y1 = min(height, y + 3)
        x0 = max(0, x - 2)
        x1 = min(width, x + 3)
        near_settlement[y0:y1, x0:x1] = True

    return has_settlement, has_port, near_settlement


def build_predictions(
    initial_states: Sequence[Dict[str, Any]],
    observation_data: Dict[str, Any],
    width: int,
    height: int,
    seeds_count: int,
) -> Dict[int, np.ndarray]:
    """Build per-seed probability tensors (H x W x 6)."""
    counts: Dict[int, np.ndarray] = observation_data["counts"]
    settlements_by_seed: Dict[int, List[Dict[str, Any]]] = observation_data.get("settlements", {})
    hidden_params = observation_data.get("hidden_params", {})
    faction_aggression = float(np.clip(float(hidden_params.get("faction_aggression", 0.0)), 0.0, 1.0))
    forest_growth_rate = float(
        np.clip(float(hidden_params.get("forest_growth_rate", 0.0)), 0.0, 1.0)
    )
    winter_severity = float(np.clip(float(hidden_params.get("winter_severity", 0.0)), 0.0, 1.0))
    trade_activity = float(np.clip(float(hidden_params.get("trade_activity", 0.0)), 0.0, 1.0))

    global_transition = compute_transition_matrix(initial_states, counts)
    global_prior = empirical_class_prior(counts, initial_states)

    predictions: Dict[int, np.ndarray] = {}

    for seed_index in range(seeds_count):
        init_grid = np.asarray(initial_states[seed_index]["grid"], dtype=np.int64)
        seed_counts = counts[seed_index]
        sample_totals = seed_counts.sum(axis=2)
        observed_mask = sample_totals > 0

        _, has_port, near_settlement = compute_settlement_maps(
            initial_states[seed_index], width=width, height=height
        )
        forest_adj = compute_forest_adjacency(init_grid)

        coastal = np.zeros((height, width), dtype=bool)
        coastal[:2, :] = True
        coastal[-2:, :] = True
        coastal[:, :2] = True
        coastal[:, -2:] = True
        coastal |= init_grid == 2

        seed_pred = np.zeros((height, width, CLASS_COUNT), dtype=np.float64)
        seed_settlement_snapshots = settlements_by_seed.get(seed_index, [])

        for y in range(height):
            for x in range(width):
                init_class = int(init_grid[y, x])
                cell_counts = seed_counts[y, x, :CLASS_COUNT].astype(np.float64)
                n_obs = int(cell_counts.sum())

                if n_obs > 0:
                    empirical = cell_counts / cell_counts.sum()
                    mode = int(np.argmax(empirical))
                    dist = np.full(CLASS_COUNT, 0.10 / (CLASS_COUNT - 1), dtype=np.float64)
                    dist[mode] = 0.90

                    # If re-queried, incorporate more of the empirical stochasticity.
                    if n_obs > 1:
                        blend = min(0.85, 0.30 + 0.10 * (n_obs - 1))
                        dist = (1.0 - blend) * dist + blend * empirical

                    seed_pred[y, x] = dist
                    continue

                if init_class == 5:
                    dist = np.full(CLASS_COUNT, 0.08 / (CLASS_COUNT - 1), dtype=np.float64)
                    dist[5] = 0.92
                    seed_pred[y, x] = dist
                    continue

                dist = global_transition[init_class].copy()

                if init_class in (1, 2):
                    dist[1] += 0.30
                    dist[3] += 0.20
                    if init_class == 2 or has_port[y, x]:
                        dist[2] += 0.10

                if near_settlement[y, x]:
                    dist[1] += 0.15
                    dist[3] += 0.10

                if coastal[y, x]:
                    dist[2] += 0.05

                if init_class == 4:
                    forest_pref = np.array([0.10, 0.02, 0.01, 0.07, 0.75, 0.05], dtype=np.float64)
                    dist = 0.55 * dist + 0.45 * forest_pref

                if init_class == 0 and forest_adj[y, x]:
                    dist[4] += 0.15

                if init_class == 3:
                    dist[4] += 0.10
                    dist[0] += 0.10

                # Mountains are static; for non-mountains, mountain transitions should be rare.
                dist[5] *= 0.35

                seed_pred[y, x] = dist

        # Hidden-parameter-driven adjustment pass.
        unobserved_mask = ~observed_mask
        if faction_aggression > 0.0:
            conflict_mask = near_settlement & unobserved_mask
            seed_pred[conflict_mask, 3] += 0.30 * faction_aggression
            seed_pred[conflict_mask, 1] -= 0.10 * faction_aggression

        if forest_growth_rate > 0.0:
            forest_growth_mask = ((init_grid == 0) | (init_grid == 3)) & forest_adj & unobserved_mask
            seed_pred[forest_growth_mask, 4] += 0.35 * forest_growth_rate

        if winter_severity > 0.0:
            winter_mask = unobserved_mask
            seed_pred[winter_mask, 3] += 0.28 * winter_severity
            seed_pred[winter_mask, 1] -= 0.18 * winter_severity

        if trade_activity > 0.0:
            trade_mask = coastal & near_settlement & unobserved_mask
            seed_pred[trade_mask, 2] += 0.30 * trade_activity

        seed_pred = np.maximum(seed_pred, 0.0)

        # Incorporate observed settlement outcomes from simulation snapshots.
        for snapshot in seed_settlement_snapshots:
            settlements_snapshot = snapshot.get("settlements", [])
            if not isinstance(settlements_snapshot, list):
                continue
            for s in settlements_snapshot:
                if not isinstance(s, dict):
                    continue
                x = s.get("x")
                y = s.get("y")
                if not isinstance(x, int) or not isinstance(y, int):
                    continue
                if not (0 <= x < width and 0 <= y < height):
                    continue

                alive = bool(s.get("alive", True))
                s_has_port = bool(s.get("has_port", False))

                if alive:
                    seed_pred[y, x, 1] += 0.5
                    if s_has_port:
                        seed_pred[y, x, 2] += 0.3
                else:
                    seed_pred[y, x, 3] += 0.5
                    seed_pred[y, x, 1] -= 0.2

                seed_pred[y, x] = np.maximum(seed_pred[y, x], 0.0)

        # Spatial smoothing for unobserved cells.
        for c in range(CLASS_COUNT):
            blurred = gaussian_filter(seed_pred[:, :, c], sigma=1.5)
            seed_pred[:, :, c] = np.where(
                observed_mask,
                seed_pred[:, :, c],
                0.65 * seed_pred[:, :, c] + 0.35 * blurred,
            )

        # Mandatory floor and renormalization.
        seed_pred = np.maximum(seed_pred, 0.01)
        seed_pred = seed_pred / seed_pred.sum(axis=2, keepdims=True)

        predictions[seed_index] = seed_pred

        output_path = Path(f"predictions_seed_{seed_index}.npy")
        np.save(output_path, seed_pred)

        total_cells = width * height
        observed_cells = int(observed_mask.sum())
        coverage = 100.0 * observed_cells / float(total_cells) if total_cells else 0.0
        alive_count = 0
        dead_count = 0
        for snapshot in seed_settlement_snapshots:
            settlements_snapshot = snapshot.get("settlements", [])
            if not isinstance(settlements_snapshot, list):
                continue
            for s in settlements_snapshot:
                if not isinstance(s, dict):
                    continue
                if bool(s.get("alive", True)):
                    alive_count += 1
                else:
                    dead_count += 1
        print(
            f"Seed {seed_index}: {coverage:.2f}% observed, "
            f"{len(seed_settlement_snapshots)} settlement snapshots, "
            f"{alive_count} alive settlements, {dead_count} dead settlements seen"
        )

    return predictions


def validate_prediction_tensor(pred: np.ndarray, seed_index: int) -> None:
    if pred.ndim != 3 or pred.shape[2] != CLASS_COUNT:
        raise SolverError(
            f"Prediction for seed {seed_index} has invalid shape {pred.shape}, "
            f"expected (height, width, {CLASS_COUNT})"
        )

    sums = pred.sum(axis=2)
    max_deviation = float(np.max(np.abs(sums - 1.0)))
    if max_deviation > 1e-3:
        raise SolverError(
            f"Prediction for seed {seed_index} does not sum to 1.0 per cell "
            f"(max deviation={max_deviation:.6f})"
        )


def summarize_submit_response(response: Any) -> str:
    if isinstance(response, dict):
        keys = [
            "status",
            "message",
            "score",
            "seed_score",
            "total_score",
            "rank",
        ]
        parts = [f"{k}={response[k]}" for k in keys if k in response]
        if parts:
            return ", ".join(parts)
        return json.dumps(response, ensure_ascii=True)
    return str(response)


def submit_all(
    session: requests.Session,
    round_id: str,
    predictions: Dict[int, np.ndarray],
    dry_run: bool = False,
) -> Dict[int, str]:
    """Submit predictions for all seeds."""
    statuses: Dict[int, str] = {}

    for seed_index in sorted(predictions.keys()):
        pred = predictions[seed_index]
        validate_prediction_tensor(pred, seed_index)

        if dry_run:
            status = f"DRY RUN: would submit seed {seed_index}"
            print(status)
            statuses[seed_index] = status
            continue

        payload = {
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": pred.tolist(),
        }

        response = request_json(session, "POST", "/astar-island/submit", payload)
        summary = summarize_submit_response(response)
        print(f"Submitted seed {seed_index}: {summary}")
        statuses[seed_index] = summary

    return statuses


def load_predictions_from_disk(seeds_count: int) -> Dict[int, np.ndarray]:
    predictions: Dict[int, np.ndarray] = {}

    for seed_index in range(seeds_count):
        path = Path(f"predictions_seed_{seed_index}.npy")
        if not path.exists():
            raise SolverError(f"Missing prediction file: {path}")
        predictions[seed_index] = np.load(path)

    return predictions


def create_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
    )
    return session


def print_summary(
    width: int,
    height: int,
    seeds_count: int,
    queries_used: int,
    observations: Dict[int, List[List[Optional[int]]]],
    statuses: Dict[int, str],
) -> None:
    print("\nSummary")
    print(f"Queries used: {queries_used}/{TOTAL_BUDGET}")

    total_cells = width * height
    for seed_index in range(seeds_count):
        observed_cells = sum(
            1
            for row in observations[seed_index]
            for value in row
            if value is not None
        )
        coverage = 100.0 * observed_cells / total_cells
        print(
            f"Coverage seed {seed_index}: {coverage:.2f}% "
            f"({observed_cells}/{total_cells})"
        )

    saved = ", ".join(f"predictions_seed_{i}.npy" for i in range(seeds_count))
    print(f"Predictions saved: {saved}")

    print("Submission status per seed:")
    for seed_index in range(seeds_count):
        print(f"  seed {seed_index}: {statuses.get(seed_index, 'not submitted')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Astar Island solver")
    parser.add_argument("--token", type=str, default=None, help="JWT bearer token")
    parser.add_argument("--dry-run", action="store_true", help="Skip submit POST calls")
    parser.add_argument(
        "--submit-only",
        action="store_true",
        help="Load saved predictions from disk and submit only",
    )
    parser.add_argument(
        "--no-query",
        action="store_true",
        help="Use only initial states (skip simulate calls)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load observations.json and skip simulate calls",
    )
    parser.add_argument(
        "--gcp-project",
        type=str,
        default="ai-nm26osl-1722",
        help="GCP project ID for Vertex AI",
    )
    args = parser.parse_args()

    token = args.token or os.getenv("ASTAR_ISLAND_TOKEN")
    if not token:
        parser.error("Token required: pass --token or set ASTAR_ISLAND_TOKEN")

    try:
        session = create_session(token)
        round_id, width, height, seeds_count, initial_states = load_round(session)

        if args.submit_only:
            predictions = load_predictions_from_disk(seeds_count)
            statuses = submit_all(session, round_id, predictions, dry_run=args.dry_run)

            # submit-only mode cannot infer query coverage from API calls this run.
            empty_observations = initialize_observations(width, height, seeds_count)
            print_summary(
                width=width,
                height=height,
                seeds_count=seeds_count,
                queries_used=0,
                observations=empty_observations,
                statuses=statuses,
            )
            return

        if args.resume:
            obs_data = json.loads(Path("observations.json").read_text(encoding="utf-8"))
            observations = {int(k): v for k, v in obs_data["latest"].items()}
            counts = {int(k): np.array(v, dtype=np.int32) for k, v in obs_data["counts"].items()}
            settlements = {int(k): v for k, v in obs_data["settlements"].items()}
            queries_used = int(obs_data["queries_used"])
            observation_data = {
                "latest": observations,
                "counts": counts,
                "settlements": settlements,
                "hidden_params": obs_data.get("hidden_params", {}),
                "queries": obs_data.get("queries", []),
                "queries_used": queries_used,
            }
            print(f"Resumed from observations.json ({queries_used} queries loaded)")
        else:
            query_plan = plan_queries(width, height, seeds_count, budget=TOTAL_BUDGET)
            observations = initialize_observations(width, height, seeds_count)

            if args.no_query:
                print("--no-query enabled: skipping simulate calls")
                observation_data = {
                    "latest": observations,
                    "counts": {
                        seed: np.zeros((height, width, CLASS_COUNT), dtype=np.int32)
                        for seed in range(seeds_count)
                    },
                    "settlements": {seed: [] for seed in range(seeds_count)},
                    "queries": [],
                    "queries_used": 0,
                }
            else:
                observation_data = run_queries(
                    session=session,
                    round_id=round_id,
                    query_plan=query_plan,
                    observations=observations,
                    initial_states=initial_states,
                    gcp_project=args.gcp_project,
                    budget=TOTAL_BUDGET,
                )

        predictions = build_predictions(
            initial_states=initial_states,
            observation_data=observation_data,
            width=width,
            height=height,
            seeds_count=seeds_count,
        )

        statuses = submit_all(
            session=session,
            round_id=round_id,
            predictions=predictions,
            dry_run=args.dry_run,
        )

        print_summary(
            width=width,
            height=height,
            seeds_count=seeds_count,
            queries_used=int(observation_data["queries_used"]),
            observations=observation_data["latest"],
            statuses=statuses,
        )

    except SolverError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
