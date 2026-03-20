#!/usr/bin/env python3
"""Solver for the Astar Island challenge (NM i AI 2026).

Architecture:
- Map all terrain codes to 6 output classes immediately on observation
- Empirical cell distribution (counts/n_obs) approximates ground truth
- Reserve queries re-sample high-entropy cells for better distribution estimates
- Settlement stats (food, population) predict collapse probability
- Bayesian posterior blends empirical samples with terrain-aware prior
"""
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
MAX_VIEWPORT = 15
TOTAL_BUDGET = 50

# Maps raw terrain codes from API responses to prediction class indices (0-5).
# This is the critical mapping — Ocean(10) and Plains(11) both map to class 0 (Empty).
# Internal codes: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain,
#                 10=Ocean (static), 11=Plains (dynamic)
RAW_CODE_TO_CLASS: Dict[int, int] = {
    0: 0,   # Empty → class 0
    1: 1,   # Settlement → class 1
    2: 2,   # Port → class 2
    3: 3,   # Ruin → class 3
    4: 4,   # Forest → class 4
    5: 5,   # Mountain → class 5
    10: 0,  # Ocean → class 0 (static, never changes)
    11: 0,  # Plains → class 0 (dynamic, can be settled/forested)
}

# Baseline priors per initial terrain code → class distribution.
# Derived from Round 2 empirical transitions + domain knowledge.
# Updated by transition learning each round.
# Format: raw_code → [p_empty, p_settlement, p_port, p_ruin, p_forest, p_mountain]
TERRAIN_PRIORS: Dict[int, np.ndarray] = {
    10: np.array([0.97, 0.01, 0.01, 0.01, 0.00, 0.00]),  # Ocean — static class 0
    11: np.array([0.84, 0.13, 0.00, 0.01, 0.02, 0.00]),  # Plains — mostly stays empty
     0: np.array([0.70, 0.10, 0.02, 0.05, 0.13, 0.00]),  # Empty land
     1: np.array([0.46, 0.46, 0.00, 0.02, 0.06, 0.00]),  # Settlement — 50/50 survive
     2: np.array([0.36, 0.18, 0.27, 0.10, 0.09, 0.00]),  # Port
     3: np.array([0.20, 0.10, 0.02, 0.35, 0.30, 0.03]),  # Ruin — reclaimed or persists
     4: np.array([0.03, 0.09, 0.02, 0.01, 0.85, 0.00]),  # Forest — mostly persists
     5: np.array([0.00, 0.00, 0.00, 0.00, 0.00, 1.00]),  # Mountain — static
}


class SolverError(RuntimeError):
    """Raised for recoverable solver failures."""


class BudgetExhaustedError(SolverError):
    """Raised when query budget is exhausted — stop querying, proceed to predict."""


def code_to_class(raw_code: int) -> int:
    """Map a raw terrain code to a prediction class index (0-5)."""
    return RAW_CODE_TO_CLASS.get(raw_code, 0)  # unknown codes → class 0


def terrain_prior(raw_code: int) -> np.ndarray:
    """Return baseline prior distribution for an initial terrain code."""
    p = TERRAIN_PRIORS.get(raw_code, None)
    if p is None:
        # Unknown code — treat as Plains-like (mostly empty, can be settled)
        p = np.array([0.80, 0.10, 0.01, 0.02, 0.07, 0.00])
    p = np.maximum(p, 0.01)
    return p / p.sum()


def request_json(
    session: requests.Session,
    method: str,
    path: str,
    payload: Optional[Dict[str, Any]] = None,
    retry_5xx: bool = True,
) -> Any:
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
                text = response.text or ""
                if "budget" in text.lower() or "exhausted" in text.lower():
                    raise BudgetExhaustedError(f"Query budget exhausted: {text}")
                raise SolverError(f"Rate limited (429) on {method} {path}: {text}")
            raise SolverError(f"HTTP {status} on {method} {path}: {response.text}")
        try:
            return response.json()
        except ValueError as exc:
            print(f"JSON parse error on {method} {path}. Raw response:")
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


def load_round(session: requests.Session) -> Tuple[str, int, int, int, List[Dict[str, Any]]]:
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
    round_weight = active.get("round_weight", detail.get("round_weight", 1.0))
    print(f"Active round: {round_number} (weight: {round_weight})")
    print(f"Map size: {width}x{height} | Seeds: {seeds_count}")
    return round_id, width, height, seeds_count, initial_states


def tiling_offsets(size: int, window: int) -> List[int]:
    if size <= window:
        return [0]
    offsets = list(range(0, size - window + 1, window))
    last = size - window
    if offsets[-1] != last:
        offsets.append(last)
    return offsets


def candidate_viewports(width: int, height: int, window: int = MAX_VIEWPORT, stride: int = 5) -> List[Tuple[int, int, int, int]]:
    x_offsets = [0] if width <= window else list(range(0, width - window + 1, stride))
    if width > window and x_offsets[-1] != width - window:
        x_offsets.append(width - window)
    y_offsets = [0] if height <= window else list(range(0, height - window + 1, stride))
    if height > window and y_offsets[-1] != height - window:
        y_offsets.append(height - window)
    candidates = [(x, y, min(window, width-x), min(window, height-y)) for y in y_offsets for x in x_offsets]
    return sorted(set(candidates), key=lambda p: (p[1], p[0], p[3], p[2]))


def plan_queries(width: int, height: int, seeds_count: int, budget: int = TOTAL_BUDGET) -> List[Tuple[int, int, int, int, int]]:
    positions = [(x, y, min(MAX_VIEWPORT, width-x), min(MAX_VIEWPORT, height-y))
                 for y in tiling_offsets(height, MAX_VIEWPORT)
                 for x in tiling_offsets(width, MAX_VIEWPORT)]
    query_plan = [(s, x, y, w, h) for s in range(seeds_count) for x, y, w, h in positions]
    if len(query_plan) > budget:
        query_plan = query_plan[:budget]
    reserve = max(0, budget - len(query_plan))
    print(f"Planned {len(query_plan)} core queries ({len(positions)} per seed), reserve budget: {reserve}")
    return query_plan


def initialize_observations(width: int, height: int, seeds_count: int) -> Dict[int, List[List[Optional[int]]]]:
    return {s: [[None]*width for _ in range(height)] for s in range(seeds_count)}


def observed_mask_from_latest(grid: List[List[Optional[int]]]) -> np.ndarray:
    h, w = len(grid), len(grid[0]) if grid else 0
    mask = np.zeros((h, w), dtype=bool)
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            if v is not None:
                mask[y, x] = True
    return mask


def compute_cell_entropy(counts_row: np.ndarray) -> float:
    """Shannon entropy of a count vector. High entropy = uncertain outcome."""
    total = float(counts_row.sum())
    if total <= 0:
        return 0.0
    p = counts_row / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def choose_reserve_query(
    initial_states: Sequence[Dict[str, Any]],
    observations: Dict[int, List[List[Optional[int]]]],
    counts: Dict[int, np.ndarray],
    candidate_positions: List[Tuple[int, int, int, int]],
    viewport_hits: Dict[int, Dict[Tuple[int, int, int, int], int]],
) -> Tuple[int, int, int, int, int]:
    """
    Choose the best reserve query.
    Priority:
    1. If any seed has <100% coverage: cover unseen cells first
    2. Otherwise: re-sample cells with highest empirical entropy
       (multiple samples of stochastic cells improve distribution estimate)
    """
    seeds = sorted(observations.keys())
    height = len(next(iter(observations.values())))
    width = len(next(iter(observations.values()))[0])
    total_cells = width * height

    masks = {s: observed_mask_from_latest(observations[s]) for s in seeds}
    coverage = {s: float(masks[s].sum()) / total_cells for s in seeds}
    min_coverage = min(coverage.values())

    if min_coverage < 1.0:
        # Still uncovered cells — prioritize coverage
        target_seed = min(seeds, key=lambda s: coverage[s])
    else:
        # Full coverage — pick seed with highest sample entropy (most uncertain)
        # Entropy = average over cells weighted by how stochastic they are
        entropy_by_seed = {}
        for s in seeds:
            cell_entropy = np.array([
                compute_cell_entropy(counts[s][y, x])
                for y in range(height) for x in range(width)
            ])
            entropy_by_seed[s] = float(cell_entropy.mean())
        target_seed = max(entropy_by_seed, key=entropy_by_seed.get)

    # Score each candidate viewport for target seed
    init_grid = np.asarray(initial_states[target_seed]["grid"], dtype=np.int64)
    target_mask = masks[target_seed]
    sample_totals = counts[target_seed].sum(axis=2).astype(np.float64)

    # Per-cell entropy from empirical counts
    cell_entropy_map = np.zeros((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            cell_entropy_map[y, x] = compute_cell_entropy(counts[target_seed][y, x])

    # Boost entropy for inherently stochastic terrain types (Settlement, Forest)
    stochastic_mask = np.isin(init_grid, [1, 2, 3, 4, 11])
    cell_entropy_map[stochastic_mask] = np.maximum(
        cell_entropy_map[stochastic_mask], 0.3
    )

    best_score, best_position = -1.0, candidate_positions[0]
    for x, y, w, h in candidate_positions:
        region_entropy = cell_entropy_map[y:y+h, x:x+w].mean()
        region_unseen = float((~target_mask[y:y+h, x:x+w]).mean())
        region_samples = float(sample_totals[y:y+h, x:x+w].mean())
        revisit_penalty = viewport_hits[target_seed].get((x, y, w, h), 0) * 0.1

        if min_coverage < 1.0:
            score = 3.0 * region_unseen + 0.5 * region_entropy - revisit_penalty
        else:
            # Re-sampling: prefer high-entropy cells, penalise over-sampled regions
            score = 2.0 * region_entropy - 0.3 * np.log1p(region_samples) - revisit_penalty

        if score > best_score:
            best_score, best_position = score, (x, y, w, h)

    x, y, w, h = best_position
    return target_seed, x, y, w, h


def infer_hidden_params(
    observations: Dict[int, List[List[Optional[int]]]],
    initial_states: Sequence[Dict[str, Any]],
    counts: Dict[int, np.ndarray],
    project: str = "ai-nm26osl-1722",
    location: str = "us-central1",
) -> Dict[str, float]:
    """Infer hidden simulator parameters via Gemini. Uses mapped class transitions."""
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
    except Exception:
        print("Vertex AI SDK unavailable; skipping hidden-parameter inference")
        return {}
    try:
        vertexai.init(project=project, location=location)
        model = GenerativeModel("gemini-2.0-flash-002")
    except Exception as exc:
        print(f"Vertex AI error: {exc}")
        return {}

    classes = {0: "Empty/Ocean/Plains", 1: "Settlement", 2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain"}
    summary_lines: List[str] = []
    for seed in sorted(observations.keys()):
        init = initial_states[seed]["grid"]
        # Build transition: initial_code → final_class
        transitions: Dict[Tuple[int, int], int] = {}
        for y, row in enumerate(observations[seed]):
            for x, raw_val in enumerate(row):
                if raw_val is None:
                    continue
                init_code = int(init[y][x])
                final_class = code_to_class(raw_val)
                key = (init_code, final_class)
                transitions[key] = transitions.get(key, 0) + 1
        compact = ", ".join(
            f"{a}->{b}({classes.get(b,'?')}):{cnt}"
            for (a, b), cnt in sorted(transitions.items())
            if cnt > 5
        )
        summary_lines.append(f"Seed {seed}: {compact}")

        # Also report settlement survival
        alive = sum(1 for snaps in [] for s in snaps for st in s.get("settlements", []) if st.get("alive", True))

    # Also report transition entropy as a summary
    settlement_survival = []
    for seed in sorted(counts.keys()):
        c = counts[seed]
        settlement_cells = np.asarray(initial_states[seed]["grid"], dtype=np.int64) == 1
        if np.any(settlement_cells):
            sc = c[settlement_cells][:, :CLASS_COUNT].sum(axis=0)
            total = sc.sum()
            if total > 0:
                surv = sc[1] / total
                ruin = sc[3] / total
                settlement_survival.append(f"seed{seed}: {surv:.0%} survive, {ruin:.0%} ruin")

    prompt = f"""You are analyzing a Norse civilization simulator that runs for 50 years.
Hidden parameters control: faction_aggression, forest_growth_rate, winter_severity, trade_activity.

Terrain classes: 0=Empty/Plains/Ocean, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain

Observed transitions (initial_code → final_class):
{chr(10).join(summary_lines)}

Settlement survival rates: {'; '.join(settlement_survival) if settlement_survival else 'unknown'}

Infer the hidden parameters. High Settlement→Ruin implies high aggression or winter severity.
High Plains→Settlement implies high expansion. High Empty→Forest implies high forest growth.
100% settlement survival implies near-zero aggression and winter severity.

Respond ONLY with JSON:
{{"faction_aggression": 0.0-1.0, "forest_growth_rate": 0.0-1.0, "winter_severity": 0.0-1.0, "trade_activity": 0.0-1.0}}"""

    try:
        response = model.generate_content(prompt)
        text = getattr(response, "text", "") or str(response)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            print("Gemini: no JSON in response")
            return {}
        parsed = json.loads(match.group())
        result = {}
        for key in ("faction_aggression", "forest_growth_rate", "winter_severity", "trade_activity"):
            try:
                result[key] = float(np.clip(float(parsed.get(key, 0.0)), 0.0, 1.0))
            except (TypeError, ValueError):
                result[key] = 0.0
        print(f"Gemini inferred: {result}")
        return result
    except Exception as exc:
        print(f"Gemini failed: {exc}")
        return {}


def _save_observations(
    round_id: str, queries_used: int,
    observations: Dict[int, List[List[Optional[int]]]],
    counts: Dict[int, np.ndarray],
    settlements: Dict[int, List[Dict[str, Any]]],
    hidden_params: Dict[str, float],
    query_log: List[Dict[str, Any]],
    seeds: List[int],
) -> None:
    """Save after every query — crash-safe."""
    Path("observations.json").write_text(json.dumps({
        "round_id": round_id,
        "queries_used": queries_used,
        "latest": {str(s): observations[s] for s in seeds},
        "counts": {str(s): counts[s].tolist() for s in seeds},
        "settlements": {str(s): settlements[s] for s in seeds},
        "hidden_params": hidden_params,
        "queries": query_log,
    }, indent=2), encoding="utf-8")


def run_queries(
    session: requests.Session,
    round_id: str,
    query_plan: List[Tuple[int, int, int, int, int]],
    observations: Dict[int, List[List[Optional[int]]]],
    initial_states: Sequence[Dict[str, Any]],
    gcp_project: str = "ai-nm26osl-1722",
    budget: int = TOTAL_BUDGET,
) -> Dict[str, Any]:
    if not observations:
        raise SolverError("Observations are empty")
    seeds = sorted(observations.keys())
    height = len(observations[seeds[0]])
    width = len(observations[seeds[0]][0]) if height else 0

    # Counts stored in CLASS_COUNT space (codes mapped to classes)
    counts: Dict[int, np.ndarray] = {
        s: np.zeros((height, width, CLASS_COUNT), dtype=np.int32) for s in seeds
    }
    # Raw codes stored separately for terrain-type based priors
    raw_codes: Dict[int, List[List[Optional[int]]]] = {
        s: [[None]*width for _ in range(height)] for s in seeds
    }
    settlements: Dict[int, List[Dict[str, Any]]] = {s: [] for s in seeds}
    # Settlement stats: track food/population for collapse prediction
    settlement_stats: Dict[int, Dict[Tuple[int, int], List[Dict[str, Any]]]] = {s: {} for s in seeds}
    query_log: List[Dict[str, Any]] = []
    hidden_params: Dict[str, float] = {}

    base_positions = sorted({(x, y, w, h) for _, x, y, w, h in query_plan}, key=lambda p: (p[1], p[0]))
    candidate_positions = sorted(
        set(base_positions) | set(candidate_viewports(width, height, MAX_VIEWPORT, stride=5)),
        key=lambda p: (p[1], p[0], p[3], p[2]),
    )
    viewport_hits: Dict[int, Dict[Tuple[int, int, int, int], int]] = {s: {} for s in seeds}
    queries_used = 0

    def execute_query(seed_index: int, x: int, y: int, w: int, h: int) -> int:
        nonlocal queries_used, hidden_params
        result = request_json(session, "POST", "/astar-island/simulate", {
            "round_id": round_id, "seed_index": seed_index,
            "viewport_x": x, "viewport_y": y, "viewport_w": w, "viewport_h": h,
        })
        grid = result.get("grid")
        if not isinstance(grid, list):
            raise SolverError(f"simulate response missing 'grid': {result}")

        new_cells = 0
        for dy, row in enumerate(grid):
            if not isinstance(row, list):
                raise SolverError("grid row not a list")
            for dx, raw_val in enumerate(row):
                if not isinstance(raw_val, int) or raw_val < 0:
                    raise SolverError(f"Bad terrain value at ({dx},{dy}): {raw_val}")
                yy, xx = y + dy, x + dx
                if yy >= height or xx >= width:
                    continue
                if observations[seed_index][yy][xx] is None:
                    new_cells += 1
                observations[seed_index][yy][xx] = raw_val
                raw_codes[seed_index][yy][xx] = raw_val
                # KEY: map raw code to class before storing in counts
                mapped = code_to_class(raw_val)
                counts[seed_index][yy, xx, mapped] += 1

        # Process settlement stats from simulate response
        snap = result.get("settlements", [])
        if isinstance(snap, list):
            settlements[seed_index].append({
                "query_index": queries_used,
                "viewport": {"x": x, "y": y, "w": w, "h": h},
                "settlements": snap,
            })
            for s in snap:
                if isinstance(s, dict):
                    sx, sy = s.get("x"), s.get("y")
                    if isinstance(sx, int) and isinstance(sy, int):
                        key = (sx, sy)
                        if key not in settlement_stats[seed_index]:
                            settlement_stats[seed_index][key] = []
                        settlement_stats[seed_index][key].append({
                            "alive": s.get("alive", True),
                            "has_port": s.get("has_port", False),
                            "population": s.get("population", None),
                            "food": s.get("food", None),
                            "wealth": s.get("wealth", None),
                            "defense": s.get("defense", None),
                            "owner_id": s.get("owner_id", None),
                        })

        queries_used += 1
        print(f"Query {queries_used}/{budget}: seed={seed_index} viewport=({x},{y},{w},{h}) -> {new_cells} new cells")
        query_log.append({"query_index": queries_used, "seed_index": seed_index,
                          "viewport": {"x": x, "y": y, "w": w, "h": h}, "new_cells": new_cells})
        viewport_hits[seed_index][(x, y, w, h)] = viewport_hits[seed_index].get((x, y, w, h), 0) + 1
        _save_observations(round_id, queries_used, observations, counts, settlements, hidden_params, query_log, seeds)
        if queries_used < budget:
            time.sleep(0.05)
        return new_cells

    # Core coverage
    for seed_index, x, y, w, h in query_plan:
        if queries_used >= budget:
            break
        try:
            execute_query(seed_index, x, y, w, h)
        except BudgetExhaustedError as exc:
            print(f"Budget exhausted (core): {exc}")
            break

    # Reserve: re-sample high-entropy cells
    while queries_used < budget:
        seed_index, x, y, w, h = choose_reserve_query(
            initial_states, observations, counts, candidate_positions, viewport_hits
        )
        try:
            execute_query(seed_index, x, y, w, h)
        except BudgetExhaustedError as exc:
            print(f"Budget exhausted (reserve): {exc}")
            break

    print(f"All queries done. Saved to observations.json ({queries_used} queries)")
    all_codes = sorted({v for g in observations.values() for row in g for v in row if v is not None})
    print(f"Discovered terrain codes: {all_codes}")

    # Gemini inference
    print("Inferring hidden params via Gemini...")
    hidden_params = infer_hidden_params(observations, initial_states, counts, project=gcp_project)
    if hidden_params:
        _save_observations(round_id, queries_used, observations, counts, settlements, hidden_params, query_log, seeds)

    return {
        "latest": observations,
        "raw_codes": raw_codes,
        "counts": counts,
        "settlements": settlements,
        "settlement_stats": settlement_stats,
        "hidden_params": hidden_params,
        "queries": query_log,
        "queries_used": queries_used,
    }


def normalize_distribution(dist: np.ndarray, floor: float = 0.01) -> np.ndarray:
    clipped = np.maximum(dist.astype(np.float64, copy=False), floor)
    total = float(clipped.sum())
    if total <= 0.0:
        return np.full(CLASS_COUNT, 1.0 / CLASS_COUNT)
    return clipped / total


def compute_forest_adjacency(init_grid: np.ndarray) -> np.ndarray:
    height, width = init_grid.shape
    forest = init_grid == 4
    adj = np.zeros((height, width), dtype=bool)
    ys, xs = np.where(forest)
    for y, x in zip(ys.tolist(), xs.tolist()):
        adj[max(0,y-1):min(height,y+2), max(0,x-1):min(width,x+2)] = True
    adj[forest] = False
    return adj


def compute_settlement_maps(initial_state: Dict[str, Any], width: int, height: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    has_settlement = np.zeros((height, width), dtype=bool)
    has_port = np.zeros((height, width), dtype=bool)
    near_settlement = np.zeros((height, width), dtype=bool)
    for s in initial_state.get("settlements", []):
        if not isinstance(s, dict):
            continue
        x, y = int(s.get("x", -1)), int(s.get("y", -1))
        if not (0 <= x < width and 0 <= y < height):
            continue
        has_settlement[y, x] = True
        has_port[y, x] = bool(s.get("has_port", False))
        near_settlement[max(0,y-2):min(height,y+3), max(0,x-2):min(width,x+3)] = True
    return has_settlement, has_port, near_settlement


def build_predictions(
    initial_states: Sequence[Dict[str, Any]],
    observation_data: Dict[str, Any],
    width: int,
    height: int,
    seeds_count: int,
    save_outputs: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Build per-seed H×W×6 probability tensors.

    Key design:
    - counts[seed][y,x] is already in class space (codes mapped via code_to_class)
    - With n_obs samples, empirical = counts/n_obs approximates true distribution
    - Bayesian posterior: blend empirical samples with terrain prior
    - alpha controls prior strength: small = trust observations, large = trust prior
    """
    counts: Dict[int, np.ndarray] = observation_data["counts"]
    settlement_stats: Dict[int, Dict] = observation_data.get("settlement_stats", {})
    hidden_params = observation_data.get("hidden_params", {})
    faction_aggression = float(np.clip(hidden_params.get("faction_aggression", 0.0), 0.0, 1.0))
    forest_growth_rate = float(np.clip(hidden_params.get("forest_growth_rate", 0.0), 0.0, 1.0))
    winter_severity = float(np.clip(hidden_params.get("winter_severity", 0.0), 0.0, 1.0))
    trade_activity = float(np.clip(hidden_params.get("trade_activity", 0.0), 0.0, 1.0))

    predictions: Dict[int, np.ndarray] = {}

    for seed_index in range(seeds_count):
        init_grid = np.asarray(initial_states[seed_index]["grid"], dtype=np.int64)
        seed_counts = counts[seed_index]  # H×W×6 in class space
        sample_totals = seed_counts.sum(axis=2)  # H×W — n_obs per cell
        observed_mask = sample_totals > 0

        _, has_port_map, near_settlement = compute_settlement_maps(
            initial_states[seed_index], width=width, height=height
        )
        forest_adj = compute_forest_adjacency(init_grid)
        seed_settlement_stats = settlement_stats.get(seed_index, {})

        # Coastal mask
        coastal = np.zeros((height, width), dtype=bool)
        coastal[:2, :] = coastal[-2:, :] = coastal[:, :2] = coastal[:, -2:] = True
        coastal |= init_grid == 2

        # Ocean mask — static, excluded from scoring, always class 0
        ocean_mask = init_grid == 10

        seed_pred = np.zeros((height, width, CLASS_COUNT), dtype=np.float64)

        for y in range(height):
            for x in range(width):
                init_code = int(init_grid[y, x])
                n_obs = int(sample_totals[y, x])
                cell_counts = seed_counts[y, x].astype(np.float64)

                # Get terrain-based prior
                prior = terrain_prior(init_code)

                # Apply domain boosts to prior
                if init_code in (1, 2):
                    # Settlement/Port: may survive or collapse to Plains(class 0)
                    prior[1] += 0.15
                    prior[0] -= 0.10
                    if init_code == 2 or has_port_map[y, x]:
                        prior[2] += 0.08
                if near_settlement[y, x] and init_code not in (10, 5):
                    prior[1] += 0.08
                if coastal[y, x] and init_code not in (10, 5):
                    prior[2] += 0.04
                if init_code == 0 and forest_adj[y, x]:
                    prior[4] += 0.10
                    prior[0] -= 0.08
                if init_code == 3:
                    prior[4] += 0.08
                    prior[0] += 0.05

                # Ocean is static — override completely
                if init_code == 10:
                    seed_pred[y, x] = normalize_distribution(
                        np.array([0.97, 0.01, 0.01, 0.01, 0.00, 0.00])
                    )
                    continue

                # Mountain is static
                if init_code == 5:
                    seed_pred[y, x] = normalize_distribution(
                        np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95])
                    )
                    continue

                prior = normalize_distribution(prior)

                if n_obs > 0:
                    # Bayesian posterior: blend empirical samples with prior
                    # alpha = prior strength. As n_obs increases, trust data more.
                    # With n_obs=1: ~60% data weight. With n_obs=5: ~85% data weight.
                    alpha = max(0.20, 0.8 / n_obs)
                    empirical = cell_counts / float(cell_counts.sum())
                    posterior = (cell_counts + alpha * prior) / (float(n_obs) + alpha)
                    seed_pred[y, x] = normalize_distribution(posterior)
                else:
                    seed_pred[y, x] = prior

        # Hidden parameter adjustments for unobserved cells
        unobs = ~observed_mask & ~ocean_mask
        if faction_aggression > 0.0:
            m = near_settlement & unobs
            seed_pred[m, 3] += 0.25 * faction_aggression
            seed_pred[m, 1] -= 0.15 * faction_aggression
        if forest_growth_rate > 0.0:
            m = (np.isin(init_grid, [0, 3, 11])) & forest_adj & unobs
            seed_pred[m, 4] += 0.30 * forest_growth_rate
        if winter_severity > 0.0:
            seed_pred[unobs, 3] += 0.20 * winter_severity
            seed_pred[unobs, 1] -= 0.12 * winter_severity
        if trade_activity > 0.0:
            m = coastal & near_settlement & unobs
            seed_pred[m, 2] += 0.25 * trade_activity
        seed_pred = np.maximum(seed_pred, 0.0)

        # Spatial smoothing only for unobserved non-ocean cells
        smooth_mask = ~observed_mask & ~ocean_mask & (init_grid != 5)
        for c in range(CLASS_COUNT):
            blurred = gaussian_filter(seed_pred[:, :, c], sigma=1.5)
            seed_pred[:, :, c] = np.where(
                smooth_mask,
                0.60 * seed_pred[:, :, c] + 0.40 * blurred,
                seed_pred[:, :, c],
            )

        # Settlement alive/dead signal — applied AFTER smoothing so it isn't diluted
        seed_settlement_snapshots = observation_data.get("settlements", {}).get(seed_index, [])
        for snapshot in seed_settlement_snapshots:
            for s in snapshot.get("settlements", []):
                if not isinstance(s, dict):
                    continue
                sx, sy = s.get("x"), s.get("y")
                if not isinstance(sx, int) or not isinstance(sy, int):
                    continue
                if not (0 <= sx < width and 0 <= sy < height):
                    continue
                alive = bool(s.get("alive", True))
                s_has_port = bool(s.get("has_port", False))
                food = s.get("food")

                if alive:
                    if s_has_port:
                        ev = np.array([0.02, 0.45, 0.38, 0.05, 0.05, 0.05])
                    else:
                        ev = np.array([0.03, 0.82, 0.04, 0.04, 0.04, 0.03])
                    if food is not None and float(food) < 0.2:
                        ev[3] += 0.15
                        ev[1] -= 0.10
                    seed_pred[sy, sx] = normalize_distribution(seed_pred[sy, sx] * ev)
                else:
                    # Dead = direct override — prior has too little Ruin to multiply up
                    seed_pred[sy, sx] = normalize_distribution(
                        np.array([0.10, 0.04, 0.01, 0.75, 0.08, 0.02])
                    )

        # settlement_stats: use alive_rate across multiple observations of same settlement
        for (sx, sy), stat_list in seed_settlement_stats.items():
            if not (0 <= sx < width and 0 <= sy < height) or not stat_list:
                continue
            foods = [s["food"] for s in stat_list if s["food"] is not None]
            alive_rate = sum(1 for s in stat_list if s["alive"]) / len(stat_list)
            settle_ev = np.array([
                (1.0 - alive_rate) * 0.5,
                alive_rate * 0.85,
                alive_rate * 0.10,
                (1.0 - alive_rate) * 0.45,
                (1.0 - alive_rate) * 0.05,
                0.00,
            ])
            settle_ev = np.maximum(settle_ev, 0.01)
            seed_pred[sy, sx] = normalize_distribution(seed_pred[sy, sx] * settle_ev)

        # Final floor (0.01 for non-static, 0.001 for static classes on static cells)
        seed_pred = np.maximum(seed_pred, 0.01)
        seed_pred = seed_pred / seed_pred.sum(axis=2, keepdims=True)

        predictions[seed_index] = seed_pred
        if save_outputs:
            np.save(Path(f"predictions_seed_{seed_index}.npy"), seed_pred)

        # Summary
        observed_cells = int(observed_mask.sum())
        coverage = 100.0 * observed_cells / (width * height)
        n_samples = int(sample_totals[observed_mask].sum()) if observed_mask.any() else 0
        avg_samples = n_samples / max(observed_cells, 1)
        seed_settlement_snapshots_count = len(seed_settlement_snapshots)
        alive_count = sum(1 for snap in seed_settlement_snapshots for s in snap.get("settlements", []) if isinstance(s, dict) and s.get("alive", True))
        dead_count = sum(1 for snap in seed_settlement_snapshots for s in snap.get("settlements", []) if isinstance(s, dict) and not s.get("alive", True))
        print(f"Seed {seed_index}: {coverage:.1f}% covered, {avg_samples:.1f} avg samples/cell, "
              f"{alive_count} alive / {dead_count} dead settlements")

    return predictions


def fetch_ground_truth(session: requests.Session, round_id: str, seeds_count: int) -> Optional[Dict[int, np.ndarray]]:
    """Fetch ground truth from analysis endpoint after round completes."""
    ground_truth = {}
    for seed_idx in range(seeds_count):
        resp = request_json(session, "GET", f"/astar-island/analysis/{round_id}/{seed_idx}", retry_5xx=False)
        if not resp or "ground_truth" not in resp:
            print(f"  Seed {seed_idx}: analysis not available")
            return None
        gt = np.array(resp["ground_truth"], dtype=np.float64)
        pred = np.array(resp.get("prediction", []), dtype=np.float64)
        score = resp.get("score")
        print(f"  Seed {seed_idx}: score={score}")
        ground_truth[seed_idx] = gt
        # Save for offline analysis
        np.save(Path(f"ground_truth_seed_{seed_idx}.npy"), gt)
        if pred.size > 0:
            np.save(Path(f"pred_analysis_seed_{seed_idx}.npy"), pred)
    return ground_truth


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def run_offline_self_check(
    initial_states: Sequence[Dict[str, Any]],
    observation_data: Dict[str, Any],
    width: int, height: int, seeds_count: int,
    holdout_fraction: float = 0.15, random_seed: int = 2026,
) -> Dict[str, Any]:
    holdout_fraction = float(np.clip(holdout_fraction, 0.05, 0.50))
    counts = observation_data["counts"]
    rng = np.random.default_rng(random_seed)
    masked_counts = {s: arr.copy() for s, arr in counts.items()}
    holdout_cells: Dict[int, np.ndarray] = {}

    for seed_index in range(seeds_count):
        observed = np.argwhere(counts[seed_index].sum(axis=2) > 0)
        if observed.size == 0:
            holdout_cells[seed_index] = np.zeros((0, 2), dtype=np.int64)
            continue
        n_hold = max(1, min(len(observed), int(round(len(observed) * holdout_fraction))))
        sel = observed[rng.choice(len(observed), n_hold, replace=False)]
        holdout_cells[seed_index] = sel
        for y, x in sel.tolist():
            masked_counts[seed_index][y, x, :] = 0

    masked_data = dict(observation_data)
    masked_data["counts"] = masked_counts
    preds = build_predictions(initial_states, masked_data, width, height, seeds_count, save_outputs=False)

    all_kl: List[float] = []
    seed_reports = {}
    for seed_index in range(seeds_count):
        pred = preds[seed_index]
        sel = holdout_cells.get(seed_index, np.zeros((0, 2), dtype=np.int64))
        if len(sel) == 0:
            seed_reports[seed_index] = {"holdout_cells": 0, "mean_kl": float("nan")}
            continue
        kls = []
        for y, x in sel.tolist():
            tc = counts[seed_index][y, x].astype(np.float64)
            tot = float(tc.sum())
            if tot <= 0:
                continue
            p = tc / tot
            q = normalize_distribution(pred[y, x], floor=1e-9)
            kls.append(_kl_divergence(p, q))
        mean_kl = float(np.mean(kls)) if kls else float("nan")
        all_kl.extend(kls)
        seed_reports[seed_index] = {"holdout_cells": len(kls), "mean_kl": mean_kl}

    overall_kl = float(np.mean(all_kl)) if all_kl else float("nan")
    # Approximate score: score = 100 * exp(-3 * weighted_kl)
    approx_score = 100.0 * float(np.exp(-3.0 * overall_kl)) if not np.isnan(overall_kl) else float("nan")
    return {"overall": {"mean_kl": overall_kl, "approx_score": approx_score}, "by_seed": seed_reports}


def print_self_check_report(report: Dict[str, Any]) -> None:
    print("\nOffline self-check:")
    o = report.get("overall", {})
    print(f"  mean_kl={o.get('mean_kl', float('nan')):.4f}  approx_score={o.get('approx_score', float('nan')):.1f}")
    for s, row in sorted(report.get("by_seed", {}).items()):
        print(f"  seed {s}: {int(row.get('holdout_cells', 0))} cells, mean_kl={row.get('mean_kl', float('nan')):.4f}")


def validate_prediction_tensor(pred: np.ndarray, seed_index: int) -> None:
    if pred.ndim != 3 or pred.shape[2] != CLASS_COUNT:
        raise SolverError(f"Seed {seed_index}: invalid shape {pred.shape}")
    sums = pred.sum(axis=2)
    max_dev = float(np.max(np.abs(sums - 1.0)))
    if max_dev > 1e-3:
        raise SolverError(f"Seed {seed_index}: doesn't sum to 1.0 (max dev={max_dev:.6f})")


def summarize_submit_response(response: Any) -> str:
    if isinstance(response, dict):
        parts = [f"{k}={response[k]}" for k in ["status", "score", "seed_score", "total_score", "rank", "message"] if k in response]
        return ", ".join(parts) if parts else json.dumps(response)
    return str(response)


def submit_all(session: requests.Session, round_id: str, predictions: Dict[int, np.ndarray], dry_run: bool = False) -> Dict[int, str]:
    statuses: Dict[int, str] = {}
    for seed_index in sorted(predictions.keys()):
        pred = predictions[seed_index]
        validate_prediction_tensor(pred, seed_index)
        if dry_run:
            print(f"DRY RUN: would submit seed {seed_index}")
            statuses[seed_index] = "dry_run"
            continue
        resp = request_json(session, "POST", "/astar-island/submit", {
            "round_id": round_id, "seed_index": seed_index, "prediction": pred.tolist()
        })
        summary = summarize_submit_response(resp)
        print(f"Submitted seed {seed_index}: {summary}")
        statuses[seed_index] = summary
    return statuses


def load_predictions_from_disk(seeds_count: int) -> Dict[int, np.ndarray]:
    predictions: Dict[int, np.ndarray] = {}
    for i in range(seeds_count):
        path = Path(f"predictions_seed_{i}.npy")
        if not path.exists():
            raise SolverError(f"Missing prediction file: {path}")
        predictions[i] = np.load(path)
    return predictions


def create_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/json", "Content-Type": "application/json"})
    return session


def print_summary(width: int, height: int, seeds_count: int, queries_used: int,
                  observations: Dict[int, List[List[Optional[int]]]], statuses: Dict[int, str]) -> None:
    print(f"\nSummary: {queries_used}/{TOTAL_BUDGET} queries used")
    total = width * height
    for s in range(seeds_count):
        obs = sum(1 for row in observations[s] for v in row if v is not None)
        print(f"  Seed {s}: {100.0*obs/total:.1f}% ({obs}/{total} cells)")
    print("Submission:", {s: statuses.get(s, "not submitted") for s in range(seeds_count)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Astar Island solver")
    parser.add_argument("--token", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--no-query", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--self-check-fraction", type=float, default=0.15)
    parser.add_argument("--gcp-project", default="ai-nm26osl-1722")
    parser.add_argument("--fetch-analysis", action="store_true",
                        help="Fetch ground truth from last completed round and exit")
    args = parser.parse_args()
    if args.check_only:
        args.self_check = True

    token = args.token or os.getenv("ASTAR_ISLAND_TOKEN")
    if not token:
        parser.error("Token required: --token or ASTAR_ISLAND_TOKEN env var")

    try:
        session = create_session(token)
        round_id, width, height, seeds_count, initial_states = load_round(session)

        # Fetch ground truth from a completed round
        if args.fetch_analysis:
            print(f"Fetching ground truth for round {round_id}...")
            gt = fetch_ground_truth(session, round_id, seeds_count)
            if gt:
                print(f"Saved ground_truth_seed_*.npy for {len(gt)} seeds")
            return

        if args.submit_only:
            preds = {i: np.load(Path(f"predictions_seed_{i}.npy")) for i in range(seeds_count)}
            statuses = submit_all(session, round_id, preds, dry_run=args.dry_run)
            obs = {s: [[None]*width for _ in range(height)] for s in range(seeds_count)}
            print_summary(width, height, seeds_count, 0, obs, statuses)
            return

        if args.resume:
            obs_data = json.loads(Path("observations.json").read_text(encoding="utf-8"))
            observation_data = {
                "latest": {int(k): v for k, v in obs_data["latest"].items()},
                "counts": {int(k): np.array(v, dtype=np.int32) for k, v in obs_data["counts"].items()},
                "settlements": {int(k): v for k, v in obs_data["settlements"].items()},
                "settlement_stats": obs_data.get("settlement_stats", {}),
                "hidden_params": obs_data.get("hidden_params", {}),
                "queries": obs_data.get("queries", []),
                "queries_used": int(obs_data["queries_used"]),
                "raw_codes": {int(k): v for k, v in obs_data.get("raw_codes", obs_data["latest"]).items()},
            }
            print(f"Resumed ({observation_data['queries_used']} queries)")
        elif args.no_query:
            print("--no-query: using initial states only")
            obs = {s: [[None]*width for _ in range(height)] for s in range(seeds_count)}
            observation_data = {
                "latest": obs,
                "raw_codes": {s: [[None]*width for _ in range(height)] for s in range(seeds_count)},
                "counts": {s: np.zeros((height, width, CLASS_COUNT), dtype=np.int32) for s in range(seeds_count)},
                "settlements": {s: [] for s in range(seeds_count)},
                "settlement_stats": {s: {} for s in range(seeds_count)},
                "hidden_params": {},
                "queries": [],
                "queries_used": 0,
            }
        else:
            query_plan = plan_queries(width, height, seeds_count, TOTAL_BUDGET)
            obs = {s: [[None]*width for _ in range(height)] for s in range(seeds_count)}
            observation_data = run_queries(
                session, round_id, query_plan, obs, initial_states,
                gcp_project=args.gcp_project, budget=TOTAL_BUDGET,
            )

        if args.self_check:
            report = run_offline_self_check(
                initial_states, observation_data, width, height, seeds_count,
                args.self_check_fraction
            )
            print_self_check_report(report)

        predictions = build_predictions(
            initial_states, observation_data, width, height, seeds_count
        )

        if args.check_only:
            print("--check-only: not submitting")
            return

        statuses = submit_all(session, round_id, predictions, dry_run=args.dry_run)
        print_summary(width, height, seeds_count, int(observation_data["queries_used"]),
                      observation_data["latest"], statuses)

    except SolverError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()