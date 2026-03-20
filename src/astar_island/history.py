from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .features import build_all_features, make_bucket_keys
from .types import NUM_CLASSES, RoundDetail
from .utils import load_json, save_json, to_jsonable

LOGGER = logging.getLogger(__name__)

TERRAIN_NAMES = {0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin", 4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains"}


def round_archive_dir(history_dir: Path, round_number: int, round_id: str) -> Path:
    return history_dir / f"round_{round_number:02d}_{round_id}"


def seed_archive_dir(history_dir: Path, round_number: int, round_id: str, seed_index: int) -> Path:
    return round_archive_dir(history_dir, round_number, round_id) / f"seed_{seed_index}"


def extract_ground_truth_tensor(payload: Any) -> np.ndarray | None:
    from .priors import extract_ground_truth_tensor as _extract_ground_truth_tensor

    return _extract_ground_truth_tensor(payload)


def extract_initial_grid(payload: Any) -> np.ndarray | None:
    if isinstance(payload, dict):
        for preferred_key in ("initial_grid", "grid", "initialStateGrid"):
            if preferred_key in payload:
                arr = _to_int_grid(payload[preferred_key])
                if arr is not None:
                    return arr
        for value in payload.values():
            arr = extract_initial_grid(value)
            if arr is not None:
                return arr
    elif isinstance(payload, list):
        arr = _to_int_grid(payload)
        if arr is not None:
            return arr
        for value in payload:
            arr = extract_initial_grid(value)
            if arr is not None:
                return arr
    return None


def _to_int_grid(value: Any) -> np.ndarray | None:
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.ndim == 2:
        return arr.astype(np.int16)
    return None


def archive_completed_rounds(api, history_dir: Path, round_ids: list[str] | None = None, max_rounds: int | None = None) -> dict[str, Any]:
    rounds = api.get_rounds(use_cache=False)
    if round_ids is None:
        eligible = [item for item in rounds if item.get("status") == "completed"]
        eligible.sort(key=lambda item: int(item.get("round_number", 0)), reverse=True)
        if max_rounds is not None:
            eligible = eligible[:max_rounds]
        round_ids = [item["id"] for item in eligible]

    archived_rounds: list[dict[str, Any]] = []
    for round_id in round_ids:
        try:
            detail = api.get_round_details(round_id, use_cache=True)
        except Exception as exc:
            LOGGER.warning("Failed to fetch round %s details: %s", round_id, exc)
            continue
        round_dir = round_archive_dir(history_dir, detail.round_number, detail.round_id)
        round_dir.mkdir(parents=True, exist_ok=True)
        save_json(round_dir / "round_detail.json", detail.raw)

        per_round_summary = archive_round_analysis(api, history_dir, detail)
        archived_rounds.append(
            {
                "round_id": detail.round_id,
                "round_number": detail.round_number,
                "status": detail.status,
                "archived_seeds": per_round_summary["archived_seeds"],
            }
        )

    manifest = {"num_rounds": len(archived_rounds), "rounds": archived_rounds}
    save_json(history_dir / "manifest.json", manifest)
    return manifest


def archive_round_analysis(api, history_dir: Path, detail: RoundDetail) -> dict[str, Any]:
    gt_all: list[np.ndarray] = []
    ig_all: list[np.ndarray] = []
    archived = 0

    for seed_index, initial_state in enumerate(detail.initial_states):
        seed_dir = seed_archive_dir(history_dir, detail.round_number, detail.round_id, seed_index)
        seed_dir.mkdir(parents=True, exist_ok=True)
        try:
            analysis = api.get_analysis(detail.round_id, seed_index)
        except Exception as exc:
            LOGGER.warning("Round %s seed %s analysis unavailable: %s", detail.round_id, seed_index, exc)
            save_json(seed_dir / "status.json", {"available": False, "error": str(exc)})
            continue

        ground_truth = extract_ground_truth_tensor(analysis)
        initial_grid = extract_initial_grid(analysis)
        if initial_grid is None:
            initial_grid = np.asarray(initial_state.grid, dtype=np.int16)
        if ground_truth is None:
            LOGGER.warning("Round %s seed %s missing ground truth tensor", detail.round_id, seed_index)
            save_json(seed_dir / "status.json", {"available": False, "error": "missing_ground_truth"})
            save_json(seed_dir / "analysis.json", to_jsonable(analysis))
            continue

        archived += 1
        gt_all.append(ground_truth)
        ig_all.append(initial_grid)
        np.save(seed_dir / "ground_truth.npy", ground_truth)
        np.save(seed_dir / "initial_grid.npy", initial_grid)
        save_json(seed_dir / "analysis.json", to_jsonable(analysis))
        save_json(
            seed_dir / "summary.json",
            {
                "available": True,
                "seed_index": seed_index,
                "score": analysis.get("score") if isinstance(analysis, dict) else None,
                "ground_truth_shape": list(ground_truth.shape),
                "initial_grid_shape": list(initial_grid.shape),
            },
        )

    round_summary = summarize_archived_round(gt_all, ig_all, detail)
    save_json(round_archive_dir(history_dir, detail.round_number, detail.round_id) / "ground_truth_priors.json", round_summary)
    return {"archived_seeds": archived, "summary": round_summary}


def summarize_archived_round(gt_all: list[np.ndarray], ig_all: list[np.ndarray], detail: RoundDetail) -> dict[str, Any]:
    if not gt_all:
        return {"round_id": detail.round_id, "round_number": detail.round_number, "available": False, "priors": []}
    gt = np.concatenate(gt_all, axis=0).reshape(-1, NUM_CLASSES)
    ig = np.concatenate(ig_all, axis=0).ravel()
    priors: list[dict[str, Any]] = []
    for code in sorted(set(ig.tolist())):
        mask = ig == code
        avg = gt[mask].mean(axis=0)
        priors.append(
            {
                "terrain_code": int(code),
                "terrain_name": TERRAIN_NAMES.get(int(code), "?"),
                "count": int(mask.sum()),
                "avg_ground_truth": [float(v) for v in avg],
            }
        )
    return {
        "round_id": detail.round_id,
        "round_number": detail.round_number,
        "available": True,
        "num_seed_tensors": len(gt_all),
        "priors": priors,
    }


def build_historical_prior_from_archive(history_dir: Path, max_rounds: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    round_dirs = sorted(history_dir.glob("round_*"), reverse=True)
    if max_rounds is not None:
        round_dirs = round_dirs[:max_rounds]

    bucket_counts: dict[str, np.ndarray] = {}
    initial_class_counts: dict[str, np.ndarray] = {}
    used_rounds: list[dict[str, Any]] = []
    used_seeds = 0

    for round_dir in round_dirs:
        detail_path = round_dir / "round_detail.json"
        if not detail_path.exists():
            continue
        detail = _round_detail_from_json(load_json(detail_path))
        features = build_all_features(detail.initial_states)
        round_used = False
        for seed_index, seed_features in features.items():
            seed_dir = round_dir / f"seed_{seed_index}"
            gt_path = seed_dir / "ground_truth.npy"
            if not gt_path.exists():
                continue
            ground_truth = np.load(gt_path)
            _accumulate_ground_truth(bucket_counts, initial_class_counts, seed_features.initial_class_grid, make_bucket_keys(seed_features), ground_truth)
            used_seeds += 1
            round_used = True
        if round_used:
            used_rounds.append({"round_id": detail.round_id, "round_number": detail.round_number})

    metadata = {"num_rounds": len(used_rounds), "num_seeds": used_seeds, "rounds": used_rounds, "source": "archive"}
    return bucket_counts, initial_class_counts, metadata


def _round_detail_from_json(payload: dict[str, Any]) -> RoundDetail:
    from .types import InitialState, Settlement

    initial_states = [
        InitialState(
            grid=np.asarray(state["grid"], dtype=np.int16),
            settlements=[Settlement.from_api(item) for item in state.get("settlements", [])],
        )
        for state in payload["initial_states"]
    ]
    return RoundDetail(
        round_id=payload["id"],
        round_number=int(payload["round_number"]),
        status=str(payload["status"]),
        map_width=int(payload["map_width"]),
        map_height=int(payload["map_height"]),
        seeds_count=int(payload["seeds_count"]),
        initial_states=initial_states,
        raw=payload,
    )


def _accumulate_ground_truth(
    bucket_counts: dict[str, np.ndarray],
    initial_class_counts: dict[str, np.ndarray],
    initial_class_grid: np.ndarray,
    bucket_keys: np.ndarray,
    ground_truth: np.ndarray,
) -> None:
    for key in np.unique(bucket_keys):
        mask = bucket_keys == key
        counts = ground_truth[mask].sum(axis=0)
        bucket_counts.setdefault(str(int(key)), np.zeros(NUM_CLASSES, dtype=np.float64))
        bucket_counts[str(int(key))] += counts
    for class_id in np.unique(initial_class_grid):
        mask = initial_class_grid == class_id
        counts = ground_truth[mask].sum(axis=0)
        initial_class_counts.setdefault(str(int(class_id)), np.zeros(NUM_CLASSES, dtype=np.float64))
        initial_class_counts[str(int(class_id))] += counts
