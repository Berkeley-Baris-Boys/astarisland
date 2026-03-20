#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import CODE_TO_CLASS, N_CLASSES, OCEAN_CODE, MOUNTAIN_CODE
from metrics import entropy_weighted_kl


EXACT_MAP_ROI = (53, 350, 866, 866)
EXACT_PALETTE = {
    0: [201, 184, 139],   # Empty
    1: [213, 118, 10],    # Settlement
    2: [13, 116, 144],    # Port
    3: [127, 30, 29],     # Ruin
    4: [45, 90, 39],      # Forest
    5: [108, 114, 128],   # Mountain
    10: [31, 58, 95],     # Ocean
    11: [201, 184, 139],  # Plains (same as Empty by design)
}

EMPTY_CODE = 0
PLAINS_CODE = 11


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def _video_info(cap: cv2.VideoCapture) -> dict[str, float]:
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = frame_count / fps if fps > 0 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_s": duration_s,
    }


def _sample_indices(frame_count: int, fps: float, interval_ms: int) -> list[int]:
    if frame_count <= 0:
        return []
    if fps <= 0:
        return list(range(frame_count))
    step = max(1, int(round((interval_ms / 1000.0) * fps)))
    return list(range(0, frame_count, step))


def _read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame_bgr = cap.read()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Failed reading frame index {idx}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _try_read_frame(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    try:
        return _read_frame(cap, idx)
    except RuntimeError:
        return None


def _first_readable_frames(
    cap: cv2.VideoCapture, idxs: list[int], needed: int = 2
) -> tuple[list[np.ndarray], list[int], list[int]]:
    frames: list[np.ndarray] = []
    ok_idxs: list[int] = []
    failed: list[int] = []
    for idx in idxs:
        frame = _try_read_frame(cap, idx)
        if frame is None:
            failed.append(int(idx))
            continue
        frames.append(frame)
        ok_idxs.append(int(idx))
        if len(frames) >= needed:
            break
    return frames, ok_idxs, failed


def _parse_roi(roi: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    if roi is None:
        return EXACT_MAP_ROI
    vals = [int(v.strip()) for v in roi.split(",")]
    if len(vals) != 4:
        raise ValueError("--roi must be 'x,y,w,h'")
    x, y, w, h = vals
    if w <= 0 or h <= 0:
        raise ValueError("ROI width/height must be positive")
    if x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("ROI is out of frame bounds")
    return x, y, w, h


def _load_palette(path: Path | None, palette_mode: str) -> tuple[np.ndarray, np.ndarray]:
    if palette_mode == "exact":
        raw: dict[str, Any] = {str(k): v for k, v in EXACT_PALETTE.items()}
    else:
        if path is None:
            raise ValueError("--palette-json is required when --palette-mode=json")
        raw = json.loads(path.read_text())
    class_ids: list[int] = []
    colors: list[list[float]] = []
    for key, rgb in raw.items():
        class_ids.append(int(key))
        if not isinstance(rgb, list) or len(rgb) != 3:
            raise ValueError(f"Invalid RGB for class {key}: {rgb}")
        colors.append([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    if not class_ids:
        raise ValueError("Palette is empty")
    return np.asarray(class_ids, dtype=np.int32), np.asarray(colors, dtype=np.float64)


def _canonicalize_empty_plains(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[out == EMPTY_CODE] = PLAINS_CODE
    return out


def _apply_color_transform(
    rgb: np.ndarray, color_matrix: np.ndarray, color_bias: np.ndarray
) -> np.ndarray:
    corrected = (rgb.astype(np.float64) @ color_matrix) + color_bias
    return np.clip(corrected, 0.0, 255.0)


def _patch_rgb_stat(patch: np.ndarray, mode: str, trim_frac: float) -> np.ndarray:
    flat = patch.reshape(-1, 3).astype(np.float64)
    if flat.shape[0] == 0:
        return np.zeros((3,), dtype=np.float64)
    if mode == "median":
        return np.median(flat, axis=0)
    if mode == "trimmed_mean":
        lo = np.quantile(flat, trim_frac, axis=0)
        hi = np.quantile(flat, 1.0 - trim_frac, axis=0)
        keep = np.all((flat >= lo[None, :]) & (flat <= hi[None, :]), axis=1)
        if np.any(keep):
            return flat[keep].mean(axis=0)
        return flat.mean(axis=0)
    raise ValueError(f"Unsupported patch stat mode: {mode}")


def _round_seed_base(dataset_root: Path, round_id: int, seed_id: int) -> Path:
    return dataset_root / f"round{int(round_id)}" / f"seed{int(seed_id)}"


def _resolve_extract_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    base = _round_seed_base(Path(args.dataset_root), args.round_id, args.seed_id)
    return base / "input" / "frames_preview"


def _resolve_decode_out_dir(args: argparse.Namespace) -> Path:
    if args.out_dir:
        return Path(args.out_dir)
    base = _round_seed_base(Path(args.dataset_root), args.round_id, args.seed_id)
    tag = args.decode_tag.strip() if args.decode_tag else "decoded"
    return base / "output" / tag


def _write_run_manifest(
    path: Path,
    *,
    video_path: Path,
    round_id: int,
    seed_id: int,
    cmd: str,
    extra: dict[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "video": str(video_path),
        "round_id": int(round_id),
        "seed_id": int(seed_id),
        "command": cmd,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))


def _decode_grid_from_frame(
    frame_rgb: np.ndarray,
    *,
    roi: tuple[int, int, int, int],
    grid_h: int,
    grid_w: int,
    class_ids: np.ndarray,
    palette_rgb: np.ndarray,
    patch_ratio: float,
    color_matrix: np.ndarray,
    color_bias: np.ndarray,
    subcell_offset_x: float,
    subcell_offset_y: float,
    patch_stat: str,
    trim_frac: float,
    prototypes_rgb: np.ndarray,
    prototype_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, rw, rh = roi
    crop = frame_rgb[y0:y0 + rh, x0:x0 + rw]
    cell_h = rh / float(grid_h)
    cell_w = rw / float(grid_w)
    grid = np.zeros((grid_h, grid_w), dtype=np.int32)
    conf = np.zeros((grid_h, grid_w), dtype=np.float32)
    max_dist = math.sqrt(3.0 * (255.0 ** 2))

    for y in range(grid_h):
        for x in range(grid_w):
            y1 = int(round(y * cell_h))
            y2 = int(round((y + 1) * cell_h))
            x1 = int(round(x * cell_w))
            x2 = int(round((x + 1) * cell_w))
            if y2 <= y1 or x2 <= x1:
                continue
            ph = max(1, int((y2 - y1) * patch_ratio))
            pw = max(1, int((x2 - x1) * patch_ratio))
            cy = int(round((y1 + y2) / 2.0 + subcell_offset_y * (y2 - y1)))
            cx = int(round((x1 + x2) / 2.0 + subcell_offset_x * (x2 - x1)))
            py1 = max(y1, cy - ph // 2)
            py2 = min(y2, py1 + ph)
            px1 = max(x1, cx - pw // 2)
            px2 = min(x2, px1 + pw)
            patch = crop[py1:py2, px1:px2]
            if patch.size == 0:
                continue
            rgb = _patch_rgb_stat(patch, patch_stat, trim_frac)
            rgb = _apply_color_transform(rgb, color_matrix, color_bias)
            z = (prototypes_rgb - rgb[None, :]) / np.maximum(prototype_scales, 1.0)
            d = np.sqrt((z ** 2).sum(axis=1))
            order = np.argsort(d)
            j = int(order[0])
            d1 = float(d[order[0]])
            d2 = float(d[order[1]]) if d.shape[0] > 1 else d1
            abs_conf = 1.0 - min(1.0, d1 / max_dist)
            margin_conf = max(0.0, min(1.0, (d2 - d1) / max(d2, 1e-6)))
            grid[y, x] = int(class_ids[j])
            conf[y, x] = float(0.5 * abs_conf + 0.5 * margin_conf)
    return grid, conf


def _fit_color_affine(
    frame_rgb: np.ndarray,
    *,
    roi: tuple[int, int, int, int],
    grid_h: int,
    grid_w: int,
    class_ids: np.ndarray,
    palette_rgb: np.ndarray,
    patch_ratio: float,
    subcell_offset_x: float,
    subcell_offset_y: float,
    patch_stat: str,
    trim_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, rw, rh = roi
    crop = frame_rgb[y0:y0 + rh, x0:x0 + rw]
    cell_h = rh / float(grid_h)
    cell_w = rw / float(grid_w)
    src: list[np.ndarray] = []
    tgt: list[np.ndarray] = []
    for y in range(grid_h):
        for x in range(grid_w):
            y1 = int(round(y * cell_h))
            y2 = int(round((y + 1) * cell_h))
            x1 = int(round(x * cell_w))
            x2 = int(round((x + 1) * cell_w))
            if y2 <= y1 or x2 <= x1:
                continue
            ph = max(1, int((y2 - y1) * patch_ratio))
            pw = max(1, int((x2 - x1) * patch_ratio))
            cy = int(round((y1 + y2) / 2.0 + subcell_offset_y * (y2 - y1)))
            cx = int(round((x1 + x2) / 2.0 + subcell_offset_x * (x2 - x1)))
            py1 = max(y1, cy - ph // 2)
            py2 = min(y2, py1 + ph)
            px1 = max(x1, cx - pw // 2)
            px2 = min(x2, px1 + pw)
            patch = crop[py1:py2, px1:px2]
            if patch.size == 0:
                continue
            rgb = _patch_rgb_stat(patch, patch_stat, trim_frac).astype(np.float64)
            d = np.sqrt(((palette_rgb - rgb[None, :]) ** 2).sum(axis=1))
            j = int(np.argmin(d))
            src.append(rgb)
            tgt.append(palette_rgb[j].astype(np.float64))
    if not src:
        return np.eye(3, dtype=np.float64), np.zeros((3,), dtype=np.float64)
    x_mat = np.asarray(src, dtype=np.float64)
    y_mat = np.asarray(tgt, dtype=np.float64)
    x_aug = np.concatenate([x_mat, np.ones((x_mat.shape[0], 1), dtype=np.float64)], axis=1)
    coeff, _, _, _ = np.linalg.lstsq(x_aug, y_mat, rcond=None)
    matrix = coeff[:3, :]
    bias = coeff[3, :]
    return matrix, bias


def _score_alignment(
    g0: np.ndarray, c0: np.ndarray, g1: np.ndarray, c1: np.ndarray
) -> float:
    change_frac = float((g0 != g1).mean())
    return float(c0.mean() + 0.4 * c1.mean() - 0.25 * change_frac)


def _refine_roi_and_offsets(
    frame0: np.ndarray,
    frame1: np.ndarray,
    *,
    roi: tuple[int, int, int, int],
    grid_h: int,
    grid_w: int,
    class_ids: np.ndarray,
    palette_rgb: np.ndarray,
    patch_ratio: float,
    patch_stat: str,
    trim_frac: float,
    refine_px: int,
    refine_step: int,
    refine_subcell: float,
) -> tuple[tuple[int, int, int, int], float, float]:
    if refine_px <= 0:
        return roi, 0.0, 0.0
    x0, y0, rw, rh = roi
    best_score = -1e9
    best = (roi, 0.0, 0.0)
    step = max(1, refine_step)
    offsets = np.arange(-refine_px, refine_px + 1, step, dtype=np.int32)
    subcells = [-refine_subcell, 0.0, refine_subcell]
    eye = np.eye(3, dtype=np.float64)
    zero = np.zeros((3,), dtype=np.float64)
    for dy in offsets.tolist():
        for dx in offsets.tolist():
            xr = x0 + int(dx)
            yr = y0 + int(dy)
            if xr < 0 or yr < 0 or xr + rw > frame0.shape[1] or yr + rh > frame0.shape[0]:
                continue
            roi_try = (xr, yr, rw, rh)
            for sy in subcells:
                for sx in subcells:
                    g0, c0 = _decode_grid_from_frame(
                        frame0,
                        roi=roi_try,
                        grid_h=grid_h,
                        grid_w=grid_w,
                        class_ids=class_ids,
                        palette_rgb=palette_rgb,
                        patch_ratio=patch_ratio,
                        color_matrix=eye,
                        color_bias=zero,
                        subcell_offset_x=sx,
                        subcell_offset_y=sy,
                        patch_stat=patch_stat,
                        trim_frac=trim_frac,
                        prototypes_rgb=palette_rgb,
                        prototype_scales=np.ones_like(palette_rgb, dtype=np.float64),
                    )
                    g1, c1 = _decode_grid_from_frame(
                        frame1,
                        roi=roi_try,
                        grid_h=grid_h,
                        grid_w=grid_w,
                        class_ids=class_ids,
                        palette_rgb=palette_rgb,
                        patch_ratio=patch_ratio,
                        color_matrix=eye,
                        color_bias=zero,
                        subcell_offset_x=sx,
                        subcell_offset_y=sy,
                        patch_stat=patch_stat,
                        trim_frac=trim_frac,
                        prototypes_rgb=palette_rgb,
                        prototype_scales=np.ones_like(palette_rgb, dtype=np.float64),
                    )
                    score = _score_alignment(g0, c0, g1, c1)
                    if score > best_score:
                        best_score = score
                        best = (roi_try, sx, sy)
    return best


def _trim_leading_static_frames(
    grids: np.ndarray,
    confs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if grids.shape[0] <= 1:
        return grids, confs
    start = 0
    for i in range(grids.shape[0] - 1):
        g0 = grids[i]
        g1 = grids[i + 1]
        all_same = np.unique(g0).size <= 1
        changed = np.any(g0 != g1)
        if changed and not all_same:
            start = i
            break
        if changed and all_same:
            start = i + 1
            break
    return grids[start:], confs[start:]


def _dedupe_by_grid_change(
    grids: np.ndarray,
    confs: np.ndarray,
    min_change_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    if grids.shape[0] <= 1:
        return grids, confs
    keep = [0]
    for i in range(1, grids.shape[0]):
        prev = grids[keep[-1]]
        cur = grids[i]
        change_frac = float((prev != cur).mean())
        if change_frac >= min_change_frac:
            keep.append(i)
    return grids[keep], confs[keep]


def _resample_to_target_steps(
    grids: np.ndarray,
    confs: np.ndarray,
    target_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if target_steps <= 0 or grids.shape[0] == 0:
        return grids, confs
    if grids.shape[0] == target_steps:
        return grids, confs
    idx = np.rint(np.linspace(0, grids.shape[0] - 1, target_steps)).astype(np.int32)
    return grids[idx], confs[idx]


def _temporal_majority_smooth(
    grids: np.ndarray, confs: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    if window <= 1 or grids.shape[0] < 3:
        return grids, confs
    out_grid = grids.copy()
    out_conf = confs.copy()
    half = window // 2
    t_max, h, w = grids.shape
    for t in range(t_max):
        t1 = max(0, t - half)
        t2 = min(t_max, t + half + 1)
        slab = grids[t1:t2]
        for y in range(h):
            for x in range(w):
                vals, cnt = np.unique(slab[:, y, x], return_counts=True)
                best_idx = int(np.argmax(cnt))
                best_val = int(vals[best_idx])
                if best_val != int(out_grid[t, y, x]):
                    out_grid[t, y, x] = best_val
                    mask = slab[:, y, x] == best_val
                    conf_slice = confs[t1:t2, y, x]
                    out_conf[t, y, x] = float(conf_slice[mask].mean()) if np.any(mask) else out_conf[t, y, x]
    return out_grid, out_conf


def _temporal_majority_smooth_conf_gated(
    grids: np.ndarray,
    confs: np.ndarray,
    window: int,
    conf_threshold: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if window <= 1 or grids.shape[0] < 3:
        return grids, confs, {"n_overridden": 0.0, "n_low_conf": 0.0, "n_from_t0": 0.0}
    out_grid = grids.copy()
    out_conf = confs.copy()
    half = window // 2
    t_max, h, w = grids.shape
    n_overridden = 0
    n_low_conf = 0
    n_from_t0 = 0
    for t in range(1, t_max):
        t1 = max(0, t - half)
        t2 = min(t_max, t + half + 1)
        slab = grids[t1:t2]
        for y in range(h):
            for x in range(w):
                if float(confs[t, y, x]) >= conf_threshold:
                    continue
                n_low_conf += 1
                vals, cnt = np.unique(slab[:, y, x], return_counts=True)
                best_idx = int(np.argmax(cnt))
                best_val = int(vals[best_idx])
                if best_val != int(out_grid[t, y, x]):
                    out_grid[t, y, x] = best_val
                    n_overridden += 1
                    if t1 == 0 and best_val == int(grids[0, y, x]):
                        n_from_t0 += 1
                    mask = slab[:, y, x] == best_val
                    conf_slice = confs[t1:t2, y, x]
                    out_conf[t, y, x] = float(conf_slice[mask].mean()) if np.any(mask) else out_conf[t, y, x]
    stats = {
        "n_overridden": float(n_overridden),
        "n_low_conf": float(n_low_conf),
        "n_from_t0": float(n_from_t0),
    }
    return out_grid, out_conf, stats


def _suppress_one_frame_flicker(
    grids: np.ndarray, confs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if grids.shape[0] < 3:
        return grids, confs
    out_grid = grids.copy()
    out_conf = confs.copy()
    for t in range(1, grids.shape[0] - 1):
        prev = out_grid[t - 1]
        cur = out_grid[t]
        nxt = out_grid[t + 1]
        mask = (prev == nxt) & (cur != prev)
        if np.any(mask):
            out_grid[t][mask] = prev[mask]
            out_conf[t][mask] = 0.5 * (out_conf[t - 1][mask] + out_conf[t + 1][mask])
    return out_grid, out_conf


def _build_prototypes_from_frame(
    frame_rgb: np.ndarray,
    *,
    roi: tuple[int, int, int, int],
    grid_h: int,
    grid_w: int,
    class_ids: np.ndarray,
    palette_rgb: np.ndarray,
    patch_ratio: float,
    subcell_offset_x: float,
    subcell_offset_y: float,
    patch_stat: str,
    trim_frac: float,
) -> np.ndarray:
    x0, y0, rw, rh = roi
    crop = frame_rgb[y0:y0 + rh, x0:x0 + rw]
    cell_h = rh / float(grid_h)
    cell_w = rw / float(grid_w)
    buckets: dict[int, list[np.ndarray]] = {int(cid): [] for cid in class_ids.tolist()}
    for y in range(grid_h):
        for x in range(grid_w):
            y1 = int(round(y * cell_h))
            y2 = int(round((y + 1) * cell_h))
            x1 = int(round(x * cell_w))
            x2 = int(round((x + 1) * cell_w))
            if y2 <= y1 or x2 <= x1:
                continue
            ph = max(1, int((y2 - y1) * patch_ratio))
            pw = max(1, int((x2 - x1) * patch_ratio))
            cy = int(round((y1 + y2) / 2.0 + subcell_offset_y * (y2 - y1)))
            cx = int(round((x1 + x2) / 2.0 + subcell_offset_x * (x2 - x1)))
            py1 = max(y1, cy - ph // 2)
            py2 = min(y2, py1 + ph)
            px1 = max(x1, cx - pw // 2)
            px2 = min(x2, px1 + pw)
            patch = crop[py1:py2, px1:px2]
            if patch.size == 0:
                continue
            rgb = _patch_rgb_stat(patch, patch_stat, trim_frac).astype(np.float64)
            d = np.sqrt(((palette_rgb - rgb[None, :]) ** 2).sum(axis=1))
            cid = int(class_ids[int(np.argmin(d))])
            buckets[cid].append(rgb)
    prototypes = palette_rgb.copy()
    for i, cid in enumerate(class_ids.tolist()):
        cells = buckets[int(cid)]
        if cells:
            prototypes[i] = np.median(np.asarray(cells, dtype=np.float64), axis=0)
    return prototypes


def _build_prototypes_from_ig(
    frame_rgb: np.ndarray,
    *,
    ig: np.ndarray,
    roi: tuple[int, int, int, int],
    grid_h: int,
    grid_w: int,
    class_ids: np.ndarray,
    palette_rgb: np.ndarray,
    patch_ratio: float,
    subcell_offset_x: float,
    subcell_offset_y: float,
    patch_stat: str,
    trim_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, rw, rh = roi
    crop = frame_rgb[y0:y0 + rh, x0:x0 + rw]
    cell_h = rh / float(grid_h)
    cell_w = rw / float(grid_w)
    buckets: dict[int, list[np.ndarray]] = {int(cid): [] for cid in class_ids.tolist()}
    for y in range(grid_h):
        for x in range(grid_w):
            y1 = int(round(y * cell_h))
            y2 = int(round((y + 1) * cell_h))
            x1 = int(round(x * cell_w))
            x2 = int(round((x + 1) * cell_w))
            if y2 <= y1 or x2 <= x1:
                continue
            ph = max(1, int((y2 - y1) * patch_ratio))
            pw = max(1, int((x2 - x1) * patch_ratio))
            cy = int(round((y1 + y2) / 2.0 + subcell_offset_y * (y2 - y1)))
            cx = int(round((x1 + x2) / 2.0 + subcell_offset_x * (x2 - x1)))
            py1 = max(y1, cy - ph // 2)
            py2 = min(y2, py1 + ph)
            px1 = max(x1, cx - pw // 2)
            px2 = min(x2, px1 + pw)
            patch = crop[py1:py2, px1:px2]
            if patch.size == 0:
                continue
            cid = int(ig[y, x])
            if cid not in buckets:
                continue
            rgb = _patch_rgb_stat(patch, patch_stat, trim_frac).astype(np.float64)
            buckets[cid].append(rgb)
    prototypes = palette_rgb.copy()
    scales = np.ones_like(palette_rgb, dtype=np.float64)
    for i, cid in enumerate(class_ids.tolist()):
        cells = buckets[int(cid)]
        if cells:
            arr = np.asarray(cells, dtype=np.float64)
            prototypes[i] = np.median(arr, axis=0)
            scales[i] = np.maximum(np.std(arr, axis=0), 6.0)
    return prototypes, scales


def _enforce_static_codes(grid: np.ndarray, ig: np.ndarray) -> None:
    ocean = ig == OCEAN_CODE
    mountain = ig == MOUNTAIN_CODE
    grid[ocean] = OCEAN_CODE
    grid[mountain] = MOUNTAIN_CODE


def cmd_probe(args: argparse.Namespace) -> None:
    cap = _open_video(Path(args.video))
    info = _video_info(cap)
    cap.release()
    print(json.dumps(info, indent=2))


def cmd_extract(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    out_dir = _resolve_extract_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = _open_video(video_path)
    info = _video_info(cap)
    idxs = _sample_indices(int(info["frame_count"]), float(info["fps"]), args.interval_ms)
    for i, idx in enumerate(idxs):
        frame = _read_frame(cap, idx)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out = out_dir / f"frame_{i:03d}_idx{idx:05d}.png"
        cv2.imwrite(str(out), bgr)
    cap.release()
    _write_run_manifest(
        out_dir / "extract_manifest.json",
        video_path=video_path,
        round_id=args.round_id,
        seed_id=args.seed_id,
        cmd="extract",
        extra={
            "interval_ms": int(args.interval_ms),
            "frames_extracted": len(idxs),
        },
    )
    print(f"Extracted {len(idxs)} frames to {out_dir}")


def _ig_path(
    data_root: Path,
    round_id: int,
    seed_id: int,
    *,
    ig_seed_offset: int = 0,
    ig_seed_id: int | None = None,
) -> Path:
    sid = int(ig_seed_id) if ig_seed_id is not None else int(seed_id) + int(ig_seed_offset)
    return data_root / f"round{int(round_id)}" / f"ig_r{int(round_id)}_seed{sid}.npy"


def _apply_decode_preset(args: argparse.Namespace) -> None:
    if not args.preset:
        return
    if args.preset == "baseline_simple":
        args.t0_anchor_mode = "off"
        args.use_ig_calibration = False
        args.classifier = "nearest_rgb"
        args.temporal_window = 1
        args.temporal_conf_gated = False
        args.suppress_flicker = False
    elif args.preset == "t0_anchor_only":
        args.t0_anchor_mode = "force"
        args.use_ig_calibration = True
        args.classifier = "nearest_prototype"
        args.temporal_window = 1
        args.temporal_conf_gated = False
        args.suppress_flicker = False
    elif args.preset == "t0_anchor_gated_temporal":
        args.t0_anchor_mode = "force"
        args.use_ig_calibration = True
        args.classifier = "nearest_prototype"
        args.temporal_window = max(3, int(args.temporal_window))
        args.temporal_conf_gated = True
        args.suppress_flicker = True
    if args.decode_tag == "decoded":
        args.decode_tag = args.preset


def cmd_decode(args: argparse.Namespace) -> None:
    _apply_decode_preset(args)
    video_path = Path(args.video)
    out_dir = _resolve_decode_out_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    palette_path = Path(args.palette_json) if args.palette_json else None
    class_ids, palette_rgb = _load_palette(palette_path, args.palette_mode)
    ig: np.ndarray | None = None
    ig_file = _ig_path(
        Path(args.data_root),
        args.round_id,
        args.seed_id,
        ig_seed_offset=args.ig_seed_offset,
        ig_seed_id=args.ig_seed_id,
    )
    if args.t0_anchor_mode == "force" or args.use_ig_calibration:
        if not ig_file.exists():
            raise FileNotFoundError(f"Initial grid required but missing: {ig_file}")
        ig = np.load(ig_file)

    cap = _open_video(video_path)
    info = _video_info(cap)
    roi = _parse_roi(args.roi, int(info["width"]), int(info["height"]))
    idxs = _sample_indices(int(info["frame_count"]), float(info["fps"]), args.interval_ms)
    if not idxs:
        raise RuntimeError("No frames sampled from video.")

    boot_frames, boot_idxs, failed_boot = _first_readable_frames(cap, idxs, needed=2)
    if not boot_frames:
        raise RuntimeError("Failed to read any sampled video frames.")
    frame0 = boot_frames[0]
    frame1 = boot_frames[1] if len(boot_frames) > 1 else boot_frames[0]
    failed_read_idxs: list[int] = list(failed_boot)

    roi, subcell_x, subcell_y = _refine_roi_and_offsets(
        frame0,
        frame1,
        roi=roi,
        grid_h=args.grid_h,
        grid_w=args.grid_w,
        class_ids=class_ids,
        palette_rgb=palette_rgb,
        patch_ratio=args.patch_ratio,
        patch_stat=args.patch_stat,
        trim_frac=args.trim_frac,
        refine_px=args.roi_refine_px,
        refine_step=args.roi_refine_step,
        refine_subcell=args.subcell_refine,
    )
    if args.use_color_affine:
        color_matrix, color_bias = _fit_color_affine(
            frame0,
            roi=roi,
            grid_h=args.grid_h,
            grid_w=args.grid_w,
            class_ids=class_ids,
            palette_rgb=palette_rgb,
            patch_ratio=args.patch_ratio,
            subcell_offset_x=subcell_x,
            subcell_offset_y=subcell_y,
            patch_stat=args.patch_stat,
            trim_frac=args.trim_frac,
        )
    else:
        color_matrix = np.eye(3, dtype=np.float64)
        color_bias = np.zeros((3,), dtype=np.float64)
    prototype_scales = np.ones_like(palette_rgb, dtype=np.float64)
    if args.use_ig_calibration and ig is not None:
        prototypes_rgb, prototype_scales = _build_prototypes_from_ig(
            frame0,
            ig=ig,
            roi=roi,
            grid_h=args.grid_h,
            grid_w=args.grid_w,
            class_ids=class_ids,
            palette_rgb=palette_rgb,
            patch_ratio=args.patch_ratio,
            subcell_offset_x=subcell_x,
            subcell_offset_y=subcell_y,
            patch_stat=args.patch_stat,
            trim_frac=args.trim_frac,
        )
    elif args.classifier == "nearest_prototype":
        prototypes_rgb = _build_prototypes_from_frame(
            frame0,
            roi=roi,
            grid_h=args.grid_h,
            grid_w=args.grid_w,
            class_ids=class_ids,
            palette_rgb=palette_rgb,
            patch_ratio=args.patch_ratio,
            subcell_offset_x=subcell_x,
            subcell_offset_y=subcell_y,
            patch_stat=args.patch_stat,
            trim_frac=args.trim_frac,
        )
    else:
        prototypes_rgb = palette_rgb.copy()

    grids: list[np.ndarray] = []
    confs: list[np.ndarray] = []
    for idx in idxs:
        frame = _try_read_frame(cap, idx)
        if frame is None:
            failed_read_idxs.append(int(idx))
            continue
        if len(grids) == 0 and args.t0_anchor_mode == "force" and ig is not None:
            grid = ig.copy().astype(np.int32)
            conf = np.ones_like(grid, dtype=np.float32)
        else:
            grid, conf = _decode_grid_from_frame(
                frame,
                roi=roi,
                grid_h=args.grid_h,
                grid_w=args.grid_w,
                class_ids=class_ids,
                palette_rgb=palette_rgb,
                patch_ratio=args.patch_ratio,
                color_matrix=color_matrix,
                color_bias=color_bias,
                subcell_offset_x=subcell_x,
                subcell_offset_y=subcell_y,
                patch_stat=args.patch_stat,
                trim_frac=args.trim_frac,
                prototypes_rgb=prototypes_rgb,
                prototype_scales=prototype_scales,
            )
            if ig is not None:
                _enforce_static_codes(grid, ig)
        if args.canonicalize_empty_plains:
            grid = _canonicalize_empty_plains(grid)
        grids.append(grid)
        confs.append(conf)
    cap.release()

    if not grids:
        raise RuntimeError("No decodable frames available after skipping unreadable frame indices.")

    grid_arr = np.stack(grids, axis=0) if grids else np.zeros((0, args.grid_h, args.grid_w), dtype=np.int32)
    conf_arr = np.stack(confs, axis=0) if confs else np.zeros((0, args.grid_h, args.grid_w), dtype=np.float32)
    raw_frames = int(grid_arr.shape[0])
    if args.trim_leading_static and raw_frames > 0:
        grid_arr, conf_arr = _trim_leading_static_frames(grid_arr, conf_arr)
    if args.min_change_frac > 0.0 and grid_arr.shape[0] > 0:
        grid_arr, conf_arr = _dedupe_by_grid_change(grid_arr, conf_arr, args.min_change_frac)
    if args.target_steps is not None and grid_arr.shape[0] > 0:
        grid_arr, conf_arr = _resample_to_target_steps(grid_arr, conf_arr, args.target_steps)
    temporal_stats = {"n_overridden": 0.0, "n_low_conf": 0.0, "n_from_t0": 0.0}
    if args.temporal_conf_gated and args.temporal_window > 1 and grid_arr.shape[0] > 0:
        grid_arr, conf_arr, temporal_stats = _temporal_majority_smooth_conf_gated(
            grid_arr,
            conf_arr,
            args.temporal_window,
            args.temporal_conf_threshold,
        )
    elif args.temporal_window > 1 and grid_arr.shape[0] > 0:
        grid_arr, conf_arr = _temporal_majority_smooth(grid_arr, conf_arr, args.temporal_window)
    if args.suppress_flicker and grid_arr.shape[0] > 0:
        grid_arr, conf_arr = _suppress_one_frame_flicker(grid_arr, conf_arr)

    np.save(out_dir / "grid_t.npy", grid_arr)
    np.save(out_dir / "confidence_t.npy", conf_arr)
    meta = {
        "video": str(video_path),
        "round_id": int(args.round_id),
        "seed_id": int(args.seed_id),
        "ig_seed_offset": int(args.ig_seed_offset),
        "ig_seed_id": int(args.ig_seed_id) if args.ig_seed_id is not None else None,
        "raw_frames_decoded": raw_frames,
        "sampled_frame_indices": len(idxs),
        "failed_read_count": int(len(set(failed_read_idxs))),
        "failed_read_indices": sorted(set(int(i) for i in failed_read_idxs)),
        "bootstrap_frame_indices": boot_idxs,
        "frames_decoded": int(grid_arr.shape[0]),
        "grid_h": int(args.grid_h),
        "grid_w": int(args.grid_w),
        "interval_ms": int(args.interval_ms),
        "trim_leading_static": bool(args.trim_leading_static),
        "min_change_frac": float(args.min_change_frac),
        "target_steps": int(args.target_steps) if args.target_steps is not None else None,
        "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
        "subcell_offset": {"x": float(subcell_x), "y": float(subcell_y)},
        "color_matrix": color_matrix.tolist(),
        "color_bias": color_bias.tolist(),
        "palette_mode": args.palette_mode,
        "classifier": args.classifier,
        "preset": args.preset,
        "patch_stat": args.patch_stat,
        "trim_frac": float(args.trim_frac),
        "use_color_affine": bool(args.use_color_affine),
        "t0_anchor_mode": args.t0_anchor_mode,
        "use_ig_calibration": bool(args.use_ig_calibration),
        "ig_path": str(ig_file) if ig_file.exists() else None,
        "canonicalize_empty_plains": bool(args.canonicalize_empty_plains),
        "temporal_window": int(args.temporal_window),
        "temporal_conf_gated": bool(args.temporal_conf_gated),
        "temporal_conf_threshold": float(args.temporal_conf_threshold),
        "suppress_flicker": bool(args.suppress_flicker),
        "temporal_stats": temporal_stats,
        "prototype_scales_mean": prototype_scales.mean(axis=0).tolist(),
        "palette_classes": [int(x) for x in class_ids.tolist()],
        "palette_json": str(palette_path) if palette_path else None,
        "mean_confidence": float(conf_arr.mean()) if conf_arr.size else 0.0,
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Decoded {grid_arr.shape[0]} frames to {out_dir}")
    print(f"Mean confidence: {meta['mean_confidence']:.4f}")
    if args.validate_t0:
        if ig_file.exists() and grid_arr.shape[0] > 0:
            ig_eval = ig if ig is not None else np.load(ig_file)
            strict_t0 = float((grid_arr[0] == ig_eval).mean())
            g0 = grid_arr[0].copy()
            ig2 = ig_eval.copy()
            g0[g0 == EMPTY_CODE] = PLAINS_CODE
            ig2[ig2 == EMPTY_CODE] = PLAINS_CODE
            collapsed_t0 = float((g0 == ig2).mean())
            print(f"t0_vs_ig_strict={strict_t0:.4f}")
            print(f"t0_vs_ig_collapsed={collapsed_t0:.4f}")
        else:
            print(f"t0 validation skipped (missing initial grid: {ig_file})")


def _load_decoded_dir(args: argparse.Namespace) -> Path:
    return _round_seed_base(Path(args.dataset_root), args.round_id, args.seed_id) / "output" / args.decode_tag


def _strict_and_collapsed_match(
    grids: np.ndarray, ig: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    strict = (grids == ig[None, :, :]).mean(axis=(1, 2))
    g2 = grids.copy()
    g2[g2 == EMPTY_CODE] = PLAINS_CODE
    ig2 = ig.copy()
    ig2[ig2 == EMPTY_CODE] = PLAINS_CODE
    collapsed = (g2 == ig2[None, :, :]).mean(axis=(1, 2))
    return strict, collapsed


def _to_prediction_probs(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    pred = np.zeros((h, w, N_CLASSES), dtype=np.float64)
    for code, cls in CODE_TO_CLASS.items():
        pred[:, :, cls][grid == code] = 1.0
    pred_sum = pred.sum(axis=2, keepdims=True)
    pred_sum[pred_sum <= 0] = 1.0
    return pred / pred_sum


def _top_confusions(
    truth: np.ndarray, pred: np.ndarray, limit: int = 10
) -> list[dict[str, int]]:
    pairs = np.stack([truth.ravel(), pred.ravel()], axis=1)
    uniq, cnt = np.unique(pairs, axis=0, return_counts=True)
    rows: list[dict[str, int]] = []
    for (t, p), c in zip(uniq.tolist(), cnt.tolist()):
        if int(t) == int(p):
            continue
        rows.append({"truth": int(t), "pred": int(p), "count": int(c)})
    rows.sort(key=lambda r: r["count"], reverse=True)
    return rows[:limit]


def cmd_eval(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root)
    dec_dir = _load_decoded_dir(args)
    grid_path = dec_dir / "grid_t.npy"
    conf_path = dec_dir / "confidence_t.npy"
    meta_path = dec_dir / "metadata.json"
    ig_path = data_root / f"round{int(args.round_id)}" / f"ig_r{int(args.round_id)}_seed{int(args.seed_id)}.npy"
    if not grid_path.exists():
        raise FileNotFoundError(f"Decoded grid file not found: {grid_path}")
    if not ig_path.exists():
        raise FileNotFoundError(f"Initial grid file not found: {ig_path}")
    grids = np.load(grid_path)
    confs = np.load(conf_path) if conf_path.exists() else np.zeros_like(grids, dtype=np.float32)
    ig = np.load(ig_path)
    strict, collapsed = _strict_and_collapsed_match(grids, ig)
    best_idx = int(np.argmax(collapsed if args.collapse_empty_plains else strict))
    top_conf_t0 = _top_confusions(ig, grids[0]) if grids.shape[0] > 0 else []
    out = {
        "round_id": int(args.round_id),
        "seed_id": int(args.seed_id),
        "frames": int(grids.shape[0]),
        "strict_first": float(strict[0]) if strict.size else 0.0,
        "strict_best": float(strict.max()) if strict.size else 0.0,
        "strict_best_idx": best_idx,
        "strict_curve": [float(x) for x in strict.tolist()],
        "collapsed_first": float(collapsed[0]) if collapsed.size else 0.0,
        "collapsed_best": float(collapsed.max()) if collapsed.size else 0.0,
        "collapsed_best_idx": int(np.argmax(collapsed)) if collapsed.size else 0,
        "collapsed_curve": [float(x) for x in collapsed.tolist()],
        "top_confusions_t0": top_conf_t0,
        "mean_confidence": float(confs.mean()) if confs.size else 0.0,
    }
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        tstats = meta.get("temporal_stats", {})
        n_over = float(tstats.get("n_overridden", 0.0))
        n_low = float(tstats.get("n_low_conf", 0.0))
        n_t0 = float(tstats.get("n_from_t0", 0.0))
        out["temporal_n_overridden"] = n_over
        out["temporal_n_low_conf"] = n_low
        out["temporal_frac_overridden_of_low_conf"] = (n_over / n_low) if n_low > 0 else 0.0
        out["temporal_frac_from_t0_among_overrides"] = (n_t0 / n_over) if n_over > 0 else 0.0

    gt_path = data_root / f"round{int(args.round_id)}" / f"gt_r{int(args.round_id)}_seed{int(args.seed_id)}.npy"
    if gt_path.exists() and grids.shape[0] > 0:
        gt = np.load(gt_path)
        curve = []
        for t in range(grids.shape[0]):
            pred_t = _to_prediction_probs(grids[t])
            score_t = entropy_weighted_kl(gt, pred_t)
            curve.append(float(score_t["score"]))
        out["gt_proxy_score_curve"] = curve
        out["gt_proxy_score_final"] = float(curve[-1])
        out["gt_proxy_score_delta_t0_to_final"] = float(curve[-1] - curve[0])

    if args.output_json:
        print(json.dumps(out, indent=2))
    else:
        print(f"round={out['round_id']} seed={out['seed_id']} frames={out['frames']}")
        print(
            f"strict first={out['strict_first']:.4f} best={out['strict_best']:.4f} "
            f"@t={out['strict_best_idx']}"
        )
        print(
            f"collapsed first={out['collapsed_first']:.4f} best={out['collapsed_best']:.4f} "
            f"@t={out['collapsed_best_idx']}"
        )
        print(f"mean_confidence={out['mean_confidence']:.4f}")
        if "gt_proxy_score_final" in out:
            print(f"gt_proxy_score_final={out['gt_proxy_score_final']:.2f}")
            print(f"gt_proxy_score_delta_t0_to_final={out['gt_proxy_score_delta_t0_to_final']:.2f}")
        if "temporal_frac_overridden_of_low_conf" in out:
            print(
                "temporal_override_lowconf="
                f"{out['temporal_frac_overridden_of_low_conf']:.4f} "
                "temporal_from_t0="
                f"{out['temporal_frac_from_t0_among_overrides']:.4f}"
            )


def cmd_benchmark(args: argparse.Namespace) -> None:
    tags = [t.strip() for t in args.decode_tags.split(",") if t.strip()]
    if not tags:
        raise ValueError("--decode-tags must contain at least one decode tag")
    dataset_root = Path(args.dataset_root)
    data_root = Path(args.data_root)
    print("=== Decoder benchmark summary ===")
    for tag in tags:
        rows: list[dict[str, float | int]] = []
        temporal_override = []
        temporal_from_t0 = []
        for round_dir in sorted(data_root.glob("round*")):
            if not round_dir.is_dir():
                continue
            try:
                round_id = int(round_dir.name.replace("round", ""))
            except ValueError:
                continue
            for seed_id in range(args.max_seeds):
                ig_path = round_dir / f"ig_r{round_id}_seed{seed_id}.npy"
                dec_dir = dataset_root / f"round{round_id}" / f"seed{seed_id}" / "output" / tag
                dec_path = dec_dir / "grid_t.npy"
                if not ig_path.exists() or not dec_path.exists():
                    continue
                ig = np.load(ig_path)
                grids = np.load(dec_path)
                strict, collapsed = _strict_and_collapsed_match(grids, ig)
                rows.append(
                    {
                        "round": round_id,
                        "seed": seed_id,
                        "strict_first": float(strict[0]),
                        "strict_best": float(strict.max()),
                        "collapsed_first": float(collapsed[0]),
                        "collapsed_best": float(collapsed.max()),
                    }
                )
                meta_path = dec_dir / "metadata.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text())
                    tstats = meta.get("temporal_stats", {})
                    n_over = float(tstats.get("n_overridden", 0.0))
                    n_low = float(tstats.get("n_low_conf", 0.0))
                    n_t0 = float(tstats.get("n_from_t0", 0.0))
                    if n_low > 0:
                        temporal_override.append(n_over / n_low)
                    if n_over > 0:
                        temporal_from_t0.append(n_t0 / n_over)
        if not rows:
            print(f"tag={tag}: no matching decoded runs")
            continue
        strict_first = np.mean([r["strict_first"] for r in rows])
        strict_best = np.mean([r["strict_best"] for r in rows])
        collapsed_first = np.mean([r["collapsed_first"] for r in rows])
        collapsed_best = np.mean([r["collapsed_best"] for r in rows])
        print(f"tag={tag} runs={len(rows)}")
        print(f"  avg strict first={strict_first:.4f} best={strict_best:.4f}")
        print(f"  avg collapsed first={collapsed_first:.4f} best={collapsed_best:.4f}")
        if temporal_override:
            print(f"  avg temporal_override_lowconf={np.mean(temporal_override):.4f}")
        if temporal_from_t0:
            print(f"  avg temporal_from_t0={np.mean(temporal_from_t0):.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Video to grid decoder for Astar Island recordings")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="Print video metadata")
    p_probe.add_argument("--video", required=True)
    p_probe.set_defaults(func=cmd_probe)

    p_extract = sub.add_parser("extract", help="Extract sampled frames to PNG")
    p_extract.add_argument("--video", required=True)
    p_extract.add_argument("--round-id", type=int, required=True)
    p_extract.add_argument("--seed-id", type=int, required=True)
    p_extract.add_argument("--dataset-root", default="data_prev_rounds/video_dataset")
    p_extract.add_argument("--out-dir", default=None, help="Optional override for output directory")
    p_extract.add_argument("--interval-ms", type=int, default=500)
    p_extract.set_defaults(func=cmd_extract)

    p_decode = sub.add_parser("decode", help="Decode sampled frames to class grids")
    p_decode.add_argument("--video", required=True)
    p_decode.add_argument("--round-id", type=int, required=True)
    p_decode.add_argument("--seed-id", type=int, required=True)
    p_decode.add_argument("--dataset-root", default="data_prev_rounds/video_dataset")
    p_decode.add_argument("--out-dir", default=None, help="Optional override for output directory")
    p_decode.add_argument("--decode-tag", default="decoded", help="Folder name under output/")
    p_decode.add_argument("--palette-json", default="testing/palette_template.json")
    p_decode.add_argument("--palette-mode", choices=["exact", "json"], default="exact")
    p_decode.add_argument(
        "--preset",
        choices=["baseline_simple", "t0_anchor_only", "t0_anchor_gated_temporal"],
        default=None,
        help="Apply reproducible decode preset and default decode tag.",
    )
    p_decode.add_argument("--classifier", choices=["nearest_rgb", "nearest_prototype"], default="nearest_rgb")
    p_decode.add_argument("--grid-h", type=int, default=40)
    p_decode.add_argument("--grid-w", type=int, default=40)
    p_decode.add_argument("--interval-ms", type=int, default=500)
    p_decode.add_argument(
        "--roi",
        default=None,
        help="x,y,w,h (defaults to exact map ROI 53,350,866,866)",
    )
    p_decode.add_argument("--patch-ratio", type=float, default=0.5, help="Cell-center patch ratio in (0,1]")
    p_decode.add_argument("--patch-stat", choices=["median", "trimmed_mean"], default="median")
    p_decode.add_argument("--trim-frac", type=float, default=0.15, help="Trim fraction per tail for trimmed mean")
    p_decode.add_argument("--canonicalize-empty-plains", action="store_true", default=True)
    p_decode.add_argument("--no-canonicalize-empty-plains", action="store_false", dest="canonicalize_empty_plains")
    p_decode.add_argument("--roi-refine-px", type=int, default=12, help="Search radius (px) around ROI for local refinement")
    p_decode.add_argument("--roi-refine-step", type=int, default=2, help="Step (px) for ROI local refinement")
    p_decode.add_argument("--subcell-refine", type=float, default=0.15, help="Fractional subcell offset search (+/- value)")
    p_decode.add_argument("--use-color-affine", action="store_true", default=False)
    p_decode.add_argument("--t0-anchor-mode", choices=["off", "force"], default="force")
    p_decode.add_argument("--use-ig-calibration", action="store_true", default=True)
    p_decode.add_argument("--no-use-ig-calibration", action="store_false", dest="use_ig_calibration")
    p_decode.add_argument("--temporal-window", type=int, default=1, help="Temporal majority smoothing window")
    p_decode.add_argument("--temporal-conf-gated", action="store_true", default=False)
    p_decode.add_argument("--temporal-conf-threshold", type=float, default=0.70)
    p_decode.add_argument("--suppress-flicker", action="store_true", default=False)
    p_decode.add_argument("--no-suppress-flicker", action="store_false", dest="suppress_flicker")
    p_decode.add_argument("--trim-leading-static", action="store_true", help="Trim initial static/non-map-like frames")
    p_decode.add_argument("--min-change-frac", type=float, default=0.0, help="Drop near-duplicate frames below this changed-cell fraction")
    p_decode.add_argument("--target-steps", type=int, default=None, help="Resample output to exactly this many time steps")
    p_decode.add_argument("--validate-t0", action="store_true", default=True)
    p_decode.add_argument("--no-validate-t0", action="store_false", dest="validate_t0")
    p_decode.add_argument("--data-root", default="data_prev_rounds", help="Root containing ig_r*_seed*.npy for t0 validation")
    p_decode.add_argument(
        "--ig-seed-offset",
        type=int,
        default=-1,
        help="Map video seed_id to IG seed index (default -1 for seed1..5 -> seed0..4).",
    )
    p_decode.add_argument(
        "--ig-seed-id",
        type=int,
        default=None,
        help="Optional explicit IG seed index override (takes precedence over offset).",
    )
    p_decode.set_defaults(func=cmd_decode)

    p_eval = sub.add_parser("eval", help="Evaluate decoded output against initial grid and optional ground truth")
    p_eval.add_argument("--round-id", type=int, required=True)
    p_eval.add_argument("--seed-id", type=int, required=True)
    p_eval.add_argument("--decode-tag", default="decoded")
    p_eval.add_argument("--dataset-root", default="data_prev_rounds/video_dataset")
    p_eval.add_argument("--data-root", default="data_prev_rounds")
    p_eval.add_argument("--collapse-empty-plains", action="store_true", default=True)
    p_eval.add_argument("--output-json", action="store_true")
    p_eval.set_defaults(func=cmd_eval)

    p_bench = sub.add_parser("benchmark", help="Benchmark decoded runs across available seeds/rounds")
    p_bench.add_argument("--decode-tags", default="decoded", help="Comma-separated decode tags to compare")
    p_bench.add_argument("--dataset-root", default="data_prev_rounds/video_dataset")
    p_bench.add_argument("--data-root", default="data_prev_rounds")
    p_bench.add_argument("--max-seeds", type=int, default=10)
    p_bench.set_defaults(func=cmd_benchmark)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "patch_ratio") and not (0.0 < args.patch_ratio <= 1.0):
        raise ValueError("--patch-ratio must be in (0, 1]")
    if hasattr(args, "trim_frac") and not (0.0 <= args.trim_frac < 0.5):
        raise ValueError("--trim-frac must be in [0, 0.5)")
    if hasattr(args, "subcell_refine") and not (0.0 <= args.subcell_refine <= 0.49):
        raise ValueError("--subcell-refine must be in [0, 0.49]")
    if hasattr(args, "temporal_window") and args.temporal_window < 1:
        raise ValueError("--temporal-window must be >= 1")
    args.func(args)


if __name__ == "__main__":
    main()

