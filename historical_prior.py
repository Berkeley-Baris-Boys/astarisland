"""
Historical prior: direct per-cell ig→gt lookup.

For each (y,x) cell in the current round, averages gt[y,x] across all
historical pairs whose ig grid most closely matches the current round's ig.

This is strictly better than the old bucket approach because:
  - It preserves exact spatial context (forest patch at (y,x), ocean
    adjacency, etc.) without lossy feature bucketing.
  - With 40 historical pairs it achieves ~98% per-cell confidence.
  - Weighting by ig-similarity ensures seeds whose starting positions
    resemble the current round contribute more.

Public API (unchanged from old version):
    from historical_prior import get_historical_prior
    hist = get_historical_prior()
    prior, conf = hist.lookup_cellwise(current_ig)        # preferred
    dist, conf  = hist.lookup(ig_code, dist, coast, fadj) # legacy compat
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Optional

import numpy as np

_DATA_DIR = Path(__file__).parent / "data_prev_rounds"

N_CLASSES = 6
_DIST_BINS = [0.0, 2.0, 4.0, 6.0, 8.0, float("inf")]
_MIN_CELLS_FOR_CONF = 20
_CONF_SCALE = 80.0


def _dist_bucket(d: float) -> int:
    for i, edge in enumerate(_DIST_BINS[1:]):
        if d < edge:
            return i
    return len(_DIST_BINS) - 2


def _forest_bucket(f: int) -> int:
    return min(int(f), 2)


class HistoricalPrior:
    """
    Precomputed per-cell prior from historical ig→gt pairs.

    Primary method: lookup_cellwise(current_ig)
      Returns H×W×6 prior and H×W confidence, computed by averaging
      the top-K most-similar historical gts per seed.

    Legacy method: lookup(ig_code, dist_to_settle, is_coastal, forest_adj)
      Returns (distribution, confidence) via old bucket table.
      Kept for backward compatibility.
    """

    def __init__(
        self,
        buckets: dict[tuple, np.ndarray],
        counts: dict[tuple, int],
        global_dist: np.ndarray,
        n_pairs: int,
        pairs: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        self._buckets = buckets
        self._counts  = counts
        self._global  = global_dist
        self.n_pairs  = n_pairs
        self._pairs   = pairs or []   # raw (ig, gt) tuples for cellwise lookup

    # ── Primary method ──────────────────────────────────────────────────────────

    def lookup_cellwise(
        self,
        current_ig: np.ndarray,
        top_k: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Direct per-cell prior for the given initial-state grid.

        Finds the `top_k` most similar historical ig grids (by cell-level
        agreement), then averages their gt arrays cell-by-cell.  If top_k is
        None, uses all pairs with matching map size.

        Returns
        -------
        prior : H×W×6 float64   — per-cell probability distributions
        conf  : H×W float64     — confidence ∈ [0, 1]
        """
        H, W = current_ig.shape
        matching = [(ig, gt) for ig, gt in self._pairs if ig.shape == (H, W)]

        if not matching:
            uniform = np.full((H, W, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float64)
            return uniform, np.zeros((H, W), dtype=np.float64)

        # ── Similarity weighting by settlement-position Jaccard ─────────────────
        # Settlement positions drive almost all dynamics variance across seeds.
        # Jaccard overlap on settlement cells gives near-1.0 for same-seed pairs
        # (identical starting positions) and near-0 for different-seed pairs.
        # This lets us find and heavily weight the "same seed, previous rounds"
        # subset, which is the ideal prior for the current seed.
        cur_settle = set(
            map(tuple, np.argwhere(np.isin(current_ig, [1, 2])).tolist())
        )

        def _jaccard(ig: np.ndarray) -> float:
            hist_settle = set(
                map(tuple, np.argwhere(np.isin(ig, [1, 2])).tolist())
            )
            union = cur_settle | hist_settle
            if not union:
                return 1.0
            return len(cur_settle & hist_settle) / len(union)

        sims = np.array([_jaccard(ig) for ig, _ in matching], dtype=np.float64)

        # If any pairs have near-identical settlement positions (same seed across
        # rounds), restrict to those — they're the gold-standard prior.
        exact_mask = sims >= 0.90
        if exact_mask.sum() >= 3:
            matching = [m for m, keep in zip(matching, exact_mask) if keep]
            sims     = sims[exact_mask]

        if top_k is not None and top_k < len(matching):
            keep_idx = np.argsort(sims)[-top_k:]
            matching  = [matching[i] for i in keep_idx]
            sims      = sims[keep_idx]

        # Softmax with high temperature to strongly prefer better-matching pairs
        sims_shifted = sims - sims.max()
        weights = np.exp(8.0 * sims_shifted)
        weights /= weights.sum()

        # ── Weighted average of gt grids ────────────────────────────────────────
        prior = np.zeros((H, W, N_CLASSES), dtype=np.float64)
        for w, (_, gt) in zip(weights, matching):
            prior += w * gt.astype(np.float64)

        prior = np.maximum(prior, 1e-6)
        prior /= prior.sum(axis=2, keepdims=True)

        # ── Per-cell confidence ─────────────────────────────────────────────────
        # High base confidence (40 real simulation outcomes).
        # Discount cells near initial settlements because their dynamics are
        # seed-specific and vary most across rounds.
        n = len(matching)
        base_conf = float(1.0 - np.exp(-n / 8.0))   # 40 pairs → ~0.99

        settle_mask = np.isin(current_ig, [1, 2])
        try:
            from scipy.ndimage import distance_transform_edt
            dist_s = (
                distance_transform_edt(~settle_mask).astype(np.float64)
                if settle_mask.any()
                else np.full((H, W), 99.0, dtype=np.float64)
            )
        except ImportError:
            dist_s = np.full((H, W), 99.0, dtype=np.float64)

        # Cells 0-2 away from an initial settlement get ~40% discount;
        # cells ≥8 away get near-zero discount.
        settle_penalty = 0.40 * np.exp(-dist_s / 3.0)
        conf = np.clip(base_conf * (1.0 - settle_penalty), 0.0, 1.0)

        return prior, conf

    # ── Legacy bucket method ────────────────────────────────────────────────────

    def lookup(
        self,
        ig_code: int,
        dist_to_settle: float,
        is_coastal: bool,
        forest_adj: int,
    ) -> tuple[np.ndarray, float]:
        db    = _dist_bucket(dist_to_settle)
        fb    = _forest_bucket(forest_adj)
        coast = int(bool(is_coastal))
        for key in [
            ("full", ig_code, db, coast, fb),
            ("dc",   ig_code, db, coast),
            ("d",    ig_code, db),
            ("i",    ig_code),
        ]:
            dist = self._buckets.get(key)
            if dist is not None:
                n    = self._counts.get(key, 0)
                conf = float(1.0 - np.exp(-n / _CONF_SCALE))
                return dist, conf
        return self._global, 0.0

    def lookup_grid(
        self,
        ig: np.ndarray,
        dist_to_settle: np.ndarray,
        coastal: np.ndarray,
        forest_adj: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        H, W = ig.shape
        prior = np.zeros((H, W, N_CLASSES), dtype=np.float64)
        conf  = np.zeros((H, W), dtype=np.float64)
        for y in range(H):
            for x in range(W):
                d, c = self.lookup(
                    int(ig[y, x]),
                    float(dist_to_settle[y, x]),
                    bool(coastal[y, x]),
                    int(forest_adj[y, x]),
                )
                prior[y, x] = d
                conf[y, x]  = c
        return prior, conf


# ── Build ──────────────────────────────────────────────────────────────────────

def _build_from_pairs(pairs: list[tuple[np.ndarray, np.ndarray]]) -> HistoricalPrior:
    from scipy.ndimage import distance_transform_edt

    raw:   dict[tuple, list[np.ndarray]] = {}
    cnts:  dict[tuple, int]              = {}
    global_acc = np.zeros(N_CLASSES, dtype=np.float64)
    global_n   = 0

    for ig, gt in pairs:
        H, W = ig.shape
        settle_mask = np.isin(ig, [1, 2])
        dist = (
            distance_transform_edt(~settle_mask).astype(np.float64)
            if settle_mask.any()
            else np.full((H, W), 999.0, dtype=np.float64)
        )
        forest = (ig == 4).astype(np.int8)
        fadj = np.zeros((H, W), dtype=int)
        fadj[1:,  :] += (ig[:-1, :] == 4)
        fadj[:-1, :] += (ig[1:,  :] == 4)
        fadj[:, 1:]  += (ig[:, :-1] == 4)
        fadj[:, :-1] += (ig[:, 1:]  == 4)
        ocean = (ig == 10)
        coast = np.zeros((H, W), dtype=bool)
        coast[1:,  :] |= ocean[:-1, :]
        coast[:-1, :] |= ocean[1:,  :]
        coast[:, 1:]  |= ocean[:, :-1]
        coast[:, :-1] |= ocean[:, 1:]
        coast &= ~ocean

        for y in range(H):
            for x in range(W):
                ig_code = int(ig[y, x])
                gt_cell = gt[y, x].astype(np.float64)
                db = _dist_bucket(float(dist[y, x]))
                fb = _forest_bucket(int(fadj[y, x]))
                c  = int(bool(coast[y, x]))
                for key in [
                    ("full", ig_code, db, c, fb),
                    ("dc",   ig_code, db, c),
                    ("d",    ig_code, db),
                    ("i",    ig_code),
                ]:
                    if key not in raw:
                        raw[key]  = []
                        cnts[key] = 0
                    raw[key].append(gt_cell)
                    cnts[key] += 1
                global_acc += gt_cell
                global_n   += 1

    buckets: dict[tuple, np.ndarray] = {}
    for key, vecs in raw.items():
        avg = np.mean(vecs, axis=0)
        avg = np.maximum(avg, 1e-6)
        avg /= avg.sum()
        buckets[key] = avg

    global_dist = np.maximum(global_acc / max(global_n, 1), 1e-6)
    global_dist /= global_dist.sum()

    return HistoricalPrior(buckets, cnts, global_dist, len(pairs), pairs=pairs)


def _load_pairs_from_dir(data_dir: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    pairs = []
    for r in range(1, 20):
        for s in range(5):
            ig_path = data_dir / f"ig_r{r}_seed{s}.npy"
            gt_path = data_dir / f"gt_r{r}_seed{s}.npy"
            if ig_path.exists() and gt_path.exists():
                pairs.append((np.load(ig_path), np.load(gt_path)))
    return pairs


@functools.lru_cache(maxsize=1)
def get_historical_prior(data_dir: Optional[str] = None) -> HistoricalPrior:
    d     = Path(data_dir) if data_dir else _DATA_DIR
    pairs = _load_pairs_from_dir(d)
    if not pairs:
        uniform = np.full(N_CLASSES, 1.0 / N_CLASSES)
        return HistoricalPrior({}, {}, uniform, 0, pairs=[])
    print(f"[historical_prior] Loaded {len(pairs)} ig→gt pairs from {d}")
    return _build_from_pairs(pairs)
