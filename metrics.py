"""
Metrics and evaluation for the Astar Island pipeline.

Logs per-round metrics to metrics.json for continuous improvement tracking.
Supports:
  - Self-check (holdout validation with KL divergence estimation)
  - Post-round analysis (compare predictions vs ground truth)
  - Cross-round learning (transition priors, parameter trends)
  - Detailed per-class and per-terrain breakdown
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from config import N_CLASSES, CLASS_NAMES, PROB_FLOOR, METRICS_FILE

# ── KL divergence helpers ─────────────────────────────────────────────────────

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> float:
    """KL(p || q) for probability vectors. Returns ∞ if q=0 and p>0."""
    p = p.astype(np.float64)
    q = np.maximum(q.astype(np.float64), eps)
    mask = p > eps
    return float(np.sum(p[mask] * (np.log(p[mask]) - np.log(q[mask]))))


def cell_entropy(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector. Used for entropy weighting."""
    p = probs.astype(np.float64)
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))


def entropy_weighted_kl(
    ground_truth: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    """
    Compute the official Astar Island scoring metric.

    score = 100 * exp(-3 * weighted_kl)
    where weighted_kl = Σ entropy(cell) * KL(gt, pred) / Σ entropy(cell)

    Returns dict with weighted_kl, score, and per-class breakdown.
    """
    H, W, C = ground_truth.shape
    assert prediction.shape == (H, W, C)

    q = np.maximum(prediction.astype(np.float64), PROB_FLOOR)
    q = q / q.sum(axis=2, keepdims=True)

    entropies = np.zeros((H, W))
    kls       = np.zeros((H, W))

    for y in range(H):
        for x in range(W):
            ent = cell_entropy(ground_truth[y, x])
            entropies[y, x] = ent
            if ent > 1e-6:
                kls[y, x] = kl_divergence(ground_truth[y, x], q[y, x])

    total_entropy = float(entropies.sum())
    if total_entropy < 1e-9:
        return {"weighted_kl": 0.0, "score": 100.0, "n_dynamic_cells": 0}

    weighted_kl = float((entropies * kls).sum()) / total_entropy
    score = float(100.0 * np.exp(-3.0 * weighted_kl))

    # Per-class breakdown (argmax cell contribution)
    argmax_gt = ground_truth.argmax(axis=2)
    class_scores = {}
    for c in range(N_CLASSES):
        mask = (argmax_gt == c) & (entropies > 1e-6)
        if mask.sum() > 0:
            ent_c = float(entropies[mask].sum())
            wkl_c = float((entropies[mask] * kls[mask]).sum()) / max(ent_c, 1e-9)
            class_scores[CLASS_NAMES[c]] = {
                "count": int(mask.sum()),
                "weighted_kl": float(wkl_c),
                "score": float(100 * np.exp(-3 * wkl_c)),
            }

    return {
        "weighted_kl": float(weighted_kl),
        "score": float(score),
        "n_dynamic_cells": int((entropies > 1e-6).sum()),
        "by_class": class_scores,
    }


# ── Self-check (holdout validation) ──────────────────────────────────────────

def holdout_self_check(
    initial_states: list[dict],
    store,
    predictions: dict[int, np.ndarray],
    holdout_fraction: float = 0.15,
    random_seed: int = 2026,
    posterior_tau: float = 2.0,
) -> dict:
    """
    Hold out a fraction of observed cells, run predictions without them,
    then measure KL divergence between held-out observations and predictions.

    Observed cell counts are sparse samples from a latent probability vector, so
    the comparison uses a smoothed posterior mean by default instead of the raw
    empirical counts. This is less prone to rewarding overconfident predictors.
    """
    from state import ObservationStore, code_to_class  # avoid circular
    import copy

    rng = np.random.default_rng(random_seed)
    held_kls: list[float] = []
    seed_reports: dict[int, dict] = {}
    global_counts = np.zeros(N_CLASSES, dtype=np.float64)

    for seed in range(store.seeds_count):
        global_counts += store.counts[seed].sum(axis=(0, 1))
    if global_counts.sum() <= 0:
        global_counts += 1.0
    global_prior = global_counts / global_counts.sum()

    for seed in range(store.seeds_count):
        counts = store.counts[seed]
        n_samp = counts.sum(axis=2)
        obs_indices = np.argwhere(n_samp > 0)

        if len(obs_indices) == 0:
            seed_reports[seed] = {"n_holdout": 0, "mean_kl": float("nan")}
            continue

        n_hold = max(1, int(round(len(obs_indices) * holdout_fraction)))
        sel = obs_indices[rng.choice(len(obs_indices), n_hold, replace=False)]

        # KL: empirical distribution at held-out cells vs prediction
        kls_seed: list[float] = []
        for y, x in sel.tolist():
            emp = counts[y, x].astype(np.float64)
            total = float(emp.sum())
            if total <= 0:
                continue
            if posterior_tau > 0.0:
                p = (emp + posterior_tau * global_prior) / (total + posterior_tau)
            else:
                p = emp / total
            q = np.maximum(predictions[seed][y, x].astype(np.float64), PROB_FLOOR)
            q = q / q.sum()
            kls_seed.append(kl_divergence(p, q))

        mean_kl = float(np.mean(kls_seed)) if kls_seed else float("nan")
        approx_score = float(100 * np.exp(-3 * mean_kl)) if not np.isnan(mean_kl) else float("nan")
        held_kls.extend(kls_seed)
        seed_reports[seed] = {
            "n_holdout": len(kls_seed),
            "mean_kl": mean_kl,
            "approx_score": approx_score,
        }

    overall_kl = float(np.mean(held_kls)) if held_kls else float("nan")
    overall_score = float(100 * np.exp(-3 * overall_kl)) if not np.isnan(overall_kl) else float("nan")

    label = "holdout validation"
    if posterior_tau > 0.0:
        label = f"smoothed holdout validation, tau={posterior_tau:g}"
    print(f"\nSelf-check ({label}):")
    print(f"  Overall KL={overall_kl:.4f}, approx score={overall_score:.1f}/100")
    for s, r in sorted(seed_reports.items()):
        kl_str = f"{r['mean_kl']:.4f}" if not np.isnan(r.get('mean_kl', float('nan'))) else "n/a"
        sc_str = f"{r['approx_score']:.1f}" if not np.isnan(r.get('approx_score', float('nan'))) else "n/a"
        print(f"  Seed {s}: n={r['n_holdout']}, KL={kl_str}, score≈{sc_str}")

    return {
        "overall": {
            "mean_kl": overall_kl,
            "approx_score": overall_score,
            "posterior_tau": posterior_tau,
        },
        "by_seed": seed_reports,
    }


# ── Metrics logger ────────────────────────────────────────────────────────────

class MetricsLogger:
    """
    Logs all metrics to metrics.json for continuous improvement across rounds.

    Entries are keyed by round_id. Each entry records:
    - Query budget usage
    - Coverage statistics
    - Estimated hidden parameters
    - Self-check results (pre-submission estimate)
    - Post-round ground truth scores (if fetched)
    - Prediction distribution summaries
    - Observation transition statistics
    - Time taken
    """

    def __init__(self, path: str = METRICS_FILE) -> None:
        self.path = Path(path)
        self._data: dict = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                self._data = {}

    def log_round_start(
        self, round_id: str, round_number: int, width: int, height: int,
        seeds_count: int, round_weight: float = 1.0,
    ) -> None:
        self._data[round_id] = {
            "round_id": round_id,
            "round_number": round_number,
            "width": width,
            "height": height,
            "seeds_count": seeds_count,
            "round_weight": round_weight,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._save()

    def log_queries(self, round_id: str, store) -> None:
        entry = self._data.get(round_id, {})
        coverage_by_seed = {}
        samples_by_seed = {}
        for s in range(store.seeds_count):
            cov = store.coverage(s)
            nsam = int(store.n_samples(s).sum())
            coverage_by_seed[str(s)] = round(cov, 4)
            samples_by_seed[str(s)] = nsam

        entry["queries_used"] = store.queries_used
        entry["coverage_by_seed"] = coverage_by_seed
        entry["samples_by_seed"] = samples_by_seed
        entry["avg_coverage"] = float(np.mean(list(coverage_by_seed.values())))
        self._data[round_id] = entry
        self._save()

    def log_params(self, round_id: str, params: dict[str, float]) -> None:
        entry = self._data.get(round_id, {})
        entry["estimated_params"] = {k: round(float(v), 4) for k, v in params.items()}
        self._data[round_id] = entry
        self._save()

    def log_self_check(self, round_id: str, self_check_result: dict) -> None:
        entry = self._data.get(round_id, {})
        entry["self_check"] = self_check_result
        self._data[round_id] = entry
        self._save()

    def log_predictions(
        self,
        round_id: str,
        predictions: dict[int, np.ndarray],
        initial_states: list[dict],
        store,
    ) -> None:
        entry = self._data.get(round_id, {})
        pred_summary: dict[str, dict] = {}
        for seed, pred in predictions.items():
            argmax = pred.argmax(axis=2)
            class_counts = {CLASS_NAMES[c]: int((argmax == c).sum()) for c in range(N_CLASSES)}
            avg_conf = float(pred.max(axis=2).mean())
            avg_entropy = float(np.mean([-np.sum(pred[y, x] * np.log(pred[y, x] + 1e-12))
                                         for y in range(store.height)
                                         for x in range(store.width)]))
            pred_summary[str(seed)] = {
                "class_counts": class_counts,
                "avg_confidence": round(avg_conf, 4),
                "avg_entropy": round(avg_entropy, 4),
            }
        entry["prediction_summary"] = pred_summary
        self._data[round_id] = entry
        self._save()

    def log_submission(self, round_id: str, statuses: dict[int, str]) -> None:
        entry = self._data.get(round_id, {})
        entry["submission_statuses"] = {str(k): v for k, v in statuses.items()}
        entry["submitted_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._data[round_id] = entry
        self._save()

    def log_ground_truth(
        self,
        round_id: str,
        ground_truth: dict[int, np.ndarray],
        predictions: dict[int, np.ndarray],
    ) -> None:
        """Log actual scores after fetching ground truth from the analysis endpoint."""
        entry = self._data.get(round_id, {})
        seed_scores: dict[str, dict] = {}

        for seed, gt in ground_truth.items():
            if seed not in predictions:
                continue
            result = entropy_weighted_kl(gt, predictions[seed])
            seed_scores[str(seed)] = result

        scores_list = [s["score"] for s in seed_scores.values() if "score" in s]
        avg_score = float(np.mean(scores_list)) if scores_list else float("nan")

        entry["ground_truth_scores"] = seed_scores
        entry["avg_score"] = round(avg_score, 4)
        entry["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._data[round_id] = entry
        self._save()

        print(f"\nGround truth scores:")
        for seed, res in sorted(seed_scores.items()):
            print(f"  Seed {seed}: score={res.get('score', 'n/a'):.1f}, "
                  f"weighted_kl={res.get('weighted_kl', 'n/a'):.4f}")
        print(f"  Average: {avg_score:.2f}/100")

    def log_transition_priors(
        self, round_id: str, store, initial_states: list[dict]
    ) -> None:
        """Log empirical code→class transition rates for cross-round learning."""
        transitions = store.transition_summary(initial_states)
        by_init: dict[int, dict[int, int]] = {}
        for (ic, fc), cnt in transitions.items():
            by_init.setdefault(ic, {})[fc] = cnt

        transition_priors: dict[str, list[float]] = {}
        for ic, fcounts in by_init.items():
            total = sum(fcounts.values())
            if total > 0:
                dist = [fcounts.get(c, 0) / total for c in range(N_CLASSES)]
                transition_priors[str(ic)] = [round(v, 4) for v in dist]

        entry = self._data.get(round_id, {})
        entry["transition_priors"] = transition_priors
        self._data[round_id] = entry
        self._save()

    def print_cross_round_summary(self) -> None:
        """Print a summary table of all logged rounds for tracking improvement."""
        completed = {
            rid: d for rid, d in self._data.items()
            if "avg_score" in d or "self_check" in d
        }
        if not completed:
            print("No completed round data in metrics log.")
            return

        print("\n── Cross-round summary ──────────────────────────────")
        print(f"{'Round':<8} {'Coverage':>9} {'SelfCheck':>10} {'Score':>7} {'Params'}")
        for rid, d in sorted(completed.items(), key=lambda x: x[1].get("round_number", 0)):
            rn   = d.get("round_number", "?")
            cov  = f"{d.get('avg_coverage', 0):.1%}"
            sc_r = d.get("self_check", {}).get("overall", {})
            sc   = f"{sc_r.get('approx_score', float('nan')):.1f}" if sc_r else "-"
            score= f"{d.get('avg_score', float('nan')):.1f}" if "avg_score" in d else "-"
            ep   = d.get("estimated_params", {})
            ep_str = " ".join(f"{k[:3]}={v:.2f}" for k, v in ep.items())
            print(f"{rn:<8} {cov:>9} {sc:>10} {score:>7}  {ep_str}")

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")


# ── Post-round analysis ───────────────────────────────────────────────────────

def fetch_and_log_analysis(
    api,
    round_id: str,
    seeds_count: int,
    predictions: dict[int, np.ndarray],
    logger: MetricsLogger,
) -> Optional[dict[int, np.ndarray]]:
    """
    Fetch ground truth from the analysis endpoint and log scores.
    Only available after a round completes.
    """
    from api import APIError  # avoid circular

    ground_truth: dict[int, np.ndarray] = {}
    for seed in range(seeds_count):
        try:
            resp = api.get_analysis(round_id, seed)
        except APIError as exc:
            print(f"  Seed {seed}: analysis not available ({exc})")
            return None

        if "ground_truth" not in resp:
            print(f"  Seed {seed}: no ground_truth in response")
            return None

        gt = np.array(resp["ground_truth"], dtype=np.float32)
        ground_truth[seed] = gt
        np.save(f"gt_fetched_seed_{seed}.npy", gt)

        api_score = resp.get("score")
        print(f"  Seed {seed}: API score = {api_score}")

    if ground_truth:
        logger.log_ground_truth(round_id, ground_truth, predictions)

    return ground_truth


def load_previous_round_data(data_dir: str) -> Optional[dict]:
    """
    Load ground truth and initial grids from a previous round data directory.
    Used for offline analysis and calibration.

    Expected files: gt_r{N}_seed{i}.npy, ig_r{N}_seed{i}.npy
    """
    from pathlib import Path
    d = Path(data_dir)
    if not d.exists():
        return None

    gt_files = sorted(d.glob("gt_*.npy"))
    ig_files = sorted(d.glob("ig_*.npy"))
    if not gt_files or not ig_files:
        return None

    ground_truths = {i: np.load(f) for i, f in enumerate(gt_files)}
    init_grids    = {i: np.load(f) for i, f in enumerate(ig_files)}

    print(f"Loaded {len(gt_files)} GT and {len(ig_files)} IG files from {data_dir}")
    return {"ground_truth": ground_truths, "initial_grids": init_grids}
