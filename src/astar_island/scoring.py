from __future__ import annotations

import math

import numpy as np

EPS = 1e-12


def cell_entropy(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    safe = np.maximum(probs, EPS)
    return -np.sum(np.where(probs > 0.0, probs * np.log(safe), 0.0), axis=-1)


def cell_kl_divergence(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    gt = np.asarray(ground_truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    safe_gt = np.maximum(gt, EPS)
    safe_pred = np.maximum(pred, EPS)
    return np.sum(np.where(gt > 0.0, gt * (np.log(safe_gt) - np.log(safe_pred)), 0.0), axis=-1)


def weighted_kl_divergence(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    gt = np.asarray(ground_truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if gt.shape != pred.shape:
        raise ValueError(f"Ground truth and prediction shapes differ: {gt.shape} vs {pred.shape}")
    entropy = cell_entropy(gt)
    weights = float(np.sum(entropy))
    if weights <= 0.0:
        return 0.0
    kl = cell_kl_divergence(gt, pred)
    return float(np.sum(entropy * kl) / weights)


def score_prediction(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    weighted_kl = weighted_kl_divergence(ground_truth, prediction)
    return float(max(0.0, min(100.0, 100.0 * math.exp(-3.0 * weighted_kl))))
