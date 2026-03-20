# Nudging the Prediction — Tweaks & Clever Ideas

Small, targeted changes that improve calibration by encoding what we already know about the world into the prediction pipeline. The core philosophy: don't let the model waste probability mass on physically impossible outcomes, and focus its attention on the cells that actually matter.

---

## The Core Insight: Not All Cells Are Equal

Before any tweaks, it helps to understand where prediction effort should be concentrated.

**Entropy** is the measure of uncertainty in a cell's prediction:

```
entropy = -Σ pᵢ log(pᵢ)
```

- **Static cells** (ocean, mountains) have near-zero entropy — the outcome is already known, so they contribute almost nothing to the score and can be handled with hard rules.
- **Dynamic cells** (settlements, ports, ruins) have high entropy — these are the cells where the model's probability estimates actually matter.

The three terrain types worth caring about are the ones that can transition between states:

| Can become → | Settlement | Port | Ruin |
|---|---|---|---|
| Settlement | ✓ | ✓ | ✓ |
| Port | ✓ | ✓ | ✓ |
| Ruin | ✓ | ✓ | ✓ |

Ocean and mountains never change. Forests are mostly stable but can slowly reclaim ruined land. Everything else is noise at the margin. The tweaks below are ordered from most impactful (hard constraints) to more subtle (shape priors).

---

## 1. Hard Mountain Constraints

**The problem:** The model could assign non-zero probability to a mountain cell becoming something else, or to a non-mountain cell becoming a mountain. Both are physically impossible — mountains are fully static.

**The fix:** A final hard clamp in `predictor.py` after all other processing:

- `P(mountain) = 1.0` on known mountain cells
- `P(mountain) = 0.0` on all non-mountain cells
- Per-cell probabilities are re-normalized after clamping

This is essentially free calibration improvement — zero model changes, just enforcing what we already know.

---

## 2. Hard Port Impossibility Constraints

**The problem:** The model could predict a port appearing inland or on a mountain — both impossible. A port requires a coastal cell.

**The fix:** Extended the same hard-constraint step in `predictor.py`:

- `P(port) = 0.0` wherever a port is physically impossible
- Mask used: `(~coastal_mask(init_grid)) | (init_grid == MOUNTAIN_CODE)`
- Per-cell probabilities are re-normalized with a defensive fallback to keep tensors valid

Like the mountain constraint, this zeroes out impossible states before they can pollute the final prediction.

---

## 3. Settlement Overprediction Reduction

**The problem:** The iterative denoising loop was applying a compounding boost to settlement probability on each pass, causing the model to over-predict settlements globally.

**The fix:** A single constant change:

```
EMPIRICAL_FULL_COVERAGE_ITER_SETTLE_BOOST: 1.15 → 1.0
```

Setting the boost to `1.0` removes the amplification entirely while keeping the same model structure. Validated via backtest sweep on rounds 2 and 3 (`budget=50`):

| Boost value | Avg score | Settlement MAE |
|---|---|---|
| `1.15` | (baseline) | higher |
| `1.10` | — | — |
| `1.05` | — | — |
| **`1.00`** | **best** | **lowest** |

`1.0` outperformed all tested values on both overall score and settlement MAE.

---

## 4. Settlement Gaussian / RBF Shape Prior

**The problem:** The model's predicted settlement distribution can be spatially noisy — high probability scattered across many cells rather than concentrated around plausible settlement clusters.

**The idea:** Real settlements cluster. A settlement is more likely to appear near other settlements than in an isolated cell. We can encode this with a distance-decay (RBF/Gaussian) prior built from the known initial settlement positions.

**The implementation:** A helper `_apply_settlement_intensity_prior(...)` in `predictor.py`:

```
SETTLEMENT_INTENSITY_BLEND_ALPHA = 0.22
SETTLEMENT_INTENSITY_SIGMA       = 2.2
```

- The prior is built from known settlement centers (`has_settle`) using Gaussian distance-decay smoothing.
- It is **scaled to the current total settlement mass** before blending — so it reshapes the spatial distribution without inflating the global settlement probability.
- Blended into settlement mass with `alpha = 0.22`.

**Result:** More blob-like, spatially coherent predictions. A/B test on rounds 2 and 3 (`budget=50`):

| Alpha | Avg score | Settlement MAE |
|---|---|---|
| `0.00` (off) | 24.7186 | 0.174922 |
| **`0.22`** | **25.5659** | **0.168616** |

A clear improvement in both overall score and settlement spatial accuracy.

---

## Validation Checklist

After each change, the following assertions are verified:

- ✅ Mountain cells are one-hot on the mountain class
- ✅ Non-mountain cells have zero mountain probability
- ✅ Physically impossible port cells have zero port probability
- ✅ Per-cell probabilities sum to `1.0` across all classes
- ✅ Syntax and lint checks pass on all modified files

---

## Summary

| Tweak | Type | Effect |
|---|---|---|
| Hard mountain constraints | Hard rule | Eliminates impossible mountain predictions |
| Hard port constraints | Hard rule | Eliminates impossible port predictions |
| Settlement boost `1.15 → 1.0` | Constant tune | Reduces global settlement overprediction |
| Settlement Gaussian prior (`α=0.22`) | Shape prior | Improves spatial clustering of settlement predictions |

The general pattern: **hard constraints first** (they're free wins), then **reduce amplification**, then **add spatial priors** to shape the distribution. Each layer builds on the last.