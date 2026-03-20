# Concrete Tweaks

Small, low-risk changes implemented to improve calibration and remove physically impossible predictions.

## 1) Hard mountain constraints
- Added a final hard clamp in `predictor.py` so:
  - `P(mountain) = 1.0` on known mountain cells.
  - `P(mountain) = 0.0` on all non-mountain cells.
- Re-normalization is applied per cell after clamping.

## 2) Hard port impossibility constraints
- Extended the same final hard-constraint step so:
  - `P(port) = 0.0` where a port is physically impossible.
  - Mask used: `(~coastal_mask(init_grid)) | (init_grid == MOUNTAIN_CODE)`.
- Per-cell probabilities are re-normalized with a defensive fallback to keep tensors valid.

## 3) Settlement overprediction reduction (single-constant tune)
- Reduced:
  - `EMPIRICAL_FULL_COVERAGE_ITER_SETTLE_BOOST: 1.15 -> 1.0`
- This removes iterative settlement amplification in full-coverage denoising while keeping the same model structure.

## 4) Settlement Gaussian/RBF shape prior
- Added a settlement-intensity prior in `predictor.py`:
  - `SETTLEMENT_INTENSITY_BLEND_ALPHA = 0.22`
  - `SETTLEMENT_INTENSITY_SIGMA = 2.2`
  - helper: `_apply_settlement_intensity_prior(...)`
- The prior is built from settlement centers (`has_settle`) using distance-decay smoothing and blended into settlement mass.
- It is scaled to current total settlement mass before blending, so this primarily changes spatial shape (more blob-like), not global magnitude.

## Validation summary
- Syntax and lint checks passed for modified files.
- Constraint assertions passed:
  - mountain one-hot on mountain cells,
  - zero mountain on non-mountain cells,
  - zero port on physically impossible cells,
  - per-cell probabilities sum to 1.
- Backtest sweep on rounds 2 and 3 (`budget=50`) favored `settle_boost=1.0` over `1.05/1.10/1.15` on overall score and settlement MAE.
- A/B test for settlement-intensity prior on rounds 2 and 3 (`budget=50`) improved:
  - `alpha=0.00`: avg score `24.7186`, settlement MAE `0.174922`
  - `alpha=0.22`: avg score `25.5659`, settlement MAE `0.168616`

