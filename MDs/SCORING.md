# Astar Island — Scoring

Your score measures how well your predicted probability distributions match the ground truth, with extra weight on the cells where outcomes are genuinely uncertain.

---

## Ground Truth

For each seed, the organisers pre-run the simulation hundreds of times with the true hidden parameters. This produces a **probability distribution per cell** — not a single outcome, but a spread across all possible terrain classes.

For example, a cell after 50 years might have ground truth:

```
[0.0, 0.60, 0.25, 0.15, 0.0, 0.0]
```

That means: 60% chance of Settlement, 25% Port, 15% Ruin. Your job is to predict that distribution as closely as possible.

---

## The Score Formula

Scoring has three layers: KL divergence per cell, entropy weighting across cells, and a final exponential transform to a 0–100 scale.

### Step 1 — KL Divergence (per cell)

KL divergence measures how far your prediction `q` is from the ground truth `p`:

```
KL(p || q) = Σ pᵢ × log(pᵢ / qᵢ)
```

Lower KL = better match. A perfect prediction yields KL = 0.

### Step 2 — Entropy Weighting

Not all cells matter equally. Static cells (ocean stays ocean, mountains stay mountains) have near-zero entropy and contribute almost nothing to the score. Only cells with genuinely uncertain outcomes are weighted meaningfully.

Entropy of a cell:

```
entropy(cell) = -Σ pᵢ × log(pᵢ)
```

Cells with higher entropy — those that could plausibly end up as a Settlement, Port, or Ruin — count more toward your final score. This focuses scoring on the interesting, dynamic parts of the map.

### Step 3 — Weighted KL & Final Score

```
weighted_kl = Σ entropy(cell) × KL(ground_truth[cell], prediction[cell])
              ─────────────────────────────────────────────────────────
                            Σ entropy(cell)

score = max(0, min(100, 100 × exp(-3 × weighted_kl)))
```

| Score | Meaning |
|-------|---------|
| `100` | Perfect — your distribution matches ground truth exactly |
| `0` | Catastrophic — KL divergence is very high |

The exponential decay means the biggest gains come from fixing your worst predictions, not polishing already-good ones.

---

## The Probability Zero Problem

> **Never assign `0.0` probability to any class.**

This is the single most important rule for scoring. The KL term `pᵢ × log(pᵢ / qᵢ)` diverges to infinity if the ground truth has `pᵢ > 0` but your prediction has `qᵢ = 0`. A single zero can return an infinite KL for that cell, destroying your score entirely.

Even if you're highly confident a cell is Forest, the ground truth — computed from hundreds of simulations — will typically assign small non-zero probability to other classes. You cannot safely assume any class is truly impossible.

**Fix:** Apply a minimum probability floor and renormalise:

```python
prediction = np.maximum(prediction, 0.01)
prediction = prediction / prediction.sum(axis=-1, keepdims=True)
```

This small floor costs almost nothing in score on cells you're confident about, but protects against catastrophic blowups on cells where the ground truth surprises you.

---

## Per-Round Score

Each round covers **5 seeds**. Your round score is the straight average across all seeds:

```
round_score = (score_seed_0 + score_seed_1 + score_seed_2 + score_seed_3 + score_seed_4) / 5
```

If you don't submit for a seed, that seed scores `0`. A uniform prediction (equal probability across all 6 classes) beats a missing submission — **always submit something for every seed**.

---

## Leaderboard Score

Your leaderboard position is determined by your **single best weighted round score** across all rounds:

```
leaderboard_score = max(round_score × round_weight) across all rounds
```

Round weights increase over time:

```
round_weight = 1.05 ^ round_number
```

Later rounds count for more. Only your best single result matters for the all-time leaderboard, so a strong performance in a later round can vault you up even if earlier rounds were weak.

A **hot streak score** (average of your last 3 rounds) is also tracked separately.

---

## Round Timeline

Each round has a prediction window of typically **2 hours 45 minutes**. After it closes:

1. Round status moves to `scoring`
2. All submitted predictions are scored against ground truth
3. Per-seed scores are averaged into a round score
4. Leaderboard updates with each team's best weighted result
5. Round status moves to `completed`

---

## Quick Reference

| Thing to remember | Detail |
|---|---|
| Never use `0.0` in any prediction | Causes infinite KL divergence |
| Minimum floor recommendation | `0.01` per class, then renormalise |
| Always submit all 5 seeds | Missing seed = score of 0 |
| Entropy weighting favours dynamic cells | Static cells (ocean, mountains) barely affect your score |
| Leaderboard uses best single round | Later rounds weighted higher (`1.05^round_number`) |