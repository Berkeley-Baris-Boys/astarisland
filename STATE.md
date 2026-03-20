# Project: Astar Island Solver

## 0. Objective
Build and run a round-time pipeline that:
1) spends the 50-query budget effectively,
2) predicts `H x W x 6` class probabilities per seed,
3) submits robustly to the challenge API.

---

## 1. Current Status (SOURCE OF TRUTH)

### ✅ Current architecture in production path
- [x] Main entrypoint is `main.py` (not `solver.py`).
- [x] Query pipeline: full-tile coverage first, then entropy-driven reserve queries.
- [x] `ObservationStore` persists raw latest grid values, per-cell class counts, and settlement snapshots.
- [x] Prediction pipeline defaults to **empirical constrained model** in `predictor.py`.
- [x] Round-level dynamics are inferred from pooled seed observations (`world_dynamics.py`).
- [x] Save/resume and submit flows are operational (`observations.json`, `predictions_seed_*.npy`).

### 🔄 Model behavior right now
- [x] Primary predictor = empirical counts + context transition priors + dynamics-adjusted geographic priors + semantic smoothing + hard constraints.
- [x] Hybrid simulator rollouts (`testing/simulator.py` Monte Carlo) are available but used only as fallback path if empirical path fails.
- [x] Heuristic-only predictor remains as last-resort fallback.
- [x] Full-coverage iterative denoiser settlement boost is neutralized (`EMPIRICAL_FULL_COVERAGE_ITER_SETTLE_BOOST = 1.0`) to reduce settlement overprediction.
- [x] Settlement intensity prior is active (`SETTLEMENT_INTENSITY_BLEND_ALPHA = 0.22`, `SETTLEMENT_INTENSITY_SIGMA = 2.2`) to improve spatial coherence.
- [x] Final hard constraints enforce terrain feasibility: mountains are exact one-hot, and port probability is zero on non-coastal/mountain cells.

### ⚠️ Important caveat
- [ ] Full year-by-year mechanics (Growth -> Conflict -> Trade -> Winter -> Environment) are not the default prediction path.
- [ ] Those mechanics are explicitly implemented in `testing/simulator.py`, but mainly power testing/fallback workflows.

---

## 2. System Architecture

### Core modules
- `main.py`: CLI orchestration, resume/check/submit/fetch-analysis modes.
- `api.py`: round loading, simulate calls, submission.
- `state.py`: observation ingestion, feature helpers, query planning, reserve viewport selection.
- `world_dynamics.py`: pooled estimation of survival/ruin/expansion/forest/fragmentation signals.
- `predictor.py`: empirical constrained predictor + hybrid/heuristic fallbacks.
- `testing/simulator.py`: explicit 5-phase simulator + Monte Carlo + parameter estimation.
- `testing/test_main_local.py`: local end-to-end test against simulator-generated GT.
- `testing/backtest_all_rounds.py`: backtesting on stored `ig`/`gt` datasets.

### Runtime flow
Auth -> load active round -> query phase (or resume/no-query) ->
estimate world dynamics -> build predictions -> optional self-check ->
submit per seed.

---

## 3. Mechanics Coverage vs `SIMULATION_MECHANICS.md`

### Incorporated strongly
- Terrain code/class mapping and static constraints (`ocean`, `mountain`).
- Geography signals tied to mechanics: coast proximity, near-settlement, forest adjacency, settlement density.
- Cross-seed shared-hidden-parameter idea via pooled `WorldDynamics`.
- Partial use of settlement snapshot signals (alive/dead, port status, low-food risk hints).

### Incorporated as proxies (not explicit phase simulation in primary path)
- Expansion pressure and hostile-world effects.
- Forest growth/reclamation tendency.
- Port development likelihood through coastal/near-settlement priors.

### Weak or mostly absent in default predictor
- Explicit trade network dynamics and tech diffusion.
- Explicit longship-driven raid range effects.
- Explicit per-year conflict targeting/conquest dynamics.
- Direct use of many hidden settlement stats as state variables over time.

---

## 4. Decisions Log (VERY IMPORTANT)

### Decision 1 (2026-03-19)
- **What:** Keep a standalone Python round runner (no service backend).
- **Why:** Fast iteration and reliable competition-time execution.

### Decision 2 (2026-03-19)
- **What:** Prioritize an empirical-constrained probabilistic predictor.
- **Why:** Robust under sparse observations and cheap at round time.

### Decision 3 (2026-03-20)
- **What:** Treat mechanics simulator as calibration/fallback/testing component, not default inference engine.
- **Why:** Empirical path has been more stable and easier to tune under strict query budget.

### Decision 4 (2026-03-20)
- **What:** Push impossible-state handling into strict post-processing constraints (mountain/port feasibility) and keep settlement shaping as a soft prior.
- **Why:** Hard constraints provide reliable calibration gains, while soft spatial priors improve map structure without making brittle assumptions.

---

## 5. Known Gaps / Risks

- `STATE.md` previously drifted from code reality; now corrected.
- Simulator constants in `config.py` and defaults in `testing/simulator.py` are not fully unified.
- Mechanics fidelity in default predictor is proxy-heavy; risk of mismatch on rounds with unusual faction/trade regimes.
- Current smoothing stack can over-flatten rare classes if not carefully calibrated.

---

## 6. Current Priorities

1. Improve mechanics fidelity without losing robustness:
   - inject more explicit conflict/trade/longship signals into primary path.
2. Calibrate transition priors with offline historical/video-derived trajectories.
3. Strengthen evaluation discipline:
   - systematic backtests (`testing/backtest_all_rounds.py`) and local smoke regression before round runs.
4. Keep state/docs synchronized with implementation details:
   - especially predictor constants, hard constraints, and fallback routing.

---

## 7. How to Run

```bash
python3 -m pip install -r requirements.txt
export ASTAR_TOKEN="<your_token>"

# Full round pipeline
python3 main.py

# Resume from observations.json
python3 main.py --resume

# Build and validate predictions without submit
python3 main.py --check-only

# Local pipeline test against simulator GT
python3 -m testing.test_main_local --budget 50 --gt-runs 100

# Backtest against stored previous rounds
python3 -m testing.backtest_all_rounds --rounds 2 3 --budget 50
```