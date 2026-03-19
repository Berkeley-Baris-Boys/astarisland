# Project: Astar Island Solver

## 0. Objective
Build a script that queries the Astar Island simulator within budget, predicts per-cell terrain class probabilities for each seed, and submits predictions to the challenge API.

---

## 1. Current Status (SOURCE OF TRUTH)

### ✅ Completed
- [x] End-to-end CLI solver in `solver.py`
- [x] API integration (round load, simulate, submit) with retry/error handling
- [x] Query planning + adaptive reserve queries
- [x] Prediction pipeline (empirical counts + heuristics + smoothing)
- [x] Save/resume flow (`observations.json`, `predictions_seed_*.npy`)
- [x] Submission validation and summary reporting

### 🔄 In Progress
- [ ] Prediction quality tuning (heuristic weights and reserve strategy)
- [ ] Better evaluation workflow across rounds/seeds

### ⏳ Next Up (priority ordered)


---

## 2. System Architecture

### Components
- API Client Layer (`request_json`, `create_session`)
- Query Engine (`plan_queries`, `run_queries`, `choose_reserve_query`)
- Prediction Engine (`build_predictions`, transition/prior helpers)
- Submission Layer (`validate_prediction_tensor`, `submit_all`)
- CLI Orchestrator (`main`)

### Flow
Auth → Load active round → Plan/Run queries (or resume) → Build predictions → Submit per seed → Print summary

---

## 3. Tasks (Atomic / AI-Executable)


---

## 4. Decisions Log (VERY IMPORTANT)

### Decision 1
- **What:** Use one standalone Python solver (no service backend)
- **Why:** Fast iteration and simple operation during challenge rounds
- **Date:** 2026-03-19

### Decision 2
- **What:** Use hybrid probabilistic heuristics instead of a trained model
- **Why:** No training pipeline required; works directly from queried evidence + priors
- **Date:** 2026-03-19

---

## 5. Known Issues / Bugs


---

## 6. Experiments / Learnings



---

## 7. How to Run

```bash
pip install requests numpy scipy
export ASTAR_ISLAND_TOKEN="<your_token>"
python solver.py --dry-run
python solver.py