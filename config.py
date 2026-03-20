"""
Configuration and constants for the Astar Island pipeline.

Terrain code system:
  Raw codes (from API): 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest,
                        5=Mountain, 10=Ocean, 11=Plains
  Prediction classes:   0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest,
                        5=Mountain
  Critical: codes 10 (Ocean) and 11 (Plains) both map to class 0.
"""
from __future__ import annotations

# ── API ───────────────────────────────────────────────────────────────────────
BASE_URL = "https://api.ainm.no"
N_CLASSES = 6
MAX_VIEWPORT = 15
TOTAL_BUDGET = 50
N_SEEDS = 5
DEFAULT_MAP_SIZE = 40
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

# ── Terrain mapping ───────────────────────────────────────────────────────────
CODE_TO_CLASS: dict[int, int] = {
    0: 0,    # Empty → Empty
    1: 1,    # Settlement → Settlement
    2: 2,    # Port → Port
    3: 3,    # Ruin → Ruin
    4: 4,    # Forest → Forest
    5: 5,    # Mountain → Mountain (static, never changes)
    10: 0,   # Ocean → Empty (static, counts as class 0)
    11: 0,   # Plains → Empty (dynamic, buildable land)
}

OCEAN_CODE = 10
MOUNTAIN_CODE = 5
PLAINS_CODE = 11
SETTLEMENT_CODE = 1
PORT_CODE = 2
RUIN_CODE = 3
FOREST_CODE = 4

# Codes that will never change class (safe to predict with high confidence)
STATIC_CODES = {OCEAN_CODE, MOUNTAIN_CODE}

# Codes representing "interesting" cells that score well
DYNAMIC_CODES = {0, 1, 2, 3, 4, 11}

# ── File paths ────────────────────────────────────────────────────────────────
OBSERVATIONS_FILE    = "observations.json"
INITIAL_STATES_FILE  = "initial_states.json"
SIM_RESULTS_FILE     = "sim_results.npz"
METRICS_FILE         = "metrics.json"
PRED_TEMPLATE        = "predictions_seed_{seed}.npy"

# ── Prediction safety ─────────────────────────────────────────────────────────
# Never let any class probability reach 0 — KL divergence becomes infinite.
# Keep this floor extremely small to avoid distorting near one-hot cells.
PROB_FLOOR = 1e-12

# ── Default hidden parameters (prior belief, updated per round) ───────────────
# Calibrated from rounds 2 & 3 observations:
#   R3 inferred: faction_aggression=0.348, forest_growth_rate=0.0,
#                winter_severity=1.0, trade_activity=0.05
DEFAULT_HIDDEN_PARAMS: dict[str, float] = {
    "faction_aggression": 0.35,
    "forest_growth_rate": 0.05,
    "winter_severity":    0.55,
    "trade_activity":     0.10,
}

# ── Simulator mechanics (see docs: Simulation Mechanics) ─────────────────────
# Initial settlement stats (hidden; defaults calibrated to plausible values)
SIM_INIT_POP     = 2.0
SIM_INIT_FOOD    = 2.5
SIM_INIT_WEALTH  = 0.8
SIM_INIT_DEFENSE = 1.2
SIM_INIT_TECH    = 1.0

# Growth phase
FOOD_FROM_FOREST      = 0.25   # food per adjacent forest tile per year
BASE_FOOD_PRODUCTION  = 0.12   # baseline food production per settlement
FOOD_CONSUMPTION_RATE = 0.08   # food consumed per population unit per year
GROWTH_FOOD_RATIO     = 1.3    # food/pop ratio needed to trigger growth
GROWTH_RATE           = 0.12   # population multiplier on growth
MAX_POPULATION        = 12.0   # hard cap on population

# Port development
PORT_MIN_POP    = 2.5
PORT_MIN_FOOD   = 1.5
PORT_BASE_PROB  = 0.12   # per year probability when conditions met

# Longship
LONGSHIP_MIN_WEALTH = 2.5
LONGSHIP_BASE_PROB  = 0.14

# Expansion (founding new settlements)
EXPAND_MIN_POP          = 4.0
EXPAND_MIN_FOOD         = 2.0
EXPAND_BASE_PROB        = 0.18
EXPAND_SEARCH_RANGE     = 4    # cells
EXPAND_POP_TRANSFER     = 0.28
EXPAND_FOOD_TRANSFER    = 0.28
EXPAND_WEALTH_TRANSFER  = 0.10
NEW_SETTLE_DEFENSE      = 0.8
NEW_SETTLE_TECH_FRAC    = 0.90  # tech_level = patron_tech * fraction

# Conflict phase
BASE_RAID_RANGE       = 4
LONGSHIP_RAID_RANGE   = 14
BASE_RAID_PROB        = 0.15   # base probability (× faction_aggression)
STARVATION_THRESHOLD  = 0.6   # food < this × pop → desperate raiding
STARVATION_BOOST      = 0.30
LOOT_WEALTH_RATE      = 0.25
LOOT_FOOD_RATE        = 0.12
RAID_DEFENSE_DAMAGE   = 0.28
ALLEGIANCE_PROB       = 0.25   # prob that a conquered settlement changes owner

# Trade phase
TRADE_RANGE       = 10
TRADE_WEALTH_GAIN = 0.12   # multiplied by trade_activity
TRADE_FOOD_GAIN   = 0.04
TECH_DIFFUSION    = 0.07

# Winter phase
BASE_WINTER_FOOD_COST = 0.35   # per population unit, × winter_severity
STARVATION_FOOD_MIN   = 0.0    # food below this → collapse

# Environment phase
RECLAIM_SEARCH_RANGE   = 4
RECLAIM_BASE_PROB      = 0.10
RECLAIM_MIN_PATRON_POP = 2.0
RECLAIM_POP_FRAC       = 0.25
RECLAIM_FOOD_FRAC      = 0.25
RECLAIM_WEALTH_FRAC    = 0.10
FOREST_RUIN_BASE_PROB  = 0.09   # × forest_growth_rate
FOREST_SPREAD_BASE_PROB= 0.03   # × forest_growth_rate (grow to empty land)
RUIN_PLAINS_PROB       = 0.04   # ruins slowly fade to empty plains

# ── Monte Carlo defaults ──────────────────────────────────────────────────────
DEFAULT_N_SIMS = 100   # simulations per seed; increase for accuracy, costs time
MIN_N_SIMS     = 30    # minimum when time is short
