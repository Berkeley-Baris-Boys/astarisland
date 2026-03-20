# Astar Island — Task Overview

## What is Astar Island?

Astar Island is a machine learning challenge where you observe a stochastic simulation of a Norse civilisation and predict the final state of the world.

The simulator evolves a procedurally generated map over 50 years. Settlements grow, expand, trade, raid each other, collapse, and interact with the environment.

Your goal is to predict, for every cell in the map, the probability distribution of terrain types after the simulation completes.

---

## Core Task

You must output a probability distribution for each cell:

- Map size: typically 40 × 40
- Output: H × W × 6 tensor
- Each cell contains probabilities for 6 classes

Classes:
- 0: Empty (Ocean, Plains, Empty)
- 1: Settlement
- 2: Port
- 3: Ruin
- 4: Forest
- 5: Mountain

Each cell’s probabilities must sum to 1.

---

## Simulation Setup

Each round consists of:

- A fixed map (same terrain layout)
- 5 different seeds (initial settlement configurations)
- Hidden parameters controlling simulation dynamics

The simulation runs for 50 years and is stochastic:
- Same initial state can lead to different outcomes
- Randomness affects growth, conflict, trade, and collapse

---

## Observations

You cannot see the full simulation directly.

Instead, you can query the simulator:
- Each query returns a small rectangular viewport of the final state
- Viewport size: between 5×5 and 15×15
- Each query runs a new stochastic simulation

Constraints:
- Maximum 50 queries per round
- Queries are shared across all 5 seeds

---

## What You See

From each query, you observe:

- A portion of the final grid
- Settlements within the viewport, including internal properties such as:
  - population
  - food
  - wealth
  - defense
  - port status
  - alive status

Each query corresponds to a different stochastic outcome.

---

## Ground Truth

The true target is not a single outcome.

Instead, for each cell:
- The organizers simulate the world many times
- This produces a probability distribution over terrain types

Example:
[0.0, 0.60, 0.25, 0.15, 0.0, 0.0]

This means:
- 60% Settlement
- 25% Port
- 15% Ruin

---

## Scoring

Predictions are evaluated using entropy-weighted KL divergence.

### KL Divergence

Measures how different your prediction is from the true distribution:

KL(p || q) = Σ pᵢ log(pᵢ / qᵢ)

Where:
- p = ground truth distribution
- q = your prediction

Lower is better.

---

### Entropy Weighting

Not all cells matter equally.

- Static cells (e.g., mountains, ocean) have near-zero entropy → ignored
- Dynamic cells (uncertain outcomes) have high entropy → weighted more

Entropy of a cell:
entropy = -Σ pᵢ log(pᵢ)

---

### Final Score

weighted_kl = Σ entropy × KL / Σ entropy

score = 100 × exp(-3 × weighted_kl)

- 100 = perfect prediction
- 0 = very poor prediction

---

## Simulation Mechanics

Each year of the simulation consists of:

1. Growth  
   Settlements grow based on surrounding terrain and resources

2. Conflict  
   Settlements raid each other and may take over or damage others

3. Trade  
   Ports trade resources, increasing wealth and stability

4. Winter  
   Harsh conditions reduce food and can cause collapse

5. Environment  
   Ruins may be reclaimed or overgrown by forest

---

## Terrain Dynamics

- Ocean and mountains are static
- Forest is mostly stable but can reclaim land
- Settlements, ports, and ruins are dynamic

The most important cells are those that can change between:
- Settlement
- Port
- Ruin

---

## Challenge Summary

You must:

- Use limited partial observations
- Infer hidden simulation dynamics
- Predict full-map probability distributions
- Handle stochastic outcomes

This is a problem of:
- partial observability
- stochastic simulation modeling
- probabilistic prediction
