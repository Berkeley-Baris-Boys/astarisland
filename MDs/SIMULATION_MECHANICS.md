# Astar Island — Simulation Mechanics

## Overview

Astar Island is a Norse-themed world simulation played out on a procedurally generated grid. Over 50 years of simulated history, settlements rise and fall, factions raid and trade, winters claim the weak, and the natural world slowly reclaims what civilisation abandons. The simulation is deterministic from a map seed, making the initial terrain fully reconstructable.

---

## The World Grid

The world is a **rectangular grid** (default **40 × 40** cells). Every cell holds one of eight terrain types, which collapse into six prediction classes used by the simulation engine.


| Internal Code | Terrain    | Prediction Class | Description                            |
| ------------- | ---------- | ---------------- | -------------------------------------- |
| `10`          | Ocean      | `0` — Empty      | Impassable water; forms the map border |
| `11`          | Plains     | `0` — Empty      | Flat, buildable land                   |
| `0`           | Empty      | `0` — Empty      | Generic unoccupied cell                |
| `1`           | Settlement | `1` — Settlement | Active Norse settlement                |
| `2`           | Port       | `2` — Port       | Coastal settlement with a harbour      |
| `3`           | Ruin       | `3` — Ruin       | Collapsed settlement                   |
| `4`           | Forest     | `4` — Forest     | Provides food to adjacent settlements  |
| `5`           | Mountain   | `5` — Mountain   | Impassable terrain                     |


**Key rules:**

- Ocean, Plains, and Empty all share prediction class `0` — the engine treats them as interchangeable "nothing" cells.
- **Mountains** are fully static and never change.
- **Forests** are mostly static but can slowly reclaim ruined land.
- The cells of real interest are those capable of transitioning: Settlements, Ports, and Ruins.

---

## Map Generation

Every map is procedurally generated from a **map seed**, which is visible during play — allowing you to reconstruct the starting terrain independently.

Generation proceeds in layers:

1. **Ocean border** — The entire perimeter is filled with impassable ocean.
2. **Fjords** — Narrow inlets of ocean cut inland from random edges, shaping the coastline.
3. **Mountain chains** — Random walks trace mountain ridges across the interior.
4. **Forest patches** — Clustered groves of forest are scattered across available land.
5. **Initial settlements** — Starting settlements are placed on valid land cells, spaced apart to prevent early overcrowding.

---

## Simulation Lifecycle

The simulation runs for **50 years**. Each year progresses through five sequential phases:

```
Growth → Conflict → Trade → Winter → Environment
```

### 1. Growth

Settlements produce food based on the terrain in adjacent cells — forests are the primary food source. When a settlement accumulates enough food and population:

- **Population grows** organically over time.
- **Ports develop** on coastal settlements, enabling sea trade and naval operations.
- **Longships are built** at prosperous ports, dramatically extending raiding and trade range.
- **Expansion occurs** — thriving settlements may found new daughter settlements on nearby empty land cells.

### 2. Conflict

Settlements raid their neighbours. The mechanics of raiding are shaped by desperation and capability:

- **Aggression scales with hunger** — low-food settlements raid more frequently and more recklessly.
- **Longships extend raiding range**, allowing coastal and island settlements to strike far-off targets.
- **Successful raids** loot resources (food and wealth) and inflict defensive damage on the target.
- **Conquest** is possible — a defeated settlement may flip allegiance and join the raiding faction's `owner_id`.

### 3. Trade

Settlements with ports can trade if they are within range of another port and not currently at war with it.

- **Both parties gain** food and wealth from a successful trade.
- **Technology diffuses** between trading partners, gradually raising the tech level of less-developed settlements.
- Trade is a key path to peaceful prosperity — ports that avoid conflict can grow wealthy and technologically advanced.

### 4. Winter

Each year ends with a winter of variable severity.

- **All settlements lose food** — the harsher the winter, the greater the loss.
- Settlements already weakened by raids or poor harvests are most vulnerable.
- **Collapse** occurs when a settlement is pushed to zero by some combination of starvation, sustained raiding, and winter severity — it becomes a **Ruin**, and its surviving population disperses to nearby friendly settlements.

### 5. Environment

The natural world responds to the state of civilisation:

- **Reclamation by nature** — Ruins left unattended are eventually swallowed by forest growth or revert to open plains.
- **Reclamation by settlements** — A nearby thriving settlement may rebuild a ruin, establishing a new outpost that inherits a portion of the patron's resources and tech level.
- **Port restoration** — Coastal ruins can be rebuilt as ports rather than plain settlements.
- If no settlement claims a ruin and no forest spreads to it, it fades back to empty plains.

---

## Settlement Properties

Each settlement tracks a rich internal state:


| Property             | Visibility     | Description                                   |
| -------------------- | -------------- | --------------------------------------------- |
| Position             | Always visible | Grid coordinates `(x, y)`                     |
| Port status          | Always visible | Whether the settlement has a harbour          |
| Population           | Query only     | Number of inhabitants                         |
| Food                 | Query only     | Current food stockpile                        |
| Wealth               | Query only     | Accumulated riches from trade and raids       |
| Defense              | Query only     | Resistance to raiding damage                  |
| Tech level           | Query only     | Accumulated knowledge, boosted by trade       |
| Longships owned      | Query only     | Number of vessels available for raiding/trade |
| Faction (`owner_id`) | Query only     | Which faction controls the settlement         |


> **Note:** Initial state reveals position and port status only. Internal statistics — population, food, wealth, defense — are hidden behind simulation queries.

---

## Terrain & Settlement Interactions

The terrain surrounding a settlement directly shapes its fate:

- 🌲 **Adjacent forests** boost food production, supporting larger populations and faster growth.
- 🏔️ **Mountains** block expansion routes and can isolate settlements, but also provide natural defensive chokepoints.
- 🌊 **Coastline access** enables port development, longship construction, and sea trade.
- 🏚️ **Nearby ruins** represent both a threat (signs of past collapse) and an opportunity (land ripe for reclamation).

---

## The Long Arc

Over 50 years, the simulation tends to produce recognisable historical patterns:

- **Early game:** Settlements spread across available land; the map fills with small, isolated outposts.
- **Mid game:** Factions consolidate; raiding intensifies as food competition increases; ports begin linking into trade networks.
- **Late game:** Strong factions dominate large territories; weak or isolated settlements collapse into ruins; nature reclaims the edges.

With high expansion rates, settlements can colonise nearly all available land within a few decades — but dense settlement increases competition and raises the stakes of each winter.

---

