from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from astar_island.api import AstarIslandAPI
from astar_island.config import AstarConfig
from astar_island.types import terrain_grid_to_class_grid
from astar_island.visualize import save_class_probability_maps, save_grid_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch post-round analysis and dump useful artifacts.")
    parser.add_argument("round_id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("analysis"))
    args = parser.parse_args()

    api = AstarIslandAPI(AstarConfig())
    detail = api.get_round_details(args.round_id, use_cache=True)
    payload = api.get_analysis(args.round_id, args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / f"analysis_seed_{args.seed}.npy", np.asarray(payload))
    save_grid_image(
        terrain_grid_to_class_grid(detail.initial_states[args.seed].grid),
        args.output / f"initial_seed_{args.seed}.png",
        "Initial terrain",
    )
    if isinstance(payload, dict) and "ground_truth" in payload:
        save_class_probability_maps(np.asarray(payload["ground_truth"], dtype=float), args.output, f"ground_truth_seed_{args.seed}")


if __name__ == "__main__":
    main()
