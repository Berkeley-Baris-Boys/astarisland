from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from astar_island.api import AstarIslandAPI
from astar_island.config import AstarConfig
from astar_island.utils import validate_prediction_tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit previously saved prediction tensors for the active or specified round.")
    parser.add_argument("run_dir", type=Path, help="Artifact directory containing prediction_seed_*.npy files.")
    parser.add_argument("--round-id", default=None, help="Optional round id. Defaults to the current active round.")
    parser.add_argument("--seed", type=int, action="append", default=None, help="Submit only specific seed index. Repeatable.")
    args = parser.parse_args()

    config = AstarConfig()
    api = AstarIslandAPI(config)
    round_id = args.round_id
    if round_id is None:
        active = api.get_active_round()
        if not active:
            raise RuntimeError("No active round found and no --round-id provided.")
        round_id = active["id"]

    seeds = args.seed if args.seed else list(range(5))
    for seed_index in seeds:
        path = args.run_dir / f"prediction_seed_{seed_index}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing prediction file: {path}")
        prediction = np.load(path)
        validate_prediction_tensor(prediction, config.predictor.min_probability)
        response = api.submit(round_id, seed_index, prediction.tolist())
        print(f"seed={seed_index} status={response.get('status', 'unknown')} round_id={response.get('round_id', round_id)}")


if __name__ == "__main__":
    main()
