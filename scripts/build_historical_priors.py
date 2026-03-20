from __future__ import annotations

import argparse
from pathlib import Path

from astar_island.api import AstarIslandAPI
from astar_island.config import AstarConfig
from astar_island.history import archive_completed_rounds
from astar_island.priors import build_historical_prior_artifact, save_historical_prior_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical feature-conditioned priors from completed rounds.")
    parser.add_argument("--max-rounds", type=int, default=9, help="Maximum number of completed rounds to ingest.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for the learned prior artifact. Defaults to ASTAR_ISLAND_HISTORICAL_PRIOR_PATH or config default.",
    )
    parser.add_argument("--refresh-archive", action="store_true", help="Fetch and persist completed-round analysis before building priors.")
    args = parser.parse_args()

    config = AstarConfig()
    api = AstarIslandAPI(config)
    if args.refresh_archive:
        archive_completed_rounds(api, config.history_dir, max_rounds=args.max_rounds)
    artifact = build_historical_prior_artifact(api, max_rounds=args.max_rounds)
    output_path = config.predictor.historical_prior_path if args.output is None else Path(args.output)
    save_historical_prior_artifact(output_path, artifact)
    print(output_path)


if __name__ == "__main__":
    main()
