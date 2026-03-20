from __future__ import annotations

import argparse

from astar_island.api import AstarIslandAPI
from astar_island.config import AstarConfig
from astar_island.history import archive_completed_rounds


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive completed rounds with initial grids, ground truths, and raw analysis payloads.")
    parser.add_argument("--max-rounds", type=int, default=9, help="Maximum number of completed rounds to archive.")
    args = parser.parse_args()

    config = AstarConfig()
    api = AstarIslandAPI(config)
    manifest = archive_completed_rounds(api, config.history_dir, max_rounds=args.max_rounds)
    print(config.history_dir)
    print(manifest)


if __name__ == "__main__":
    main()
