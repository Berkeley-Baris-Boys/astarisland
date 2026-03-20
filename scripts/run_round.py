from __future__ import annotations

import argparse

from astar_island.config import AstarConfig
from astar_island.submit import run_active_round


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Astar Island round pipeline.")
    parser.add_argument("--no-submit", action="store_true", help="Build predictions without submitting.")
    parser.add_argument("--no-plots", action="store_true", help="Skip visualization output.")
    args = parser.parse_args()
    config = AstarConfig()
    run_dir = run_active_round(config, submit=not args.no_submit, make_plots=not args.no_plots)
    print(run_dir)


if __name__ == "__main__":
    main()
