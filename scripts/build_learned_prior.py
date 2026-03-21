from __future__ import annotations

import argparse
from pathlib import Path

from astar_island.config import AstarConfig
from astar_island.learned_prior import build_learned_prior_artifact_from_archive, save_learned_prior_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a learned hierarchical prior from archived completed rounds.")
    parser.add_argument("--history-dir", type=Path, default=None, help="Archive directory. Defaults to ASTAR_ISLAND_HISTORY_DIR or config.")
    parser.add_argument("--output", type=Path, default=None, help="Output path. Defaults to ASTAR_ISLAND_LEARNED_PRIOR_PATH or config.")
    parser.add_argument("--holdout-round", type=int, default=None, help="Optional round number to exclude from training.")
    parser.add_argument("--l2-active", type=float, default=0.6)
    parser.add_argument("--l2-forest", type=float, default=0.6)
    parser.add_argument("--l2-active-type", type=float, default=0.8)
    parser.add_argument("--maxiter", type=int, default=300)
    args = parser.parse_args()

    config = AstarConfig()
    history_dir = config.history_dir if args.history_dir is None else args.history_dir
    output_path = config.predictor.learned_prior_path if args.output is None else args.output

    artifact = build_learned_prior_artifact_from_archive(
        history_dir,
        holdout_round_number=args.holdout_round,
        l2_active=args.l2_active,
        l2_forest=args.l2_forest,
        l2_active_type=args.l2_active_type,
        maxiter=args.maxiter,
    )
    save_learned_prior_artifact(output_path, artifact)
    print(output_path)


if __name__ == "__main__":
    main()
