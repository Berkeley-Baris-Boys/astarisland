from __future__ import annotations

import argparse
from pathlib import Path

from astar_island.config import AstarConfig
from astar_island.residual_calibrator import build_residual_calibrator_artifact_from_archive, save_residual_calibrator_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a residual calibrator from archived prediction/ground-truth pairs.")
    parser.add_argument("--history-dir", type=Path, default=None, help="Archive directory. Defaults to ASTAR_ISLAND_HISTORY_DIR or config.")
    parser.add_argument("--output", type=Path, default=None, help="Output path. Defaults to ASTAR_ISLAND_RESIDUAL_CALIBRATOR_PATH or config.")
    parser.add_argument("--holdout-round", type=int, default=None, help="Optional round number to exclude from training.")
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--min-samples-leaf", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2-regularization", type=float, default=0.2)
    args = parser.parse_args()

    config = AstarConfig()
    history_dir = config.history_dir if args.history_dir is None else args.history_dir
    output_path = config.predictor.residual_calibrator_path if args.output is None else args.output

    artifact = build_residual_calibrator_artifact_from_archive(
        history_dir,
        holdout_round_number=args.holdout_round,
        max_iter=args.max_iter,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        learning_rate=args.learning_rate,
        l2_regularization=args.l2_regularization,
    )
    save_residual_calibrator_artifact(output_path, artifact)
    print(output_path)


if __name__ == "__main__":
    main()
