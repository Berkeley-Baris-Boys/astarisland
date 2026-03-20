from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from astar_island.visualize import save_class_probability_maps, save_grid_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a saved prediction tensor.")
    parser.add_argument("prediction", type=Path)
    parser.add_argument("--output", type=Path, default=Path("inspection"))
    args = parser.parse_args()

    prediction = np.load(args.prediction)
    args.output.mkdir(parents=True, exist_ok=True)
    save_grid_image(np.argmax(prediction, axis=-1), args.output / "argmax.png", "Argmax classes")
    save_class_probability_maps(prediction, args.output, "prediction")


if __name__ == "__main__":
    main()
