from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .types import CLASS_NAMES


TERRAIN_COLORS = {
    0: "#d9ccb8",
    1: "#c26d3a",
    2: "#3b82f6",
    3: "#5b4a42",
    4: "#2e7d32",
    5: "#707070",
}


def save_grid_image(grid: np.ndarray, path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = plt.matplotlib.colors.ListedColormap([TERRAIN_COLORS[i] for i in range(6)])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=cmap, interpolation="nearest", vmin=0, vmax=5)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_heatmap(values: np.ndarray, path: Path, title: str, cmap: str = "viridis") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(values, cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_class_probability_maps(prediction: np.ndarray, output_dir: Path, prefix: str) -> None:
    for class_index, class_name in enumerate(CLASS_NAMES):
        save_heatmap(
            prediction[..., class_index],
            output_dir / f"{prefix}_{class_name}.png",
            title=f"{prefix} {class_name}",
            cmap="magma",
        )
