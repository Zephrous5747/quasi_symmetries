"""Plot mixed seniority+quartet pool scan: summary graph and highlighted variance maps."""

from __future__ import annotations

from quasi_symmetries.config import heatmap_optimization_dir, scans_dir, table_path

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from scripts.plot.orbital_heatmaps import (
    SystemJob,
    _increasing_orbital_ticks,
    _reversed_orbital_ticks,
    _system_title,
    _variance_display_coords,
    _variance_for_display,
)


def _highlight_mixed_pool(
    ax: plt.Axes,
    singles: list[int],
    quartets: list[tuple[int, int]],
    n_spatial: int,
) -> None:
    for orbital in singles:
        row, col = _variance_display_coords(orbital, orbital, n_spatial)
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=1.4,
                zorder=10,
            )
        )
    for p, q in quartets:
        if p > q:
            p, q = q, p
        row, col = _variance_display_coords(p, q, n_spatial)
        ax.add_patch(
            Rectangle(
                (col - 0.5, row - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=1.4,
                zorder=10,
            )
        )


def plot_cost_scan(csv_path: Path, output_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: float(r["Geometry_Param"]))

    xs = [float(r["Geometry_Param"]) for r in rows]
    v_id = [float(r["V_Identity"]) for r in rows]
    v_opt = [float(r["V_Optimized"]) for r in rows]

    fig, ax = plt.subplots(figsize=(8.0, 4.5), constrained_layout=True)
    ax.plot(xs, v_id, marker="o", label="Canonical (identity rotation)")
    ax.plot(xs, v_opt, marker="s", label="Mixed pool optimized")
    ax.set_xlabel("H2O OH bond length (Å)")
    ax.set_ylabel(r"Pool variance $\sum 1-\langle s\rangle^2$")
    pool = rows[0]
    singles = [int(v) for v in pool["Pool_Singles"].split()]
    quartets = pool["Pool_Quartets"].split()
    op_label = ", ".join(f"$s_{i}$" for i in singles)
    op_label += ", " + ", ".join(
        f"$s_{{{p}{q}}}$" for p, q in (edge.split("-") for edge in quartets)
    )
    ax.set_title(f"Mixed operator pool: {op_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {output_path}")


def plot_geometry_variance(
    npz_path: Path,
    *,
    job: SystemJob,
    output_path: Path,
    vmin: float = 1e-4,
    vmax: float = 1.0,
) -> None:
    data = np.load(npz_path)
    singles = [int(v) for v in data["pool_singles"]]
    quartets = [tuple(int(v) for v in edge) for edge in data["pool_quartets"]]
    n_spatial = int(data["variance_canonical"].shape[0])

    fig = plt.figure(figsize=(9.5, 4.2), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig)
    x_ticks = _increasing_orbital_ticks(n_spatial)
    y_ticks = _reversed_orbital_ticks(n_spatial)
    im = None

    for col, (title, key) in enumerate(
        (("Canonical", "variance_canonical"), ("Mixed pool", "variance_mixed"))
    ):
        ax = fig.add_subplot(gs[0, col])
        display = _variance_for_display(np.asarray(data[key], dtype=float))
        display = np.ma.masked_less(display, vmin / 10)
        im = ax.imshow(
            display,
            aspect="equal",
            cmap="viridis",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            origin="lower",
        )
        ax.set_title(title)
        ax.set_xlabel("Orbital index")
        ax.set_ylabel("Orbital index")
        ax.set_xticks(range(n_spatial))
        ax.set_yticks(range(n_spatial))
        ax.set_xticklabels(x_ticks)
        ax.set_yticklabels(y_ticks)
        if key == "variance_mixed":
            _highlight_mixed_pool(ax, singles, quartets, n_spatial)

    cbar = fig.colorbar(im, ax=fig.axes, shrink=0.9)
    cbar.set_label(r"Parity variance $1-\langle s\rangle^2$")
    singles_txt = ", ".join(f"$s_{i}$" for i in singles)
    quartets_txt = ", ".join(f"$s_{{{p}{q}}}$" for p, q in quartets)
    fig.suptitle(
        f"Mixed pool ($s_0,s_1,s_2$ + quartets) | {_system_title(job)}\n"
        f"Cost terms: {singles_txt}, {quartets_txt} (boxed on optimized panel)"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=table_path("h2o", "mixed_pool_summary.csv"),
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=heatmap_optimization_dir("h2o"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=scans_dir("h2o"),
    )
    parser.add_argument("--vmin", type=float, default=1e-4)
    parser.add_argument("--vmax", type=float, default=1.0)
    args = parser.parse_args()

    plot_cost_scan(args.csv, args.output_dir / "mixed_pool_cost_scan.png")


if __name__ == "__main__":
    main()
