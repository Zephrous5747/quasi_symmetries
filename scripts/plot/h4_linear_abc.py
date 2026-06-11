from quasi_symmetries.config import CACHE_DIR, IMAGES_DIR, OPT_RESULTS_DIR, TABLES_DIR
"""Plot H4 linear diagnostics comparing canonical, fixed-abc, and shared-abc runs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


SERIES_STYLES = {
    "initial": {
        "label": "Initial (canonical orbitals)",
        "color": "tab:gray",
        "marker": "o",
        "linestyle": "--",
    },
    "fixed": {
        "label": "Fixed-abc optimization",
        "color": "tab:blue",
        "marker": "s",
        "linestyle": "-",
    },
    "shared": {
        "label": "Shared-abc optimization",
        "color": "tab:orange",
        "marker": "^",
        "linestyle": "-",
    },
}


def _as_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return math.nan
    return float(value)


def read_summary_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _points(rows: list[dict[str, str]], *, optimized: bool) -> list[dict[str, float]]:
    suffix = "Optimized" if optimized else "Identity"
    points: list[dict[str, float]] = []
    for row in rows:
        e_fci = _as_float(row, "E_FCI")
        e_dec = _as_float(row, f"Edec_{suffix}")
        points.append(
            {
                "x": _as_float(row, "Geometry_Param"),
                "variance": _as_float(row, f"V_{suffix}"),
                "comm_sq": _as_float(row, f"Sum_CommSq_{suffix}"),
                "edec_error": abs(e_dec - e_fci),
                "k": _as_float(row, f"Kcoupled_{suffix}"),
            }
        )
    return sorted(points, key=lambda point: point["x"])


def plot_h4_linear_abc_diagnostics(
    fixed_csv: Path,
    shared_csv: Path,
    output_path: Path,
    *,
    title: str = "H4 linear fixed/shared abc diagnostics",
) -> None:
    fixed_rows = read_summary_rows(fixed_csv)
    shared_rows = read_summary_rows(shared_csv)
    if not fixed_rows:
        raise ValueError(f"No data rows found in {fixed_csv}")
    if not shared_rows:
        raise ValueError(f"No data rows found in {shared_csv}")

    series = {
        "initial": _points(fixed_rows, optimized=False),
        "fixed": _points(fixed_rows, optimized=True),
        "shared": _points(shared_rows, optimized=True),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    panels = [
        ("variance", r"Variance $\sum_{(p,q)} V_{pq}$", True),
        ("comm_sq", r"Commutator score $\sum_{(p,q)} \|[H,s_{pq}]\psi\|^2$", True),
        ("edec_error", r"Decoupled energy error $|E_{\mathrm{dec}}-E_{\mathrm{FCI}}|$ (Ha)", True),
        ("k", r"Coupled-sector dimension $K$", False),
    ]

    for series_name, points in series.items():
        style = SERIES_STYLES[series_name]
        xs = [point["x"] for point in points]
        for ax, (field, ylabel, use_log) in zip(axes.ravel(), panels):
            ys = [point[field] for point in points]
            ax.plot(
                xs,
                ys,
                marker=style["marker"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=4.5,
                label=style["label"],
            )
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            if use_log:
                positive = [y for y in ys if y > 0 and math.isfinite(y)]
                if positive:
                    ax.set_yscale("log")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"H--H spacing $R$ ($\AA$)")

    axes[0, 0].legend(loc="best", frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed-csv",
        type=Path,
        default=TABLES_DIR / 'h4_linear_quasi_symmetry_fixed_abc.csv"),
        help="Fixed-abc H4 linear summary CSV.",
    )
    parser.add_argument(
        "--shared-csv",
        type=Path,
        default=TABLES_DIR / 'h4_linear_quasi_symmetry_shared_abc.csv"),
        help="Shared-abc H4 linear summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMAGES_DIR / 'quartets/h4_linear_abc_diagnostics.png"),
        help="Output figure path.",
    )
    args = parser.parse_args()
    plot_h4_linear_abc_diagnostics(args.fixed_csv, args.shared_csv, args.output)


if __name__ == "__main__":
    main()
