from quasi_symmetries.config import CACHE_DIR, IMAGES_DIR, OPT_RESULTS_DIR, TABLES_DIR
"""Plot quartet baseline diagnostics from summary CSV output."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


BASELINES = ("greedy", "ring", "balanced_tree")
BASELINE_LABELS = {
    "greedy": "Greedy",
    "ring": "Ring",
    "balanced_tree": "Balanced tree",
    "unoptimized_quartets": "Unoptimized quartets",
    "fixed_abc_quadratics": "Seniorities",
}
BASELINE_COLORS = {
    "greedy": "#0072B2",
    "ring": "#E69F00",
    "balanced_tree": "#CC79A7",
    "unoptimized_quartets": "#999999",
    "fixed_abc_quadratics": "#000000",
}
BASELINE_MARKERS = {
    "greedy": "o",
    "ring": "s",
    "balanced_tree": "^",
    "unoptimized_quartets": "D",
    "fixed_abc_quadratics": "x",
}
BASELINE_LINESTYLES = {
    "greedy": "-",
    "ring": "-",
    "balanced_tree": "-",
    "unoptimized_quartets": "--",
    "fixed_abc_quadratics": ":",
}


def _as_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return math.nan
    return float(value)


def read_summary_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _baseline_points(rows: list[dict[str, str]], baseline: str) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for row in rows:
        if row.get("Baseline") != baseline:
            continue
        e_fci = _as_float(row, "E_FCI")
        e_dec = _as_float(row, "Edec_Optimized")
        points.append(
            {
                "x": _as_float(row, "Geometry_Param"),
                "variance": _as_float(row, "V_Optimized"),
                "comm_sq": _as_float(row, "Sum_CommSq_Optimized"),
                "edec_error": abs(e_dec - e_fci),
                "k": _as_float(row, "Kcoupled_Optimized"),
            }
        )
    return sorted(points, key=lambda point: point["x"])


def _identity_quartet_points(rows: list[dict[str, str]], baseline: str = "greedy") -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for row in rows:
        if row.get("Baseline") != baseline:
            continue
        e_fci = _as_float(row, "E_FCI")
        e_dec = _as_float(row, "Edec_Identity")
        points.append(
            {
                "x": _as_float(row, "Geometry_Param"),
                "variance": _as_float(row, "V_Identity"),
                "comm_sq": _as_float(row, "Sum_CommSq_Identity"),
                "edec_error": abs(e_dec - e_fci),
                "k": _as_float(row, "Kcoupled_Identity"),
            }
        )
    return sorted(points, key=lambda point: point["x"])


def _quadratic_fixed_abc_points(rows: list[dict[str, str]]) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for row in rows:
        e_fci = _as_float(row, "E_FCI")
        e_dec = _as_float(row, "Edec_Optimized")
        points.append(
            {
                "x": _as_float(row, "Geometry_Param"),
                "variance": _as_float(row, "V_Optimized"),
                "comm_sq": _as_float(row, "Sum_CommSq_Optimized"),
                "edec_error": abs(e_dec - e_fci),
                "k": _as_float(row, "Kcoupled_Optimized"),
            }
        )
    return sorted(points, key=lambda point: point["x"])


def plot_quartet_baseline_diagnostics(
    csv_path: Path,
    output_path: Path,
    *,
    fixed_abc_csv_path: Path | None = None,
    title: str = "H4 linear quartet baseline diagnostics",
) -> None:
    rows = read_summary_rows(csv_path)
    if not rows:
        raise ValueError(f"No data rows found in {csv_path}")

    series = {
        "unoptimized_quartets": _identity_quartet_points(rows),
        **{baseline: _baseline_points(rows, baseline) for baseline in BASELINES},
    }
    if fixed_abc_csv_path is not None:
        fixed_abc_rows = read_summary_rows(fixed_abc_csv_path)
        if not fixed_abc_rows:
            raise ValueError(f"No data rows found in {fixed_abc_csv_path}")
        series["fixed_abc_quadratics"] = _quadratic_fixed_abc_points(fixed_abc_rows)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    panels = [
        ("variance", r"Variance diagnostic", True),
        ("comm_sq", r"Commutator score", True),
        ("edec_error", r"Decoupled energy error $|E_{\mathrm{dec}}-E_{\mathrm{FCI}}|$ (Ha)", True),
        ("k", r"Coupled-sector dimension $K$", False),
    ]

    for baseline, points in series.items():
        if not points:
            continue

        xs = [point["x"] for point in points]
        for ax, (field, ylabel, use_log) in zip(axes.ravel(), panels):
            ys = [point[field] for point in points]
            ax.plot(
                xs,
                ys,
                marker=BASELINE_MARKERS[baseline],
                color=BASELINE_COLORS[baseline],
                linestyle=BASELINE_LINESTYLES[baseline],
                linewidth=1.8,
                markersize=4.5,
                label=BASELINE_LABELS[baseline],
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
        "--csv",
        type=Path,
        default=OPT_RESULTS_DIR / 'h4_linear_quartet_baseline_summary.csv"),
        help="Quartet baseline summary CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMAGES_DIR / 'quartets/h4_linear_quartet_baseline_diagnostics.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--fixed-abc-csv",
        type=Path,
        default=TABLES_DIR / 'h4_linear_quasi_symmetry_fixed_abc.csv"),
        help="Quadratics fixed-abc summary CSV to overlay as a baseline.",
    )
    args = parser.parse_args()
    plot_quartet_baseline_diagnostics(args.csv, args.output, fixed_abc_csv_path=args.fixed_abc_csv)


if __name__ == "__main__":
    main()
