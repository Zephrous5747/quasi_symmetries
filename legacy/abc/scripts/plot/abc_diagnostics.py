"""Plot fixed/shared/local abc diagnostics and commutator correlations."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from quasi_symmetries.config import IMAGES_DIR, LEGACY_ABC_TABLES_DIR

SYSTEMS = {
    "h4_linear": {
        "label": "H4 linear",
        "x_label": r"H--H spacing $R$ ($\AA$)",
        "fixed_csv": LEGACY_ABC_TABLES_DIR / "h4_linear_quasi_symmetry_fixed_abc.csv",
        "shared_csv": LEGACY_ABC_TABLES_DIR / "h4_linear_quasi_symmetry_shared_abc.csv",
        "local_csv": LEGACY_ABC_TABLES_DIR / "h4_linear_quasi_symmetry_local_abc.csv",
        "output": IMAGES_DIR / "quadratics/h4_linear_abc_diagnostics.png",
        "scatter_output": IMAGES_DIR / "quadratics/h4_linear_commutator_scatter.png",
    },
    "lih": {
        "label": "LiH",
        "x_label": r"Bond length $R$ ($\AA$)",
        "fixed_csv": LEGACY_ABC_TABLES_DIR / "lih_quasi_symmetry_fixed_abc.csv",
        "shared_csv": LEGACY_ABC_TABLES_DIR / "lih_quasi_symmetry_shared_abc.csv",
        "local_csv": LEGACY_ABC_TABLES_DIR / "lih_quasi_symmetry_local_abc.csv",
        "output": IMAGES_DIR / "quadratics/lih_abc_diagnostics.png",
        "scatter_output": IMAGES_DIR / "quadratics/lih_commutator_scatter.png",
    },
    "h2o": {
        "label": "H2O",
        "x_label": r"O--H bond length scale",
        "fixed_csv": LEGACY_ABC_TABLES_DIR / "h2o_quasi_symmetry_fixed_abc.csv",
        "shared_csv": LEGACY_ABC_TABLES_DIR / "h2o_quasi_symmetry_shared_abc.csv",
        "local_csv": LEGACY_ABC_TABLES_DIR / "h2o_quasi_symmetry_local_abc.csv",
        "output": IMAGES_DIR / "quadratics/h2o_abc_diagnostics.png",
        "scatter_output": IMAGES_DIR / "quadratics/h2o_commutator_scatter.png",
    },
}

CASE_STYLES = {
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
    "local": {
        "label": "Local-abc optimization",
        "color": "tab:green",
        "marker": "D",
        "linestyle": "-.",
    },
}

SYSTEM_MARKERS = {
    "h4_linear": "o",
    "lih": "s",
    "h2o": "^",
}

SCATTER_PANELS = [
    ("k", r"Coupled-sector dimension $K$", False),
    ("edec_error", r"Decoupled energy error $|E_{\mathrm{dec}}-E_{\mathrm{FCI}}|$ (Ha)", True),
    ("ebo_error", r"Energy diagnostic error $|E_{\mathrm{BO}}-E_{\mathrm{FCI}}|$ (Ha)", True),
]

SCATTER_TITLES = {
    "k": "K",
    "edec_error": r"$|E_{\mathrm{dec}}-E_{\mathrm{FCI}}|$",
    "ebo_error": r"$|E_{\mathrm{BO}}-E_{\mathrm{FCI}}|$",
}


def _as_float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    if value == "":
        return math.nan
    return float(value)


def read_summary_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _case_points(
    system_name: str,
    rows: list[dict[str, str]],
    *,
    case_name: str,
    optimized: bool,
) -> list[dict[str, float | str]]:
    suffix = "Optimized" if optimized else "Identity"
    points: list[dict[str, float | str]] = []
    for row in rows:
        e_fci = _as_float(row, "E_FCI")
        e_dec = _as_float(row, f"Edec_{suffix}")
        e_bo = _as_float(row, f"EBO_{suffix}")
        points.append(
            {
                "system": system_name,
                "case": case_name,
                "x": _as_float(row, "Geometry_Param"),
                "variance": _as_float(row, f"V_{suffix}"),
                "comm_sq": _as_float(row, f"Sum_CommSq_{suffix}"),
                "edec": e_dec,
                "edec_error": abs(e_dec - e_fci),
                "ebo": e_bo,
                "ebo_error": abs(e_bo - e_fci),
                "k": _as_float(row, f"Kcoupled_{suffix}"),
            }
        )
    return sorted(points, key=lambda point: float(point["x"]))


def load_system_cases(system_name: str, config: dict[str, object]) -> dict[str, list[dict[str, float | str]]]:
    fixed_rows = read_summary_rows(Path(config["fixed_csv"]))
    shared_rows = read_summary_rows(Path(config["shared_csv"]))
    local_rows = read_summary_rows(Path(config["local_csv"]))
    if not fixed_rows:
        raise ValueError(f"No data rows found in {config['fixed_csv']}")
    if not shared_rows:
        raise ValueError(f"No data rows found in {config['shared_csv']}")
    if not local_rows:
        raise ValueError(f"No data rows found in {config['local_csv']}")

    return {
        "initial": _case_points(system_name, fixed_rows, case_name="initial", optimized=False),
        "fixed": _case_points(system_name, fixed_rows, case_name="fixed", optimized=True),
        "shared": _case_points(system_name, shared_rows, case_name="shared", optimized=True),
        "local": _case_points(system_name, local_rows, case_name="local", optimized=True),
    }


def plot_system_diagnostics(
    system_name: str,
    config: dict[str, object],
    cases: dict[str, list[dict[str, float | str]]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    panels = [
        ("variance", r"Variance $\sum_{(p,q)} V_{pq}$", True),
        ("comm_sq", r"Commutator score $C = \sum_{(p,q)} \|[H,s_{pq}]\psi\|^2$", True),
        ("edec_error", r"Decoupled energy error $|E_{\mathrm{dec}}-E_{\mathrm{FCI}}|$ (Ha)", True),
        ("k", r"Coupled-sector dimension $K$", False),
    ]

    for case_name, points in cases.items():
        style = CASE_STYLES[case_name]
        xs = [float(point["x"]) for point in points]
        for ax, (field, ylabel, use_log) in zip(axes.ravel(), panels):
            ys = [float(point[field]) for point in points]
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
            if use_log and any(y > 0 and math.isfinite(y) for y in ys):
                ax.set_yscale("log")

    for ax in axes[-1, :]:
        ax.set_xlabel(str(config["x_label"]))

    axes[0, 0].legend(loc="best", frameon=False)
    fig.suptitle(f"{config['label']} fixed/shared/local abc diagnostics")
    fig.tight_layout()
    output_path = Path(config["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pearson_correlation(xs: list[float], ys: list[float]) -> float:
    pairs = [
        (x, y)
        for x, y in zip(xs, ys)
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(pairs) < 2:
        return math.nan

    clean_xs = [x for x, _ in pairs]
    clean_ys = [y for _, y in pairs]
    mean_x = sum(clean_xs) / len(clean_xs)
    mean_y = sum(clean_ys) / len(clean_ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    x_norm = math.sqrt(sum((x - mean_x) ** 2 for x in clean_xs))
    y_norm = math.sqrt(sum((y - mean_y) ** 2 for y in clean_ys))
    if x_norm == 0 or y_norm == 0:
        return math.nan
    return numerator / (x_norm * y_norm)


def _correlation_label(correlations: dict[str, float]) -> str:
    return "\n".join(
        f"{CASE_STYLES[case]['label'].split()[0]} r={value:.3f}"
        for case, value in correlations.items()
        if math.isfinite(value)
    )


def plot_system_commutator_scatter(
    system_name: str,
    config: dict[str, object],
    cases: dict[str, list[dict[str, float | str]]],
) -> list[dict[str, str]]:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), sharey=True)
    correlation_rows: list[dict[str, str]] = []

    for ax, (x_field, x_label, use_log_x) in zip(axes, SCATTER_PANELS):
        panel_correlations: dict[str, float] = {}
        for case_name, points in cases.items():
            style = CASE_STYLES[case_name]
            finite_points = [
                point
                for point in points
                if math.isfinite(float(point[x_field]))
                and math.isfinite(float(point["comm_sq"]))
                and (not use_log_x or float(point[x_field]) > 0)
            ]
            xs = [float(point[x_field]) for point in finite_points]
            ys = [float(point["comm_sq"]) for point in finite_points]
            corr = pearson_correlation(xs, ys)
            panel_correlations[case_name] = corr
            correlation_rows.append(
                {
                    "system": system_name,
                    "x_metric": x_field,
                    "case": case_name,
                    "pearson_r": f"{corr:.12g}" if math.isfinite(corr) else "",
                    "n": str(len(xs)),
                }
            )

            ax.scatter(
                xs,
                ys,
                color=style["color"],
                marker=style["marker"],
                s=48,
                alpha=0.82,
                edgecolors="none",
                label=style["label"],
            )

        ax.set_xlabel(x_label)
        ax.set_yscale("log")
        if use_log_x:
            ax.set_xscale("log")
        ax.set_title(f"C vs {SCATTER_TITLES[x_field]}\n{_correlation_label(panel_correlations)}")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(r"Commutator score $C$")
    axes[0].legend(loc="best", frameon=False)
    fig.suptitle(f"{config['label']} commutator correlations")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    output_path = Path(config["scatter_output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return correlation_rows


def plot_cross_system_scatter(
    all_points: list[dict[str, float | str]],
    output_path: Path,
    correlations_csv: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), sharey=True)

    correlation_rows: list[dict[str, str]] = []
    for ax, (x_field, x_label, use_log_x) in zip(axes, SCATTER_PANELS):
        panel_correlations: dict[str, float] = {}
        for case_name, style in CASE_STYLES.items():
            case_points = [point for point in all_points if point["case"] == case_name]
            xs = [float(point[x_field]) for point in case_points]
            ys = [float(point["comm_sq"]) for point in case_points]
            corr = pearson_correlation(xs, ys)
            panel_correlations[case_name] = corr
            correlation_rows.append(
                {
                    "x_metric": x_field,
                    "case": case_name,
                    "pearson_r": f"{corr:.12g}" if math.isfinite(corr) else "",
                    "n": str(len(xs)),
                }
            )

            for system_name, system_config in SYSTEMS.items():
                system_points = [
                    point
                    for point in case_points
                    if point["system"] == system_name
                    and math.isfinite(float(point[x_field]))
                    and math.isfinite(float(point["comm_sq"]))
                ]
                if not system_points:
                    continue
                ax.scatter(
                    [float(point[x_field]) for point in system_points],
                    [float(point["comm_sq"]) for point in system_points],
                    color=style["color"],
                    marker=SYSTEM_MARKERS[system_name],
                    s=42,
                    alpha=0.78,
                    edgecolors="none",
                    label=f"{style['label']} - {system_config['label']}",
                )

        ax.set_xlabel(x_label)
        ax.set_yscale("log")
        if use_log_x:
            ax.set_xscale("log")
        ax.set_title(f"C vs {SCATTER_TITLES[x_field]}\n{_correlation_label(panel_correlations)}")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(r"Commutator score $C$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle("Cross-system commutator correlations")
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    correlations_csv.parent.mkdir(parents=True, exist_ok=True)
    with correlations_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["x_metric", "case", "pearson_r", "n"])
        writer.writeheader()
        writer.writerows(correlation_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--correlations-csv",
        type=Path,
        default=TABLES_DIR / 'abc_system_commutator_correlations.csv"),
        help="Output CSV for per-system Pearson correlation coefficients.",
    )
    args = parser.parse_args()

    correlation_rows: list[dict[str, str]] = []
    for system_name, config in SYSTEMS.items():
        cases = load_system_cases(system_name, config)
        plot_system_diagnostics(system_name, config, cases)
        correlation_rows.extend(plot_system_commutator_scatter(system_name, config, cases))

    args.correlations_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.correlations_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "x_metric", "case", "pearson_r", "n"])
        writer.writeheader()
        writer.writerows(correlation_rows)


if __name__ == "__main__":
    main()
