"""Plot N2 mixed-pool vs parity-seniority operator diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt

from quasi_symmetries.config import IMAGES_DIR, TABLES_DIR

SERIES = (
    ("seniority_canonical", "Parity Seniorities (canonical)", "#000000", "x", ":"),
    ("mixed_canonical", "Mixed Pool (canonical)", "#999999", "D", "--"),
    ("seniority_optimized", "Parity Seniorities (optimized)", "#E69F00", "s", "-."),
    ("mixed_optimized", "Mixed Pool (optimized)", "#009E73", "o", "-"),
)

DEFAULT_TITLE = (
    r"N$_2$ operator diagnostics: mixed pool selected "
    r"$s_0$–$s_3$ + quartets $s_{58}$, $s_{67}$, $s_{49}$"
)


def _as_float(row: dict[str, str], field: str) -> float:
    if not field:
        return math.nan
    value = row.get(field, "")
    if value == "":
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return sorted(csv.DictReader(handle), key=lambda row: float(row["Geometry_Param"]))


def _points(
    rows: list[dict[str, str]],
    *,
    variance_field: str,
    comm_field: str,
    edec_field: str,
    k_field: str,
) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for row in rows:
        e_fci = _as_float(row, "E_FCI")
        e_dec = _as_float(row, edec_field)
        points.append(
            {
                "x": _as_float(row, "Geometry_Param"),
                "variance": _as_float(row, variance_field),
                "comm_sq": _as_float(row, comm_field),
                "edec_error": abs(e_dec - e_fci) if math.isfinite(e_dec) and math.isfinite(e_fci) else math.nan,
                "k": _as_float(row, k_field),
            }
        )
    return points


def _has_finite(values: list[float]) -> bool:
    return any(math.isfinite(value) for value in values)


def plot_n2_operator_diagnostics(
    *,
    seniority_csv: Path,
    mixed_pool_csv: Path,
    output_path: Path,
    title: str = DEFAULT_TITLE,
) -> None:
    seniority_rows = _read_rows(seniority_csv)
    mixed_rows = _read_rows(mixed_pool_csv)
    if not seniority_rows or not mixed_rows:
        raise ValueError("Both CSV inputs must contain at least one geometry row.")

    seniority_has_energy = bool(seniority_rows[0].get("Edec_Optimized", "").strip())
    seniority_comm = "Sum_CommSq_Action" if seniority_has_energy else ""
    seniority_edec_id = "Edec_Identity" if seniority_has_energy else ""
    seniority_edec_opt = "Edec_Optimized" if seniority_has_energy else ""
    seniority_k_id = "Kcoupled_Identity" if seniority_has_energy else ""
    seniority_k_opt = "Kcoupled_Optimized" if seniority_has_energy else ""

    series = {
        "seniority_canonical": _points(
            seniority_rows,
            variance_field="V_Identity",
            comm_field="",
            edec_field=seniority_edec_id,
            k_field=seniority_k_id,
        ),
        "seniority_optimized": _points(
            seniority_rows,
            variance_field="V_Optimized",
            comm_field=seniority_comm,
            edec_field=seniority_edec_opt,
            k_field=seniority_k_opt,
        ),
        "mixed_canonical": _points(
            mixed_rows,
            variance_field="V_Identity",
            comm_field="",
            edec_field="Edec_Identity",
            k_field="Kcoupled_Identity",
        ),
        "mixed_optimized": _points(
            mixed_rows,
            variance_field="V_Optimized",
            comm_field="Sum_CommSq_Action",
            edec_field="Edec_Optimized",
            k_field="Kcoupled_Optimized",
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True)
    panels = [
        ("variance", "Variance diagnostic", True),
        ("comm_sq", "Commutator score", True),
        ("edec_error", r"Decoupled energy error $|E_{\mathrm{dec}}-E_{\mathrm{FCI}}|$ (Ha)", True),
        ("k", r"Coupled-sector dimension $K$", False),
    ]

    for key, label, color, marker, linestyle in SERIES:
        points = series[key]
        if not points:
            continue
        xs = [point["x"] for point in points]
        for ax, (field, ylabel, use_log) in zip(axes.ravel(), panels):
            ys = [point[field] for point in points]
            if not _has_finite(ys):
                continue
            ax.plot(
                xs,
                ys,
                marker=marker,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                markersize=4.5,
                label=label,
            )
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
            if use_log:
                positive = [y for y in ys if y > 0 and math.isfinite(y)]
                if positive:
                    ax.set_yscale("log")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"N$_2$ bond length ($\AA$)")

    axes[0, 0].legend(loc="best", frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seniority-csv",
        type=Path,
        default=TABLES_DIR / "n2_parity_seniority_summary.csv",
    )
    parser.add_argument(
        "--mixed-pool-csv",
        type=Path,
        default=TABLES_DIR / "n2_mixed_pool_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=IMAGES_DIR / "quartets/n2_mixed_pool_diagnostics.png",
    )
    args = parser.parse_args()
    plot_n2_operator_diagnostics(
        seniority_csv=args.seniority_csv,
        mixed_pool_csv=args.mixed_pool_csv,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
