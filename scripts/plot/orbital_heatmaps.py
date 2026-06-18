"""Plot MO-coefficient and quartet-variance heatmaps across orbital frames."""

from __future__ import annotations

from quasi_symmetries.config import IMAGES_DIR, LEGACY_ABC_OPT_RESULTS_DIR, LEGACY_ABC_TABLES_DIR, OPT_RESULTS_DIR, TABLES_DIR

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from quasi_symmetries.hamiltonian.cache import load_reference_state
from quasi_symmetries.hamiltonian.geometry import N2_REPRESENTATIVE_GRID, default_grid_for_molecule
from quasi_symmetries.optimization import (
    build_U_from_thetas,
    closed_shell_hf_bitstring,
    popcount,
    solve_cisd_state,
)
from quasi_symmetries.optimization.quartet import quartet_parity_expectations, single_parity_expectations


QUARTET_BASELINES = ("greedy", "ring", "balanced_tree")

ROW1_PANELS = (
    ("Canonical", "canonical"),
    ("Seniority", "seniority"),
)

ROW2_PANELS = (
    ("Greedy", "greedy"),
    ("Ring", "ring"),
    ("Balanced tree", "balanced_tree"),
)


@dataclass(frozen=True)
class SystemJob:
    molecule: str
    x: float
    geometry_kwargs: dict


DEFAULT_JOBS = (
    SystemJob("h2o", 1.6433333333333333, {"hoh_angle_deg": 104.5}),
    *(SystemJob("n2", float(x), {}) for x in N2_REPRESENTATIVE_GRID),
)


def _geometry_tag(x: float) -> str:
    return f"{x:.6g}".replace(".", "p")


def _default_paths(molecule: str) -> tuple[Path, Path]:
    mol = molecule.lower()
    if mol == "h2o":
        seniority = TABLES_DIR / "h2o_parity_seniority_diagnostics.csv"
    elif mol == "n2":
        seniority = TABLES_DIR / "n2_parity_seniority_summary.csv"
    else:
        seniority = LEGACY_ABC_TABLES_DIR / f"{mol}_quasi_symmetry_fixed_abc.csv"
    quartet = TABLES_DIR / f"{mol}_quartet_baseline_summary.csv"
    return seniority, quartet


def _load_csv_row(csv_path: Path, x: float) -> dict[str, str]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["Geometry_Param"]) - x) <= 1e-9:
                return row
    raise ValueError(f"No row with Geometry_Param={x} in {csv_path}")


def _u_from_csv_row(row: dict[str, str]) -> np.ndarray:
    return np.asarray(json.loads(row["U_Spatial"]), dtype=np.complex128)


def _u_from_thetas_row(row: dict[str, str], n_spatial: int) -> np.ndarray:
    thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
    pairs = [tuple(pair) for pair in json.loads(row["Rotation_Pairs_JSON"])]
    return build_U_from_thetas(n_spatial, thetas, pairs)


def _load_seniority_unitary(
    molecule: str,
    x: float,
    n_spatial: int,
    *,
    seniority_csv: Path,
    seniority_npz: Path | None,
) -> np.ndarray:
    if seniority_npz is not None and seniority_npz.is_file():
        return np.asarray(np.load(seniority_npz)["u_spatial"], dtype=np.complex128)
    row = _load_csv_row(seniority_csv, x)
    if "U_Spatial" in row:
        return _u_from_csv_row(row)
    return _u_from_thetas_row(row, n_spatial)


def _parse_quartet_edges(value: str) -> list[tuple[int, int]]:
    return [tuple(int(part) for part in edge.split("-")) for edge in value.split()]


def _load_quartet_edges(
    x: float,
    *,
    quartet_csv: Path,
) -> dict[str, list[tuple[int, int]]]:
    rows_by_baseline: dict[str, dict[str, str]] = {}
    with quartet_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["Geometry_Param"]) - x) > 1e-9:
                continue
            baseline = row.get("Baseline", "")
            if baseline in QUARTET_BASELINES:
                rows_by_baseline[baseline] = row

    missing = [name for name in QUARTET_BASELINES if name not in rows_by_baseline]
    if missing:
        raise ValueError(f"Missing quartet baselines {missing} for x={x} in {quartet_csv}")

    return {
        baseline: _parse_quartet_edges(rows_by_baseline[baseline]["Edges"])
        for baseline in QUARTET_BASELINES
    }


def _load_quartet_unitaries(
    x: float,
    n_spatial: int,
    *,
    quartet_csv: Path,
) -> dict[str, np.ndarray]:
    rows_by_baseline: dict[str, dict[str, str]] = {}
    with quartet_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if abs(float(row["Geometry_Param"]) - x) > 1e-9:
                continue
            baseline = row.get("Baseline", "")
            if baseline in QUARTET_BASELINES:
                rows_by_baseline[baseline] = row

    missing = [name for name in QUARTET_BASELINES if name not in rows_by_baseline]
    if missing:
        raise ValueError(f"Missing quartet baselines {missing} for x={x} in {quartet_csv}")

    return {
        baseline: _u_from_thetas_row(rows_by_baseline[baseline], n_spatial)
        for baseline in QUARTET_BASELINES
    }


def _load_all_unitaries(
    job: SystemJob,
    n_spatial: int,
    *,
    seniority_csv: Path,
    quartet_csv: Path,
    seniority_npz: Path | None,
) -> dict[str, np.ndarray]:
    unitaries = {
        "canonical": np.eye(n_spatial, dtype=np.complex128),
        "seniority": _load_seniority_unitary(
            job.molecule,
            job.x,
            n_spatial,
            seniority_csv=seniority_csv,
            seniority_npz=seniority_npz,
        ),
    }
    unitaries.update(
        _load_quartet_unitaries(job.x, n_spatial, quartet_csv=quartet_csv)
    )
    return unitaries


def _load_mo_coefficients(ref: dict, job: SystemJob) -> tuple[np.ndarray, list[str]]:
    mo_coeff = ref.get("mo_coeff")
    ao_labels = ref.get("ao_labels")
    if mo_coeff is not None and ao_labels is not None:
        return np.asarray(mo_coeff, dtype=np.float64), list(ao_labels)

    sidecar = (
        Path("opt_results")
        / "mo_coeff_cache"
        / f"{job.molecule}_{_geometry_tag(job.x)}_mo_coeff.npz"
    )
    if sidecar.is_file():
        data = np.load(sidecar, allow_pickle=True)
        return np.asarray(data["mo_coeff"], dtype=np.float64), list(data["ao_labels"])

    raise FileNotFoundError(
        f"MO coefficients missing for {job.molecule} x={job.x}. "
        "Run scripts/colab_precompute_mo_coefficients.py on Colab."
    )


def rotated_mo_coefficients(mo_coeff: np.ndarray, u_spatial: np.ndarray) -> np.ndarray:
    rotated = np.asarray(mo_coeff, dtype=np.float64) @ np.asarray(u_spatial, dtype=np.complex128)
    return np.real(rotated).astype(np.float64, copy=False)


def parity_variance_matrix(
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    u_spatial: np.ndarray,
    n_spatial: int,
) -> np.ndarray:
    matrix = np.full((n_spatial, n_spatial), np.nan, dtype=float)
    singles = single_parity_expectations(v_sub, basis_bitstrings, u_spatial, n_spatial)
    for orbital, stats in enumerate(singles):
        matrix[orbital, orbital] = stats.variance

    for p in range(n_spatial):
        for q in range(p + 1, n_spatial):
            stats = quartet_parity_expectations(
                v_sub,
                basis_bitstrings,
                u_spatial,
                n_spatial,
                [(p, q)],
            )[0]
            matrix[p, q] = stats.variance
    return matrix


def _variance_for_display(matrix: np.ndarray) -> np.ndarray:
    """Reverse row order only; x-axis columns stay in increasing orbital index."""
    return np.ma.masked_invalid(matrix[::-1, :])


def _variance_display_coords(p: int, q: int, n_spatial: int) -> tuple[int, int]:
    """Map stored variance-matrix indices to imshow display coordinates."""
    row = n_spatial - 1 - p
    col = q
    return row, col


def _highlight_quartet_cost_edges(
    ax: plt.Axes,
    edges: list[tuple[int, int]],
    n_spatial: int,
) -> None:
    for p, q in edges:
        if p == q:
            continue
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


def _increasing_orbital_ticks(n_spatial: int) -> list[str]:
    return [str(i) for i in range(n_spatial)]


def _reversed_orbital_ticks(n_spatial: int) -> list[str]:
    return [str(n_spatial - 1 - i) for i in range(n_spatial)]


def _format_ao_label(label: str) -> str:
    parts = label.split()
    if len(parts) >= 3:
        return f"{parts[0]} {parts[1]} {parts[2]}"
    return label


def _system_title(job: SystemJob) -> str:
    if job.molecule == "h2o":
        angle = job.geometry_kwargs.get("hoh_angle_deg", 104.5)
        return f"H2O OH={job.x:.4f} Å, HOH={angle:.1f}°"
    if job.molecule == "n2":
        return f"N2 bond={job.x:.4f} Å"
    return f"{job.molecule} x={job.x:.4f}"


def plot_orbital_coefficient_grid(
    mo_coeff: np.ndarray,
    ao_labels: list[str],
    unitaries: dict[str, np.ndarray],
    *,
    job: SystemJob,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(14.0, 7.5), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0])

    vmax = 1.0
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im = None
    panels = list(ROW1_PANELS) + list(ROW2_PANELS)

    for index, (title, key) in enumerate(panels):
        row = 0 if index < 2 else 1
        col = index if index < 2 else index - 2
        ax = fig.add_subplot(gs[row, col])
        coeff = rotated_mo_coefficients(mo_coeff, unitaries[key])
        coeff = coeff[:, ::-1]
        im = ax.imshow(coeff, aspect="auto", cmap="PuOr_r", norm=norm, origin="upper")
        ax.set_title(title)
        ax.set_xlabel("Molecular orbital")
        ax.set_xticks(range(coeff.shape[1]))
        ax.set_xticklabels(_reversed_orbital_ticks(coeff.shape[1]))
        ax.set_yticks(range(len(ao_labels)))
        ax.set_yticklabels([_format_ao_label(label) for label in ao_labels], fontsize=8)

    cbar = fig.colorbar(im, ax=fig.axes, shrink=0.88)
    cbar.set_label("MO coefficient")
    fig.suptitle(f"Canonical orbitals | {_system_title(job)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _variance_panel_keys() -> list[tuple[str, str]]:
    return list(ROW1_PANELS) + list(ROW2_PANELS)


def plot_variance_grid_from_matrices(
    variance_by_key: dict[str, np.ndarray],
    n_spatial: int,
    *,
    job: SystemJob,
    output_path: Path,
    vmin: float = 1e-4,
    vmax: float = 1.0,
    quartet_cost_edges: dict[str, list[tuple[int, int]]] | None = None,
) -> None:
    fig = plt.figure(figsize=(14.0, 7.5), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0])

    im = None
    x_ticks = _increasing_orbital_ticks(n_spatial)
    y_ticks = _reversed_orbital_ticks(n_spatial)

    for index, (title, key) in enumerate(_variance_panel_keys()):
        row = 0 if index < 2 else 1
        col = index if index < 2 else index - 2
        ax = fig.add_subplot(gs[row, col])
        display = _variance_for_display(variance_by_key[key])
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
        if quartet_cost_edges is not None and key in quartet_cost_edges:
            _highlight_quartet_cost_edges(ax, quartet_cost_edges[key], n_spatial)

    cbar = fig.colorbar(im, ax=fig.axes, shrink=0.88)
    cbar.set_label(r"Parity variance $1-\langle s\rangle^2$")
    fig.suptitle(
        "Quartet parity variance on FCI state\n"
        f"(diagonal: single-orbital $s_p$; off-diagonal: quartet $s_{{pq}}$) | "
        f"{_system_title(job)}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_variance_grid(
    v_sub: np.ndarray,
    basis_bitstrings: list[int],
    unitaries: dict[str, np.ndarray],
    n_spatial: int,
    *,
    job: SystemJob,
    output_path: Path,
    vmin: float = 1e-4,
    vmax: float = 1.0,
    quartet_cost_edges: dict[str, list[tuple[int, int]]] | None = None,
) -> None:
    variance_by_key = {
        key: parity_variance_matrix(v_sub, basis_bitstrings, unitaries[key], n_spatial)
        for _, key in _variance_panel_keys()
    }
    plot_variance_grid_from_matrices(
        variance_by_key,
        n_spatial,
        job=job,
        output_path=output_path,
        vmin=vmin,
        vmax=vmax,
        quartet_cost_edges=quartet_cost_edges,
    )


def _npz_path_for_job(job: SystemJob, output_dir: Path) -> Path:
    mol = job.molecule.lower()
    return output_dir / mol / f"{mol}_{_geometry_tag(job.x)}_heatmap_data.npz"


def process_variance_from_npz(
    job: SystemJob,
    *,
    output_dir: Path,
    npz_path: Path | None = None,
    quartet_csv: Path | None = None,
    vmin: float = 1e-4,
    vmax: float = 1.0,
) -> None:
    npz_path = _npz_path_for_job(job, output_dir) if npz_path is None else npz_path
    if not npz_path.is_file():
        raise FileNotFoundError(f"Precomputed variance data not found: {npz_path}")

    data = np.load(npz_path)
    variance_by_key = {
        key: np.asarray(data[f"variance_{key}"], dtype=float)
        for _, key in _variance_panel_keys()
    }
    n_spatial = next(iter(variance_by_key.values())).shape[0]

    if quartet_csv is None:
        _, quartet_csv = _default_paths(job.molecule)
    quartet_cost_edges = _load_quartet_edges(job.x, quartet_csv=quartet_csv)

    mol = job.molecule.lower()
    tag = _geometry_tag(job.x)
    variance_path = output_dir / mol / f"{mol}_{tag}_parity_variance.png"
    plot_variance_grid_from_matrices(
        variance_by_key,
        n_spatial,
        job=job,
        output_path=variance_path,
        vmin=vmin,
        vmax=vmax,
        quartet_cost_edges=quartet_cost_edges,
    )
    print(f"[ok] wrote {variance_path} (from {npz_path.name})")


def process_system(
    job: SystemJob,
    *,
    cache_dir: str,
    output_dir: Path,
    seniority_csv: Path | None = None,
    quartet_csv: Path | None = None,
    seniority_npz: Path | None = None,
    vmin: float = 1e-4,
    vmax: float = 1.0,
) -> None:
    default_seniority, default_quartet = _default_paths(job.molecule)
    seniority_csv = default_seniority if seniority_csv is None else seniority_csv
    quartet_csv = default_quartet if quartet_csv is None else quartet_csv

    if seniority_npz is None:
        candidate = (
            LEGACY_ABC_OPT_RESULTS_DIR
            / f"{job.molecule}_seniority_rotations"
            / f"{job.molecule}_{_geometry_tag(job.x)}.npz"
        )
        seniority_npz = candidate if candidate.is_file() else None

    ref = load_reference_state(
        job.molecule,
        job.x,
        cache_dir=cache_dir,
        load_hamiltonian=False,
        load_full_hamiltonian=False,
        compute_rdms=False,
        popcount_fn=popcount,
        solve_cisd_fn=solve_cisd_state,
        hf_bitstring_fn=closed_shell_hf_bitstring,
        **job.geometry_kwargs,
    )
    n_spatial = ref["n_spatial"]
    unitaries = _load_all_unitaries(
        job,
        n_spatial,
        seniority_csv=seniority_csv,
        quartet_csv=quartet_csv,
        seniority_npz=seniority_npz,
    )

    tag = _geometry_tag(job.x)
    mol = job.molecule.lower()
    out_root = output_dir / mol
    out_root.mkdir(parents=True, exist_ok=True)

    quartet_cost_edges = _load_quartet_edges(job.x, quartet_csv=quartet_csv)
    variance_path = out_root / f"{mol}_{tag}_parity_variance.png"
    plot_variance_grid(
        ref["v_sub"],
        ref["basis_bitstrings"],
        unitaries,
        n_spatial,
        job=job,
        output_path=variance_path,
        vmin=vmin,
        vmax=vmax,
        quartet_cost_edges=quartet_cost_edges,
    )
    print(f"[ok] wrote {variance_path}")

    variance_data = {
        f"variance_{key}": parity_variance_matrix(
            ref["v_sub"], ref["basis_bitstrings"], unitaries[key], n_spatial
        )
        for key in unitaries
    }
    np.savez(
        out_root / f"{mol}_{tag}_heatmap_data.npz",
        geometry_param=job.x,
        **{f"u_{key}": unitaries[key] for key in unitaries},
        **variance_data,
    )

    try:
        mo_coeff, ao_labels = _load_mo_coefficients(ref, job)
    except FileNotFoundError as exc:
        print(f"[skip] {mol} x={job.x}: orbital-coefficient heatmaps ({exc})")
        return

    orbital_path = out_root / f"{mol}_{tag}_canonical_orbitals.png"
    plot_orbital_coefficient_grid(
        mo_coeff,
        ao_labels,
        unitaries,
        job=job,
        output_path=orbital_path,
    )
    print(f"[ok] wrote {orbital_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="hamiltonian_cache")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=IMAGES_DIR / "orbital_heatmaps",
    )
    parser.add_argument(
        "--molecule",
        choices=("h2o", "n2"),
        help="If set, only plot this molecule (with --x or default grid).",
    )
    parser.add_argument("--x", type=float, help="Single geometry parameter.")
    parser.add_argument(
        "--all-geometries",
        action="store_true",
        help="Plot every geometry on the default scan grid for --molecule.",
    )
    parser.add_argument(
        "--from-npz",
        action="store_true",
        help="Replot variance heatmaps from saved *_heatmap_data.npz only.",
    )
    parser.add_argument("--vmin", type=float, default=1e-4)
    parser.add_argument("--vmax", type=float, default=1.0)
    args = parser.parse_args()

    if args.molecule is None:
        jobs = list(DEFAULT_JOBS)
    elif args.molecule == "h2o":
        if args.all_geometries:
            xs = [float(x) for x in default_grid_for_molecule("h2o")]
        elif args.x is not None:
            xs = [args.x]
        else:
            xs = [1.6433333333333333]
        jobs = [SystemJob("h2o", x, {"hoh_angle_deg": 104.5}) for x in xs]
    else:
        if args.all_geometries:
            xs = [float(x) for x in default_grid_for_molecule("n2")]
        elif args.x is not None:
            xs = [args.x]
        else:
            xs = [float(v) for v in N2_REPRESENTATIVE_GRID]
        jobs = [SystemJob("n2", x, {}) for x in xs]

    for job in jobs:
        print(f"[run] {job.molecule} x={job.x}", flush=True)
        if args.from_npz:
            process_variance_from_npz(
                job,
                output_dir=args.output_dir,
                vmin=args.vmin,
                vmax=args.vmax,
            )
            continue
        process_system(
            job,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            vmin=args.vmin,
            vmax=args.vmax,
        )


if __name__ == "__main__":
    main()
