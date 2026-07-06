"""Generate canonical parity-variance heatmaps for selectable molecules/bases/backends."""

from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle

from quasi_symmetries.config import CACHE_DIR, IMAGES_DIR, table_path
from quasi_symmetries.hamiltonian.geometry import basis_cache_slug, default_grid_for_molecule
from quasi_symmetries.reference.backends import (
    ExactCIReference,
    default_backend_output_dir,
)
from quasi_symmetries.reference.dmrg import DMRGRunConfig, run_block2_dmrg
from quasi_symmetries.reference.dmrg_parity import dmrg_parity_variance_matrix
from quasi_symmetries.symmetry.labels import MoleculeSymmetryLabels, irrep_blocks
from scripts.plot.orbital_heatmaps import (
    SystemJob,
    _increasing_orbital_ticks,
    _reversed_orbital_ticks,
    _system_title,
    _variance_display_coords,
    _variance_for_display,
)

HOH_ANGLE_DEG = 104.5
SUPPORTED_MOLECULES = ("h2o", "n2")
SUPPORTED_BASES = ("sto-3g", "6-31g", "6-31g*", "cc-pvdz")
SUPPORTED_WAVEFUNCTIONS = ("exact", "dmrg")

SUMMARY_FIELDS = [
    "Molecule",
    "Basis",
    "Wavefunction",
    "Geometry_Param",
    "n_spatial",
    "n_electrons",
    "dim_sub",
    "E_HF",
    "Reference_Energy",
    "Reference_Energy_Label",
    "Bond_Dimension",
    "Max_Sweeps",
    "Sum_Diag_Variance",
    "Validation_MaxAbsDiff",
    "Validation_RMSDiff",
    "Elapsed_Seconds",
]


def _geometry_tag(x: float) -> str:
    return f"{x:.6g}".replace(".", "p")


def _geometry_kwargs(molecule: str) -> dict[str, Any]:
    if molecule.lower() == "h2o":
        return {"hoh_angle_deg": HOH_ANGLE_DEG}
    return {}


def _system_job(molecule: str, x: float) -> SystemJob:
    return SystemJob(molecule.lower(), x, _geometry_kwargs(molecule))


def _highlight_irrep_blocks(ax: plt.Axes, labels: MoleculeSymmetryLabels | None, n_spatial: int) -> None:
    if labels is None:
        return
    for block in irrep_blocks(labels.irrep_labels):
        for orbital in block:
            row, col = _variance_display_coords(orbital, orbital, n_spatial)
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="white",
                    linewidth=0.8,
                    linestyle="--",
                    zorder=9,
                )
            )


def plot_canonical_variance_heatmap(
    *,
    variance: np.ndarray,
    job: SystemJob,
    output_path: Path,
    basis: str,
    wavefunction: str,
    labels: MoleculeSymmetryLabels | None = None,
    vmin: float = 1e-4,
    vmax: float = 1.0,
) -> None:
    n_spatial = variance.shape[0]
    side = min(14.0, 4.0 + 0.38 * n_spatial)
    fig, ax = plt.subplots(figsize=(side, side * 0.88), constrained_layout=True)
    display = _variance_for_display(np.asarray(variance, dtype=float))
    display = np.ma.masked_less(display, vmin / 10)
    im = ax.imshow(
        display,
        aspect="equal",
        cmap="viridis",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        origin="lower",
    )
    ax.set_xlabel("Orbital index")
    ax.set_ylabel("Orbital index")
    ax.set_xticks(range(n_spatial))
    ax.set_yticks(range(n_spatial))
    ax.set_xticklabels(_increasing_orbital_ticks(n_spatial))
    ax.set_yticklabels(_reversed_orbital_ticks(n_spatial))
    _highlight_irrep_blocks(ax, labels, n_spatial)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label(r"Parity variance $1-\langle s\rangle^2$")
    fig.suptitle(
        f"Canonical parity variance ({basis}, {wavefunction}) | {_system_title(job)}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _basis_min_orbitals(molecule: str, basis: str) -> int:
    normalized = basis.strip().lower()
    mol = molecule.lower()
    if mol == "h2o":
        if normalized in {"sto-3g", "sto3g"}:
            return 7
        if normalized == "6-31g":
            return 13
        if normalized == "6-31g*":
            return 18
        if normalized == "cc-pvdz":
            return 24
    if mol == "n2":
        if normalized in {"sto-3g", "sto3g"}:
            return 10
        if normalized == "6-31g":
            return 18
        if normalized == "6-31g*":
            return 24
    return 1


def _validate_basis_size(*, molecule: str, basis: str, n_spatial: int, cache_hint: str) -> None:
    minimum = _basis_min_orbitals(molecule, basis)
    if int(n_spatial) < minimum:
        raise ValueError(
            f"Reference has n_spatial={n_spatial} for basis={basis!r} ({cache_hint}). "
            f"Expected at least {minimum}; regenerate the requested basis/backend."
        )


def _validation_stats(dmrg_variance: np.ndarray, exact_variance: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(dmrg_variance) & np.isfinite(exact_variance)
    diff = np.asarray(dmrg_variance[mask] - exact_variance[mask], dtype=float)
    if diff.size == 0:
        return {"Validation_MaxAbsDiff": float("nan"), "Validation_RMSDiff": float("nan")}
    return {
        "Validation_MaxAbsDiff": float(np.max(np.abs(diff))),
        "Validation_RMSDiff": float(np.sqrt(np.mean(diff**2))),
    }


def _exact_reference(
    *,
    molecule: str,
    x: float,
    basis: str,
    cache_dir: Path,
    overwrite: bool,
    skip_generation: bool,
) -> ExactCIReference:
    return ExactCIReference.from_cache_or_generate(
        molecule=molecule,
        x=x,
        basis=basis,
        cache_dir=cache_dir,
        overwrite=overwrite,
        skip_generation=skip_generation,
        geometry_kwargs=_geometry_kwargs(molecule),
    )


def _run_exact(
    *,
    molecule: str,
    x: float,
    basis: str,
    cache_dir: Path,
    overwrite: bool,
    skip_generation: bool,
) -> tuple[np.ndarray, MoleculeSymmetryLabels | None, dict[str, Any]]:
    reference = _exact_reference(
        molecule=molecule,
        x=x,
        basis=basis,
        cache_dir=cache_dir,
        overwrite=overwrite,
        skip_generation=skip_generation,
    )
    variance = reference.canonical_variance_matrix()
    meta = reference.metadata
    _validate_basis_size(
        molecule=molecule,
        basis=basis,
        n_spatial=meta.n_spatial,
        cache_hint=meta.cache_path or "",
    )
    return variance, reference.symmetry_labels, {
        "n_spatial": meta.n_spatial,
        "n_electrons": meta.n_electrons,
        "dim_sub": meta.dim_sub,
        "E_HF": meta.energy_hf,
        "Reference_Energy": meta.reference_energy,
        "Reference_Energy_Label": meta.reference_energy_label,
        "Bond_Dimension": "",
        "Max_Sweeps": "",
    }


def _run_dmrg(
    *,
    molecule: str,
    x: float,
    basis: str,
    dmrg_cache_root: Path,
    bond_dimension: int,
    max_sweeps: int,
    n_threads: int,
    overwrite: bool,
) -> tuple[np.ndarray, MoleculeSymmetryLabels | None, dict[str, Any], Any]:
    dmrg_kwargs: dict[str, Any] = {
        "molecule": molecule,
        "x": x,
        "basis": basis,
        "cache_root": dmrg_cache_root,
        "bond_dimension": bond_dimension,
        "max_sweeps": max_sweeps,
        "n_threads": n_threads,
        "overwrite": overwrite,
    }
    if molecule.lower() == "h2o":
        dmrg_kwargs["hoh_angle_deg"] = HOH_ANGLE_DEG
    reference = run_block2_dmrg(DMRGRunConfig(**dmrg_kwargs))
    variance = dmrg_parity_variance_matrix(reference)
    _validate_basis_size(
        molecule=molecule,
        basis=basis,
        n_spatial=reference.n_spatial,
        cache_hint=str(reference.metadata_path),
    )
    return variance, reference.symmetry_labels, {
        "n_spatial": reference.n_spatial,
        "n_electrons": reference.n_electrons,
        "dim_sub": int(reference.metadata["dim_sub"]),
        "E_HF": reference.energy_hf,
        "Reference_Energy": reference.energy_dmrg,
        "Reference_Energy_Label": "E_DMRG",
        "Bond_Dimension": int(reference.metadata["bond_dimension"]),
        "Max_Sweeps": int(reference.metadata["max_sweeps"]),
    }, reference


def run_geometry(
    x: float,
    *,
    molecule: str,
    basis: str,
    wavefunction: str,
    cache_dir: Path,
    dmrg_cache_root: Path,
    heatmap_dir: Path,
    overwrite: bool,
    skip_generation: bool,
    skip_plots: bool,
    validate_ci_vector: bool,
    bond_dimension: int,
    max_sweeps: int,
    n_threads: int,
    vmin: float,
    vmax: float,
) -> dict[str, Any]:
    molecule = molecule.lower()
    start = time.perf_counter()
    validation: dict[str, float] = {
        "Validation_MaxAbsDiff": float("nan"),
        "Validation_RMSDiff": float("nan"),
    }
    dmrg_reference = None
    if wavefunction == "exact":
        variance, labels, metrics = _run_exact(
            molecule=molecule,
            x=x,
            basis=basis,
            cache_dir=cache_dir,
            overwrite=overwrite,
            skip_generation=skip_generation,
        )
    elif wavefunction == "dmrg":
        variance, labels, metrics, dmrg_reference = _run_dmrg(
            molecule=molecule,
            x=x,
            basis=basis,
            dmrg_cache_root=dmrg_cache_root,
            bond_dimension=bond_dimension,
            max_sweeps=max_sweeps,
            n_threads=n_threads,
            overwrite=overwrite,
        )
        if validate_ci_vector:
            exact_reference = _exact_reference(
                molecule=molecule,
                x=x,
                basis=basis,
                cache_dir=cache_dir,
                overwrite=False,
                skip_generation=skip_generation,
            )
            validation = _validation_stats(variance, exact_reference.canonical_variance_matrix())
    else:
        raise ValueError(f"Unsupported wavefunction backend: {wavefunction!r}")

    tag = _geometry_tag(x)
    if not skip_plots:
        output_png = heatmap_dir / f"{molecule}_{tag}_canonical_variance.png"
        plot_canonical_variance_heatmap(
            variance=variance,
            job=_system_job(molecule, x),
            output_path=output_png,
            basis=basis,
            wavefunction=wavefunction,
            labels=labels,
            vmin=vmin,
            vmax=vmax,
        )
        npz_payload: dict[str, Any] = {
            "geometry_param": x,
            "basis": basis,
            "wavefunction_backend": wavefunction,
            "n_spatial": int(metrics["n_spatial"]),
            "n_electrons": int(metrics["n_electrons"]),
            "dim_sub": int(metrics["dim_sub"]),
            "reference_energy": float(metrics["Reference_Energy"]),
            "reference_energy_label": str(metrics["Reference_Energy_Label"]),
            "variance_canonical": variance,
        }
        if wavefunction == "dmrg":
            npz_payload["bond_dimension"] = int(metrics["Bond_Dimension"])
            npz_payload["max_sweeps"] = int(metrics["Max_Sweeps"])
            if dmrg_reference is not None:
                npz_payload["dmrg_metadata_path"] = str(dmrg_reference.metadata_path)
        if labels is not None:
            npz_payload["irrep_labels"] = np.asarray(labels.irrep_labels, dtype=str)
        np.savez(heatmap_dir / f"{molecule}_{tag}_canonical_variance_data.npz", **npz_payload)
        print(
            f"[ok] wrote {output_png} "
            f"(basis={basis}, wavefunction={wavefunction}, n_spatial={metrics['n_spatial']})",
            flush=True,
        )

    elapsed = time.perf_counter() - start
    return {
        "Molecule": molecule,
        "Basis": basis,
        "Wavefunction": wavefunction,
        "Geometry_Param": x,
        **metrics,
        **validation,
        "Sum_Diag_Variance": float(np.nansum(np.diag(variance))),
        "Elapsed_Seconds": elapsed,
    }


def _default_cache_dir(basis: str, wavefunction: str) -> Path:
    if wavefunction == "exact":
        slug = basis_cache_slug(basis)
        return CACHE_DIR if not slug else CACHE_DIR / slug
    return CACHE_DIR


def _default_summary_csv(molecule: str, basis: str, wavefunction: str) -> Path:
    return table_path(
        molecule,
        f"{wavefunction}_canonical_variance_summary.csv",
        basis,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=SUPPORTED_MOLECULES, default="h2o")
    parser.add_argument("--basis", choices=SUPPORTED_BASES, default="6-31g")
    parser.add_argument("--wavefunction", choices=SUPPORTED_WAVEFUNCTIONS, default="exact")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--dmrg-cache-root", type=Path, default=Path("dmrg_cache"))
    parser.add_argument("--heatmap-dir", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--validate-ci-vector", action="store_true")
    parser.add_argument("--dmrg-bond-dimension", type=int, default=512)
    parser.add_argument("--dmrg-max-sweeps", type=int, default=20)
    parser.add_argument("--dmrg-threads", type=int, default=1)
    parser.add_argument("--vmin", type=float, default=1e-4)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument(
        "--x",
        type=float,
        action="append",
        help="Restrict to specific geometry parameter value(s). Default: full molecule grid.",
    )
    args = parser.parse_args()
    molecule = args.molecule.lower()

    cache_dir = args.cache_dir or _default_cache_dir(args.basis, args.wavefunction)
    heatmap_dir = args.heatmap_dir or default_backend_output_dir(
        IMAGES_DIR / "orbital_heatmaps",
        molecule,
        args.basis,
        args.wavefunction,
    )
    summary_csv = args.summary_csv or _default_summary_csv(molecule, args.basis, args.wavefunction)
    grid = [float(v) for v in args.x] if args.x else [float(v) for v in default_grid_for_molecule(molecule)]

    heatmap_dir.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    worker_count = max(1, min(args.max_workers, len(grid)))
    # Block2/pyblock2 is not thread-safe: run one DMRG driver per process.
    executor_cls = ProcessPoolExecutor if args.wavefunction == "dmrg" else ThreadPoolExecutor
    if args.wavefunction == "dmrg":
        print(
            f"[pool] DMRG backend: {worker_count} process worker(s) over {len(grid)} geometries",
            flush=True,
        )
    else:
        print(
            f"[pool] exact backend: {worker_count} thread worker(s) over {len(grid)} geometries",
            flush=True,
        )

    rows: list[dict[str, Any]] = []
    with executor_cls(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                run_geometry,
                x,
                molecule=molecule,
                basis=args.basis,
                wavefunction=args.wavefunction,
                cache_dir=cache_dir,
                dmrg_cache_root=args.dmrg_cache_root,
                heatmap_dir=heatmap_dir,
                overwrite=args.overwrite,
                skip_generation=args.skip_generation,
                skip_plots=args.skip_plots,
                validate_ci_vector=args.validate_ci_vector,
                bond_dimension=args.dmrg_bond_dimension,
                max_sweeps=args.dmrg_max_sweeps,
                n_threads=args.dmrg_threads,
                vmin=args.vmin,
                vmax=args.vmax,
            ): x
            for x in grid
        }
        for future in as_completed(futures):
            x = futures[future]
            row = future.result()
            print(
                f"[ok] {molecule} x={x:.6g}: basis={args.basis} wavefunction={args.wavefunction} "
                f"n_spatial={row['n_spatial']} E_ref={row['Reference_Energy']:.10f} "
                f"sum_diag_V={row['Sum_Diag_Variance']:.6g} ({row['Elapsed_Seconds']:.1f}s)",
                flush=True,
            )
            rows.append(row)

    rows.sort(key=lambda row: float(row["Geometry_Param"]))
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in SUMMARY_FIELDS})
    print(f"[ok] wrote {summary_csv} ({len(rows)} rows)", flush=True)


if __name__ == "__main__":
    main()
