"""H2O parity-seniority and mixed-pool optimization with variance heatmaps."""

from __future__ import annotations

from quasi_symmetries.config import CACHE_DIR, heatmap_optimization_dir, table_path

import argparse
import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle

from quasi_symmetries.diagnostics.mixed_pool import mixed_pool_energy_indicators
from quasi_symmetries.hamiltonian.cache import load_reference_state
from quasi_symmetries.hamiltonian.geometry import default_grid_for_molecule
from quasi_symmetries.optimization import (
    build_U_from_thetas,
    closed_shell_hf_bitstring,
    popcount,
    solve_cisd_state,
)
from quasi_symmetries.optimization.quartet import (
    MixedOperatorPool,
    mixed_pool_cost_for_u,
    normalize_edge,
    optimize_mixed_operator_pool,
)
from quasi_symmetries.symmetry.labels import load_symmetry_labels
from scripts.plot.orbital_heatmaps import (
    SystemJob,
    _increasing_orbital_ticks,
    _reversed_orbital_ticks,
    _system_title,
    _variance_display_coords,
    _variance_for_display,
    parity_variance_matrix,
)

HOH_ANGLE_DEG = 104.5

H2O_USER_MIXED_POOL = MixedOperatorPool(
    singles=(0, 1, 2),
    quartets=(normalize_edge((3, 6)), normalize_edge((4, 5))),
)

SUMMARY_FIELDS = [
    "Workflow",
    "Molecule",
    "Geometry_Param",
    "E_HF",
    "E_FCI",
    "E_CISD",
    "n_spatial",
    "Pool_Singles",
    "Pool_Quartets",
    "V_Identity",
    "V_Optimized",
    "Single_Expectations",
    "Single_Variances",
    "Quartet_Expectations",
    "Quartet_Variances",
    "Thetas_JSON",
    "Rotation_Pairs_JSON",
    "Optimizer_Success",
    "Optimizer_Status",
    "Optimizer_Message",
    "Optimizer_Nit",
    "Optimizer_Nfev",
    "N_Restarts",
    "Elapsed_Seconds",
]

H2O_HEATMAP_DIR = heatmap_optimization_dir("h2o")
H2O_SENIORITY_CSV = table_path("h2o", "parity_seniority_diagnostics.csv")
H2O_MIXED_SUMMARY_CSV = table_path("h2o", "mixed_pool_summary.csv")
H2O_MIXED_ENERGY_CSV = table_path("h2o", "mixed_pool_energy_diagnostics.csv")


def _load_seniority_rows(csv_path: Path = H2O_SENIORITY_CSV) -> list[dict]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Missing seniority diagnostics {csv_path}. Run the seniority phase first."
        )
    with csv_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _seniority_row_for_x(rows: list[dict], x: float, tol: float = 1e-9) -> dict:
    for row in rows:
        if abs(float(row["Geometry_Param"]) - x) <= tol:
            return row
    raise KeyError(f"No seniority row for geometry x={x:.6g}")


def _u_from_optimization_row(row: dict, n_spatial: int) -> np.ndarray:
    pairs = [tuple(pair) for pair in json.loads(row["Rotation_Pairs_JSON"])]
    thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
    return build_U_from_thetas(n_spatial, thetas, pairs)


def _thetas_from_optimization_row(row: dict) -> np.ndarray:
    return np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)


def _geometry_tag(x: float) -> str:
    return f"{x:.6g}".replace(".", "p")


def _json_list(values) -> str:
    return json.dumps([float(v) for v in values], separators=(",", ":"))


def _edge_json(edges) -> str:
    return json.dumps([[int(p), int(q)] for p, q in edges], separators=(",", ":"))


def _pool_labels(pool: MixedOperatorPool) -> tuple[str, str]:
    singles = " ".join(str(i) for i in pool.singles)
    quartets = " ".join(f"{p}-{q}" for p, q in pool.quartets)
    return singles, quartets


def _system_job(x: float) -> SystemJob:
    return SystemJob("h2o", x, {"hoh_angle_deg": HOH_ANGLE_DEG})


def _load_reference(x: float, *, cache_dir: str, load_hamiltonian: bool = False) -> dict:
    return load_reference_state(
        "h2o",
        x,
        cache_dir=cache_dir,
        load_hamiltonian=load_hamiltonian,
        load_full_hamiltonian=False,
        compute_rdms=False,
        popcount_fn=popcount,
        solve_cisd_fn=solve_cisd_state,
        hf_bitstring_fn=closed_shell_hf_bitstring,
        hoh_angle_deg=HOH_ANGLE_DEG,
    )


def _irrep_labels(ref: dict):
    labels = load_symmetry_labels(ref, molecule="h2o")
    return labels.irrep_labels if labels is not None else None


def _summary_row(
    *,
    workflow: str,
    ref: dict,
    pool: MixedOperatorPool,
    best: dict,
    v_identity: float,
    elapsed: float,
) -> dict:
    res = best["res"]
    singles_label, quartets_label = _pool_labels(pool)
    return {
        "Workflow": workflow,
        "Molecule": "h2o",
        "Geometry_Param": ref["geometry_param"],
        "E_HF": ref["energy_hf"],
        "E_FCI": ref["energy_fci"],
        "E_CISD": ref["energy_cisd"],
        "n_spatial": ref["n_spatial"],
        "Pool_Singles": singles_label,
        "Pool_Quartets": quartets_label,
        "V_Identity": v_identity,
        "V_Optimized": float(best["cost"]),
        "Single_Expectations": _json_list(stat.expectation for stat in best["single_stats"]),
        "Single_Variances": _json_list(stat.variance for stat in best["single_stats"]),
        "Quartet_Expectations": _json_list(stat.expectation for stat in best["quartet_stats"]),
        "Quartet_Variances": _json_list(stat.variance for stat in best["quartet_stats"]),
        "Thetas_JSON": _json_list(res.x),
        "Rotation_Pairs_JSON": _edge_json(best["pairs"]),
        "Optimizer_Success": bool(getattr(res, "success", False)),
        "Optimizer_Status": getattr(res, "status", ""),
        "Optimizer_Message": str(getattr(res, "message", "")),
        "Optimizer_Nit": getattr(res, "nit", ""),
        "Optimizer_Nfev": getattr(res, "nfev", ""),
        "N_Restarts": int(best["n_restarts"]),
        "Elapsed_Seconds": elapsed,
        "_u_spatial": best["u_spatial"],
        "_pool": pool,
    }


def _optimize_geometry(
    x: float,
    *,
    pool: MixedOperatorPool,
    cache_dir: str,
    n_restarts: int,
    maxfev: int | None,
    initial_thetas: np.ndarray | None = None,
    baseline_u: np.ndarray | None = None,
) -> dict:
    start = time.perf_counter()
    ref = _load_reference(x, cache_dir=cache_dir, load_hamiltonian=True)
    ref["geometry_param"] = x
    n_spatial = ref["n_spatial"]
    v_sub = ref["v_sub"]
    basis = ref["basis_bitstrings"]
    if baseline_u is None:
        baseline_u = np.eye(n_spatial, dtype=np.complex128)
    v_baseline = mixed_pool_cost_for_u(v_sub, basis, baseline_u, n_spatial, pool)

    best = optimize_mixed_operator_pool(
        v_sub,
        basis,
        n_spatial,
        pool,
        n_restarts=n_restarts,
        include_zero_start=True,
        initial_thetas=initial_thetas,
        parallel=False,
        maxfev=maxfev,
        irrep_labels=_irrep_labels(ref),
    )
    workflow = "parity_seniority" if not pool.quartets else "mixed_pool"
    row = _summary_row(
        workflow=workflow,
        ref=ref,
        pool=pool,
        best=best,
        v_identity=v_baseline,
        elapsed=time.perf_counter() - start,
    )
    row.update(mixed_pool_energy_indicators(ref, pool, best["u_spatial"]))
    row["Energy_Diagnostics_Seconds"] = row["Elapsed_Seconds"]
    return row


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(SUMMARY_FIELDS)
        for row in rows:
            for key in row:
                if key not in fieldnames and not key.startswith("_"):
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _highlight_pool(
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
        p, q = (p, q) if p < q else (q, p)
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


def plot_two_panel_variance(
    *,
    variance_canonical: np.ndarray,
    variance_optimized: np.ndarray,
    job: SystemJob,
    output_path: Path,
    left_title: str = "Canonical",
    right_title: str = "Optimized",
    highlight_singles: list[int] | None = None,
    highlight_quartets: list[tuple[int, int]] | None = None,
    suptitle: str | None = None,
    vmin: float = 1e-4,
    vmax: float = 1.0,
) -> None:
    n_spatial = variance_canonical.shape[0]
    fig = plt.figure(figsize=(9.5, 4.2), constrained_layout=True)
    gs = GridSpec(1, 2, figure=fig)
    x_ticks = _increasing_orbital_ticks(n_spatial)
    y_ticks = _reversed_orbital_ticks(n_spatial)
    im = None

    for col, (title, matrix) in enumerate(
        ((left_title, variance_canonical), (right_title, variance_optimized))
    ):
        ax = fig.add_subplot(gs[0, col])
        display = _variance_for_display(np.asarray(matrix, dtype=float))
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
        if col == 1 and highlight_singles is not None and highlight_quartets is not None:
            _highlight_pool(ax, highlight_singles, highlight_quartets, n_spatial)

    cbar = fig.colorbar(im, ax=fig.axes, shrink=0.9)
    cbar.set_label(r"Parity variance $1-\langle s\rangle^2$")
    fig.suptitle(suptitle or f"Parity variance | {_system_title(job)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_variance_npz(
    path: Path,
    *,
    x: float,
    u_canonical: np.ndarray,
    u_optimized: np.ndarray,
    variance_canonical: np.ndarray,
    variance_optimized: np.ndarray,
    pool: MixedOperatorPool | None = None,
    thetas: np.ndarray | None = None,
    pairs: list[tuple[int, int]] | None = None,
) -> None:
    payload = {
        "geometry_param": x,
        "u_canonical": u_canonical,
        "u_optimized": u_optimized,
        "variance_canonical": variance_canonical,
        "variance_optimized": variance_optimized,
    }
    if pool is not None:
        payload["pool_singles"] = np.asarray(pool.singles, dtype=int)
        payload["pool_quartets"] = np.asarray(pool.quartets, dtype=int)
    if thetas is not None:
        payload["thetas"] = thetas
    if pairs is not None:
        payload["rotation_pairs"] = np.asarray(pairs, dtype=int)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **payload)


def run_seniority_phase(
    grid: list[float],
    *,
    cache_dir: str,
    n_restarts: int,
    maxfev: int | None,
    max_workers: int,
    heatmap_dir: Path,
) -> list[dict]:
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(grid))) as executor:
        futures = {
            executor.submit(
                _optimize_geometry,
                x,
                pool=MixedOperatorPool(
                    singles=tuple(range(_load_reference(x, cache_dir=cache_dir)["n_spatial"])),
                    quartets=(),
                ),
                cache_dir=cache_dir,
                n_restarts=n_restarts,
                maxfev=maxfev,
            ): x
            for x in grid
        }
        for future in as_completed(futures):
            x = futures[future]
            row = future.result()
            print(
                f"[seniority] h2o x={x:.6g}: V {row['V_Identity']:.6g} -> {row['V_Optimized']:.6g}, "
                f"Edec err {abs(float(row['Edec_Optimized']) - float(row['E_FCI'])):.2e}, "
                f"K={row['Kcoupled_Optimized']} ({row['Elapsed_Seconds']:.1f}s)",
                flush=True,
            )
            rows.append(row)

    rows.sort(key=lambda row: float(row["Geometry_Param"]))
    _write_csv(H2O_SENIORITY_CSV, rows)
    print(f"[ok] wrote {H2O_SENIORITY_CSV}", flush=True)

    for row in rows:
        x = float(row["Geometry_Param"])
        tag = _geometry_tag(x)
        ref = _load_reference(x, cache_dir=cache_dir)
        n_spatial = ref["n_spatial"]
        u_canonical = np.eye(n_spatial, dtype=np.complex128)
        u_opt = row["_u_spatial"]
        var_id = parity_variance_matrix(ref["v_sub"], ref["basis_bitstrings"], u_canonical, n_spatial)
        var_opt = parity_variance_matrix(ref["v_sub"], ref["basis_bitstrings"], u_opt, n_spatial)
        npz_path = heatmap_dir / f"h2o_{tag}_seniority_data.npz"
        pairs = [tuple(pair) for pair in json.loads(row["Rotation_Pairs_JSON"])]
        thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
        _save_variance_npz(
            npz_path,
            x=x,
            u_canonical=u_canonical,
            u_optimized=u_opt,
            variance_canonical=var_id,
            variance_optimized=var_opt,
            thetas=thetas,
            pairs=pairs,
        )
        job = _system_job(x)
        plot_two_panel_variance(
            variance_canonical=var_id,
            variance_optimized=var_opt,
            job=job,
            output_path=heatmap_dir / f"h2o_{tag}_seniority_variance.png",
            left_title="Canonical",
            right_title="Parity seniorities optimized",
            suptitle=(
                f"All parity seniorities (C2v blocked rotations) | {_system_title(job)}"
            ),
        )
        print(f"[ok] wrote {heatmap_dir / f'h2o_{tag}_seniority_variance.png'}", flush=True)
    return rows


def run_mixed_phase(
    grid: list[float],
    *,
    cache_dir: str,
    n_restarts: int,
    maxfev: int | None,
    max_workers: int,
    heatmap_dir: Path,
    seniority_rows: list[dict],
    pool: MixedOperatorPool = H2O_USER_MIXED_POOL,
) -> list[dict]:
    warm_by_x: dict[float, np.ndarray] = {}
    baseline_u_by_x: dict[float, np.ndarray] = {}

    for x in grid:
        ref = _load_reference(x, cache_dir=cache_dir)
        n_spatial = ref["n_spatial"]
        seniority_row = _seniority_row_for_x(seniority_rows, x)
        u_seniority = _u_from_optimization_row(seniority_row, n_spatial)
        warm_by_x[x] = _thetas_from_optimization_row(seniority_row)
        baseline_u_by_x[x] = u_seniority
        print(
            f"[pool] x={x:.6g}: singles={pool.singles}, quartets={pool.quartets} "
            f"(fixed H2O mixed pool, seniority warm start)",
            flush=True,
        )

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(grid))) as executor:
        futures = {
            executor.submit(
                _optimize_geometry,
                x,
                pool=pool,
                cache_dir=cache_dir,
                n_restarts=n_restarts,
                maxfev=maxfev,
                initial_thetas=warm_by_x[x],
                baseline_u=baseline_u_by_x[x],
            ): x
            for x in grid
        }
        for future in as_completed(futures):
            x = futures[future]
            row = future.result()
            print(
                f"[mixed] h2o x={x:.6g}: V {row['V_Identity']:.6g} -> {row['V_Optimized']:.6g}, "
                f"Edec err {abs(float(row['Edec_Optimized']) - float(row['E_FCI'])):.2e}, "
                f"K={row['Kcoupled_Optimized']} ({row['Elapsed_Seconds']:.1f}s)",
                flush=True,
            )
            rows.append(row)

    rows.sort(key=lambda row: float(row["Geometry_Param"]))
    _write_csv(H2O_MIXED_SUMMARY_CSV, rows, fieldnames=SUMMARY_FIELDS)
    print(f"[ok] wrote {H2O_MIXED_SUMMARY_CSV}", flush=True)
    _write_csv(H2O_MIXED_ENERGY_CSV, rows)
    print(f"[ok] wrote {H2O_MIXED_ENERGY_CSV}", flush=True)

    for row in rows:
        x = float(row["Geometry_Param"])
        tag = _geometry_tag(x)
        pool_row = row["_pool"]
        ref = _load_reference(x, cache_dir=cache_dir)
        n_spatial = ref["n_spatial"]
        seniority_row = _seniority_row_for_x(seniority_rows, x)
        u_seniority = _u_from_optimization_row(seniority_row, n_spatial)
        u_opt = row["_u_spatial"]
        var_seniority = parity_variance_matrix(
            ref["v_sub"], ref["basis_bitstrings"], u_seniority, n_spatial
        )
        var_opt = parity_variance_matrix(ref["v_sub"], ref["basis_bitstrings"], u_opt, n_spatial)
        pairs = [tuple(pair) for pair in json.loads(row["Rotation_Pairs_JSON"])]
        thetas = np.asarray(json.loads(row["Thetas_JSON"]), dtype=float)
        path = heatmap_dir / f"h2o_{tag}_mixed_pool_data.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            geometry_param=x,
            u_canonical=np.eye(n_spatial, dtype=np.complex128),
            u_seniority=u_seniority,
            u_optimized=u_opt,
            variance_canonical=parity_variance_matrix(
                ref["v_sub"],
                ref["basis_bitstrings"],
                np.eye(n_spatial, dtype=np.complex128),
                n_spatial,
            ),
            variance_seniority=var_seniority,
            variance_optimized=var_opt,
            pool_singles=np.asarray(pool_row.singles, dtype=int),
            pool_quartets=np.asarray(pool_row.quartets, dtype=int),
            thetas=thetas,
            rotation_pairs=np.asarray(pairs, dtype=int),
        )
        singles = list(pool_row.singles)
        quartets = list(pool_row.quartets)
        singles_txt = ", ".join(f"$s_{i}$" for i in singles)
        quartets_txt = ", ".join(f"$s_{{{p}{q}}}$" for p, q in quartets)
        job = _system_job(x)
        plot_two_panel_variance(
            variance_canonical=var_seniority,
            variance_optimized=var_opt,
            job=job,
            output_path=heatmap_dir / f"h2o_{tag}_mixed_pool_variance.png",
            left_title="Seniority optimized",
            right_title="Mixed pool optimized",
            highlight_singles=singles,
            highlight_quartets=quartets,
            suptitle=(
                f"Mixed pool ({singles_txt}, {quartets_txt}) | {_system_title(job)}"
            ),
        )
        print(f"[ok] wrote {heatmap_dir / f'h2o_{tag}_mixed_pool_variance.png'}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    parser.add_argument("--heatmap-dir", type=Path, default=H2O_HEATMAP_DIR)
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--optimizer-maxfev", type=int, default=500)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--skip-seniority", action="store_true")
    parser.add_argument("--skip-mixed", action="store_true")
    parser.add_argument(
        "--x",
        type=float,
        action="append",
        help="Restrict to specific OH bond length(s). Default: full H2O grid.",
    )
    args = parser.parse_args()

    if args.x:
        grid = [float(v) for v in args.x]
    else:
        grid = [float(v) for v in default_grid_for_molecule("h2o")]

    args.heatmap_dir.mkdir(parents=True, exist_ok=True)
    H2O_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

    seniority_rows: list[dict] = []
    if not args.skip_seniority:
        seniority_rows = run_seniority_phase(
            grid,
            cache_dir=args.cache_dir,
            n_restarts=args.n_restarts,
            maxfev=args.optimizer_maxfev,
            max_workers=args.max_workers,
            heatmap_dir=args.heatmap_dir,
        )
    elif not args.skip_mixed:
        seniority_rows = _load_seniority_rows()
    if not args.skip_mixed:
        run_mixed_phase(
            grid,
            cache_dir=args.cache_dir,
            n_restarts=args.n_restarts,
            maxfev=args.optimizer_maxfev,
            max_workers=args.max_workers,
            heatmap_dir=args.heatmap_dir,
            seniority_rows=seniority_rows,
        )


if __name__ == "__main__":
    main()
