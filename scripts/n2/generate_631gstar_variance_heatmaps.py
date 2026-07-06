"""N2 6-31G* full fixed-N FCI is infeasible — report dim_sub estimate and exit."""

from __future__ import annotations

import argparse
import math
import sys

from quasi_symmetries.hamiltonian.geometry import (
    N2_REPRESENTATIVE_GRID,
    get_geometry_and_description,
)
from quasi_symmetries.symmetry.labels import molecule_point_group
from quasi_symmetries.optimization import fixed_n_subspace_dim

DEFAULT_BASIS = "6-31g*"
DENSE_SUBSPACE_MAX = 8_000


def _estimate_n2_631gstar(molecule: str, x: float, basis: str) -> tuple[int, int, int]:
    try:
        from pyscf import gto, scf
    except ImportError as exc:
        raise SystemExit(
            "PySCF is required for the N2 feasibility estimate. "
            "Install pyscf and openfermionpyscf."
        ) from exc

    geometry, _ = get_geometry_and_description(molecule, x)
    point_group = molecule_point_group(molecule)
    mol = gto.M(
        atom=[(atom, coords) for atom, coords in geometry],
        basis=basis,
        symmetry=point_group,
        charge=0,
        spin=0,
    )
    mol.build()
    mf = scf.RHF(mol).run(verbose=0)
    n_spatial = mf.mo_coeff.shape[1]
    n_electrons = mol.nelectron
    dim_sub = fixed_n_subspace_dim(n_spatial, n_electrons)
    return n_spatial, n_electrons, dim_sub


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default=DEFAULT_BASIS)
    parser.add_argument(
        "--x",
        type=float,
        default=float(N2_REPRESENTATIVE_GRID[0]),
        help="Bond length (Å) for the PySCF sizing estimate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Exit 0 after printing the feasibility report (default exits 1).",
    )
    args = parser.parse_args()

    n_spatial, n_electrons, dim_sub = _estimate_n2_631gstar("n2", args.x, args.basis)
    log10_dim = math.log10(dim_sub) if dim_sub > 0 else 0.0

    print("=== N2 6-31G* fixed-N FCI feasibility ===")
    print(f"Geometry:           N2 bond = {args.x:.4f} Å")
    print(f"Basis:              {args.basis}")
    print(f"Point group:        {molecule_point_group('n2')}")
    print(f"n_spatial:          {n_spatial}")
    print(f"n_electrons:        {n_electrons}")
    print(f"dim_sub:            {dim_sub:,}  (~10^{log10_dim:.1f})")
    print(f"dense_sub_limit:    {DENSE_SUBSPACE_MAX:,}")
    print()
    print("Full fixed-N FCI at this basis is NOT supported by the current pipeline.")
    print("Reason: dim_sub is far beyond sparse FCI / variance workflows used for STO-3G and H2O 6-31G*.")
    print()
    print("Possible future paths:")
    print("  - active-space FCIDUMP + smaller electron count")
    print("  - DMRG (or similar) reference wavefunction instead of exact FCI")
    print("  - smaller basis (e.g. STO-3G) for full-CI quasi-symmetry optimization")
    print()
    print("No Hamiltonian caches or variance heatmaps were written.")

    return 0 if args.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
