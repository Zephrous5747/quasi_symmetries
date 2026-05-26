"""PySCF-backed Hamiltonian generation for offline caching."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse.linalg as spla
from openfermion import MolecularData, get_fermion_operator, get_sparse_operator, jordan_wigner
from openfermionpyscf import run_pyscf

from hamiltonian_cache import cache_path, save_reference_state
from hamiltonian_geometry import get_geometry_and_description, iter_scan_points
from optimization_abc_utils import (
    BASIS,
    CHARGE,
    MULTIPLICITY,
    closed_shell_hf_bitstring,
    compute_spin_rdms_from_statevector,
    compute_spin_rdms_from_subspace_state,
    fixed_n_subspace_dim,
    popcount,
    solve_cisd_state,
    use_dense_subspace_ops,
)


def build_reference_state_with_pyscf(
    *,
    geometry: list,
    description: str,
    basis: str = BASIS,
    charge: int = CHARGE,
    multiplicity: int = MULTIPLICITY,
    popcount_fn=popcount,
    solve_cisd_fn=solve_cisd_state,
    hf_bitstring_fn=closed_shell_hf_bitstring,
) -> dict[str, Any]:
    """Run PySCF and build fixed-N Hamiltonian, FCI ground state, and CISD reference."""
    mol = MolecularData(
        geometry=geometry,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        description=description,
    )
    mol = run_pyscf(mol, run_scf=True, run_fci=False, run_cisd=True)
    energy_hf = float(mol.hf_energy)
    orbital_energies = np.asarray(mol.orbital_energies, dtype=float)

    n_electrons = mol.n_electrons
    n_spatial = mol.n_orbitals
    n_qubits = 2 * n_spatial
    dim = 1 << n_qubits
    use_dense = use_dense_subspace_ops(n_spatial, n_electrons)
    dim_sub = fixed_n_subspace_dim(n_spatial, n_electrons)

    h_interaction = mol.get_molecular_hamiltonian()
    h_fermion = get_fermion_operator(h_interaction)
    h_qubit = jordan_wigner(h_fermion)
    h_full = get_sparse_operator(h_qubit, n_qubits).tocsc()

    basis_bitstrings = [
        bitstring for bitstring in range(dim) if popcount_fn(bitstring) == n_electrons
    ]
    basis_idx = np.array(basis_bitstrings, dtype=int)
    h_sub = h_full[basis_idx, :][:, basis_idx].tocsc()
    evals, evecs = spla.eigsh(h_sub, k=1, which="SA")
    energy_fci = float(np.real(evals[0]))
    v_sub = evecs[:, 0]

    h_full_keep = h_full if use_dense else None
    psi_full: np.ndarray | None
    if use_dense:
        psi_full = np.zeros(dim, dtype=np.complex128)
        psi_full[basis_idx] = v_sub
        psi_full /= np.linalg.norm(psi_full)
    else:
        psi_full = None
        del h_full

    hf_bitstring = hf_bitstring_fn(n_electrons, n_spatial)
    energy_cisd, _, _ = solve_cisd_fn(h_sub, basis_bitstrings, hf_bitstring, n_qubits)

    if use_dense:
        gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_statevector(psi_full, n_spatial)
    else:
        gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_subspace_state(
            v_sub, basis_bitstrings, n_spatial
        )

    return {
        "mol": mol,
        "description": description,
        "basis": basis,
        "charge": charge,
        "multiplicity": multiplicity,
        "energy_hf": energy_hf,
        "orbital_energies": orbital_energies,
        "n_electrons": n_electrons,
        "n_spatial": n_spatial,
        "n_qubits": n_qubits,
        "h_sub": h_sub,
        "h_full": h_full_keep,
        "basis_bitstrings": basis_bitstrings,
        "energy_fci": energy_fci,
        "v_sub": v_sub,
        "psi_full": psi_full,
        "energy_cisd": energy_cisd,
        "use_dense": use_dense,
        "dim_sub": dim_sub,
        "gamma_a": gamma_a,
        "gamma_b": gamma_b,
        "gamma_ab": gamma_ab,
    }


def generate_and_save(
    molecule: str,
    x: float,
    cache_dir: str | Path = "hamiltonian_cache",
    *,
    basis: str = BASIS,
    charge: int = CHARGE,
    multiplicity: int = MULTIPLICITY,
    overwrite: bool = False,
    **geometry_kwargs: Any,
) -> Path:
    """Generate one geometry point and write HDF5 cache."""
    out_path = cache_path(molecule, x, cache_dir=cache_dir, **geometry_kwargs)
    if out_path.is_file() and not overwrite:
        print(f"[skip] {out_path.name} already exists")
        return out_path

    geometry, description = get_geometry_and_description(molecule, x, **geometry_kwargs)
    ref = build_reference_state_with_pyscf(
        geometry=geometry,
        description=description,
        basis=basis,
        charge=charge,
        multiplicity=multiplicity,
    )
    save_reference_state(
        ref,
        out_path,
        molecule=molecule,
        x=x,
        geometry_kwargs=geometry_kwargs,
    )
    print(
        f"[ok] {out_path.name} | E_HF={ref['energy_hf']:.8f} "
        f"E_FCI={ref['energy_fci']:.8f} dim_sub={ref['dim_sub']}"
    )
    return out_path


def generate_scan(
    molecule: str,
    grid=None,
    cache_dir: str | Path = "hamiltonian_cache",
    overwrite: bool = False,
    **kwargs: Any,
) -> list[Path]:
    """Generate HDF5 caches for all points on a molecule grid."""
    written: list[Path] = []
    for x, geom_kwargs in iter_scan_points(molecule, grid=grid, **kwargs):
        written.append(
            generate_and_save(
                molecule,
                x,
                cache_dir=cache_dir,
                overwrite=overwrite,
                **geom_kwargs,
            )
        )
    return written
