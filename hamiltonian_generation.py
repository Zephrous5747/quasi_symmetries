"""PySCF-backed Hamiltonian generation for offline caching."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from openfermion import MolecularData, get_fermion_operator, get_sparse_operator, jordan_wigner
from openfermionpyscf import run_pyscf

from hamiltonian_cache import cache_path, save_reference_state
from hamiltonian_geometry import get_geometry_and_description, iter_scan_points
from optimization_abc_utils import (
    BASIS,
    CHARGE,
    MULTIPLICITY,
    apply_annihilate,
    apply_create,
    closed_shell_hf_bitstring,
    compute_spin_rdms_from_statevector,
    compute_spin_rdms_from_subspace_state,
    fixed_n_subspace_dim,
    popcount,
    solve_cisd_state,
    use_dense_subspace_ops,
)


def apply_fermion_term_to_bitstring(bitstring: int, term: tuple, n_qubits: int) -> tuple[int | None, int]:
    """Apply an OpenFermion term to a determinant bitstring."""
    out = int(bitstring)
    sign = 1
    for mode, action in reversed(term):
        if action == 0:
            out, term_sign = apply_annihilate(out, mode, n_qubits)
        else:
            out, term_sign = apply_create(out, mode, n_qubits)
        if out is None:
            return None, 0
        sign *= term_sign
    return out, sign


def build_fixed_n_hamiltonian_direct(h_fermion, basis_bitstrings: list[int], n_qubits: int) -> sp.csc_matrix:
    """Build the fixed-N Hamiltonian without allocating the full Fock-space operator."""
    det_to_idx = {int(bitstring): idx for idx, bitstring in enumerate(basis_bitstrings)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[complex] = []
    dim_sub = len(basis_bitstrings)

    for term, coef in h_fermion.terms.items():
        coef = complex(coef)
        if term == ():
            rows.extend(range(dim_sub))
            cols.extend(range(dim_sub))
            data.extend([coef] * dim_sub)
            continue

        for col, ket in enumerate(basis_bitstrings):
            bra, sign = apply_fermion_term_to_bitstring(ket, term, n_qubits)
            if bra is None:
                continue
            row = det_to_idx.get(int(bra))
            if row is None:
                continue
            rows.append(row)
            cols.append(col)
            data.append(coef * sign)

    h_sub = sp.coo_matrix((data, (rows, cols)), shape=(dim_sub, dim_sub), dtype=np.complex128)
    h_sub = h_sub.tocsc()
    h_sub = 0.5 * (h_sub + h_sub.getH())
    return h_sub.tocsc()


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
    compute_rdms: bool = False,
) -> dict[str, Any]:
    """Run PySCF and build fixed-N Hamiltonian, FCI ground state, and CISD reference."""
    mol = MolecularData(
        geometry=geometry,
        basis=basis,
        multiplicity=multiplicity,
        charge=charge,
        description=description,
    )
    mol = run_pyscf(mol, run_scf=True, run_fci=False, run_cisd=False)
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

    basis_bitstrings = [
        bitstring for bitstring in range(dim) if popcount_fn(bitstring) == n_electrons
    ]
    basis_idx = np.array(basis_bitstrings, dtype=int)

    h_full_keep = None
    if use_dense:
        h_qubit = jordan_wigner(h_fermion)
        h_full = get_sparse_operator(h_qubit, n_qubits).tocsc()
        h_sub = h_full[basis_idx, :][:, basis_idx].tocsc()
        h_full_keep = h_full
    else:
        print(
            f"  [memory] fixed-N subspace dim={dim_sub} > dense limit; "
            "building h_sub directly without h_full."
        )
        h_sub = build_fixed_n_hamiltonian_direct(h_fermion, basis_bitstrings, n_qubits)
        gc.collect()

    evals, evecs = spla.eigsh(h_sub, k=1, which="SA")
    energy_fci = float(np.real(evals[0]))
    v_sub = evecs[:, 0]

    psi_full: np.ndarray | None
    if compute_rdms and use_dense:
        psi_full = np.zeros(dim, dtype=np.complex128)
        psi_full[basis_idx] = v_sub
        psi_full /= np.linalg.norm(psi_full)
    else:
        psi_full = None

    hf_bitstring = hf_bitstring_fn(n_electrons, n_spatial)
    energy_cisd, _, _ = solve_cisd_fn(h_sub, basis_bitstrings, hf_bitstring, n_qubits)

    if compute_rdms and use_dense and psi_full is not None:
        gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_statevector(psi_full, n_spatial)
    elif compute_rdms:
        gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_subspace_state(
            v_sub, basis_bitstrings, n_spatial
        )
    else:
        gamma_a = gamma_b = gamma_ab = None

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
