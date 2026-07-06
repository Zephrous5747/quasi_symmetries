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

from quasi_symmetries.config import BASIS, CHARGE, DEFAULT_CACHE_DIR, MULTIPLICITY
from quasi_symmetries.fermion.bitstring import apply_annihilate, apply_create, closed_shell_hf_bitstring
from quasi_symmetries.hamiltonian.cache import cache_path, save_reference_state
from quasi_symmetries.hamiltonian.geometry import get_geometry_and_description, iter_scan_points
from quasi_symmetries.symmetry.labels import (
    MoleculeSymmetryLabels,
    extract_labels_from_pyscf,
    fallback_sto3g_irrep_labels,
    labels_from_irrep_list,
    molecule_point_group,
)
from quasi_symmetries.optimization import (
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


def _attach_symmetry_labels(
    *,
    molecule: str | None,
    geometry: list,
    n_spatial: int,
    basis: str,
    charge: int,
    multiplicity: int,
) -> MoleculeSymmetryLabels | None:
    mol_name = (molecule or "").lower()
    if molecule_point_group(mol_name) is None:
        return None
    spin = multiplicity - 1
    try:
        return extract_labels_from_pyscf(
            geometry,
            mol_name,
            basis=basis,
            charge=charge,
            spin=spin,
        )
    except ImportError:
        pass
    except Exception as exc:
        print(f"  [symmetry] PySCF symmetry labeling failed ({exc}); using fallback table.")
    try:
        labels = fallback_sto3g_irrep_labels(mol_name, n_spatial)
    except ValueError:
        return None
    return labels_from_irrep_list(mol_name, labels)


def _pyscf_nuclear_energy(mol, mf) -> float:
    """Return nuclear repulsion energy across PySCF versions."""
    for source in (mf, mol):
        enuc = getattr(source, "energy_nuc", None)
        if enuc is None:
            continue
        if callable(enuc):
            enuc = enuc()
        return float(enuc)
    raise AttributeError("Could not obtain PySCF nuclear repulsion energy.")


def _real_mo_coefficients(mf) -> np.ndarray:
    """Return real MO coefficients for ao2mo (PySCF rejects complex128 arrays)."""
    mo = np.asarray(mf.mo_coeff)
    if np.iscomplexobj(mo):
        imag_max = float(np.max(np.abs(np.imag(mo))))
        if imag_max > 1e-8:
            raise ValueError(
                f"MO coefficients have significant imaginary part (max|Im|={imag_max:.2e})."
            )
        mo = mo.real
    return np.asarray(mo, dtype=np.float64, order="C")


def _pyscf_mo_integrals(mol, mf) -> tuple[np.ndarray, np.ndarray]:
    """One- and two-body MO integrals in OpenFermion InteractionOperator layout."""
    from pyscf import ao2mo

    mo = _real_mo_coefficients(mf)
    nmo = mo.shape[1]
    h1 = mo.T @ mf.get_hcore() @ mo
    eri_compressed = ao2mo.kernel(mol, mo)
    if getattr(eri_compressed, "ndim", 1) == 4:
        eri_mo = np.asarray(eri_compressed, dtype=float)
    else:
        eri_mo = ao2mo.restore(1, eri_compressed, nmo)
    # OpenFermion: h[p,q,r,s] = (ps|qr) = pyscf_eri[p,s,q,r]
    two_body = np.asarray(eri_mo.transpose(0, 2, 3, 1), order="C", dtype=float)
    return h1, two_body


def _build_fermion_hamiltonian_from_pyscf_mf(mol, mf):
    """Build an OpenFermion FermionOperator in the SCF MO basis."""
    from openfermion import InteractionOperator, get_fermion_operator
    from openfermion.chem.molecular_data import spinorb_from_spatial

    h1_spatial, two_body_spatial = _pyscf_mo_integrals(mol, mf)
    one_body, two_body = spinorb_from_spatial(h1_spatial, two_body_spatial)
    interaction = InteractionOperator(
        constant=_pyscf_nuclear_energy(mol, mf),
        one_body_tensor=one_body,
        two_body_tensor=0.5 * two_body,
    )
    return get_fermion_operator(interaction)


def hf_energy_on_subspace(
    h_sub: sp.spmatrix,
    basis_bitstrings: list[int],
    *,
    n_electrons: int,
    n_spatial: int,
    hf_bitstring_fn=closed_shell_hf_bitstring,
) -> float:
    """Expectation value of the HF determinant on the fixed-N subspace."""
    n_qubits = 2 * n_spatial
    hf_bitstring = int(hf_bitstring_fn(n_electrons, n_spatial))
    if hf_bitstring not in basis_bitstrings:
        raise ValueError("HF determinant is outside the fixed-N subspace.")
    index = basis_bitstrings.index(hf_bitstring)
    vector = np.zeros(len(basis_bitstrings), dtype=np.complex128)
    vector[index] = 1.0
    return float(np.real(np.vdot(vector, h_sub.dot(vector))))


def validate_reference_energies(
    ref: dict[str, Any],
    *,
    tol: float = 5e-4,
) -> dict[str, float]:
    """Return energy diagnostics and raise if basic variational bounds fail."""
    e_hf_sub = hf_energy_on_subspace(
        ref["h_sub"],
        ref["basis_bitstrings"],
        n_electrons=int(ref["n_electrons"]),
        n_spatial=int(ref["n_spatial"]),
    )
    e_hf = float(ref["energy_hf"])
    e_fci = float(ref["energy_fci"])
    e_cisd = float(ref["energy_cisd"])
    diagnostics = {
        "E_HF_reported": e_hf,
        "E_HF_on_sub": e_hf_sub,
        "E_FCI": e_fci,
        "E_CISD": e_cisd,
        "delta_HF_reported_vs_sub": e_hf - e_hf_sub,
        "delta_FCI_vs_HF_sub": e_fci - e_hf_sub,
    }
    if e_fci > e_hf_sub + tol:
        raise ValueError(
            "FCI energy exceeds HF energy on h_sub: "
            f"E_FCI={e_fci:.8f}, E_HF_on_sub={e_hf_sub:.8f}"
        )
    if abs(e_hf - e_hf_sub) > max(tol, 5e-3):
        raise ValueError(
            "Reported HF energy does not match <HF|H|HF> on h_sub: "
            f"reported={e_hf:.8f}, on_sub={e_hf_sub:.8f}"
        )
    return diagnostics


def _try_symmetry_adapted_hamiltonian(
    *,
    molecule: str,
    geometry: list,
    basis: str,
    charge: int,
    multiplicity: int,
) -> tuple[Any, MoleculeSymmetryLabels, float] | None:
    """When PySCF is available, rebuild H in symmetry-adapted MOs (C2v for H2O, D2h for N2)."""
    mol_name = molecule.lower()
    if molecule_point_group(mol_name) is None:
        return None
    try:
        from pyscf import gto, scf

        spin = multiplicity - 1
        labels = extract_labels_from_pyscf(
            geometry,
            mol_name,
            basis=basis,
            charge=charge,
            spin=spin,
        )
        mol = gto.M(
            atom=[(atom, coords) for atom, coords in geometry],
            basis=basis,
            symmetry=labels.point_group,
            charge=charge,
            spin=spin,
        )
        mol.build()
        mf = scf.RHF(mol).run(verbose=0)
        h_fermion = _build_fermion_hamiltonian_from_pyscf_mf(mol, mf)
        return h_fermion, labels, float(mf.e_tot)
    except ImportError:
        return None
    except Exception as exc:
        print(f"  [symmetry] Could not rebuild symmetry-adapted Hamiltonian: {exc}")
        return None


def build_reference_state_with_pyscf(
    *,
    geometry: list,
    description: str,
    basis: str = BASIS,
    charge: int = CHARGE,
    multiplicity: int = MULTIPLICITY,
    molecule: str | None = None,
    use_symmetry: bool = True,
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

    symmetry_labels: MoleculeSymmetryLabels | None = None
    symmetry_energy_hf: float | None = None
    h_fermion = None
    enforce_point_group = (
        use_symmetry
        and molecule is not None
        and molecule.lower() in {"h2o", "n2"}
    )
    if use_symmetry and molecule is not None:
        sym_payload = _try_symmetry_adapted_hamiltonian(
            molecule=molecule,
            geometry=geometry,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
        )
        if sym_payload is not None:
            h_fermion, symmetry_labels, symmetry_energy_hf = sym_payload
        elif enforce_point_group:
            pg = molecule_point_group(molecule)
            raise RuntimeError(
                f"Point-group symmetry ({pg}) adaptation failed for {molecule}; "
                "install PySCF and regenerate caches in symmetry-adapted MOs."
            )

    if h_fermion is None:
        h_interaction = mol.get_molecular_hamiltonian()
        h_fermion = get_fermion_operator(h_interaction)
    elif symmetry_energy_hf is not None:
        energy_hf = symmetry_energy_hf

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

    if symmetry_labels is None and use_symmetry and molecule is not None:
        symmetry_labels = _attach_symmetry_labels(
            molecule=molecule,
            geometry=geometry,
            n_spatial=n_spatial,
            basis=basis,
            charge=charge,
            multiplicity=multiplicity,
        )

    energy_checks = validate_reference_energies(
        {
            "h_sub": h_sub,
            "basis_bitstrings": basis_bitstrings,
            "n_electrons": n_electrons,
            "n_spatial": n_spatial,
            "energy_hf": energy_hf,
            "energy_fci": energy_fci,
            "energy_cisd": energy_cisd,
        }
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
        "symmetry_labels": symmetry_labels,
        "energy_checks": energy_checks,
    }


def generate_and_save(
    molecule: str,
    x: float,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    basis: str = BASIS,
    charge: int = CHARGE,
    multiplicity: int = MULTIPLICITY,
    overwrite: bool = False,
    **geometry_kwargs: Any,
) -> Path:
    """Generate one geometry point and write HDF5 cache."""
    cache_kwargs = {**geometry_kwargs, "basis": basis}
    out_path = cache_path(molecule, x, cache_dir=cache_dir, **cache_kwargs)
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
        molecule=molecule,
        use_symmetry=molecule.lower() in {"h2o", "n2"},
    )
    save_reference_state(
        ref,
        out_path,
        molecule=molecule,
        x=x,
        geometry_kwargs=cache_kwargs,
    )
    print(
        f"[ok] {out_path.name} | E_HF={ref['energy_hf']:.8f} "
        f"E_FCI={ref['energy_fci']:.8f} dim_sub={ref['dim_sub']}"
    )
    return out_path


def generate_scan(
    molecule: str,
    grid=None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    overwrite: bool = False,
    *,
    basis: str = BASIS,
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
                basis=basis,
                **geom_kwargs,
            )
        )
    return written
