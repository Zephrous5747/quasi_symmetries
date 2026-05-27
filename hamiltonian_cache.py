"""Load precomputed fixed-N Hamiltonian data from HDF5 (no PySCF required)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sp

from hamiltonian_geometry import cache_filename, hamiltonian_cache_basename
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


DEFAULT_CACHE_DIR = "hamiltonian_cache"


def cache_path(
    molecule: str,
    x: float,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    **kwargs: Any,
) -> Path:
    return Path(cache_dir) / cache_filename(molecule, x, **kwargs)


def list_cached_hamiltonians(cache_dir: str | Path = DEFAULT_CACHE_DIR) -> list[str]:
    root = Path(cache_dir)
    if not root.is_dir():
        return []
    return sorted(p.stem for p in root.glob("*.h5"))


def save_reference_state(
    ref: dict[str, Any],
    path: str | Path,
    *,
    molecule: str,
    x: float,
    geometry_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Write reference-state dict produced by hamiltonian_generation.build_reference_state_with_pyscf."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    h_sub = ref["h_sub"].tocsr()
    v_sub = np.asarray(ref["v_sub"], dtype=np.complex128).ravel()
    basis_bitstrings = np.asarray(ref["basis_bitstrings"], dtype=np.int64)
    orbital_energies = np.asarray(ref.get("orbital_energies", []), dtype=float)

    meta = {
        "molecule": molecule.lower(),
        "geometry_param": float(x),
        "geometry_kwargs": geometry_kwargs or {},
        "basis": ref.get("basis", BASIS),
        "charge": int(ref.get("charge", CHARGE)),
        "multiplicity": int(ref.get("multiplicity", MULTIPLICITY)),
        "description": ref.get("description", ""),
        "basename": hamiltonian_cache_basename(molecule, x, **(geometry_kwargs or {})),
    }

    with h5py.File(path, "w") as handle:
        handle.attrs["meta_json"] = json.dumps(meta)
        handle.attrs["energy_hf"] = float(ref["energy_hf"])
        handle.attrs["energy_fci"] = float(ref["energy_fci"])
        handle.attrs["energy_cisd"] = float(ref["energy_cisd"])
        handle.attrs["n_electrons"] = int(ref["n_electrons"])
        handle.attrs["n_spatial"] = int(ref["n_spatial"])
        handle.attrs["n_qubits"] = int(ref["n_qubits"])
        handle.attrs["use_dense"] = bool(ref["use_dense"])
        handle.attrs["dim_sub"] = int(ref["dim_sub"])

        handle.create_dataset("h_sub_data", data=h_sub.data)
        handle.create_dataset("h_sub_indices", data=h_sub.indices)
        handle.create_dataset("h_sub_indptr", data=h_sub.indptr)
        handle.attrs["h_sub_shape"] = h_sub.shape

        handle.create_dataset("v_sub", data=v_sub)
        handle.create_dataset("basis_bitstrings", data=basis_bitstrings)
        if orbital_energies.size:
            handle.create_dataset("orbital_energies", data=orbital_energies)

        if ref.get("h_full") is not None:
            h_full = ref["h_full"].tocsr()
            handle.create_dataset("h_full_data", data=h_full.data)
            handle.create_dataset("h_full_indices", data=h_full.indices)
            handle.create_dataset("h_full_indptr", data=h_full.indptr)
            handle.attrs["h_full_shape"] = h_full.shape

    return path


def load_reference_state(
    molecule: str,
    x: float,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    popcount_fn=popcount,
    solve_cisd_fn=solve_cisd_state,
    hf_bitstring_fn=closed_shell_hf_bitstring,
    load_hamiltonian: bool = True,
    load_full_hamiltonian: bool = True,
    compute_rdms: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Load a precomputed reference state from HDF5.

    By default, recomputes RDMs from the stored FCI vector (no PySCF).
    Memory-heavy consumers can disable Hamiltonian loading and RDM construction
    when they only need the fixed-N FCI vector and determinant basis.
    """
    path = cache_path(molecule, x, cache_dir=cache_dir, **kwargs)
    if not path.is_file():
        basename = hamiltonian_cache_basename(molecule, x, **kwargs)
        raise FileNotFoundError(
            f"Hamiltonian cache not found: {path}\n"
            f"Expected basename '{basename}.h5'. "
            f"Generate it with generate_hamiltonians.py on a machine with PySCF."
        )

    with h5py.File(path, "r") as handle:
        meta = json.loads(handle.attrs["meta_json"])
        energy_hf = float(handle.attrs["energy_hf"])
        energy_fci = float(handle.attrs["energy_fci"])
        energy_cisd = float(handle.attrs["energy_cisd"])
        n_electrons = int(handle.attrs["n_electrons"])
        n_spatial = int(handle.attrs["n_spatial"])
        n_qubits = int(handle.attrs["n_qubits"])
        use_dense = bool(handle.attrs["use_dense"])
        dim_sub = int(handle.attrs["dim_sub"])

        h_sub = None
        if load_hamiltonian:
            shape = tuple(int(v) for v in handle.attrs["h_sub_shape"])
            h_sub = sp.csr_matrix(
                (
                    handle["h_sub_data"][()],
                    handle["h_sub_indices"][()],
                    handle["h_sub_indptr"][()],
                ),
                shape=shape,
            ).tocsc()

        v_sub = np.asarray(handle["v_sub"][()], dtype=np.complex128)
        basis_bitstrings = [int(b) for b in handle["basis_bitstrings"][()]]
        orbital_energies = (
            np.asarray(handle["orbital_energies"][()], dtype=float)
            if "orbital_energies" in handle
            else np.array([], dtype=float)
        )

        h_full = None
        if load_full_hamiltonian and "h_full_data" in handle:
            full_shape = tuple(int(v) for v in handle.attrs["h_full_shape"])
            h_full = sp.csr_matrix(
                (
                    handle["h_full_data"][()],
                    handle["h_full_indices"][()],
                    handle["h_full_indptr"][()],
                ),
                shape=full_shape,
            ).tocsc()

    dim = 1 << n_qubits

    psi_full: np.ndarray | None
    if compute_rdms and use_dense:
        basis_idx = np.array(basis_bitstrings, dtype=int)
        psi_full = np.zeros(dim, dtype=np.complex128)
        psi_full[basis_idx] = v_sub
        psi_full /= np.linalg.norm(psi_full)
    else:
        psi_full = None

    if compute_rdms and use_dense and psi_full is not None:
        gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_statevector(psi_full, n_spatial)
    elif compute_rdms:
        gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_subspace_state(
            v_sub, basis_bitstrings, n_spatial
        )
    else:
        gamma_a = gamma_b = gamma_ab = None

    if not use_dense and compute_rdms:
        print(
            f"  [memory] fixed-N subspace dim={dim_sub} > dense limit; "
            "skipping dense H / R / entropy / sector-energy diagnostics."
        )

    return {
        "mol": None,
        "meta": meta,
        "energy_hf": energy_hf,
        "n_electrons": n_electrons,
        "n_spatial": n_spatial,
        "n_qubits": n_qubits,
        "h_sub": h_sub,
        "h_full": h_full,
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
        "orbital_energies": orbital_energies,
        "cache_path": str(path),
    }


def verify_cache_matches_request(
    molecule: str,
    x: float,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    **kwargs: Any,
) -> Path:
    """Load metadata only and confirm basename matches requested geometry."""
    path = cache_path(molecule, x, cache_dir=cache_dir, **kwargs)
    if not path.is_file():
        raise FileNotFoundError(f"Missing cache file: {path}")
    expected = hamiltonian_cache_basename(molecule, x, **kwargs)
    with h5py.File(path, "r") as handle:
        meta = json.loads(handle.attrs["meta_json"])
    if meta.get("basename") != expected:
        raise ValueError(
            f"Cache basename mismatch for {path}: "
            f"file has '{meta.get('basename')}', expected '{expected}'."
        )
    return path
