"""Small backend boundary for canonical reference-state heatmaps.

The existing exact-CI cache remains the source of truth for ``wavefunction=exact``.
DMRG backends provide the same metadata and a canonical variance matrix without
pretending to own a full determinant vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quasi_symmetries.config import normalized_basis_slug
from quasi_symmetries.hamiltonian.cache import cache_path, load_reference_state
from quasi_symmetries.hamiltonian.geometry import basis_cache_slug
from quasi_symmetries.optimization import closed_shell_hf_bitstring, popcount, solve_cisd_state
from quasi_symmetries.symmetry.labels import MoleculeSymmetryLabels, load_symmetry_labels
from scripts.plot.orbital_heatmaps import parity_variance_matrix


@dataclass(frozen=True)
class ReferenceMetadata:
    molecule: str
    basis: str
    geometry_param: float
    n_spatial: int
    n_electrons: int
    energy_hf: float
    reference_energy: float
    wavefunction_backend: str
    reference_energy_label: str
    dim_sub: int | None = None
    cache_path: str | None = None
    bond_dimension: int | None = None
    max_sweeps: int | None = None


def default_backend_output_dir(root: Path, molecule: str, basis: str, backend: str) -> Path:
    return (
        root
        / molecule.lower()
        / normalized_basis_slug(basis)
        / "canonical"
        / backend.lower()
    )


class ExactCIReference:
    """Adapter for the existing HDF5 exact-CI reference cache."""

    wavefunction_backend = "exact"
    reference_energy_label = "E_FCI"

    def __init__(self, ref: dict[str, Any]) -> None:
        self.ref = ref
        meta = ref.get("meta", {})
        self.metadata = ReferenceMetadata(
            molecule=str(meta.get("molecule", "h2o")),
            basis=str(meta.get("basis", "sto-3g")),
            geometry_param=float(meta.get("geometry_param", 0.0)),
            n_spatial=int(ref["n_spatial"]),
            n_electrons=int(ref["n_electrons"]),
            energy_hf=float(ref["energy_hf"]),
            reference_energy=float(ref["energy_fci"]),
            wavefunction_backend=self.wavefunction_backend,
            reference_energy_label=self.reference_energy_label,
            dim_sub=int(ref["dim_sub"]),
            cache_path=str(ref.get("cache_path", "")),
        )

    @classmethod
    def from_cache_or_generate(
        cls,
        *,
        molecule: str,
        x: float,
        basis: str,
        cache_dir: str | Path,
        overwrite: bool,
        skip_generation: bool,
        geometry_kwargs: dict[str, Any],
    ) -> "ExactCIReference":
        cache_kwargs = {**geometry_kwargs, "basis": basis}
        if not skip_generation:
            from quasi_symmetries.hamiltonian.generation import generate_and_save

            generate_and_save(
                molecule,
                x,
                cache_dir=cache_dir,
                basis=basis,
                overwrite=overwrite,
                **geometry_kwargs,
            )
        else:
            expected = cache_path(molecule, x, cache_dir=cache_dir, **cache_kwargs)
            if not expected.is_file():
                raise FileNotFoundError(f"Missing exact-CI cache: {expected}")

        ref = load_reference_state(
            molecule,
            x,
            cache_dir=cache_dir,
            load_hamiltonian=False,
            load_full_hamiltonian=False,
            compute_rdms=False,
            popcount_fn=popcount,
            solve_cisd_fn=solve_cisd_state,
            hf_bitstring_fn=closed_shell_hf_bitstring,
            **cache_kwargs,
        )
        meta_basis = str(ref.get("meta", {}).get("basis", "")).strip().lower()
        expected_basis = basis.strip().lower()
        if meta_basis not in {"", expected_basis}:
            raise ValueError(
                f"Cache basis mismatch: metadata has {meta_basis!r}, expected {basis!r}."
            )
        return cls(ref)

    @property
    def symmetry_labels(self) -> MoleculeSymmetryLabels | None:
        return load_symmetry_labels(self.ref, molecule=self.metadata.molecule)

    def canonical_variance_matrix(self) -> np.ndarray:
        n_spatial = self.metadata.n_spatial
        u_canonical = np.eye(n_spatial, dtype=np.complex128)
        return parity_variance_matrix(
            self.ref["v_sub"],
            self.ref["basis_bitstrings"],
            u_canonical,
            n_spatial,
        )
