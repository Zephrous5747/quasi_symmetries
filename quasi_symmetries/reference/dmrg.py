"""Block2/pyblock2 DMRG reference generation.

This module is intentionally optional: importing quasi_symmetries does not
require Block2. Runtime errors mention the missing dependency or unsupported
operator API at the point the DMRG backend is selected.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from quasi_symmetries.config import CHARGE, MULTIPLICITY
from quasi_symmetries.hamiltonian.geometry import basis_cache_slug, get_geometry_and_description
from quasi_symmetries.symmetry.labels import (
    MoleculeSymmetryLabels,
    extract_labels_from_pyscf,
    molecule_point_group,
)


class DMRGBackendError(RuntimeError):
    """Raised when the optional Block2 backend cannot run."""


@dataclass(frozen=True)
class DMRGRunConfig:
    molecule: str
    x: float
    basis: str
    cache_root: Path
    hoh_angle_deg: float = 104.5
    bond_dimension: int = 512
    max_sweeps: int = 20
    charge: int = CHARGE
    multiplicity: int = MULTIPLICITY
    n_threads: int = 1
    overwrite: bool = False

    @property
    def spin(self) -> int:
        return int(self.multiplicity) - 1


@dataclass
class DMRGReference:
    metadata_path: Path
    cache_dir: Path
    metadata: dict[str, Any]
    symmetry_labels: MoleculeSymmetryLabels | None
    driver: Any | None = None
    mps: Any | None = None

    @property
    def n_spatial(self) -> int:
        return int(self.metadata["n_spatial"])

    @property
    def n_electrons(self) -> int:
        return int(self.metadata["n_electrons"])

    @property
    def energy_hf(self) -> float:
        return float(self.metadata["energy_hf"])

    @property
    def energy_dmrg(self) -> float:
        return float(self.metadata["energy_dmrg"])

    @property
    def parity_json_path(self) -> Path:
        return self.cache_dir / "canonical_parity_expectations.json"


def _geometry_kwargs(config: DMRGRunConfig) -> dict[str, Any]:
    if config.molecule.lower() == "h2o":
        return {"hoh_angle_deg": config.hoh_angle_deg}
    return {}


def _geometry_tag(x: float) -> str:
    return f"x_{x:.6g}".replace(".", "p")


def dmrg_cache_dir(config: DMRGRunConfig) -> Path:
    basis_slug = basis_cache_slug(config.basis) or "sto3g"
    return (
        Path(config.cache_root)
        / config.molecule.lower()
        / basis_slug
        / _geometry_tag(config.x)
    )


def _load_pyblock2():
    try:
        from pyblock2.driver.core import DMRGDriver, SymmetryTypes
    except ImportError as exc:
        raise DMRGBackendError(
            "pyblock2 is required for wavefunction=dmrg. Install Block2/pyblock2 "
            "in the Trillium venv before running this backend."
        ) from exc
    return DMRGDriver, SymmetryTypes


def _pyscf_integrals_and_labels(config: DMRGRunConfig):
    from pyscf import ao2mo, gto, scf

    geometry, description = get_geometry_and_description(
        config.molecule,
        config.x,
        **_geometry_kwargs(config),
    )
    point_group = molecule_point_group(config.molecule)
    mol = gto.M(
        atom=[(atom, coords) for atom, coords in geometry],
        basis=config.basis,
        symmetry=point_group,
        charge=config.charge,
        spin=config.spin,
    )
    mol.build()
    mf = scf.RHF(mol).run(verbose=0)
    mo = np.asarray(mf.mo_coeff, dtype=float, order="C")
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.kernel(mol, mo)
    g2e = ao2mo.restore(1, eri, mo.shape[1])
    labels = extract_labels_from_pyscf(
        geometry,
        config.molecule,
        basis=config.basis,
        charge=config.charge,
        spin=config.spin,
    )
    return {
        "description": description,
        "mol": mol,
        "mf": mf,
        "h1e": np.asarray(h1e, dtype=float),
        "g2e": np.asarray(g2e, dtype=float),
        "ecore": float(mol.energy_nuc()),
        "labels": labels,
    }


def _block2_orb_sym(labels: MoleculeSymmetryLabels) -> list[int] | None:
    """Best-effort Block2 irrep IDs; let Block2 run without them if mapping fails."""
    try:
        from pyscf import symm
    except ImportError:
        return None

    out: list[int] = []
    for label in labels.irrep_labels:
        try:
            out.append(int(symm.irrep_name2id(labels.point_group, label)))
        except Exception:
            return None
    return out


def _sweep_schedule(config: DMRGRunConfig) -> tuple[list[int], list[float], list[float]]:
    max_bond = max(1, int(config.bond_dimension))
    ramp = sorted({max(32, max_bond // 4), max(64, max_bond // 2), max_bond})
    bond_dims = [dim for dim in ramp for _ in range(max(1, config.max_sweeps // len(ramp)))]
    while len(bond_dims) < config.max_sweeps:
        bond_dims.append(max_bond)
    noises = [1e-4] * min(4, config.max_sweeps) + [1e-5] * max(0, config.max_sweeps - 8)
    while len(noises) < config.max_sweeps:
        noises.append(0.0)
    thresholds = [1e-7] * min(4, config.max_sweeps) + [1e-8] * max(0, config.max_sweeps - 8)
    while len(thresholds) < config.max_sweeps:
        thresholds.append(1e-9)
    return bond_dims[: config.max_sweeps], noises[: config.max_sweeps], thresholds[: config.max_sweeps]


def run_block2_dmrg(config: DMRGRunConfig) -> DMRGReference:
    cache_dir = dmrg_cache_dir(config)
    metadata_path = cache_dir / "metadata.json"
    if metadata_path.is_file() and not config.overwrite:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        labels = None
        if "irrep_labels" in metadata:
            labels = MoleculeSymmetryLabels(
                molecule=metadata["molecule"],
                point_group=metadata["point_group"],
                irrep_labels=tuple(metadata["irrep_labels"]),
                inversion_parity=tuple(metadata.get("inversion_parity", [1] * int(metadata["n_spatial"]))),
                mo_coefficients=None,
            )
        return DMRGReference(metadata_path, cache_dir, metadata, labels)

    if config.overwrite and cache_dir.is_dir():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scratch = cache_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)

    DMRGDriver, SymmetryTypes = _load_pyblock2()
    payload = _pyscf_integrals_and_labels(config)
    labels: MoleculeSymmetryLabels = payload["labels"]
    mol = payload["mol"]
    n_spatial = int(payload["h1e"].shape[0])
    n_electrons = int(mol.nelectron)
    dim_sub = math.comb(2 * n_spatial, n_electrons)
    orb_sym = _block2_orb_sym(labels)

    driver = DMRGDriver(
        scratch=str(scratch),
        symm_type=SymmetryTypes.SZ,
        n_threads=int(config.n_threads),
    )
    initialize_kwargs: dict[str, Any] = {
        "n_sites": n_spatial,
        "n_elec": n_electrons,
        "spin": config.spin,
    }
    if orb_sym is not None:
        initialize_kwargs["orb_sym"] = orb_sym
    driver.initialize_system(**initialize_kwargs)

    mpo = driver.get_qc_mpo(
        h1e=payload["h1e"],
        g2e=payload["g2e"],
        ecore=float(payload["ecore"]),
        iprint=1,
    )
    ket = driver.get_random_mps(tag="GS", bond_dim=int(config.bond_dimension), nroots=1)
    bond_dims, noises, thresholds = _sweep_schedule(config)
    energy = driver.dmrg(
        mpo,
        ket,
        n_sweeps=int(config.max_sweeps),
        bond_dims=bond_dims,
        noises=noises,
        thrds=thresholds,
        iprint=1,
    )

    metadata = {
        "wavefunction_backend": "dmrg",
        "reference_energy_label": "E_DMRG",
        "molecule": config.molecule.lower(),
        "basis": config.basis,
        "geometry_param": float(config.x),
        "geometry_kwargs": _geometry_kwargs(config),
        "description": payload["description"],
        "point_group": labels.point_group,
        "irrep_labels": list(labels.irrep_labels),
        "inversion_parity": list(labels.inversion_parity),
        "n_spatial": n_spatial,
        "n_qubits": 2 * n_spatial,
        "n_electrons": n_electrons,
        "dim_sub": dim_sub,
        "energy_hf": float(payload["mf"].e_tot),
        "energy_dmrg": float(energy),
        "bond_dimension": int(config.bond_dimension),
        "max_sweeps": int(config.max_sweeps),
        "sweep_bond_dimensions": bond_dims,
        "sweep_noises": noises,
        "sweep_thresholds": thresholds,
        "block2_scratch": str(scratch),
        "orb_sym": orb_sym,
        "config": {**asdict(config), "cache_root": str(config.cache_root)},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return DMRGReference(metadata_path, cache_dir, metadata, labels, driver=driver, mps=ket)
