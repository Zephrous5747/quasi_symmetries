"""Symmetry-adapted MO irrep labels and molecule-specific operator maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

# C2v mirror characters: which irreps are odd under each reflection.
# Convention matches PySCF C2v with the molecular plane in xy and C2 along z.
_C2V_MIRROR_A = frozenset({"A2", "B2"})  # sigma_a
_C2V_MIRROR_B = frozenset({"A2", "B1"})  # sigma_b

# Documented STO-3G fallback labels (energy ordering, symmetry-adapted MOs).
_STO3G_FALLBACK_IRREPS: dict[str, list[str]] = {
    "h2o": ["A1", "A1", "B2", "A1", "B1", "A1", "B1"],
    "n2": ["Ag", "Ag", "B1u", "B2u", "B3u", "Ag", "B1u", "B2u", "B3u", "Ag"],
}

_POINT_GROUPS = {
    "h2o": "C2v",
    # Finite D2h subgroup used by PySCF for homonuclear diatomics (sigma/g/u labels).
    "n2": "D2h",
}


def molecule_point_group(molecule: str) -> str | None:
    return _POINT_GROUPS.get(molecule.lower())


def normalize_irrep_label(label: str) -> str:
    text = str(label).strip()
    if not text:
        return text
    if text[-1].lower() in {"g", "u"} and len(text) >= 2:
        return text[0].upper() + text[1:-1] + text[-1].lower()
    if len(text) >= 2 and text[0].isalpha() and text[1].isdigit():
        return text[0].upper() + text[1:]
    return text.upper()


def fallback_sto3g_irrep_labels(molecule: str, n_spatial: int) -> list[str]:
    mol = molecule.lower()
    labels = _STO3G_FALLBACK_IRREPS.get(mol)
    if labels is None:
        raise ValueError(f"No fallback irrep table for molecule '{molecule}'.")
    if len(labels) != n_spatial:
        raise ValueError(
            f"Fallback irrep count {len(labels)} does not match n_spatial={n_spatial} for {molecule}."
        )
    return [normalize_irrep_label(item) for item in labels]


def inversion_parity_from_irrep(label: str) -> int:
    """Return +1 (gerade) or -1 (ungerade) for diatomic D2h labels."""
    normalized = normalize_irrep_label(label)
    last = normalized[-1].lower()
    if last == "u":
        return -1
    if last == "g":
        return +1
    raise ValueError(f"Cannot infer inversion parity from irrep label '{label}'.")


def irrep_blocks(irrep_labels: Iterable[str]) -> list[list[int]]:
    """Group spatial orbital indices by normalized irrep label."""
    blocks: dict[str, list[int]] = {}
    for index, label in enumerate(irrep_labels):
        blocks.setdefault(normalize_irrep_label(label), []).append(index)
    return [blocks[key] for key in sorted(blocks)]


def symmetry_adapted_pair_list(n_spatial: int, irrep_labels: Iterable[str]) -> list[tuple[int, int]]:
    """Givens rotation pairs that preserve symmetry-adapted orbital blocks."""
    labels = [normalize_irrep_label(label) for label in irrep_labels]
    pairs: list[tuple[int, int]] = []
    for block in irrep_blocks(labels):
        for left in range(len(block)):
            for right in range(left + 1, len(block)):
                pairs.append((block[left], block[right]))
    if len(pairs) != len(set(pairs)):
        raise ValueError("Duplicate symmetry-adapted rotation pairs generated.")
    if n_spatial and not pairs and n_spatial > 1:
        # Single-orbital irreps only: no intra-block rotations.
        return []
    return pairs


def _indices_with_irrep(irrep_labels: Iterable[str], target: str) -> list[int]:
    target_n = normalize_irrep_label(target)
    return [
        index
        for index, label in enumerate(irrep_labels)
        if normalize_irrep_label(label) == target_n
    ]


def h2o_mirror_orbital_sets(irrep_labels: Iterable[str]) -> dict[str, tuple[int, ...]]:
    """
    Orbital sets whose pair-parity product implements each C2v mirror.

    Uses the character table when A2 is absent (STO-3G minimal basis).
    When A2, B1, and B2 are all present, also exposes Q1=b1*a2 and Q2=b2*a2.
    """
    labels = [normalize_irrep_label(label) for label in irrep_labels]
    sigma_a = tuple(index for index, label in enumerate(labels) if label in _C2V_MIRROR_A)
    sigma_b = tuple(index for index, label in enumerate(labels) if label in _C2V_MIRROR_B)

    a2_indices = _indices_with_irrep(labels, "A2")
    b1_indices = _indices_with_irrep(labels, "B1")
    b2_indices = _indices_with_irrep(labels, "B2")

    result: dict[str, tuple[int, ...]] = {
        "sigma_a": sigma_a,
        "sigma_b": sigma_b,
    }
    if a2_indices and b1_indices:
        result["Q1"] = (b1_indices[0], a2_indices[0])
    if a2_indices and b2_indices:
        result["Q2"] = (b2_indices[0], a2_indices[0])
    return result


def h2o_exact_quartet_edges(irrep_labels: Iterable[str]) -> list[tuple[int, int]]:
    """Exact H2O mirror quartets when the minimal b+a2 pairs exist."""
    mirrors = h2o_mirror_orbital_sets(irrep_labels)
    edges: list[tuple[int, int]] = []
    for key in ("Q1", "Q2"):
        orbitals = mirrors.get(key)
        if orbitals and len(orbitals) == 2:
            p, q = int(orbitals[0]), int(orbitals[1])
            edges.append((p, q) if p < q else (q, p))
    return edges


def n2_ungerade_orbital_indices(irrep_labels: Iterable[str]) -> list[int]:
    return [
        index
        for index, label in enumerate(irrep_labels)
        if inversion_parity_from_irrep(label) < 0
    ]


@dataclass(frozen=True)
class MoleculeSymmetryLabels:
    molecule: str
    point_group: str
    irrep_labels: tuple[str, ...]
    inversion_parity: tuple[int, ...]
    mo_coefficients: np.ndarray | None = None
  # symmetry-adapted MO coefficients when available

    def validate(self, n_spatial: int) -> None:
        if len(self.irrep_labels) != n_spatial:
            raise ValueError(
                f"Expected {n_spatial} irrep labels, got {len(self.irrep_labels)}."
            )
        if len(self.inversion_parity) != n_spatial:
            raise ValueError(
                f"Expected {n_spatial} inversion parities, got {len(self.inversion_parity)}."
            )

    def irrep_blocks(self) -> list[list[int]]:
        return irrep_blocks(self.irrep_labels)

    def symmetry_pairs(self, n_spatial: int) -> list[tuple[int, int]]:
        return symmetry_adapted_pair_list(n_spatial, self.irrep_labels)

    def h2o_a1_indices(self) -> list[int]:
        return _indices_with_irrep(self.irrep_labels, "A1")

    def h2o_mirror_sets(self) -> dict[str, tuple[int, ...]]:
        return h2o_mirror_orbital_sets(self.irrep_labels)

    def n2_ungerade_indices(self) -> list[int]:
        return n2_ungerade_orbital_indices(self.irrep_labels)


def _inversion_parities_for_labels(irrep_labels: Iterable[str]) -> list[int]:
    values: list[int] = []
    for label in irrep_labels:
        normalized = normalize_irrep_label(label)
        if normalized.endswith(("g", "u")):
            values.append(inversion_parity_from_irrep(normalized))
        else:
            values.append(+1)
    return values


def labels_from_irrep_list(molecule: str, irrep_labels: Iterable[str]) -> MoleculeSymmetryLabels:
    labels = tuple(normalize_irrep_label(label) for label in irrep_labels)
    point_group = molecule_point_group(molecule)
    if point_group is None:
        raise ValueError(f"Molecule '{molecule}' has no configured point group.")
    return MoleculeSymmetryLabels(
        molecule=molecule.lower(),
        point_group=point_group,
        irrep_labels=labels,
        inversion_parity=tuple(_inversion_parities_for_labels(labels)),
        mo_coefficients=None,
    )


def _mo_irrep_labels_from_pyscf(mol, mo_coeff: np.ndarray) -> list[str]:
    """Return normalized irrep label for each MO column (PySCF version tolerant)."""
    from pyscf import symm
    import inspect

    mo = np.asarray(mo_coeff, dtype=np.complex128)
    signature = inspect.signature(symm.label_orb_symm)
    if len(signature.parameters) >= 4:
        raw = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
        if isinstance(raw, (str, int)):
            raw = [raw]
        return [normalize_irrep_label(label) for label in raw]

    return [
        normalize_irrep_label(symm.label_orb_symm(mol, mo[:, index]))
        for index in range(mol.nao)
    ]


def extract_labels_from_pyscf(
    geometry: list[tuple[str, tuple[float, float, float]]],
    molecule: str,
    *,
    basis: str,
    charge: int,
    spin: int,
) -> MoleculeSymmetryLabels:
    """Build symmetry labels from a PySCF symmetry-adapted SCF calculation."""
    from pyscf import gto, scf, symm

    mol_name = molecule.lower()
    point_group = molecule_point_group(mol_name)
    if point_group is None:
        raise ValueError(f"No point group configured for molecule '{molecule}'.")

    mol = gto.M(
        atom=[(atom, coords) for atom, coords in geometry],
        basis=basis,
        symmetry=point_group,
        charge=charge,
        spin=spin,
    )
    mol.build()
    mf = scf.RHF(mol).run(verbose=0)
    irrep_labels = _mo_irrep_labels_from_pyscf(mol, mf.mo_coeff)
    return MoleculeSymmetryLabels(
        molecule=mol_name,
        point_group=point_group,
        irrep_labels=tuple(irrep_labels),
        inversion_parity=tuple(_inversion_parities_for_labels(irrep_labels)),
        mo_coefficients=np.asarray(mf.mo_coeff, dtype=np.complex128),
    )


def symmetry_labels_available(ref: dict[str, Any]) -> bool:
    labels = ref.get("symmetry_labels")
    return isinstance(labels, MoleculeSymmetryLabels)


def load_symmetry_labels(
    ref: dict[str, Any],
    *,
    molecule: str | None = None,
) -> MoleculeSymmetryLabels | None:
    """Return symmetry labels from a reference dict, with fallback tables if needed."""
    cached = ref.get("symmetry_labels")
    if isinstance(cached, MoleculeSymmetryLabels):
        cached.validate(int(ref["n_spatial"]))
        return cached

    mol = (molecule or ref.get("meta", {}).get("molecule") or "").lower()
    if mol not in _POINT_GROUPS:
        return None

    n_spatial = int(ref["n_spatial"])
    try:
        labels = fallback_sto3g_irrep_labels(mol, n_spatial)
    except ValueError:
        return None
    return labels_from_irrep_list(mol, labels)
