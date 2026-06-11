"""Geometry definitions, scan grids, and Hamiltonian cache naming."""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np

# Representative N2 bond lengths (Å): equilibrium, stretched, dissociative.
N2_BOND_EQUILIBRIUM = 1.2
N2_BOND_STRONGLY_CORRELATED = 1.4
N2_BOND_DISSOCIATIVE = 2.2
N2_REPRESENTATIVE_GRID = (
    N2_BOND_EQUILIBRIUM,
    N2_BOND_STRONGLY_CORRELATED,
    N2_BOND_DISSOCIATIVE,
)

SUPPORTED_MOLECULES = frozenset(
    {"lih", "h2o", "h4_linear", "h4_square", "h4_rectangle", "n2"}
)


def _encode_geometry_param(value: float, scale: int) -> str:
    """Encode a float as an integer string at fixed decimal resolution."""
    return str(int(round(float(value) * scale)))


def hamiltonian_cache_basename(molecule: str, x: float, **kwargs: Any) -> str:
    """
    Basename for cached Hamiltonian HDF5 files (no extension).

    Examples:
        lih x=1.60           -> lih_16
        h2o x=0.958, 104.5°  -> h2o_958_1045
        h4_rectangle x=1.1, ar=1.5 -> h4_rectangle_11_15
    """
    mol = molecule.lower()
    if mol not in SUPPORTED_MOLECULES:
        raise ValueError(
            f"Unsupported molecule '{molecule}'. "
            f"Choose from: {sorted(SUPPORTED_MOLECULES)}"
        )

    if mol == "lih":
        return f"lih_{_encode_geometry_param(x, 10)}"

    if mol == "h2o":
        angle = float(kwargs.get("hoh_angle_deg", 104.5))
        return f"h2o_{_encode_geometry_param(x, 1000)}_{_encode_geometry_param(angle, 10)}"

    if mol == "h4_rectangle":
        aspect_ratio = float(kwargs.get("aspect_ratio", 1.5))
        return (
            f"h4_rectangle_{_encode_geometry_param(x, 10)}"
            f"_{_encode_geometry_param(aspect_ratio, 10)}"
        )

    if mol in {"h4_linear", "h4_square", "n2"}:
        return f"{mol}_{_encode_geometry_param(x, 10)}"

    raise ValueError(f"Unsupported molecule '{molecule}'.")


def cache_filename(molecule: str, x: float, **kwargs: Any) -> str:
    """Return basename + '.h5' for a geometry point."""
    return f"{hamiltonian_cache_basename(molecule, x, **kwargs)}.h5"


def default_grid_for_molecule(molecule: str) -> np.ndarray:
    if molecule == "lih":
        return np.linspace(0.8, 6.0, 10)
    if molecule == "h2o":
        return np.linspace(0.958, 2.5, 10)
    if molecule == "h4_linear":
        return np.linspace(0.6, 5.0, 10)
    if molecule in {"h4_square", "h4_rectangle"}:
        return np.linspace(0.6, 3.0, 10)
    if molecule == "n2":
        return np.array(N2_REPRESENTATIVE_GRID, dtype=float)
    raise ValueError(f"Unsupported molecule '{molecule}'")


def build_lih_geometry(li_h_bond_angstrom: float):
    r = li_h_bond_angstrom / 2.0
    return [
        ("Li", (0.0, 0.0, -r)),
        ("H", (0.0, 0.0, +r)),
    ]


def build_h2o_geometry(oh_bond_angstrom: float, hoh_angle_deg: float = 104.5):
    angle_rad = math.radians(hoh_angle_deg / 2.0)
    xh = oh_bond_angstrom * math.sin(angle_rad)
    yh = oh_bond_angstrom * math.cos(angle_rad)
    return [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (xh, yh, 0.0)),
        ("H", (-xh, yh, 0.0)),
    ]


def build_h4_linear_geometry(h_h_bond_angstrom: float):
    d = h_h_bond_angstrom
    coords = [-1.5 * d, -0.5 * d, 0.5 * d, 1.5 * d]
    return [("H", (0.0, 0.0, z)) for z in coords]


def build_h4_square_geometry(side_angstrom: float):
    s = side_angstrom / 2.0
    return [
        ("H", (-s, -s, 0.0)),
        ("H", (+s, -s, 0.0)),
        ("H", (+s, +s, 0.0)),
        ("H", (-s, +s, 0.0)),
    ]


def build_h4_rectangle_geometry(long_side_angstrom: float, aspect_ratio: float = 1.5):
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive.")
    a = long_side_angstrom / 2.0
    b = (long_side_angstrom / aspect_ratio) / 2.0
    return [
        ("H", (-a, -b, 0.0)),
        ("H", (+a, -b, 0.0)),
        ("H", (+a, +b, 0.0)),
        ("H", (-a, +b, 0.0)),
    ]


def build_n2_geometry(bond_angstrom: float):
    r = bond_angstrom / 2.0
    return [
        ("N", (0.0, 0.0, -r)),
        ("N", (0.0, 0.0, +r)),
    ]


def get_geometry_and_description(molecule: str, x: float, **kwargs: Any) -> tuple[list, str]:
    mol = molecule.lower()

    if mol == "lih":
        return build_lih_geometry(x), f"LiH_Bond{x:.4f}"

    if mol == "h2o":
        angle = kwargs.get("hoh_angle_deg", 104.5)
        return build_h2o_geometry(x, hoh_angle_deg=angle), f"H2O_OH{x:.4f}"

    if mol == "h4_linear":
        return build_h4_linear_geometry(x), f"H4_linear_d{x:.4f}"

    if mol == "h4_square":
        return build_h4_square_geometry(x), f"H4_square_side{x:.4f}"

    if mol == "h4_rectangle":
        aspect_ratio = kwargs.get("aspect_ratio", 1.5)
        return (
            build_h4_rectangle_geometry(x, aspect_ratio=aspect_ratio),
            f"H4_rectangle_long{x:.4f}_ar{aspect_ratio:.3f}",
        )

    if mol == "n2":
        return build_n2_geometry(x), f"N2_Bond{x:.4f}"

    raise ValueError(
        f"Unsupported molecule '{molecule}'. "
        f"Choose from: {sorted(SUPPORTED_MOLECULES)}"
    )


def iter_scan_points(
    molecule: str,
    grid: Iterable[float] | None = None,
    **kwargs: Any,
) -> list[tuple[float, dict[str, Any]]]:
    """Return (x, geometry_kwargs) pairs for a molecule scan."""
    if grid is None:
        grid = default_grid_for_molecule(molecule)
    geom_kwargs = {k: v for k, v in kwargs.items() if k in ("hoh_angle_deg", "aspect_ratio")}
    return [(float(x), geom_kwargs) for x in grid]
