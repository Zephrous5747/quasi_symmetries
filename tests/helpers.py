"""Shared fixtures for small-system parent-Hamiltonian tests."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from hamiltonian_cache import DEFAULT_CACHE_DIR, load_reference_state
from hamiltonian_geometry import get_geometry_and_description
from optimization_abc_utils import closed_shell_hf_bitstring, popcount, solve_cisd_state


H4_MOLECULE = "h4_linear"
H4_GEOMETRY = 1.1

LIH_MOLECULE = "lih"
LIH_GEOMETRY = 1.4

STANDARD_ABC = (
    1.0 / 6.0**0.5,
    1.0 / 6.0**0.5,
    -2.0 / 6.0**0.5,
)

VAR_TOL = 1e-6
OPT_MAXFEV = 3000
OPT_MAXITER = 400
N_RESTARTS = 3
K_COUPLED_TOL = 1e-3
PLANTED_SEED = 107

# LiH fixed-N subspace is larger; keep optimization budgets lighter in tests.
LIH_N_RESTARTS = 2
LIH_OPT_MAXFEV = 400
LIH_OPT_MAXITER = 150


def load_h4_reference(
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> dict:
    """Single-system fixture: H4 linear chain at 1.1 Angstrom."""
    return load_small_reference(H4_MOLECULE, H4_GEOMETRY, cache_dir=cache_dir)


def load_lih_reference(
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> dict:
    """Single-system fixture: LiH at 1.4 Angstrom (sto-3g cache lih_14.h5)."""
    return load_small_reference(LIH_MOLECULE, LIH_GEOMETRY, cache_dir=cache_dir)


def load_small_reference(
    molecule: str,
    x: float,
    *,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    **geometry_kwargs,
) -> dict:
    """Load cached reference state or build with PySCF for a small geometry."""
    cache_dir = Path(cache_dir)
    try:
        return load_reference_state(
            molecule,
            x,
            cache_dir=str(cache_dir),
            popcount_fn=popcount,
            solve_cisd_fn=solve_cisd_state,
            hf_bitstring_fn=closed_shell_hf_bitstring,
            **geometry_kwargs,
        )
    except FileNotFoundError:
        try:
            from hamiltonian_generation import build_reference_state_with_pyscf
        except ImportError as exc:
            raise FileNotFoundError(
                f"No cache for {molecule} x={x} and PySCF stack unavailable ({exc})."
            ) from exc
        geometry, description = get_geometry_and_description(molecule, x, **geometry_kwargs)
        return build_reference_state_with_pyscf(
            geometry=geometry,
            description=description,
            popcount_fn=popcount,
            solve_cisd_fn=solve_cisd_state,
            hf_bitstring_fn=closed_shell_hf_bitstring,
        )


def hf_state_vector(ref: dict) -> np.ndarray:
    """Closed-shell HF determinant in the fixed-N subspace ordering."""
    hf_bitstring = closed_shell_hf_bitstring(ref["n_electrons"], ref["n_spatial"])
    det_to_idx = {int(b): i for i, b in enumerate(ref["basis_bitstrings"])}
    idx = det_to_idx[int(hf_bitstring)]
    psi = np.zeros(len(ref["basis_bitstrings"]), dtype=np.complex128)
    psi[idx] = 1.0
    return psi


def molecular_fermion_hamiltonian(molecule: str, x: float, **geometry_kwargs) -> tuple[object, int, int]:
    """Return (h_fermion, n_spatial, n_qubits) for a small geometry via PySCF."""
    try:
        from openfermion import MolecularData, get_fermion_operator
        from openfermionpyscf import run_pyscf
        from optimization_abc_utils import BASIS, CHARGE, MULTIPLICITY
    except ImportError as exc:
        raise unittest.SkipTest(f"PySCF stack unavailable: {exc}") from exc

    geometry, description = get_geometry_and_description(molecule, x, **geometry_kwargs)
    mol = MolecularData(
        geometry=geometry,
        basis=BASIS,
        multiplicity=MULTIPLICITY,
        charge=CHARGE,
        description=description,
    )
    mol = run_pyscf(mol, run_scf=True, run_fci=False, run_cisd=False)
    h_fermion = get_fermion_operator(mol.get_molecular_hamiltonian())
    n_spatial = mol.n_orbitals
    return h_fermion, n_spatial, 2 * n_spatial


def physical_hamiltonian_dense(ref: dict) -> np.ndarray:
    """Fixed-N physical Hamiltonian as a dense matrix."""
    h_sub = ref["h_sub"]
    return np.asarray(
        h_sub.toarray() if hasattr(h_sub, "toarray") else h_sub,
        dtype=np.complex128,
    )


def optimization_budget(ref: dict) -> dict[str, int]:
    """Powell restart / iteration budget keyed by fixed-N subspace size."""
    dim = len(ref["basis_bitstrings"])
    if dim > 200:
        return {
            "n_restarts": LIH_N_RESTARTS,
            "maxfev": LIH_OPT_MAXFEV,
            "maxiter": LIH_OPT_MAXITER,
        }
    return {
        "n_restarts": N_RESTARTS,
        "maxfev": OPT_MAXFEV,
        "maxiter": OPT_MAXITER,
    }


def quartet_sectors_from_edges(
    basis_bitstrings: list[int],
    edges: list[tuple[int, int]],
    n_spatial: int,
) -> dict[tuple[int, ...], list[int]]:
    from quartet_optimization_utils import quartet_parity_diagonal

    sectors: dict[tuple[int, ...], list[int]] = {}
    for index, bitstring in enumerate(basis_bitstrings):
        key = tuple(
            int(quartet_parity_diagonal([int(bitstring)], edge, n_spatial)[0])
            for edge in edges
        )
        sectors.setdefault(key, []).append(index)
    return sectors


def k_coupled_for_edges(
    h_dense: np.ndarray,
    basis_bitstrings: list[int],
    edges: list[tuple[int, int]],
    n_spatial: int,
    energy_target: float,
    *,
    tol: float = 1e-3,
) -> dict[str, float | int | bool]:
    from optimization_abc_utils import coupled_energy_test, diagonalize_sector_blocks

    sectors = quartet_sectors_from_edges(basis_bitstrings, edges, n_spatial)
    sector_data = diagonalize_sector_blocks(h_dense, sectors)
    e_coupled, k_coupled, converged, _ = coupled_energy_test(
        h_dense,
        sector_data,
        E_exact=energy_target,
        tol=tol,
    )
    return {
        "Kcoupled": int(k_coupled),
        "Ecoupled": float(e_coupled) if e_coupled is not None else float("nan"),
        "Converged": bool(converged),
        "NumSectors": len(sectors),
    }


def k_coupled_for_seniority(
    h_dense: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
    n_qubits: int,
    energy_target: float,
    a: float,
    b: float,
    c: float,
    *,
    tol: float = 1e-3,
) -> dict[str, float | int | bool]:
    from optimization_abc_utils import build_generalized_sectors, coupled_energy_test, diagonalize_sector_blocks

    sectors = build_generalized_sectors(basis_bitstrings, n_spatial, n_qubits, a, b, c)
    sector_data = diagonalize_sector_blocks(h_dense, sectors)
    e_coupled, k_coupled, converged, _ = coupled_energy_test(
        h_dense,
        sector_data,
        E_exact=energy_target,
        tol=tol,
    )
    return {
        "Kcoupled": int(k_coupled),
        "Ecoupled": float(e_coupled) if e_coupled is not None else float("nan"),
        "Converged": bool(converged),
        "NumSectors": len(sectors),
    }


def rotate_h_and_state(
    h_dense: np.ndarray,
    psi: np.ndarray,
    u_spatial: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
) -> tuple[np.ndarray, np.ndarray]:
    from quartet_optimization_utils import orbital_rotation_representation_R_fast

    rotation = orbital_rotation_representation_R_fast(u_spatial, basis_bitstrings, n_spatial)
    psi_rot = rotation.conj().T @ psi
    h_rot = rotation.conj().T @ h_dense @ rotation
    h_rot = 0.5 * (h_rot + h_rot.conj().T)
    return h_rot, psi_rot


def random_unitary(n_spatial: int, seed: int) -> np.ndarray:
    """Haar-like random unitary via QR of a Gaussian matrix."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_spatial, n_spatial)) + 1j * rng.standard_normal((n_spatial, n_spatial))
    q, r = np.linalg.qr(z)
    phase = np.diag(r) / np.abs(np.diag(r))
    return q @ np.diag(phase)


QUARTET_TOPOLOGY_NAMES: tuple[str, ...] = ("matching", "ring", "hub", "balanced_tree")
PARENT_PROTOCOL_NAMES: tuple[str, ...] = (*QUARTET_TOPOLOGY_NAMES, "seniority")
OPTIMIZATION_PROTOCOL_NAMES: tuple[str, ...] = PARENT_PROTOCOL_NAMES


def quartet_topology_builders() -> dict[str, object]:
    """Return quartet edge constructors keyed by topology name."""
    from quartet_optimization_utils import (
        balanced_tree_plus_edges,
        hub_edges,
        matching_edges,
        ring_edges,
    )

    return {
        "matching": matching_edges,
        "ring": ring_edges,
        "hub": hub_edges,
        "balanced_tree": balanced_tree_plus_edges,
    }


def verify_parent_ground_state(
    h_parent: np.ndarray,
    psi: np.ndarray,
    energy: float,
    *,
    tol: float = 1e-8,
) -> None:
    """Raise if psi is not an eigenvector of h_parent with eigenvalue energy."""
    h = 0.5 * (h_parent + h_parent.conj().T)
    residual = float(np.linalg.norm(h @ psi - energy * psi))
    if residual > tol:
        raise ValueError(
            f"Parent ground-state residual {residual:.3e} exceeds tol={tol:.1e}."
        )
    rayleigh = float(np.real(np.vdot(psi, h @ psi)))
    if abs(rayleigh - energy) > tol:
        raise ValueError(
            f"Rayleigh quotient {rayleigh:.12f} != declared energy {energy:.12f}."
        )


def build_parent_hamiltonian(ref: dict, parent_protocol: str) -> dict:
    """
    Parent Hamiltonian H_P from Eq. (9)-(10) term removal on the physical Hamiltonian.

    The returned psi is the lowest eigenvector of H_P (via parent_ground_state), not FCI.

    Quartet protocols: project onto operators commuting with hat R_p hat R_q on topology edges.
    Seniority protocol: DOCI pair-parity parent (commutes with every R^pair_p).
    """
    from parity_parent_hamiltonians import (
        parent_ground_state,
        project_h_sub_to_pair_parent,
        project_h_sub_to_polynomial_parent,
    )

    builders = quartet_topology_builders()
    h_phys = physical_hamiltonian_dense(ref)
    basis_bitstrings = ref["basis_bitstrings"]
    n_spatial = ref["n_spatial"]

    if parent_protocol in builders:
        edges = builders[parent_protocol](n_spatial)
        h_parent = project_h_sub_to_polynomial_parent(
            h_phys,
            basis_bitstrings,
            n_spatial,
            edges,
        )
        energy, psi = parent_ground_state(h_parent)
        verify_parent_ground_state(h_parent, psi, energy)
        return {
            "h_dense": h_parent,
            "psi": psi,
            "energy": energy,
            "parent_protocol": parent_protocol,
            "construction": "quartet_polynomial_parent",
            "edges": edges,
        }

    if parent_protocol == "seniority":
        h_parent = project_h_sub_to_pair_parent(
            h_phys,
            basis_bitstrings,
            n_spatial,
        )
        energy, psi = parent_ground_state(h_parent)
        verify_parent_ground_state(h_parent, psi, energy)
        return {
            "h_dense": h_parent,
            "psi": psi,
            "energy": energy,
            "parent_protocol": parent_protocol,
            "construction": "pair_parity_parent",
        }

    raise ValueError(f"Unknown parent protocol {parent_protocol!r}.")


def seniority_optimization_params(n_spatial: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Standard ABC x0 and Givens pairs for seniority variance optimization."""
    from optimization_abc_utils import pair_list_for_n

    pairs = pair_list_for_n(n_spatial)
    m = len(pairs)
    x0 = np.zeros(m + 2, dtype=float)
    x0[m] = float(np.arccos(-2.0 / np.sqrt(6.0)))
    x0[m + 1] = np.pi / 4.0
    return x0, pairs


def optimize_seniority_protocol(
    psi: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
    *,
    n_restarts: int = N_RESTARTS,
    random_seed: int = PLANTED_SEED,
    maxfev: int = OPT_MAXFEV,
    maxiter: int = OPT_MAXITER,
) -> dict[str, float | np.ndarray]:
    from quartet_optimization_utils import rotate_state_to_orbital_frame
    from scipy.optimize import minimize

    from optimization_abc_utils import (
        ANGLE_INIT_SCALE,
        build_U_from_thetas,
        compute_spin_rdms_from_subspace_state,
        variance_restricted,
    )

    seniority_x0, pairs = seniority_optimization_params(n_spatial)

    def objective(thetas: np.ndarray) -> float:
        u = build_U_from_thetas(n_spatial, thetas, pairs)
        rotated = rotate_state_to_orbital_frame(
            psi,
            basis_bitstrings,
            u,
            n_spatial,
        )
        gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_subspace_state(
            rotated,
            basis_bitstrings,
            n_spatial,
        )
        cost, _, _, _, _, _ = variance_restricted(
            gamma_a,
            gamma_b,
            gamma_ab,
            seniority_x0,
            pairs,
        )
        return float(cost)

    rng = np.random.default_rng(random_seed)
    best_cost = float("inf")
    best_u = np.eye(n_spatial, dtype=np.complex128)
    for _ in range(max(1, int(n_restarts))):
        x0 = ANGLE_INIT_SCALE * rng.standard_normal(len(pairs))
        res = minimize(
            objective,
            x0=x0,
            method="Powell",
            options={"maxiter": maxiter, "maxfev": maxfev, "disp": False},
        )
        trial = float(objective(res.x))
        if trial < best_cost:
            best_cost = trial
            best_u = build_U_from_thetas(n_spatial, res.x, pairs)
    return {"cost": best_cost, "u_spatial": best_u}


def k_after_seniority_optimization(
    h_dense: np.ndarray,
    psi: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
    n_qubits: int,
    energy_target: float,
    *,
    n_restarts: int = N_RESTARTS,
    random_seed: int = PLANTED_SEED,
    maxfev: int = OPT_MAXFEV,
    maxiter: int = OPT_MAXITER,
    tol: float = K_COUPLED_TOL,
) -> dict[str, float | int | bool | np.ndarray]:
    a, b, c = STANDARD_ABC
    opt = optimize_seniority_protocol(
        psi,
        basis_bitstrings,
        n_spatial,
        n_restarts=n_restarts,
        random_seed=random_seed,
        maxfev=maxfev,
        maxiter=maxiter,
    )
    h_opt, _ = rotate_h_and_state(
        h_dense,
        psi,
        opt["u_spatial"],
        basis_bitstrings,
        n_spatial,
    )
    k_info = k_coupled_for_seniority(
        h_opt,
        basis_bitstrings,
        n_spatial,
        n_qubits,
        energy_target,
        a,
        b,
        c,
        tol=tol,
    )
    return {
        "variance": float(opt["cost"]),
        "u_spatial": opt["u_spatial"],
        "h_opt": h_opt,
        **k_info,
    }


def confuse_hamiltonian_and_state(
    h_dense: np.ndarray,
    psi: np.ndarray,
    u_spatial: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same orbital rotation to H and its reference state."""
    return rotate_h_and_state(h_dense, psi, u_spatial, basis_bitstrings, n_spatial)


def optimize_quartet_protocol(
    psi: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
    edges: list[tuple[int, int]],
    *,
    n_restarts: int = N_RESTARTS,
    random_seed: int = PLANTED_SEED,
    maxfev: int = OPT_MAXFEV,
    maxiter: int = OPT_MAXITER,
) -> dict[str, float | np.ndarray]:
    from quartet_optimization_utils import optimize_fixed_edge_quartets

    best = optimize_fixed_edge_quartets(
        psi,
        basis_bitstrings,
        n_spatial,
        edges,
        n_restarts=n_restarts,
        random_seed=random_seed,
        include_zero_start=True,
        parallel=False,
        maxfev=maxfev,
        maxiter=maxiter,
    )
    return {
        "cost": float(best["cost"]),
        "u_spatial": best["u_spatial"],
    }


def k_after_quartet_optimization(
    h_dense: np.ndarray,
    psi: np.ndarray,
    basis_bitstrings: list[int],
    n_spatial: int,
    edges: list[tuple[int, int]],
    energy_target: float,
    *,
    n_restarts: int = N_RESTARTS,
    random_seed: int = PLANTED_SEED,
    maxfev: int = OPT_MAXFEV,
    maxiter: int = OPT_MAXITER,
    tol: float = K_COUPLED_TOL,
) -> dict[str, float | int | bool | np.ndarray]:
    opt = optimize_quartet_protocol(
        psi,
        basis_bitstrings,
        n_spatial,
        edges,
        n_restarts=n_restarts,
        random_seed=random_seed,
        maxfev=maxfev,
        maxiter=maxiter,
    )
    h_opt, _ = rotate_h_and_state(
        h_dense,
        psi,
        opt["u_spatial"],
        basis_bitstrings,
        n_spatial,
    )
    k_info = k_coupled_for_edges(
        h_opt,
        basis_bitstrings,
        edges,
        n_spatial,
        energy_target,
        tol=tol,
    )
    return {
        "variance": float(opt["cost"]),
        "u_spatial": opt["u_spatial"],
        "h_opt": h_opt,
        **k_info,
    }


def run_parent_hamiltonian_survey(
    ref: dict,
    *,
    apply_confusion: bool = True,
    confusion_seed: int = PLANTED_SEED,
    n_restarts: int | None = None,
    random_seed: int = PLANTED_SEED,
    maxfev: int | None = None,
    maxiter: int | None = None,
) -> list[dict[str, object]]:
    """
    For each Eq. (9)-(10) parent H_P, optionally confuse with a random unitary,
    then optimize every protocol and record K_coupled and parity variance.
    """
    budget = optimization_budget(ref)
    n_restarts = budget["n_restarts"] if n_restarts is None else n_restarts
    maxfev = budget["maxfev"] if maxfev is None else maxfev
    maxiter = budget["maxiter"] if maxiter is None else maxiter

    builders = quartet_topology_builders()
    n_spatial = ref["n_spatial"]
    basis_bitstrings = ref["basis_bitstrings"]
    a, b, c = STANDARD_ABC

    u_confusion = random_unitary(n_spatial, confusion_seed) if apply_confusion else np.eye(
        n_spatial, dtype=np.complex128
    )

    rows: list[dict[str, object]] = []
    for parent_protocol in PARENT_PROTOCOL_NAMES:
        parent = build_parent_hamiltonian(ref, parent_protocol)
        h_dense = parent["h_dense"]
        psi = parent["psi"]
        energy_target = parent["energy"]
        h_work, psi_work = confuse_hamiltonian_and_state(
            h_dense,
            psi,
            u_confusion,
            basis_bitstrings,
            n_spatial,
        )

        for protocol_name in OPTIMIZATION_PROTOCOL_NAMES:
            if protocol_name == "seniority":
                k_before = k_coupled_for_seniority(
                    h_work,
                    basis_bitstrings,
                    n_spatial,
                    ref["n_qubits"],
                    energy_target,
                    a,
                    b,
                    c,
                    tol=K_COUPLED_TOL,
                )
                opt = k_after_seniority_optimization(
                    h_work,
                    psi_work,
                    basis_bitstrings,
                    n_spatial,
                    ref["n_qubits"],
                    energy_target,
                    n_restarts=n_restarts,
                    random_seed=random_seed,
                    maxfev=maxfev,
                    maxiter=maxiter,
                )
            else:
                edges = builders[protocol_name](n_spatial)
                k_before = k_coupled_for_edges(
                    h_work,
                    basis_bitstrings,
                    edges,
                    n_spatial,
                    energy_target,
                    tol=K_COUPLED_TOL,
                )
                opt = k_after_quartet_optimization(
                    h_work,
                    psi_work,
                    basis_bitstrings,
                    n_spatial,
                    edges,
                    energy_target,
                    n_restarts=n_restarts,
                    random_seed=random_seed,
                    maxfev=maxfev,
                    maxiter=maxiter,
                )

            rows.append(
                {
                    "ParentProtocol": parent_protocol,
                    "Construction": parent["construction"],
                    "OptimizationProtocol": protocol_name,
                    "Confused": apply_confusion,
                    "EnergyTarget": float(energy_target),
                    "K_Before": int(k_before["Kcoupled"]),
                    "K_After": int(opt["Kcoupled"]),
                    "VarianceAfter": float(opt["variance"]),
                    "SubspaceDim": len(basis_bitstrings),
                    "NSpatial": n_spatial,
                }
            )

    return rows


# Backward-compatible alias for older imports.
run_topology_parent_survey = run_parent_hamiltonian_survey
