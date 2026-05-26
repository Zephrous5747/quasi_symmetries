"""Unified optimization workflows for quasi-symmetry experiments."""

import csv
from typing import Any, Iterable
import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import optimization_different_abc_utils as local_utils
from hamiltonian_cache import DEFAULT_CACHE_DIR, load_reference_state
from hamiltonian_geometry import default_grid_for_molecule as _default_grid_for_molecule
from optimization_abc_utils import (
    ANGLE_INIT_SCALE,
    EVAL_STATE_SPECIFIC_COMMUTATIVITY,
    MAXITER,
    N_RESTARTS,
    OP_COEF_TOL,
    OPT_METHOD,
    RANDOM_SEED,
    analyze_individual_symmetry_operators_with_leakage,
    analyze_individual_symmetry_operators_with_leakage_subspace,
    build_generalized_sectors,
    closed_shell_hf_bitstring,
    optimize_variance_restricted,
    orbital_rotation_representation_R,
    pair_list_for_n,
    popcount,
    shannon_block_decomposition,
    shared_abc_energy_indicators,
    skipped_energy_sector_diagnostics,
    solve_cisd_state,
    variance_restricted,
)


WORKFLOW_FIXED_ABC = "fixed_abc"
WORKFLOW_SHARED_ABC = "shared_abc"
WORKFLOW_LOCAL_ABC = "local_abc"
VALID_WORKFLOWS = {WORKFLOW_FIXED_ABC, WORKFLOW_SHARED_ABC, WORKFLOW_LOCAL_ABC}

# Representative N2 bond lengths (Å): equilibrium, stretched (strong correlation), dissociative.
N2_BOND_EQUILIBRIUM = 1.2
N2_BOND_STRONGLY_CORRELATED = 1.4
N2_BOND_DISSOCIATIVE = 2.2
N2_REPRESENTATIVE_GRID = (
    N2_BOND_EQUILIBRIUM,
    N2_BOND_STRONGLY_CORRELATED,
    N2_BOND_DISSOCIATIVE,
)


def _aufbau_spin_occupation(n_spatial: int, n_up: int, n_down: int) -> np.ndarray:
    occ = np.zeros(2 * n_spatial, dtype=int)
    occ[0 : 2 * n_up : 2] = 1
    occ[1 : 2 * n_down : 2] = 1
    return occ


def _split_workflow_kwargs(kwargs: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Extract cache directory and geometry-only kwargs from workflow kwargs."""
    cache_dir = str(kwargs.pop("hamiltonian_cache_dir", DEFAULT_CACHE_DIR))
    geom_kw = {k: kwargs[k] for k in ("hoh_angle_deg", "aspect_ratio") if k in kwargs}
    return cache_dir, geom_kw


def _default_csv_name(molecule: str, workflow: str) -> str:
    suffix = {
        WORKFLOW_FIXED_ABC: "fixed_abc",
        WORKFLOW_SHARED_ABC: "shared_abc",
        WORKFLOW_LOCAL_ABC: "local_abc",
    }[workflow]
    return f"{molecule}_quasi_symmetry_{suffix}.csv"


def _skipped_entropy_energy_fields() -> dict[str, Any]:
    """CSV fields when dense entropy / sector-energy diagnostics are skipped."""
    nan = float("nan")
    return {
        "Coarse_Entropy_Identity": nan,
        "Coarse_Entropy_Optimized": nan,
        "Fine_Entropy_Identity": nan,
        "Fine_Entropy_Optimized": nan,
        "Edec_Identity": nan,
        "Edec_Optimized": nan,
        "Ecoupled_Identity": nan,
        "Ecoupled_Optimized": nan,
        "Kcoupled_Identity": 0,
        "Kcoupled_Optimized": 0,
        "EBO_Identity": nan,
        "EBO_Optimized": nan,
        "NumSectors_Identity": 0,
        "NumSectors_Optimized": 0,
        "DenseDiagnosticsSkipped": True,
    }


def _prepare_reference_state(
    molecule: str,
    x: float,
    *,
    cache_dir: str,
    popcount_fn=popcount,
    solve_cisd_fn=solve_cisd_state,
    hf_bitstring_fn=closed_shell_hf_bitstring,
    **geometry_kwargs: Any,
) -> dict[str, Any]:
    """Load precomputed fixed-N Hamiltonian / FCI / CISD reference from HDF5 cache."""
    return load_reference_state(
        molecule,
        x,
        cache_dir=cache_dir,
        popcount_fn=popcount_fn,
        solve_cisd_fn=solve_cisd_fn,
        hf_bitstring_fn=hf_bitstring_fn,
        **geometry_kwargs,
    )


def _compute_spin_rdms_from_ref(ref: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if ref["use_dense"]:
        return compute_spin_rdms_from_statevector(ref["psi_full"], ref["n_spatial"])
    return compute_spin_rdms_from_subspace_state(
        ref["v_sub"], ref["basis_bitstrings"], ref["n_spatial"]
    )


def _run_shared_abc_commutativity(
    ref: dict[str, Any],
    molecule: str,
    u_spatial: np.ndarray,
    a: float,
    b: float,
    c: float,
    label: str,
) -> tuple[float, float, dict[str, Any]]:
    if not EVAL_STATE_SPECIFIC_COMMUTATIVITY:
        return 0.0, 0.0, {"sum_exp": np.nan}

    if ref["use_dense"]:
        result = analyze_individual_symmetry_operators_with_leakage(
            ref["h_full"],
            ref["psi_full"],
            u_spatial,
            ref["n_spatial"],
            ref["n_qubits"],
            a,
            b,
            c,
            label=f"{molecule} / {label}",
            tol=OP_COEF_TOL,
            check_eigenstate=True,
        )
    else:
        result = analyze_individual_symmetry_operators_with_leakage_subspace(
            ref["h_sub"],
            ref["v_sub"],
            ref["basis_bitstrings"],
            u_spatial,
            ref["n_spatial"],
            ref["n_qubits"],
            a,
            b,
            c,
            label=f"{molecule} / {label}",
            tol=OP_COEF_TOL,
            check_eigenstate=True,
        )
    return float(result["sum_comm_sq"]), float(np.real(result["sum_exp"])), result


def _run_dense_entropy_and_energy(
    ref: dict[str, Any],
    molecule: str,
    energy_fci: float,
    a_id: float,
    b_id: float,
    c_id: float,
    u_identity: np.ndarray,
    a_opt: float,
    b_opt: float,
    c_opt: float,
    u_optimized: np.ndarray,
    label_identity: str,
    label_optimized: str,
) -> dict[str, Any]:
    h_identity = ref["h_sub"].toarray().astype(np.complex128)
    psi_identity = ref["v_sub"] / np.linalg.norm(ref["v_sub"])
    n_spatial = ref["n_spatial"]
    n_qubits = ref["n_qubits"]
    basis_bitstrings = ref["basis_bitstrings"]

    sectors_identity = build_generalized_sectors(
        basis_bitstrings, n_spatial, n_qubits, a_id, b_id, c_id
    )
    entropy_fine_identity, entropy_coarse_identity, _ = shannon_block_decomposition(
        h_identity, psi_identity, sectors_identity
    )

    sectors_optimized = build_generalized_sectors(
        basis_bitstrings, n_spatial, n_qubits, a_opt, b_opt, c_opt
    )
    r_opt = orbital_rotation_representation_R(u_optimized, basis_bitstrings, n_spatial)
    h_rot = r_opt.conj().T @ (h_identity @ r_opt)
    h_rot = 0.5 * (h_rot + h_rot.conj().T)
    psi_rot = r_opt.conj().T @ psi_identity
    entropy_fine_optimized, entropy_coarse_optimized, _ = shannon_block_decomposition(
        h_rot, psi_rot, sectors_optimized
    )

    energy_identity = shared_abc_energy_indicators(
        H_dense=h_identity,
        basis_bitstrings=basis_bitstrings,
        n_spatial=n_spatial,
        n_qubits=n_qubits,
        a=a_id,
        b=b_id,
        c=c_id,
        U_spatial=u_identity,
        E_exact=energy_fci,
        tol=1e-3,
        label=f"{molecule} / {label_identity}",
    )
    energy_optimized = shared_abc_energy_indicators(
        H_dense=h_identity,
        basis_bitstrings=basis_bitstrings,
        n_spatial=n_spatial,
        n_qubits=n_qubits,
        a=a_opt,
        b=b_opt,
        c=c_opt,
        U_spatial=u_optimized,
        E_exact=energy_fci,
        tol=1e-3,
        label=f"{molecule} / {label_optimized}",
    )

    return {
        "Coarse_Entropy_Identity": entropy_coarse_identity,
        "Coarse_Entropy_Optimized": entropy_coarse_optimized,
        "Fine_Entropy_Identity": entropy_fine_identity,
        "Fine_Entropy_Optimized": entropy_fine_optimized,
        "Edec_Identity": energy_identity.E_dec_min,
        "Edec_Optimized": energy_optimized.E_dec_min,
        "Ecoupled_Identity": energy_identity.E_coupled,
        "Ecoupled_Optimized": energy_optimized.E_coupled,
        "Kcoupled_Identity": energy_identity.K_coupled,
        "Kcoupled_Optimized": energy_optimized.K_coupled,
        "EBO_Identity": energy_identity.E_BO,
        "EBO_Optimized": energy_optimized.E_BO,
        "NumSectors_Identity": energy_identity.n_sectors,
        "NumSectors_Optimized": energy_optimized.n_sectors,
        "DenseDiagnosticsSkipped": False,
    }


def _abc_to_angles(a: float, b: float, c: float) -> tuple[float, float]:
    norm = float(np.sqrt(a * a + b * b + c * c))
    if norm == 0.0:
        raise ValueError("The fixed (a, b, c) vector must be non-zero.")
    a_n, b_n, c_n = a / norm, b / norm, c / norm
    phi1 = float(np.arccos(np.clip(c_n, -1.0, 1.0)))
    phi2 = float(np.mod(np.arctan2(b_n, a_n), 2.0 * np.pi))
    return phi1, phi2


def _optimize_unitary_only(
    gamma_a: np.ndarray,
    gamma_b: np.ndarray,
    gamma_ab: np.ndarray,
    a: float,
    b: float,
    c: float,
) -> dict[str, Any]:
    np.random.seed(RANDOM_SEED)
    n_spatial = gamma_a.shape[0]
    pairs = pair_list_for_n(n_spatial)
    n_angles = len(pairs)
    phi1, phi2 = _abc_to_angles(a, b, c)

    def objective(thetas: np.ndarray) -> float:
        x_params = np.concatenate([thetas, np.array([phi1, phi2])])
        value, _, _, _, _, _ = variance_restricted(gamma_a, gamma_b, gamma_ab, x_params, pairs)
        return value

    best: dict[str, Any] | None = None
    for restart in range(N_RESTARTS):
        x0 = np.zeros(n_angles)
        if restart > 0:
            x0 = ANGLE_INIT_SCALE * np.random.randn(n_angles)

        result = minimize(
            objective,
            x0=x0,
            method=OPT_METHOD,
            options={"maxiter": MAXITER, "disp": False},
        )
        score = objective(result.x)
        if best is None or score < best["V"]:
            best = {"res": result, "V": score, "pairs": pairs, "phi1": phi1, "phi2": phi2}

    assert best is not None
    return best


def evaluate_single_point_fixed_abc(
    molecule: str,
    x: float,
    fixed_abc: tuple[float, float, float] = (1.0/math.sqrt(6), 1.0/math.sqrt(6), -2.0/math.sqrt(6)),
    **kwargs: Any,
) -> dict[str, Any]:
    cache_dir, geom_kw = _split_workflow_kwargs(kwargs)
    ref = _prepare_reference_state(molecule, x, cache_dir=cache_dir, **geom_kw)
    gamma_a, gamma_b, gamma_ab = ref["gamma_a"], ref["gamma_b"], ref["gamma_ab"]
    n_spatial = ref["n_spatial"]
    energy_fci = ref["energy_fci"]
    pairs = pair_list_for_n(n_spatial)
    n_angles = len(pairs)

    a_fixed, b_fixed, c_fixed = fixed_abc
    phi1, phi2 = _abc_to_angles(a_fixed, b_fixed, c_fixed)
    x_identity = np.zeros(n_angles + 2)
    x_identity[n_angles] = phi1
    x_identity[n_angles + 1] = phi2
    v_identity, _, u_identity, a_n, b_n, c_n = variance_restricted(
        gamma_a, gamma_b, gamma_ab, x_identity, pairs
    )

    best = _optimize_unitary_only(gamma_a, gamma_b, gamma_ab, a_fixed, b_fixed, c_fixed)
    x_opt = np.concatenate([best["res"].x, np.array([best["phi1"], best["phi2"]])])
    v_optimized, _, u_optimized, _, _, _ = variance_restricted(
        gamma_a, gamma_b, gamma_ab, x_opt, best["pairs"]
    )

    comm_sq_identity, sum_sexp_id, _ = _run_shared_abc_commutativity(
        ref, molecule, u_identity, a_n, b_n, c_n, "fixed_abc_identity"
    )
    comm_sq_optimized, sum_sexp_opt, _ = _run_shared_abc_commutativity(
        ref, molecule, u_optimized, a_n, b_n, c_n, "fixed_abc_optimized"
    )

    if ref["use_dense"]:
        post = _run_dense_entropy_and_energy(
            ref,
            molecule,
            energy_fci,
            a_n,
            b_n,
            c_n,
            u_identity,
            a_n,
            b_n,
            c_n,
            u_optimized,
            "fixed_abc_identity",
            "fixed_abc_optimized",
        )
    else:
        post = _skipped_entropy_energy_fields()

    return {
        "Workflow": WORKFLOW_FIXED_ABC,
        "Molecule": molecule,
        "Geometry_Param": x,
        "E_HF": ref["energy_hf"],
        "E_FCI": energy_fci,
        "E_CISD": ref["energy_cisd"],
        "V_Identity": v_identity,
        "V_Optimized": v_optimized,
        "a": a_n,
        "b": b_n,
        "c": c_n,
        "Sum_CommSq_Identity": comm_sq_identity,
        "Sum_CommSq_Optimized": comm_sq_optimized,
        "Sum_Sexp_Identity": sum_sexp_id if EVAL_STATE_SPECIFIC_COMMUTATIVITY else np.nan,
        "Sum_Sexp_Optimized": sum_sexp_opt if EVAL_STATE_SPECIFIC_COMMUTATIVITY else np.nan,
        **post,
    }


def evaluate_single_point_shared_abc(
    molecule: str,
    x: float,
    **kwargs: Any,
) -> dict[str, Any]:
    cache_dir, geom_kw = _split_workflow_kwargs(kwargs)
    ref = _prepare_reference_state(molecule, x, cache_dir=cache_dir, **geom_kw)
    gamma_a, gamma_b, gamma_ab = ref["gamma_a"], ref["gamma_b"], ref["gamma_ab"]
    energy_fci = ref["energy_fci"]
    pairs = pair_list_for_n(ref["n_spatial"])
    n_pairs = len(pairs)
    x_identity = np.zeros(n_pairs + 2)
    x_identity[n_pairs] = np.arccos(-2.0 / np.sqrt(6.0))
    x_identity[n_pairs + 1] = np.pi / 4.0
    v_identity, _, u_identity, a_identity, b_identity, c_identity = variance_restricted(
        gamma_a, gamma_b, gamma_ab, x_identity, pairs
    )

    best = optimize_variance_restricted(gamma_a, gamma_b, gamma_ab)
    v_optimized, _, u_optimized, a_opt, b_opt, c_opt = variance_restricted(
        gamma_a, gamma_b, gamma_ab, best["res"].x, best["pairs"]
    )

    comm_sq_identity, sum_sexp_id, _ = _run_shared_abc_commutativity(
        ref, molecule, u_identity, a_identity, b_identity, c_identity, "identity"
    )
    comm_sq_optimized, sum_sexp_opt, _ = _run_shared_abc_commutativity(
        ref, molecule, u_optimized, a_opt, b_opt, c_opt, "optimized"
    )

    if ref["use_dense"]:
        post = _run_dense_entropy_and_energy(
            ref,
            molecule,
            energy_fci,
            a_identity,
            b_identity,
            c_identity,
            u_identity,
            a_opt,
            b_opt,
            c_opt,
            u_optimized,
            "identity",
            "optimized",
        )
    else:
        post = _skipped_entropy_energy_fields()

    return {
        "Molecule": molecule,
        "Geometry_Param": x,
        "E_HF": ref["energy_hf"],
        "E_FCI": energy_fci,
        "E_CISD": ref["energy_cisd"],
        "V_Identity": v_identity,
        "V_Optimized": v_optimized,
        "a": a_opt,
        "b": b_opt,
        "c": c_opt,
        "Sum_CommSq_Identity": comm_sq_identity,
        "Sum_CommSq_Optimized": comm_sq_optimized,
        "Sum_Sexp_Identity": sum_sexp_id if EVAL_STATE_SPECIFIC_COMMUTATIVITY else np.nan,
        "Sum_Sexp_Optimized": sum_sexp_opt if EVAL_STATE_SPECIFIC_COMMUTATIVITY else np.nan,
        **post,
    }


def _run_local_abc_commutativity(
    ref: dict[str, Any],
    u_spatial: np.ndarray,
    local_abcs,
) -> tuple[float, float]:
    if not local_utils.EVAL_STATE_SPECIFIC_COMMUTATIVITY:
        return 0.0, float("nan")

    psi = np.asarray(ref["psi_full"] if ref["use_dense"] else ref["v_sub"], dtype=np.complex128)
    psi = psi / np.linalg.norm(psi)
    h_mat = ref["h_full"] if ref["use_dense"] else ref["h_sub"]

    sum_exp = 0.0 + 0.0j
    sum_comm_sq = 0.0

    for i in range(ref["n_spatial"]):
        si_ferm = local_utils.build_single_local_operator(
            u_spatial, ref["n_spatial"], i, local_abcs, tol=local_utils.OP_COEF_TOL
        )
        si_full = local_utils.fermion_to_sparse_qubit(si_ferm, ref["n_qubits"])
        if ref["use_dense"]:
            si_mat = si_full
        else:
            si_mat = local_utils.restrict_operator_to_subspace(si_full, ref["basis_bitstrings"])

        sum_exp += np.vdot(psi, si_mat.dot(psi))
        comm_sq_i, _ = local_utils.comm_state_norm_sq(
            h_mat, si_mat, psi, check_eigenstate=False
        )
        sum_comm_sq += comm_sq_i

    return float(sum_comm_sq), float(np.real(sum_exp))


def evaluate_single_point_local_abc(
    molecule: str,
    x: float,
    **kwargs: Any,
) -> dict[str, Any]:
    cache_dir, geom_kw = _split_workflow_kwargs(kwargs)
    ref = _prepare_reference_state(
        molecule,
        x,
        cache_dir=cache_dir,
        popcount_fn=local_utils.popcount,
        solve_cisd_fn=local_utils.solve_cisd_state,
        hf_bitstring_fn=local_utils.closed_shell_hf_bitstring,
        **geom_kw,
    )
    gamma_a, gamma_b, gamma_ab = ref["gamma_a"], ref["gamma_b"], ref["gamma_ab"]
    n_spatial = ref["n_spatial"]
    n_qubits = ref["n_qubits"]
    basis_bitstrings = ref["basis_bitstrings"]
    pairs = local_utils.pair_list_for_n(n_spatial)
    n_pairs = len(pairs)

    x_identity = np.zeros(n_pairs + 2 * n_spatial)
    for i in range(n_spatial):
        x_identity[n_pairs + 2 * i] = np.arccos(-2.0 / np.sqrt(6.0))
        x_identity[n_pairs + 2 * i + 1] = np.pi / 4.0

    v_identity, _, u_identity, local_abcs_identity = local_utils.variance_restricted_local_abc(
        gamma_a, gamma_b, gamma_ab, x_identity, pairs
    )
    best = local_utils.optimize_variance_restricted_local_abc(gamma_a, gamma_b, gamma_ab)
    v_optimized, _, u_optimized, local_abcs_optimized = local_utils.variance_restricted_local_abc(
        gamma_a, gamma_b, gamma_ab, best["res"].x, best["pairs"]
    )

    comm_sq_identity, sum_sexp_id = _run_local_abc_commutativity(
        ref, u_identity, local_abcs_identity
    )
    comm_sq_optimized, sum_sexp_opt = _run_local_abc_commutativity(
        ref, u_optimized, local_abcs_optimized
    )

    nan = float("nan")
    if ref["use_dense"]:
        h_identity = ref["h_sub"].toarray().astype(np.complex128)
        psi_identity = ref["v_sub"] / np.linalg.norm(ref["v_sub"])
        sectors_identity = local_utils.build_generalized_sectors_local_abc(
            basis_bitstrings, n_spatial, n_qubits, local_abcs_identity
        )
        entropy_fine_identity, entropy_coarse_identity, _ = local_utils.shannon_block_decomposition(
            h_identity, psi_identity, sectors_identity
        )
        sectors_optimized = local_utils.build_generalized_sectors_local_abc(
            basis_bitstrings, n_spatial, n_qubits, local_abcs_optimized
        )
        r_opt = local_utils.orbital_rotation_representation_R(
            u_optimized, basis_bitstrings, n_spatial
        )
        h_rot = r_opt.conj().T @ (h_identity @ r_opt)
        h_rot = 0.5 * (h_rot + h_rot.conj().T)
        psi_rot = r_opt.conj().T @ psi_identity
        entropy_fine_optimized, entropy_coarse_optimized, _ = local_utils.shannon_block_decomposition(
            h_rot, psi_rot, sectors_optimized
        )
        dense_skipped = False
    else:
        entropy_coarse_identity = entropy_coarse_optimized = nan
        entropy_fine_identity = entropy_fine_optimized = nan
        dense_skipped = True

    row: dict[str, Any] = {
        "Molecule": molecule,
        "Geometry_Param": x,
        "E_HF": ref["energy_hf"],
        "E_FCI": ref["energy_fci"],
        "E_CISD": ref["energy_cisd"],
        "V_Identity": v_identity,
        "V_Optimized": v_optimized,
        "Sum_CommSq_Identity": comm_sq_identity,
        "Sum_CommSq_Optimized": comm_sq_optimized,
        "Sum_Sexp_Identity": sum_sexp_id if local_utils.EVAL_STATE_SPECIFIC_COMMUTATIVITY else nan,
        "Sum_Sexp_Optimized": sum_sexp_opt if local_utils.EVAL_STATE_SPECIFIC_COMMUTATIVITY else nan,
        "Coarse_Entropy_Identity": entropy_coarse_identity,
        "Coarse_Entropy_Optimized": entropy_coarse_optimized,
        "Fine_Entropy_Identity": entropy_fine_identity,
        "Fine_Entropy_Optimized": entropy_fine_optimized,
        "DenseDiagnosticsSkipped": dense_skipped,
    }
    for i, (a_i, b_i, c_i) in enumerate(local_abcs_identity):
        row[f"a_id_{i}"] = a_i
        row[f"b_id_{i}"] = b_i
        row[f"c_id_{i}"] = c_i
    for i, (a_i, b_i, c_i) in enumerate(local_abcs_optimized):
        row[f"a_opt_{i}"] = a_i
        row[f"b_opt_{i}"] = b_i
        row[f"c_opt_{i}"] = c_i
    return row


def evaluate_single_point(
    workflow: str,
    molecule: str,
    x: float,
    **kwargs: Any,
) -> dict[str, Any]:
    if workflow == WORKFLOW_FIXED_ABC:
        return evaluate_single_point_fixed_abc(molecule=molecule, x=x, **kwargs)
    if workflow == WORKFLOW_SHARED_ABC:
        row = evaluate_single_point_shared_abc(molecule=molecule, x=x, **kwargs)
        row["Workflow"] = WORKFLOW_SHARED_ABC
        return row
    if workflow == WORKFLOW_LOCAL_ABC:
        row = evaluate_single_point_local_abc(molecule=molecule, x=x, **kwargs)
        row["Workflow"] = WORKFLOW_LOCAL_ABC
        return row
    raise ValueError(f"Unsupported workflow '{workflow}'. Choose from: {sorted(VALID_WORKFLOWS)}")


def _fieldnames_for_workflow(workflow: str, molecule: str, **kwargs: Any) -> list[str] | None:
    if workflow == WORKFLOW_LOCAL_ABC:
        builder = getattr(local_utils, "get_fieldnames_for_molecule", None)
        if callable(builder):
            names = builder(molecule, **kwargs)
            if "Workflow" not in names:
                names = ["Workflow", *names]
            return names
    return None


def run_scan(
    workflow: str,
    molecule: str,
    grid: Iterable[float],
    csv_filename: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    if workflow not in VALID_WORKFLOWS:
        raise ValueError(f"Unsupported workflow '{workflow}'. Choose from: {sorted(VALID_WORKFLOWS)}")

    results: list[dict[str, Any]] = []
    fieldnames = _fieldnames_for_workflow(workflow, molecule, **kwargs)

    if csv_filename is not None:
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as handle:
            writer: csv.DictWriter[str] | None = None
            for x in grid:
                try:
                    row = evaluate_single_point(workflow=workflow, molecule=molecule, x=float(x), **kwargs)
                    results.append(row)
                    if writer is None:
                        names = fieldnames or list(row.keys())
                        writer = csv.DictWriter(handle, fieldnames=names)
                        writer.writeheader()
                    writer.writerow(row)
                    handle.flush()
                except Exception as exc:
                    print(f"[{workflow}/{molecule}] Error at x={x}: {exc}")
        return results

    for x in grid:
        try:
            row = evaluate_single_point(workflow=workflow, molecule=molecule, x=float(x), **kwargs)
            results.append(row)
        except Exception as exc:
            print(f"[{workflow}/{molecule}] Error at x={x}: {exc}")
    return results


def main(
    workflow: str = WORKFLOW_SHARED_ABC,
    molecule: str = "lih",
    grid: Iterable[float] | None = None,
    csv_filename: str | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    molecule_name = molecule.lower()
    if workflow not in VALID_WORKFLOWS:
        raise ValueError(f"Unsupported workflow '{workflow}'. Choose from: {sorted(VALID_WORKFLOWS)}")
    if grid is None:
        grid = _default_grid_for_molecule(molecule_name)
    if csv_filename is None:
        csv_filename = _default_csv_name(molecule_name, workflow)
    return run_scan(
        workflow=workflow,
        molecule=molecule_name,
        grid=grid,
        csv_filename=csv_filename,
        **kwargs,
    )


def collect_orbital_diagram_data(
    molecule: str,
    x: float,
    fixed_abc: tuple[float, float, float] = (1.0 / math.sqrt(6), 1.0 / math.sqrt(6), -2.0 / math.sqrt(6)),
    **kwargs: Any,
) -> dict[str, Any]:
    """Collect HF/fixed/shared orbital-diagram data for one geometry point."""
    molecule_name = molecule.lower()
    cache_dir, geom_kw = _split_workflow_kwargs(kwargs)
    ref = _prepare_reference_state(molecule_name, x, cache_dir=cache_dir, **geom_kw)

    energy_hf = ref["energy_hf"]
    energy_fci = ref["energy_fci"]
    n_spatial = ref["n_spatial"]
    n_electrons = ref["n_electrons"]
    n_up = n_electrons // 2
    n_down = n_electrons // 2
    gamma_a, gamma_b, gamma_ab = ref["gamma_a"], ref["gamma_b"], ref["gamma_ab"]
    pairs = pair_list_for_n(n_spatial)
    n_pairs = len(pairs)

    a_fixed, b_fixed, c_fixed = fixed_abc
    best_fixed = _optimize_unitary_only(gamma_a, gamma_b, gamma_ab, a_fixed, b_fixed, c_fixed)
    x_fixed_opt = np.concatenate([best_fixed["res"].x, np.array([best_fixed["phi1"], best_fixed["phi2"]])])
    _, _, u_fixed_opt, a_fixed_opt, b_fixed_opt, c_fixed_opt = variance_restricted(
        gamma_a, gamma_b, gamma_ab, x_fixed_opt, best_fixed["pairs"]
    )

    best_shared = optimize_variance_restricted(gamma_a, gamma_b, gamma_ab)
    v_shared_opt, _, u_shared_opt, a_shared_opt, b_shared_opt, c_shared_opt = variance_restricted(
        gamma_a, gamma_b, gamma_ab, best_shared["res"].x, best_shared["pairs"]
    )

    # Frozen-density rotated Fock spectrum:
    # start from canonical HF Fock in spatial-MO basis (diag of mo energies),
    # rotate representation by U, and diagonalize to obtain frame-resolved levels.
    spatial_hf_energies = np.asarray(ref.get("orbital_energies", []), dtype=float)
    if spatial_hf_energies.size == 0:
        raise ValueError(
            "orbital_energies missing from cache; regenerate Hamiltonian HDF5 with "
            "generate_hamiltonians.py."
        )
    if spatial_hf_energies.shape[0] != n_spatial:
        raise ValueError("HF orbital energies are inconsistent with n_spatial.")
    fock_hf = np.diag(spatial_hf_energies)

    def _expand_spatial_to_spin(spatial_vals: np.ndarray) -> np.ndarray:
        spin_vals = np.empty(2 * len(spatial_vals), dtype=float)
        spin_vals[0::2] = spatial_vals
        spin_vals[1::2] = spatial_vals
        return spin_vals

    def _frame_aufbau_spin_occupation() -> np.ndarray:
        return _aufbau_spin_occupation(n_spatial=n_spatial, n_up=n_up, n_down=n_down)

    def _frame_payload(U: np.ndarray, label: str) -> dict[str, Any]:
        fock_rot = U.T @ fock_hf @ U
        fock_rot = 0.5 * (fock_rot + fock_rot.T)
        spatial_levels = np.linalg.eigvalsh(fock_rot)
        one_body = _expand_spatial_to_spin(spatial_levels)
        occ = _frame_aufbau_spin_occupation()
        return {
            "label": label,
            "U": U,
            "one_body": one_body,
            "one_body_shifted": one_body,
            "occ": occ.astype(int),
        }

    frames = [
        _frame_payload(np.eye(n_spatial), "HF"),
        _frame_payload(u_fixed_opt, "Fixed-optimized"),
        _frame_payload(u_shared_opt, "Shared-optimized"),
    ]

    return {
        "molecule": molecule_name,
        "x": x,
        "energy_hf": energy_hf,
        "n_spatial": n_spatial,
        "n_up": n_up,
        "n_down": n_down,
        "energy_fci": energy_fci,
        "abc_fixed_optimized": (a_fixed_opt, b_fixed_opt, c_fixed_opt),
        "abc_shared_optimized": (a_shared_opt, b_shared_opt, c_shared_opt),
        "V_fixed_optimized": float(best_fixed["V"]),
        "V_shared_optimized": v_shared_opt,
        "frames": frames,
    }


def plot_orbital_diagram_for_point(
    molecule: str,
    x: float,
    save_path: str | None = None,
    fixed_abc: tuple[float, float, float] = (1.0 / math.sqrt(6), 1.0 / math.sqrt(6), -2.0 / math.sqrt(6)),
    **kwargs: Any,
) -> dict[str, Any]:
    """Plot HF/fixed/shared frozen-density Fock orbital energies."""
    data = collect_orbital_diagram_data(
        molecule=molecule,
        x=x,
        fixed_abc=fixed_abc,
        **kwargs,
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    x_positions = np.arange(len(data["frames"]))

    for frame_idx, frame in enumerate(data["frames"]):
        energies = frame["one_body_shifted"]
        occ = frame["occ"]
        for mode, energy in enumerate(energies):
            y = float(energy)
            color = "tab:blue" if int(occ[mode]) == 1 else "0.5"
            ax.hlines(y, frame_idx - 0.25, frame_idx + 0.25, color=color, linewidth=2.0)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f["label"] for f in data["frames"]])
    ax.set_xlim(-0.5, len(data["frames"]) - 0.2)
    ax.set_ylabel("Frozen-density Fock orbital energy")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title(f"{data['molecule']} orbital diagram | x={x:.4f}")
    ax.plot([], [], color="tab:blue", lw=2.0, label="occupied")
    ax.plot([], [], color="0.5", lw=2.0, label="virtual")
    ax.legend(loc="best")

    fig.tight_layout()

    if save_path is None:
        save_path = f"{data['molecule']}_orbital_diagram_hf_fixed_shared_x{x:.4f}.png"
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    data["plot_path"] = save_path
    return data

