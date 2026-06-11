"""Optimization utilities — re-exports common symbols."""

from quasi_symmetries.optimization.rotations import (
    build_U_from_thetas,
    build_U_from_thetas_symmetry_blocked,
    orbital_rotation_representation_R,
    pair_list_for_n,
    symmetry_blocked_pair_list,
)
from quasi_symmetries.config import ANGLE_INIT_SCALE
from quasi_symmetries.optimization.variance import (
    OptLog,
    optimize_variance_restricted,
    variance_restricted,
)
from quasi_symmetries.fermion.bitstring import (
    closed_shell_hf_bitstring,
    mode_is_occupied,
    popcount,
)
from quasi_symmetries.optimization.rotations import check_R_vs_direct_seniority
from quasi_symmetries.fermion.subspace import (
    fixed_n_subspace_dim,
    restrict_operator_to_subspace,
    use_dense_subspace_ops,
)
from quasi_symmetries.fermion.rdms import (
    compute_spin_rdms_from_statevector,
    compute_spin_rdms_from_subspace_state,
)
from quasi_symmetries.fermion.operators import (
    build_total_operator,
    comm_expect_comm_sq_abs,
    fermion_to_sparse_qubit,
    rotated_seniority_orbital_fermion,
)
from quasi_symmetries.diagnostics.sparse_energy import (
    SparseSubspaceHamiltonian,
    coupled_energy_lazy,
    decoupled_energy_lazy,
    energy_sector_diagnostics_sparse,
)
from quasi_symmetries.diagnostics.energy_sectors import (
    EnergySectorDiagnostics,
    analyze_individual_symmetry_operators_with_leakage_subspace,
    bo_like_coupled_energy_test,
    build_generalized_sectors,
    build_sectors_with_exact_symmetries,
    comm_state_norm_sq,
    coupled_energy_test,
    decoupled_energy_test,
    diagonalize_sector_blocks,
    energy_sector_diagnostics_symmetry_restricted,
    shannon_block_decomposition,
    shared_abc_energy_indicators,
    skipped_energy_sector_diagnostics,
    solve_cisd_state,
)
from quasi_symmetries.config import (
    BASIS,
    CHARGE,
    EVAL_STATE_SPECIFIC_COMMUTATIVITY,
    MAXITER,
    MULTIPLICITY,
    N_RESTARTS,
    OP_COEF_TOL,
    OPT_METHOD,
    RANDOM_SEED,
)

__all__ = [
    "ANGLE_INIT_SCALE", "BASIS", "CHARGE", "EVAL_STATE_SPECIFIC_COMMUTATIVITY",
    "MAXITER", "MULTIPLICITY", "N_RESTARTS", "OP_COEF_TOL", "OPT_METHOD",
    "RANDOM_SEED", "EnergySectorDiagnostics", "OptLog",
    "analyze_individual_symmetry_operators_with_leakage_subspace",
    "bo_like_coupled_energy_test", "build_U_from_thetas",
    "build_U_from_thetas_symmetry_blocked", "build_generalized_sectors",
    "build_sectors_with_exact_symmetries",
    "check_R_vs_direct_seniority",
    "build_total_operator", "closed_shell_hf_bitstring", "comm_expect_comm_sq_abs",
    "comm_state_norm_sq",
    "compute_spin_rdms_from_statevector", "compute_spin_rdms_from_subspace_state",
    "coupled_energy_lazy", "coupled_energy_test", "decoupled_energy_lazy",
    "decoupled_energy_test", "diagonalize_sector_blocks",
    "energy_sector_diagnostics_sparse", "energy_sector_diagnostics_symmetry_restricted",
    "fermion_to_sparse_qubit", "fixed_n_subspace_dim", "mode_is_occupied",
    "optimize_variance_restricted",
    "orbital_rotation_representation_R", "pair_list_for_n", "popcount",
    "symmetry_blocked_pair_list",
    "restrict_operator_to_subspace", "rotated_seniority_orbital_fermion",
    "shannon_block_decomposition", "shared_abc_energy_indicators",
    "skipped_energy_sector_diagnostics", "solve_cisd_state", "SparseSubspaceHamiltonian",
    "use_dense_subspace_ops",
    "variance_restricted",
]
