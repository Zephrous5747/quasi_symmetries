"""Scan H4 confusion tolerance: epsilon-scaled U_rand recovery vs parent protocol."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optimization_abc_utils import build_U_from_thetas, pair_list_for_n
from quartet_optimization_utils import quartet_cost_for_u
from tests.helpers import (
    K_COUPLED_TOL,
    PARENT_PROTOCOL_NAMES,
    PLANTED_SEED,
    STANDARD_ABC,
    build_parent_hamiltonian,
    confuse_hamiltonian_and_state,
    k_after_quartet_optimization,
    k_after_seniority_optimization,
    k_coupled_for_edges,
    k_coupled_for_seniority,
    load_h4_reference,
    quartet_topology_builders,
)

DEFAULT_OUTPUT = ROOT / "opt_results" / "h4_confusion_epsilon_scan.csv"
FIELDNAMES = [
    "Epsilon",
    "ParentProtocol",
    "EnergyTarget",
    "VarPre",
    "KPre",
    "VarPost",
    "KPost",
    "Recovered",
]


def normalized_confusion_direction(n_spatial: int, seed: int) -> np.ndarray:
    """Fixed unit direction in Givens-angle space for epsilon scaling."""
    pairs = pair_list_for_n(n_spatial)
    rng = np.random.default_rng(seed)
    direction = rng.standard_normal(len(pairs))
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("Zero confusion direction.")
    return direction / norm


def confusion_unitary(n_spatial: int, epsilon: float, direction: np.ndarray) -> np.ndarray:
    pairs = pair_list_for_n(n_spatial)
    thetas = epsilon * direction
    return build_U_from_thetas(n_spatial, thetas, pairs)


def pre_confusion_metrics(
    ref: dict,
    parent_protocol: str,
    h_work: np.ndarray,
    psi_work: np.ndarray,
    energy_target: float,
) -> tuple[float, int]:
    n_spatial = ref["n_spatial"]
    basis = ref["basis_bitstrings"]
    a, b, c = STANDARD_ABC
    if parent_protocol == "seniority":
        from quartet_optimization_utils import rotate_state_to_orbital_frame
        from optimization_abc_utils import compute_spin_rdms_from_subspace_state, variance_restricted

        pairs = pair_list_for_n(n_spatial)
        m = len(pairs)
        seniority_x0 = np.zeros(m + 2, dtype=float)
        seniority_x0[m] = float(np.arccos(-2.0 / np.sqrt(6.0)))
        seniority_x0[m + 1] = np.pi / 4.0
        rotated = rotate_state_to_orbital_frame(
            psi_work, basis, np.eye(n_spatial, dtype=np.complex128), n_spatial
        )
        gamma_a, gamma_b, gamma_ab = compute_spin_rdms_from_subspace_state(
            rotated, basis, n_spatial
        )
        var_pre, *_ = variance_restricted(gamma_a, gamma_b, gamma_ab, seniority_x0, pairs)
        k_pre = int(
            k_coupled_for_seniority(
                h_work, basis, n_spatial, ref["n_qubits"], energy_target, a, b, c, tol=K_COUPLED_TOL
            )["Kcoupled"]
        )
        return float(var_pre), k_pre

    edges = quartet_topology_builders()[parent_protocol](n_spatial)
    var_pre = quartet_cost_for_u(
        psi_work, basis, np.eye(n_spatial, dtype=np.complex128), n_spatial, edges
    )
    k_pre = int(
        k_coupled_for_edges(
            h_work, basis, edges, n_spatial, energy_target, tol=K_COUPLED_TOL
        )["Kcoupled"]
    )
    return float(var_pre), k_pre


def evaluate_parent_at_epsilon(
    ref: dict,
    parent_protocol: str,
    parent: dict,
    epsilon: float,
    direction: np.ndarray,
    *,
    n_restarts: int,
    random_seed: int,
    maxfev: int,
    maxiter: int,
    parallel_restarts: bool,
) -> dict[str, object]:
    n_spatial = ref["n_spatial"]
    basis = ref["basis_bitstrings"]
    u_conf = confusion_unitary(n_spatial, epsilon, direction)
    h_work, psi_work = confuse_hamiltonian_and_state(
        parent["h_dense"], parent["psi"], u_conf, basis, n_spatial
    )
    energy_target = parent["energy"]
    var_pre, k_pre = pre_confusion_metrics(ref, parent_protocol, h_work, psi_work, energy_target)

    if parent_protocol == "seniority":
        opt = k_after_seniority_optimization(
            h_work,
            psi_work,
            basis,
            n_spatial,
            ref["n_qubits"],
            energy_target,
            n_restarts=n_restarts,
            random_seed=random_seed,
            maxfev=maxfev,
            maxiter=maxiter,
        )
    else:
        edges = quartet_topology_builders()[parent_protocol](n_spatial)
        from quartet_optimization_utils import optimize_fixed_edge_quartets

        best = optimize_fixed_edge_quartets(
            psi_work,
            basis,
            n_spatial,
            edges,
            n_restarts=n_restarts,
            random_seed=random_seed,
            include_zero_start=True,
            parallel=parallel_restarts,
            maxfev=maxfev,
            maxiter=maxiter,
        )
        h_opt, _ = confuse_hamiltonian_and_state(
            h_work,
            psi_work,
            best["u_spatial"],
            basis,
            n_spatial,
        )
        k_info = k_coupled_for_edges(
            h_opt, basis, edges, n_spatial, energy_target, tol=K_COUPLED_TOL
        )
        opt = {"variance": float(best["cost"]), **k_info}

    var_post = float(opt["variance"])
    k_post = int(opt["Kcoupled"])
    recovered = var_post < 1e-6 and k_post == 1
    return {
        "Epsilon": epsilon,
        "ParentProtocol": parent_protocol,
        "EnergyTarget": energy_target,
        "VarPre": var_pre,
        "KPre": k_pre,
        "VarPost": var_post,
        "KPost": k_post,
        "Recovered": int(recovered),
    }


def write_rows(path: Path, rows: list[dict[str, object]], *, write_header: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
        handle.flush()


def load_completed_epsilons(path: Path) -> set[float]:
    if not path.is_file():
        return set()
    with open(path, newline="", encoding="utf-8") as handle:
        return {float(row["Epsilon"]) for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epsilon-min", type=float, default=0.3)
    parser.add_argument("--epsilon-max", type=float, default=1.5)
    parser.add_argument("--epsilon-step", type=float, default=0.1)
    parser.add_argument("--direction-seed", type=int, default=PLANTED_SEED)
    parser.add_argument("--opt-seed", type=int, default=PLANTED_SEED)
    parser.add_argument("--n-restarts", type=int, default=3)
    parser.add_argument("--maxfev", type=int, default=3000)
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--workers", type=int, default=len(PARENT_PROTOCOL_NAMES))
    parser.add_argument("--no-parallel-restarts", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Skip epsilons already present in the output CSV.",
    )
    args = parser.parse_args()

    ref = load_h4_reference()
    direction = normalized_confusion_direction(ref["n_spatial"], args.direction_seed)
    parents = {name: build_parent_hamiltonian(ref, name) for name in PARENT_PROTOCOL_NAMES}

    epsilons = np.arange(args.epsilon_min, args.epsilon_max + 0.5 * args.epsilon_step, args.epsilon_step)
    epsilons = [float(round(float(e), 10)) for e in epsilons]

    completed = load_completed_epsilons(args.output) if args.append else set()
    if completed:
        epsilons = [e for e in epsilons if e not in completed]

    write_header = not args.output.is_file() or not args.append
    if not args.append and args.output.exists():
        args.output.unlink()
        write_header = True

    print(f"Writing {args.output}", flush=True)
    print(f"Direction seed={args.direction_seed}, opt seed={args.opt_seed}", flush=True)
    print(f"Epsilons: {epsilons}", flush=True)

    for index, epsilon in enumerate(epsilons):
        rows: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    evaluate_parent_at_epsilon,
                    ref,
                    parent_protocol,
                    parents[parent_protocol],
                    epsilon,
                    direction,
                    n_restarts=args.n_restarts,
                    random_seed=args.opt_seed,
                    maxfev=args.maxfev,
                    maxiter=args.maxiter,
                    parallel_restarts=not args.no_parallel_restarts,
                ): parent_protocol
                for parent_protocol in PARENT_PROTOCOL_NAMES
            }
            for future in as_completed(futures):
                parent_protocol = futures[future]
                row = future.result()
                rows.append(row)
                print(
                    f"eps={epsilon:.1f} {parent_protocol:14} "
                    f"pre K={row['KPre']} post K={row['KPost']} "
                    f"var={row['VarPost']:.4f} recovered={row['Recovered']}",
                    flush=True,
                )

        rows.sort(key=lambda row: PARENT_PROTOCOL_NAMES.index(str(row["ParentProtocol"])))
        write_rows(args.output, rows, write_header=(write_header and index == 0))

    print(f"Done. Results in {args.output}", flush=True)


if __name__ == "__main__":
    main()
