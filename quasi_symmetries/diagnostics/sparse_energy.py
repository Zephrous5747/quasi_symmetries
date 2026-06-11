"""Lazy sparse-matrix energy-sector diagnostics (N2-scale systems)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


class SparseSubspaceHamiltonian:
    """Thin wrapper around a scipy sparse fixed-N Hamiltonian."""

    def __init__(self, h_sub) -> None:
        self.h_sub = h_sub
        self.shape = h_sub.shape

    def dot(self, vector: np.ndarray) -> np.ndarray:
        return self.h_sub.dot(vector)

    def sector_block_dense(self, idxs: Iterable[int]) -> np.ndarray:
        idxs_arr = np.asarray(list(idxs), dtype=np.int64)
        block = self.h_sub[np.ix_(idxs_arr, idxs_arr)].toarray().astype(np.complex128)
        return 0.5 * (block + block.conj().T)


def decoupled_energy_lazy(
    h_op: SparseSubspaceHamiltonian | Any,
    sectors_dict: dict,
) -> tuple[float, object, int]:
    """Decoupled minimum energy across symmetry sectors (block-wise)."""
    best_E = None
    best_key = None
    best_dim = 0

    for key, idxs in sectors_dict.items():
        blk = h_op.sector_block_dense(idxs)
        e0 = float(np.linalg.eigvalsh(blk)[0])
        if best_E is None or e0 < best_E:
            best_E = e0
            best_key = key
            best_dim = len(idxs)

    return float(best_E), best_key, best_dim


def _diagonalize_sector_blocks_lazy(
    h_op: Any,
    sectors_dict: dict,
) -> dict:
    sector_data = {}
    dim = h_op.shape[0]
    for key, idxs in sectors_dict.items():
        blk = h_op.sector_block_dense(idxs)
        evals, evecs = np.linalg.eigh(blk)
        evecs_full = []
        for j in range(evecs.shape[1]):
            v = np.zeros(dim, dtype=np.complex128)
            v[np.asarray(idxs, dtype=int)] = evecs[:, j]
            evecs_full.append(v)
        sector_data[key] = {"idxs": idxs, "evals": evals, "evecs_full": evecs_full}
    return sector_data


def coupled_energy_lazy(
    h_op: Any,
    sectors_dict: dict,
    E_exact: float | None = None,
    tol: float = 1e-8,
    max_total_vectors: int | None = None,
) -> tuple[float, int, bool, list]:
    """Coupled-energy test using lazy sector block diagonalization."""
    sector_data = _diagonalize_sector_blocks_lazy(h_op, sectors_dict)
    dim = h_op.shape[0]

    candidates = []
    for key, data in sector_data.items():
        for e, v in zip(data["evals"], data["evecs_full"]):
            candidates.append((float(e), key, v))

    candidates.sort(key=lambda t: t[0])

    if max_total_vectors is None:
        max_total_vectors = len(candidates)

    chosen_vecs = []
    chosen_keys = []
    E_proj = None
    K_final = 0
    converged = False

    for K in range(1, min(max_total_vectors, len(candidates)) + 1):
        e, key, v = candidates[K - 1]
        chosen_vecs.append(v)
        chosen_keys.append(key)

        V = np.column_stack(chosen_vecs)
        H_proj = np.zeros((K, K), dtype=np.complex128)
        for i in range(K):
            Hv_i = h_op.dot(chosen_vecs[i])
            for j in range(K):
                H_proj[i, j] = np.vdot(chosen_vecs[j], Hv_i)
        H_proj = 0.5 * (H_proj + H_proj.conj().T)

        evals_proj = np.linalg.eigvalsh(H_proj)
        E_proj = float(evals_proj[0])
        K_final = K

        if E_exact is not None and abs(E_proj - E_exact) <= tol:
            converged = True
            break

    return E_proj, K_final, converged, chosen_keys[:K_final]


def energy_sector_diagnostics_sparse(
    h_op: Any,
    sectors: dict,
    energy_fci: float,
    *,
    tol: float = 1e-3,
    lazy: bool = False,
) -> dict[str, float | int | bool]:
    """Energy indicators from lazy sector blocks."""
    del lazy  # interface compatibility; all paths are lazy here
    e_dec, _, _ = decoupled_energy_lazy(h_op, sectors)
    e_coupled, k_coupled, converged, _ = coupled_energy_lazy(
        h_op,
        sectors,
        E_exact=energy_fci,
        tol=tol,
    )
    return {
        "Edec": float(e_dec),
        "Ecoupled": float(e_coupled),
        "Kcoupled": int(k_coupled),
        "Coupled_Converged": bool(converged),
    }
