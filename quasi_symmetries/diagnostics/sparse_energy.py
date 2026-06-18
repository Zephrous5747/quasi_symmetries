"""Lazy sparse-matrix energy-sector diagnostics (N2-scale systems)."""

from __future__ import annotations

import os
import time
from typing import Any, Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from quasi_symmetries.config import SPARSE_ENERGY_MAX_WORKERS, STATES_PER_SECTOR


def _default_workers() -> int:
    return SPARSE_ENERGY_MAX_WORKERS


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


def _sector_eigenpairs(blk: np.ndarray, states_per_sector: int | None) -> tuple[np.ndarray, np.ndarray]:
    dim = blk.shape[0]
    k = dim if states_per_sector is None else min(int(states_per_sector), dim)
    if k <= dim - 2 and k < dim:
        evals, evecs = spla.eigsh(blk, k=k, which="SA")
        order = np.argsort(evals)
        return np.asarray(evals[order], dtype=float), np.asarray(evecs[:, order], dtype=np.complex128)
    evals, evecs = np.linalg.eigh(blk)
    return np.asarray(evals, dtype=float), np.asarray(evecs, dtype=np.complex128)


def _diagonalize_one_sector(
    h_op: Any,
    key: object,
    idxs: list[int],
    dim: int,
    *,
    states_per_sector: int | None = STATES_PER_SECTOR,
) -> tuple[object, dict[str, Any]]:
    blk = h_op.sector_block_dense(idxs)
    evals, evecs = _sector_eigenpairs(blk, states_per_sector)
    idxs_arr = np.asarray(idxs, dtype=int)
    evecs_full = []
    for j in range(evecs.shape[1]):
        v = np.zeros(dim, dtype=np.complex128)
        v[idxs_arr] = evecs[:, j]
        evecs_full.append(v)
    return key, {"idxs": idxs, "evals": evals, "evecs_full": evecs_full}


def _identity_sector_worker(args: tuple) -> tuple[object, dict[str, Any]]:
    key, idxs, dim, states_per_sector = args
    global _IDENTITY_H_SUB_PAYLOAD
    h_data, h_indices, h_indptr, shape = _IDENTITY_H_SUB_PAYLOAD
    h_sub = sp.csc_matrix((h_data, h_indices, h_indptr), shape=shape)
    h_op = SparseSubspaceHamiltonian(h_sub)
    return _diagonalize_one_sector(
        h_op,
        key,
        idxs,
        dim,
        states_per_sector=states_per_sector,
    )


_IDENTITY_H_SUB_PAYLOAD: tuple | None = None


def _init_identity_worker(payload: tuple) -> None:
    global _IDENTITY_H_SUB_PAYLOAD
    _IDENTITY_H_SUB_PAYLOAD = payload


def _can_parallelize_identity(h_op: Any) -> bool:
    return isinstance(h_op, SparseSubspaceHamiltonian)


def _diagonalize_sector_blocks_lazy(
    h_op: Any,
    sectors_dict: dict,
    *,
    max_workers: int | None = None,
    states_per_sector: int | None = STATES_PER_SECTOR,
) -> dict:
    dim = h_op.shape[0]
    items = list(sectors_dict.items())
    if not items:
        return {}

    workers = max_workers if max_workers is not None else _default_workers()
    if workers > 1 and _can_parallelize_identity(h_op):
        from concurrent.futures import ProcessPoolExecutor

        h_sub = h_op.h_sub
        payload = (h_sub.data, h_sub.indices, h_sub.indptr, h_sub.shape)
        tasks = [(key, idxs, dim, states_per_sector) for key, idxs in items]
        sector_data: dict = {}
        with ProcessPoolExecutor(
            max_workers=min(workers, len(items)),
            initializer=_init_identity_worker,
            initargs=(payload,),
        ) as executor:
            for key, data in executor.map(_identity_sector_worker, tasks):
                sector_data[key] = data
        return sector_data

    sector_data = {}
    for key, idxs in items:
        _, data = _diagonalize_one_sector(
            h_op,
            key,
            idxs,
            dim,
            states_per_sector=states_per_sector,
        )
        sector_data[key] = data
    return sector_data


_ROTATED_H_PAYLOAD: tuple | None = None


def _init_rotated_h_worker(payload: tuple) -> None:
    global _ROTATED_H_PAYLOAD
    _ROTATED_H_PAYLOAD = payload


def _rotated_h_column_worker(col_index: int) -> tuple[int, np.ndarray]:
    global _ROTATED_H_PAYLOAD
    if _ROTATED_H_PAYLOAD is None:
        raise RuntimeError("Rotated-H worker payload is not initialized.")
    h_data, h_indices, h_indptr, shape, u_spatial, basis_bitstrings, n_spatial = _ROTATED_H_PAYLOAD
    from quasi_symmetries.diagnostics.n2_action import OrbitalRotationAction, RotatedHamiltonian

    h_sub = sp.csc_matrix((h_data, h_indices, h_indptr), shape=shape)
    action = OrbitalRotationAction(
        np.asarray(u_spatial, dtype=np.complex128),
        list(basis_bitstrings),
        int(n_spatial),
    )
    rot_h = RotatedHamiltonian(h_sub, action)
    dim = int(shape[0])
    vector = np.zeros(dim, dtype=np.complex128)
    vector[col_index] = 1.0
    return col_index, rot_h.dot(vector)


def _rotated_h_column_batch_worker(col_indices: list[int]) -> list[tuple[int, np.ndarray]]:
    return [_rotated_h_column_worker(col_index) for col_index in col_indices]


def build_rotated_h_sub_csc(
    h_sub: sp.spmatrix,
    action: Any,
    *,
    max_workers: int | None = None,
    profile: bool = False,
) -> sp.csc_matrix:
    """Build H_rot = R^dagger H R as CSC by applying the rotation once per column."""
    if not hasattr(action, "u_spatial"):
        raise ValueError(
            "build_rotated_h_sub_csc requires an OrbitalRotationAction with "
            "u_spatial, basis_bitstrings, and n_spatial attributes."
        )

    if not isinstance(h_sub, sp.csc_matrix):
        h_sub = h_sub.tocsc()
    dim = h_sub.shape[0]
    workers = max_workers if max_workers is not None else _default_workers()
    u_spatial = action.u_spatial
    basis_bitstrings = action.basis_bitstrings
    n_spatial = action.n_spatial

    t0 = time.perf_counter()
    payload = (
        h_sub.data,
        h_sub.indices,
        h_sub.indptr,
        h_sub.shape,
        np.asarray(u_spatial, dtype=np.complex128),
        [int(b) for b in basis_bitstrings],
        int(n_spatial),
    )
    columns: list[np.ndarray | None] = [None] * dim
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor

        chunk_size = max(1, (dim + workers - 1) // workers)
        batches = [
            list(range(start, min(start + chunk_size, dim)))
            for start in range(0, dim, chunk_size)
        ]
        with ProcessPoolExecutor(
            max_workers=min(workers, len(batches)),
            initializer=_init_rotated_h_worker,
            initargs=(payload,),
        ) as executor:
            for batch in executor.map(_rotated_h_column_batch_worker, batches):
                for col_index, column in batch:
                    columns[col_index] = column
    else:
        global _ROTATED_H_PAYLOAD
        _ROTATED_H_PAYLOAD = payload
        for col_index in range(dim):
            _, column = _rotated_h_column_worker(col_index)
            columns[col_index] = column
        _ROTATED_H_PAYLOAD = None

    data: list[complex] = []
    rows: list[int] = []
    cols: list[int] = []
    for col_index, column in enumerate(columns):
        if column is None:
            continue
        for row_index, value in enumerate(column):
            if value != 0:
                rows.append(row_index)
                cols.append(col_index)
                data.append(value)
    h_rot = sp.csc_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128)
    h_rot = 0.5 * (h_rot + h_rot.conj().T).tocsc()
    elapsed = time.perf_counter() - t0
    if profile:
        print(
            f"  [profile] build_rotated_h_sub_csc dim={dim} nnz={h_sub.nnz}->{h_rot.nnz} "
            f"in {elapsed:.1f}s",
            flush=True,
        )
    return h_rot


def _all_sector_eigenpair_candidates(
    sector_data,
) -> list[tuple[float, object, np.ndarray, int]]:
    candidates: list[tuple[float, object, np.ndarray, int]] = []
    for key, data in sector_data.items():
        for block_index, (energy, vector) in enumerate(
            zip(data["evals"], data["evecs_full"])
        ):
            candidates.append((float(energy), key, vector, int(block_index)))
    candidates.sort(key=lambda item: item[0])
    return candidates


def decoupled_energy_from_sector_data(
    sector_data: dict,
) -> tuple[float, object, int]:
    best_E = None
    best_key = None
    best_dim = 0
    for key, data in sector_data.items():
        e0 = float(data["evals"][0])
        dim = len(data["idxs"])
        if best_E is None or e0 < best_E:
            best_E = e0
            best_key = key
            best_dim = dim
    return float(best_E), best_key, best_dim


def decoupled_energy_lazy(
    h_op: SparseSubspaceHamiltonian | Any,
    sectors_dict: dict,
    *,
    sector_data: dict | None = None,
    max_workers: int | None = None,
    states_per_sector: int | None = STATES_PER_SECTOR,
) -> tuple[float, object, int]:
    """Decoupled minimum energy across symmetry sectors (block-wise)."""
    if sector_data is not None:
        return decoupled_energy_from_sector_data(sector_data)

    best_E = None
    best_key = None
    best_dim = 0

    for key, idxs in sectors_dict.items():
        blk = h_op.sector_block_dense(idxs)
        e0 = float(_sector_eigenpairs(blk, states_per_sector)[0][0])
        if best_E is None or e0 < best_E:
            best_E = e0
            best_key = key
            best_dim = len(idxs)

    return float(best_E), best_key, best_dim


def coupled_energy_lazy(
    h_op: Any,
    sectors_dict: dict,
    E_exact: float | None = None,
    tol: float = 1e-8,
    max_total_vectors: int | None = None,
    coupling_tol: float = 1e-12,
    energy_change_tol: float = 1e-12,
    *,
    sector_data: dict | None = None,
    max_workers: int | None = None,
    states_per_sector: int | None = STATES_PER_SECTOR,
) -> tuple[float, int, bool, list]:
    """Coupled-energy test using lazy sector block diagonalization."""
    if sector_data is None:
        sector_data = _diagonalize_sector_blocks_lazy(
            h_op,
            sectors_dict,
            max_workers=max_workers,
            states_per_sector=states_per_sector,
        )
    candidates = _all_sector_eigenpair_candidates(sector_data)
    if not candidates:
        return None, 0, False, []

    if max_total_vectors is None:
        max_total_vectors = len(candidates)

    chosen_vecs: list[np.ndarray] = []
    chosen_keys: list[tuple[object, int]] = []
    chosen_indices: set[int] = set()
    e_proj: float | None = None
    converged = False

    while True:
        added_this_pass = False
        for index, (_energy, key, vec, block_index) in enumerate(candidates):
            if index in chosen_indices:
                continue
            if len(chosen_vecs) >= max_total_vectors:
                break

            if chosen_vecs:
                if _max_coupling_to_span_lazy(h_op, vec, chosen_vecs) <= coupling_tol:
                    continue
                e_new = _projected_ground_energy_lazy(h_op, [*chosen_vecs, vec])
                if e_proj is not None:
                    if E_exact is not None:
                        if abs(e_new - E_exact) >= abs(e_proj - E_exact) - energy_change_tol:
                            continue
                    elif abs(e_new - e_proj) <= energy_change_tol:
                        continue
            else:
                e_new = _projected_ground_energy_lazy(h_op, [vec])

            chosen_indices.add(index)
            chosen_vecs.append(vec)
            chosen_keys.append((key, block_index))
            e_proj = e_new
            added_this_pass = True

            if E_exact is not None and abs(e_proj - E_exact) <= tol:
                converged = True
                break

        if converged:
            break
        if not added_this_pass or len(chosen_vecs) >= max_total_vectors:
            break

    if E_exact is not None and e_proj is not None and abs(e_proj - E_exact) <= tol:
        converged = True

    return e_proj, len(chosen_vecs), converged, chosen_keys


def _projected_ground_energy_lazy(h_op: Any, vecs: list[np.ndarray]) -> float:
    k = len(vecs)
    h_proj = np.zeros((k, k), dtype=np.complex128)
    for i in range(k):
        h_v_i = h_op.dot(vecs[i])
        for j in range(k):
            h_proj[i, j] = np.vdot(vecs[j], h_v_i)
    h_proj = 0.5 * (h_proj + h_proj.conj().T)
    return float(np.linalg.eigvalsh(h_proj)[0])


def _max_coupling_to_span_lazy(
    h_op: Any,
    candidate: np.ndarray,
    chosen_vecs: list[np.ndarray],
) -> float:
    if not chosen_vecs:
        return float("inf")
    h_cand = h_op.dot(candidate)
    return max(float(abs(np.vdot(chosen, h_cand))) for chosen in chosen_vecs)


def energy_sector_diagnostics_sparse(
    h_op: Any,
    sectors: dict,
    energy_fci: float,
    *,
    tol: float = 1e-3,
    lazy: bool = False,
    max_workers: int | None = None,
    states_per_sector: int | None = STATES_PER_SECTOR,
    profile: bool = False,
) -> dict[str, float | int | bool]:
    """Energy indicators from lazy sector blocks."""
    del lazy  # interface compatibility; all paths are lazy here
    timings: dict[str, float] = {}
    workers = max_workers if max_workers is not None else _default_workers()

    t0 = time.perf_counter()
    sector_data = _diagonalize_sector_blocks_lazy(
        h_op,
        sectors,
        max_workers=workers,
        states_per_sector=states_per_sector,
    )
    timings["diagonalize_seconds"] = time.perf_counter() - t0
    if profile:
        print(
            f"  [profile] diagonalized {len(sector_data)} sectors in "
            f"{timings['diagonalize_seconds']:.1f}s",
            flush=True,
        )

    t0 = time.perf_counter()
    e_dec, _, _ = decoupled_energy_from_sector_data(sector_data)
    timings["edec_seconds"] = time.perf_counter() - t0

    if abs(e_dec - energy_fci) <= tol:
        if profile:
            print(
                f"  [profile] early K=1: |Edec-FCI|={abs(e_dec - energy_fci):.3e}",
                flush=True,
            )
        result = {
            "Edec": float(e_dec),
            "Ecoupled": float(e_dec),
            "Kcoupled": 1,
            "Coupled_Converged": True,
        }
        if profile:
            timings["coupled_seconds"] = 0.0
            timings["total_seconds"] = sum(
                value for key, value in timings.items() if key.endswith("_seconds")
            )
            result["_profile"] = timings  # type: ignore[assignment]
        return result

    t0 = time.perf_counter()
    e_coupled, k_coupled, converged, _ = coupled_energy_lazy(
        h_op,
        sectors,
        E_exact=energy_fci,
        tol=tol,
        sector_data=sector_data,
    )
    timings["coupled_seconds"] = time.perf_counter() - t0
    if profile:
        print(
            f"  [profile] coupled K={k_coupled} converged={converged} in "
            f"{timings['coupled_seconds']:.1f}s",
            flush=True,
        )
    if not converged and profile:
        print(
            f"  [warn] K did not converge within tol={tol}; "
            f"consider increasing STATES_PER_SECTOR (currently {states_per_sector})",
            flush=True,
        )

    result = {
        "Edec": float(e_dec),
        "Ecoupled": float(e_coupled),
        "Kcoupled": int(k_coupled),
        "Coupled_Converged": bool(converged),
    }
    if profile:
        timings["total_seconds"] = sum(
            value for key, value in timings.items() if key.endswith("_seconds")
        )
        result["_profile"] = timings  # type: ignore[assignment]
    return result
