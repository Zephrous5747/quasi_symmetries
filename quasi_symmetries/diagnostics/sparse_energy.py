"""Lazy sparse-matrix energy-sector diagnostics (N2-scale systems)."""

from __future__ import annotations

import os
import time
from typing import Any, Iterable

import numpy as np


def _default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 1)))


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


def _diagonalize_one_sector(
    h_op: Any,
    key: object,
    idxs: list[int],
    dim: int,
) -> tuple[object, dict[str, Any]]:
    blk = h_op.sector_block_dense(idxs)
    evals, evecs = np.linalg.eigh(blk)
    idxs_arr = np.asarray(idxs, dtype=int)
    evecs_full = []
    for j in range(evecs.shape[1]):
        v = np.zeros(dim, dtype=np.complex128)
        v[idxs_arr] = evecs[:, j]
        evecs_full.append(v)
    return key, {"idxs": idxs, "evals": evals, "evecs_full": evecs_full}


def _identity_sector_worker(args: tuple) -> tuple[object, dict[str, Any]]:
    key, idxs, dim = args
    import scipy.sparse as sp

    global _IDENTITY_H_SUB_PAYLOAD
    h_data, h_indices, h_indptr, shape = _IDENTITY_H_SUB_PAYLOAD
    h_sub = sp.csc_matrix((h_data, h_indices, h_indptr), shape=shape)
    h_op = SparseSubspaceHamiltonian(h_sub)
    return _diagonalize_one_sector(h_op, key, idxs, dim)


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
        tasks = [(key, idxs, dim) for key, idxs in items]
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
        _, data = _diagonalize_one_sector(h_op, key, idxs, dim)
        sector_data[key] = data
    return sector_data


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
) -> tuple[float, object, int]:
    """Decoupled minimum energy across symmetry sectors (block-wise)."""
    if sector_data is not None:
        return decoupled_energy_from_sector_data(sector_data)

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
) -> tuple[float, int, bool, list]:
    """Coupled-energy test using lazy sector block diagonalization."""
    if sector_data is None:
        sector_data = _diagonalize_sector_blocks_lazy(
            h_op,
            sectors_dict,
            max_workers=max_workers,
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
    profile: bool = False,
) -> dict[str, float | int | bool]:
    """Energy indicators from lazy sector blocks."""
    del lazy  # interface compatibility; all paths are lazy here
    timings: dict[str, float] = {}
    workers = max_workers if max_workers is not None else _default_workers()

    t0 = time.perf_counter()
    sector_data = _diagonalize_sector_blocks_lazy(h_op, sectors, max_workers=workers)
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
