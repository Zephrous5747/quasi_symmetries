"""MPS-native canonical parity measurements for Block2 references."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from quasi_symmetries.reference.dmrg import DMRGBackendError, DMRGReference


class DMRGParityError(RuntimeError):
    """Raised when canonical parity cannot be measured from the MPS."""


def _add_number_product_sz(builder: Any, orbitals: list[int], coef: float) -> None:
    """Add coef * prod_i (cd_i + CD_i) using Block2 SZ/U(1) operator strings."""
    if not orbitals:
        return
    for picks in product(("cd", "CD"), repeat=len(orbitals)):
        indices: list[int] = []
        for orbital in orbitals:
            indices.extend([int(orbital), int(orbital)])
        builder.add_term("".join(picks), indices, float(coef))


def _parity_terms(orbitals: Iterable[int]) -> tuple[float, list[tuple[list[int], float]]]:
    """Expand prod_i (1 - 4 E_ii + 2 E_ii E_ii).

    For a spatial orbital with N_i in {0, 1, 2}, (-1)^N_i = 1 - 4 N_i + 2 N_i^2.
    Block2 evaluates the spin-free E_ii products directly as MPS/MPO expectations.
    """
    constant = 0.0
    terms: list[tuple[list[int], float]] = []
    choices = [((), 1.0), ((0,), -4.0), ((0, 0), 2.0)]
    orbital_list = [int(orbital) for orbital in orbitals]
    for local_choices in product(choices, repeat=len(orbital_list)):
        factors: list[int] = []
        coef = 1.0
        for orbital, (pattern, local_coef) in zip(orbital_list, local_choices):
            coef *= local_coef
            factors.extend(orbital for _ in pattern)
        if not factors:
            constant += coef
        else:
            terms.append((factors, coef))
    return constant, terms


def _expectation_with_constant(driver: Any, ket: Any, constant: float, terms: list[tuple[list[int], float]]) -> float:
    impo = driver.get_identity_mpo()
    norm = float(np.real(driver.expectation(ket, impo, ket)))
    if norm == 0:
        raise DMRGParityError("Cannot measure parity on a zero-norm Block2 MPS.")
    if not terms:
        return float(constant)
    builder = driver.expr_builder()
    for orbitals, coef in terms:
        _add_number_product_sz(builder, orbitals, coef)
    expr = builder.finalize(adjust_order=True, fermionic_ops="cdCD")
    try:
        from pyblock2.driver.core import MPOAlgorithmTypes

        mpo = driver.get_mpo(expr, algo_type=MPOAlgorithmTypes.FastBipartite, iprint=0)
    except ImportError:
        mpo = driver.get_mpo(expr, iprint=0)
    value = float(np.real(driver.expectation(ket, mpo, ket))) / norm
    return float(np.clip(constant + value, -1.0, 1.0))


def parity_expectation(driver: Any, ket: Any, orbitals: Iterable[int]) -> float:
    constant, terms = _parity_terms(orbitals)
    return _expectation_with_constant(driver, ket, constant, terms)


def _load_cached_matrix(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return np.asarray(payload["variance_canonical"], dtype=float)


def _save_cached_matrix(path: Path, matrix: np.ndarray, expectations: dict[str, float]) -> None:
    path.write_text(
        json.dumps(
            {
                "variance_canonical": np.asarray(matrix, dtype=float).tolist(),
                "expectations": expectations,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def dmrg_parity_variance_matrix(reference: DMRGReference, *, use_cache: bool = True) -> np.ndarray:
    cache_path = reference.parity_json_path
    if use_cache and cache_path.is_file():
        return _load_cached_matrix(cache_path)
    if reference.driver is None or reference.mps is None:
        raise DMRGBackendError(
            "DMRG metadata was loaded from disk without a live Block2 driver/MPS. "
            "Re-run with generation enabled, or provide cached canonical_parity_expectations.json."
        )

    n_spatial = int(reference.n_spatial)
    matrix = np.full((n_spatial, n_spatial), np.nan, dtype=float)
    expectations: dict[str, float] = {}
    for orbital in range(n_spatial):
        expval = parity_expectation(reference.driver, reference.mps, [orbital])
        expectations[f"s_{orbital}"] = expval
        matrix[orbital, orbital] = max(0.0, 1.0 - expval**2)

    for p in range(n_spatial):
        for q in range(p + 1, n_spatial):
            expval = parity_expectation(reference.driver, reference.mps, [p, q])
            expectations[f"s_{p}_{q}"] = expval
            matrix[p, q] = max(0.0, 1.0 - expval**2)

    _save_cached_matrix(cache_path, matrix, expectations)
    return matrix
