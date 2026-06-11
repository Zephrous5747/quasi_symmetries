"""Fixed-N subspace dimension and restriction."""

from __future__ import annotations

import math

import numpy as np

from quasi_symmetries.config import DENSE_SUBSPACE_MAX

def fixed_n_subspace_dim(n_spatial: int, n_electrons: int) -> int:
    return math.comb(2 * n_spatial, n_electrons)
def use_dense_subspace_ops(n_spatial: int, n_electrons: int) -> bool:
    return fixed_n_subspace_dim(n_spatial, n_electrons) <= DENSE_SUBSPACE_MAX
def restrict_operator_to_subspace(op_mat, basis_bitstrings):
    idx = np.asarray(basis_bitstrings, dtype=int)
    return op_mat[idx, :][:, idx].tocsc()
