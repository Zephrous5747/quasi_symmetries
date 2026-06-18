"""Parity variance optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from quasi_symmetries.config import (
    ANGLE_INIT_SCALE,
    MAXITER,
    N_RESTARTS,
    OPT_METHOD,
    RANDOM_SEED,
)
from quasi_symmetries.optimization.rotations import build_U_from_thetas, pair_list_for_n


@dataclass
class OptLog:
    V: list
    nOmega: list
    x: list
def variance_restricted(gamma_a, gamma_b, Gamma_ab, x_params, pairs):
    n = gamma_a.shape[0]
    m = len(pairs)

    # Unpack orbital rotations and operator angles
    thetas = x_params[:m]
    phi1, phi2 = x_params[m], x_params[m+1]

    # Spherical parameterization for sqrt(a^2 + b^2 + c^2) = 1
    a = np.sin(phi1) * np.cos(phi2)
    b = np.sin(phi1) * np.sin(phi2)
    c = np.cos(phi1)

    U = build_U_from_thetas(n, thetas, pairs)
    Ua = U.T @ gamma_a @ U
    Ub = U.T @ gamma_b @ U

    exp_vals = np.zeros(n, dtype=float)
    V_total = 0.0

    for i in range(n):
        u = U[:, i]
        G_i = np.einsum("p,q,r,s,pqrs->", u, u, u, u, Gamma_ab, optimize=True).real
        N_a = Ua[i, i].real
        N_b = Ub[i, i].real

        # < \tilde{\Omega}_i >
        exp_omega = a * N_a + b * N_b + c * G_i
        exp_vals[i] = exp_omega

        # < \tilde{\Omega}_i^2 >
        exp_omega_sq = a**2 * N_a + b**2 * N_b + (2*a*b + 2*a*c + 2*b*c + c**2) * G_i

        # Exact Variance: <O^2> - <O>^2
        V_total += float(exp_omega_sq - exp_omega**2)

    return V_total, exp_vals, U, a, b, c
def optimize_variance_restricted(gamma_a, gamma_b, Gamma_ab, pairs=None):
    np.random.seed(RANDOM_SEED)
    n = gamma_a.shape[0]
    pairs = pair_list_for_n(n) if pairs is None else list(pairs)
    m = len(pairs)
    num_params = m + 2 # +2 for phi1, phi2

    def obj(x):
        V, _, _, _, _, _ = variance_restricted(gamma_a, gamma_b, Gamma_ab, x, pairs)
        return V

    best = None
    for r in range(N_RESTARTS):
        x0 = np.zeros(num_params)
        if r == 0:
            # Initialize close to standard seniority: a=1, b=1, c=-2 (Normalized by sqrt(6))
            x0[m] = np.arccos(-2.0 / np.sqrt(6.0)) # phi1 for c
            x0[m+1] = np.pi / 4.0                  # phi2 for a, b
        else:
            x0[:m] = ANGLE_INIT_SCALE * np.random.randn(m)
            x0[m] = np.random.uniform(0, np.pi)
            x0[m+1] = np.random.uniform(0, 2*np.pi)

        log = OptLog(V=[], nOmega=[], x=[])

        def callback(xk):
            V, nO, _, _, _, _ = variance_restricted(gamma_a, gamma_b, Gamma_ab, xk, pairs)
            log.V.append(V); log.nOmega.append(nO); log.x.append(np.array(xk, copy=True))

        if OPT_METHOD.upper() == "POWELL":
            res = minimize(obj, x0=x0, method="Powell", options={"maxiter": MAXITER, "disp": False})
            callback(res.x)
        else:
            res = minimize(obj, x0=x0, method=OPT_METHOD, options={"maxiter": MAXITER, "disp": False}, callback=callback)

        V_fin = obj(res.x)
        if best is None or V_fin < best["V"]:
            best = {"res": res, "log": log, "V": V_fin, "pairs": pairs, "a": np.sin(res.x[m])*np.cos(res.x[m+1]), "b": np.sin(res.x[m])*np.sin(res.x[m+1]), "c": np.cos(res.x[m])}

    return best
