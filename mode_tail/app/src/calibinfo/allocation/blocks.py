"""Per-light Fisher blocks for the allocation experiment (prediction side only).

Structure (CI04 whitened per-light form, V1b identity): for analysis light k,

    A_k = diag(sqrt(w_k) * s_k)          (P x P, diagonal)
    B_k = sqrt(w_k)[:, None] * B_phi[k]  (P x 3)
    u_k = A_k^T B_k = (w_k * s_k)[:, None] * B_phi[k]

    DeltaF(Lam) = sum_k A_k^T A_k - sum_k u_k K_k u_k^T
                = diag(Finf) - sum_k u_k K_k u_k^T,
    K_k = (M0_k + Lam_k)^{-1},  M0_k = B_k^T B_k = B_phi[k]^T diag(w_k) B_phi[k].

Lam_k is the per-light precision block (3x3, base Lam0_k = Sigma_phi^{-1}; the
regime multiplies a selected light's block). Every routine here is prediction-side:
inputs are the whitened linearization assets and precision blocks only.

Numerical discipline: no raw inv/solve on possibly-singular matrices -- symmetric
inverses go through `sym_inv` (eigh with the house relative tolerance).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def sym_inv(M, tol_rel=1e-12):
    """Symmetric pseudo-inverse via eigh (never np.linalg.inv/solve)."""
    w, V = np.linalg.eigh(M)
    tol = max(float(w.max()), 1.0) * tol_rel
    winv = np.where(w > tol, 1.0 / np.where(w > tol, w, 1.0), 0.0)
    return (V * winv) @ V.T


class LightBlocks:
    """Per-light prediction-side blocks shared by all selection policies."""

    def __init__(self, u, M0, lam0, finf, active):
        self.u = np.asarray(u, float)          # (L, P, 3)
        self.M0 = np.asarray(M0, float)        # (L, 3, 3)
        self.lam0 = np.asarray(lam0, float)    # (L, 3, 3) base precision (update direction)
        self.finf = np.asarray(finf, float)    # (P,)
        self.active = np.asarray(active, bool)  # (L,)
        self.L = self.u.shape[0]
        self.P = self.u.shape[1]

    def assemble(self, lam):
        """DeltaF for per-light precision blocks lam (L, 3, 3)."""
        DF = np.diag(self.finf)
        for k in range(self.L):
            if not self.active[k]:
                continue
            K = sym_inv(self.M0[k] + lam[k])
            DF -= (self.u[k] @ K) @ self.u[k].T
        return DF

    def rank3_update(self, DF, k, lam_old, lam_new):
        """DeltaF + u_k (K_old - K_new) u_k^T for a precision change on light k."""
        K_old = sym_inv(self.M0[k] + lam_old)
        K_new = sym_inv(self.M0[k] + lam_new)
        return DF + (self.u[k] @ (K_old - K_new)) @ self.u[k].T


def objective_of(DF, kind, tol_rel=1e-12):
    """Scalar Fisher objective of a candidate DeltaF.

    e: lambda_min (plain smallest eigenvalue)
    a: trace(pinv(DF)), Moore-Penrose with positive subspace eig > tol_rel * max
    d: logdet of the positive subspace (sum of log eigenvalues above the threshold)
    """
    ev = np.linalg.eigvalsh(DF)
    if kind == "e":
        return float(ev[0])
    tol = tol_rel * max(float(ev[-1]), 1.0)
    pos = ev > tol
    if kind == "a":
        return float(np.sum(1.0 / ev[pos]))
    if kind == "d":
        return float(np.sum(np.log(ev[pos])))
    raise ValueError(f"unknown objective kind: {kind}")


def mode_gains(u, M0, lam, lam0, active, modes):
    """g[k, j] = a_j^T (dDeltaF/dlambda_k) a_j with update direction Lam0_k.

    dDeltaF/dlambda_k = u_k K_k Lam0_k K_k u_k^T (PSD), so
    g[k, j] = (K_k u_k^T a_j)^T Lam0_k (K_k u_k^T a_j) >= 0.
    Inactive lights have u_k = 0 -> g = 0. u/M0/lam/lam0: (L, ...); modes: (P, m).
    Returns (L, m).
    """
    L, m = u.shape[0], modes.shape[1]
    G = np.zeros((L, m))
    for k in range(L):
        if not active[k]:
            continue
        K = sym_inv(M0[k] + lam[k])
        V = modes.T @ u[k]                     # (m, 3): rows u_k^T a_j
        Kv = K @ V.T                           # (3, m)
        L0 = lam0[k]
        for j in range(m):
            G[k, j] = float(Kv[:, j] @ (L0 @ Kv[:, j]))
    return G
